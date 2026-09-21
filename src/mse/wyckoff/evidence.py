"""证据聚合引擎 (Spec §2.4)。

把一组 Evidence 聚合成 (probability, confidence, reason):

    probability = 加权几何平均(memberships, weights)   —— "是该事件"的可能性
                = exp( Σ wᵢ·ln(mᵢ) / Σ wᵢ )
      * 几何平均 → 任一关键证据接近 0 会显著拉低总分 (符合"缺一不可")。
      * 必要条件 (necessary): 隶属度 < necessary_floor → 一票否决, probability=0。

    confidence  = f(完整性, 一致性, 数据质量)             —— 对判断有多确定
      * 完整性 completeness = 命中证据数 / 期望证据数
      * 一致性 agreement    = 证据方向是否集中 (低离散 = 高一致)
      * 数据质量 data_quality = 该窗口数据是否有停牌/缺失 (由调用方传入)

probability 与 confidence 独立输出 (Spec §2.3)。本模块为纯计算, 无 LLM。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mse.core.models.evidence import Evidence
from mse.wyckoff.membership import clamp01
from mse.wyckoff.params import AggregationParams

# 规则集/引擎版本 (Spec §8 版本化)。规则或聚合逻辑变更时递增。
ENGINE_VERSION = "phase2-0.1.0"


@dataclass(frozen=True)
class AggregateResult:
    """聚合产物。reason 为模板化摘要, 供 WyckoffEvent 填充。"""

    probability: float
    confidence: float
    reason: str
    hit_count: int
    expected_count: int
    vetoed: bool = False


def _weighted_geometric_mean(memberships: list[float], weights: list[float], eps: float) -> float:
    """exp( Σ wᵢ·ln(max(mᵢ,eps)) / Σ wᵢ )。"""
    wsum = sum(weights)
    if wsum <= 0:
        return 0.0
    acc = 0.0
    for m, w in zip(memberships, weights):
        acc += w * math.log(max(m, eps))
    return clamp01(math.exp(acc / wsum))


def _agreement(memberships: list[float], weights: list[float]) -> float:
    """方向一致性 = 1 - 加权标准差。证据隶属度越集中 → 越一致。"""
    if len(memberships) < 2:
        return 1.0
    wsum = sum(weights)
    if wsum <= 0:
        return 1.0
    mean = sum(m * w for m, w in zip(memberships, weights)) / wsum
    var = sum(w * (m - mean) ** 2 for m, w in zip(memberships, weights)) / wsum
    return clamp01(1.0 - math.sqrt(var))


def aggregate(
    evidences: list[Evidence],
    *,
    data_quality: float = 1.0,
    params: AggregationParams = AggregationParams(),
) -> AggregateResult:
    """聚合证据 → probability / confidence / reason (Spec §2.4)。"""
    if not evidences:
        return AggregateResult(0.0, 0.0, "无证据", 0, 0)

    memberships = [e.membership for e in evidences]
    weights = [e.weight for e in evidences]

    # ── 必要条件一票否决 (Spec §2.4) ──────────────────────────
    vetoed = False
    veto_notes: list[str] = []
    for e in evidences:
        if e.necessary and e.membership < params.necessary_floor:
            vetoed = True
            veto_notes.append(f"{e.rule_id}={e.membership:.2f}<floor")

    probability = 0.0 if vetoed else _weighted_geometric_mean(memberships, weights, params.eps)

    # ── confidence 三分量 ─────────────────────────────────────
    hit_count = sum(1 for m in memberships if m >= params.hit_threshold)
    expected_count = len(evidences)
    completeness = hit_count / expected_count
    agreement = _agreement(memberships, weights)
    dq = clamp01(data_quality)

    wc, wa, wq = params.conf_w_completeness, params.conf_w_agreement, params.conf_w_quality
    wtot = wc + wa + wq
    confidence = clamp01((wc * completeness + wa * agreement + wq * dq) / wtot)

    reason = _build_reason(evidences, probability, confidence, vetoed, veto_notes)
    return AggregateResult(
        probability=round(probability, 4),
        confidence=round(confidence, 4),
        reason=reason,
        hit_count=hit_count,
        expected_count=expected_count,
        vetoed=vetoed,
    )


def _build_reason(
    evidences: list[Evidence],
    probability: float,
    confidence: float,
    vetoed: bool,
    veto_notes: list[str],
) -> str:
    """模板化摘要 (Spec §2.5): 按 weight×membership 排序取前若干条命中证据。"""
    if vetoed:
        return f"否决(必要条件不足): {'; '.join(veto_notes)}"
    ranked = sorted(evidences, key=lambda e: e.weight * e.membership, reverse=True)
    top = [e for e in ranked if e.membership > 0][:3]
    parts = [f"{e.rule_id}(m={e.membership:.2f},w={e.weight:.2f})" for e in top]
    body = "; ".join(parts) if parts else "无显著证据"
    return f"p={probability:.2f} c={confidence:.2f} | {body}"
