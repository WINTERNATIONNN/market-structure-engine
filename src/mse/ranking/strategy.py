"""Ranking 层 —— 把 Scanner/Scoring 产出的候选按不同**策略**排序 + 过滤。

Scanner 内部只做了默认的 composite 降序; 本层将"如何排"正式化为可选策略, 并统一
承载过滤 (最小分/最小 confidence/方向)。红线: 纯计算, 零 LLM, **不重新打分** —— 只在
既有分数上排序/筛选。

分层: ranking → analysis → scanner → scoring → ...
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from mse.ranking.params import RankingParams
from mse.scoring import CandidateProfile


class RankingStrategy(str, Enum):
    """排名策略 —— 决定排序主键与 (可选) 方向过滤。"""

    COMPOSITE = "composite"        # 综合分 (= max(springboard, transition))
    SPRINGBOARD = "springboard"    # 看涨吸筹 setup 强度
    TRANSITION = "transition"      # 近期阶段转折强度
    BULLISH = "bullish"            # 仅看涨方向, 按综合分
    BEARISH = "bearish"            # 仅看跌方向, 按转折分


class Ranking(BaseModel):
    """一次排名的产物 —— 策略名 + 过滤后已排序的候选。"""

    model_config = ConfigDict(frozen=True)

    strategy: RankingStrategy
    total_in: int = Field(..., ge=0, description="过滤前候选数")
    items: list[CandidateProfile] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)


# 每种策略的 (排序主键取值函数, 方向过滤)。主键均"越大越靠前"。
def _key_composite(p: CandidateProfile) -> tuple:
    return (p.composite_score, p.springboard_score, p.transition_score)


def _key_springboard(p: CandidateProfile) -> tuple:
    return (p.springboard_score, p.composite_score)


def _key_transition(p: CandidateProfile) -> tuple:
    return (p.transition_score, p.composite_score)


_STRATEGY = {
    RankingStrategy.COMPOSITE: (_key_composite, None),
    RankingStrategy.SPRINGBOARD: (_key_springboard, None),
    RankingStrategy.TRANSITION: (_key_transition, None),
    RankingStrategy.BULLISH: (_key_composite, "bullish"),
    RankingStrategy.BEARISH: (_key_transition, "bearish"),
}


def _passes_direction(p: CandidateProfile, want: str | None) -> bool:
    if want is None:
        return True
    if want == "bullish":
        # 看涨: 明确看涨转折, 或有 setup 分而非看跌。
        return p.transition_direction == "bullish" or (
            p.transition_direction != "bearish" and p.springboard_score > 0.0
        )
    # 看跌: 明确看跌转折。
    return p.transition_direction == "bearish"


def rank(
    profiles: list[CandidateProfile],
    *,
    strategy: RankingStrategy = RankingStrategy.COMPOSITE,
    params: RankingParams = RankingParams(),
) -> Ranking:
    """按策略过滤 + 排序, 截断到 top_n。ticker 升序作稳定 tie-break。"""
    key_fn, direction = _STRATEGY[strategy]

    filtered = [
        p for p in profiles
        if p.composite_score >= params.min_composite
        and p.phase_confidence >= params.min_confidence
        and _passes_direction(p, direction)
    ]
    # 主键降序; 平手时 ticker 升序 (稳定排序: 先按 ticker 升序, 再按主键降序)。
    ordered = sorted(filtered, key=lambda p: p.ticker)
    ordered = sorted(ordered, key=key_fn, reverse=True)
    return Ranking(strategy=strategy, total_in=len(profiles), items=ordered[: params.top_n])
