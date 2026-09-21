"""Analysis 层 —— 市场级派生视图 (对 Scanner 产出的 CandidateProfile 列表做汇总/分类)。

红线一致: 纯计算, 零 LLM, 输出结构化 (非布尔)。本层**不重新打分**, 只在既有 composite/
springboard/transition/direction 之上做**分类与聚合**:

    * MarketBreadth —— 阶段直方图 + 多空计数 (市场广度快照)。
    * MarketSummary —— 分桶: 看涨 setup / 看跌预警 / 新鲜转折 / 综合榜首。

分层: analysis → scanner → scoring → ... (见 pyproject.toml importlinter)。
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from mse.analysis.params import AnalysisParams
from mse.core.enums import WyckoffPhase
from mse.scoring import CandidateProfile


def classify_direction(profile: CandidateProfile, params: AnalysisParams) -> str:
    """把候选归为 'bullish' / 'bearish' / 'neutral' (只读既有分数, 不重算)。

    优先看 FSM 转折方向; 无明确转折时, 用 springboard 判断是否为看涨 setup。
    """
    if profile.transition_direction == "bearish":
        return "bearish"
    if profile.transition_direction == "bullish":
        return "bullish"
    if profile.springboard_score >= params.bullish_springboard:
        return "bullish"
    return "neutral"


def _is_fresh_transition(profile: CandidateProfile, params: AnalysisParams) -> bool:
    lt = profile.last_transition
    return lt is not None and lt.bars_ago <= params.fresh_transition_bars


class MarketBreadth(BaseModel):
    """市场广度快照 —— 阶段分布 + 多空计数。"""

    model_config = ConfigDict(frozen=True)

    as_of: date | None = Field(default=None, description="基准日 (取候选中最新 as_of)")
    analyzed: int = Field(..., ge=0, description="纳入统计的候选数")
    phase_counts: dict[str, int] = Field(default_factory=dict, description="阶段 → 数量")
    bullish: int = Field(..., ge=0)
    bearish: int = Field(..., ge=0)
    neutral: int = Field(..., ge=0)

    def phase_pct(self, phase: WyckoffPhase | str) -> float:
        """某阶段占比 (0..1); analyzed=0 → 0。"""
        if self.analyzed == 0:
            return 0.0
        key = phase.value if isinstance(phase, WyckoffPhase) else phase
        return self.phase_counts.get(key, 0) / self.analyzed


class MarketSummary(BaseModel):
    """市场分析汇总 —— 广度 + 三类分桶 + 综合榜首 (均为已排序的 CandidateProfile 列表)。"""

    model_config = ConfigDict(frozen=True)

    breadth: MarketBreadth
    bullish_setups: list[CandidateProfile] = Field(default_factory=list)
    bearish_warnings: list[CandidateProfile] = Field(default_factory=list)
    fresh_transitions: list[CandidateProfile] = Field(default_factory=list)
    top_overall: list[CandidateProfile] = Field(default_factory=list)


def analyze_market(
    profiles: list[CandidateProfile],
    *,
    params: AnalysisParams = AnalysisParams(),
    top_n: int = 25,
) -> MarketSummary:
    """把候选列表汇总成 MarketSummary (广度 + 分桶)。纯计算, 不重新打分。"""
    analyzed = len(profiles)

    phase_counts: dict[str, int] = {}
    bullish = bearish = neutral = 0
    for p in profiles:
        phase_counts[p.current_phase.value] = phase_counts.get(p.current_phase.value, 0) + 1
        d = classify_direction(p, params)
        if d == "bullish":
            bullish += 1
        elif d == "bearish":
            bearish += 1
        else:
            neutral += 1

    as_of = max((p.as_of for p in profiles), default=None)
    breadth = MarketBreadth(
        as_of=as_of, analyzed=analyzed, phase_counts=phase_counts,
        bullish=bullish, bearish=bearish, neutral=neutral,
    )

    def by_composite(seq: list[CandidateProfile]) -> list[CandidateProfile]:
        return sorted(seq, key=lambda p: (p.composite_score, p.springboard_score, p.ticker),
                      reverse=True)

    bullish_setups = by_composite(
        [p for p in profiles if classify_direction(p, params) == "bullish"
         and p.composite_score >= params.watch_composite]
    )
    bearish_warnings = by_composite(
        [p for p in profiles if classify_direction(p, params) == "bearish"
         and p.composite_score >= params.watch_composite]
    )
    fresh_transitions = by_composite([p for p in profiles if _is_fresh_transition(p, params)])
    top_overall = by_composite(list(profiles))[:top_n]

    return MarketSummary(
        breadth=breadth,
        bullish_setups=bullish_setups[:top_n],
        bearish_warnings=bearish_warnings[:top_n],
        fresh_transitions=fresh_transitions[:top_n],
        top_overall=top_overall,
    )
