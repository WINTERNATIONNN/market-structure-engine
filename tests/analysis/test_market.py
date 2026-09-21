"""Analysis 层离线单测 —— 合成 CandidateProfile 喂 analyze_market / classify_direction。"""

from __future__ import annotations

from datetime import date

from mse.analysis import AnalysisParams, analyze_market, classify_direction
from mse.core.enums import WyckoffPhase
from mse.scoring import CandidateProfile, TransitionRef

_AS_OF = date(2026, 8, 8)


def _profile(
    ticker: str,
    phase: WyckoffPhase,
    *,
    composite: float,
    springboard: float = 0.0,
    transition: float = 0.0,
    direction: str | None = None,
    bars_ago: int | None = None,
) -> CandidateProfile:
    lt = None
    if bars_ago is not None:
        lt = TransitionRef(
            from_phase=WyckoffPhase.ACCUMULATION, to_phase=phase,
            entered_date=_AS_OF, bars_ago=bars_ago, probability=0.8,
        )
    return CandidateProfile(
        ticker=ticker, as_of=_AS_OF, current_phase=phase,
        phase_probability=0.6, phase_confidence=0.5,
        springboard_score=springboard, transition_score=transition,
        transition_direction=direction, composite_score=composite,
        last_transition=lt, tags=[f"阶段={phase.value}"],
    )


def test_classify_direction() -> None:
    p = AnalysisParams()
    assert classify_direction(_profile("A", WyckoffPhase.MARKUP, composite=0.8,
                                       direction="bullish"), p) == "bullish"
    assert classify_direction(_profile("B", WyckoffPhase.DISTRIBUTION, composite=0.8,
                                       direction="bearish"), p) == "bearish"
    # 无明确转折但 springboard 够高 → 看涨 setup。
    assert classify_direction(_profile("C", WyckoffPhase.DISTRIBUTION, composite=0.6,
                                       springboard=0.6), p) == "bullish"
    # 啥都不强 → 中性。
    assert classify_direction(_profile("D", WyckoffPhase.UNDEFINED, composite=0.1), p) == "neutral"


def test_breadth_counts_and_pct() -> None:
    profiles = [
        _profile("A", WyckoffPhase.MARKUP, composite=0.8, direction="bullish"),
        _profile("B", WyckoffPhase.MARKUP, composite=0.7, direction="bullish"),
        _profile("C", WyckoffPhase.DISTRIBUTION, composite=0.8, direction="bearish"),
        _profile("D", WyckoffPhase.UNDEFINED, composite=0.1),
    ]
    summary = analyze_market(profiles)
    b = summary.breadth
    assert b.analyzed == 4
    assert b.bullish == 2 and b.bearish == 1 and b.neutral == 1
    assert b.phase_counts["markup"] == 2
    assert b.phase_pct(WyckoffPhase.MARKUP) == 0.5
    assert b.as_of == _AS_OF


def test_buckets_sorted_and_filtered() -> None:
    profiles = [
        _profile("BULL_HI", WyckoffPhase.MARKUP, composite=0.9, direction="bullish", bars_ago=1),
        _profile("BULL_LO", WyckoffPhase.MARKUP, composite=0.55, direction="bullish"),
        _profile("BEAR", WyckoffPhase.DISTRIBUTION, composite=0.8, direction="bearish", bars_ago=2),
        _profile("WEAK_BULL", WyckoffPhase.MARKUP, composite=0.2, direction="bullish"),  # 低于 watch
    ]
    summary = analyze_market(profiles)
    # 看涨桶: 只含 ≥ watch(0.5) 的, 且按 composite 降序。
    bull_tickers = [p.ticker for p in summary.bullish_setups]
    assert bull_tickers == ["BULL_HI", "BULL_LO"]
    assert [p.ticker for p in summary.bearish_warnings] == ["BEAR"]
    # 新鲜转折: bars_ago ≤ 4。
    fresh = {p.ticker for p in summary.fresh_transitions}
    assert fresh == {"BULL_HI", "BEAR"}


def test_empty_market_safe() -> None:
    summary = analyze_market([])
    assert summary.breadth.analyzed == 0
    assert summary.breadth.phase_pct(WyckoffPhase.MARKUP) == 0.0
    assert summary.top_overall == []
