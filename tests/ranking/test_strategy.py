"""Ranking 层离线单测 —— 合成 CandidateProfile 喂 rank(各策略)。"""

from __future__ import annotations

from datetime import date

from mse.core.enums import WyckoffPhase
from mse.ranking import RankingParams, RankingStrategy, rank
from mse.scoring import CandidateProfile

_AS_OF = date(2026, 8, 8)


def _profile(
    ticker: str,
    *,
    composite: float,
    springboard: float = 0.0,
    transition: float = 0.0,
    direction: str | None = None,
    confidence: float = 0.5,
) -> CandidateProfile:
    return CandidateProfile(
        ticker=ticker, as_of=_AS_OF, current_phase=WyckoffPhase.MARKUP,
        phase_probability=0.6, phase_confidence=confidence,
        springboard_score=springboard, transition_score=transition,
        transition_direction=direction, composite_score=composite,
    )


def _pool() -> list[CandidateProfile]:
    return [
        _profile("AAA", composite=0.9, springboard=0.3, transition=0.9, direction="bullish"),
        _profile("BBB", composite=0.8, springboard=0.8, transition=0.2, direction="bullish"),
        _profile("CCC", composite=0.7, springboard=0.1, transition=0.7, direction="bearish"),
        _profile("DDD", composite=0.4, springboard=0.4, transition=0.0),
    ]


def test_composite_ordering() -> None:
    r = rank(_pool(), strategy=RankingStrategy.COMPOSITE)
    assert [p.ticker for p in r.items] == ["AAA", "BBB", "CCC", "DDD"]
    assert r.strategy == RankingStrategy.COMPOSITE
    assert r.total_in == 4 and r.count == 4


def test_springboard_strategy() -> None:
    r = rank(_pool(), strategy=RankingStrategy.SPRINGBOARD)
    assert r.items[0].ticker == "BBB"  # springboard 最高


def test_transition_strategy() -> None:
    r = rank(_pool(), strategy=RankingStrategy.TRANSITION)
    assert r.items[0].ticker == "AAA"  # transition 最高


def test_bullish_filter() -> None:
    r = rank(_pool(), strategy=RankingStrategy.BULLISH)
    tickers = [p.ticker for p in r.items]
    assert "CCC" not in tickers  # 看跌被过滤
    assert tickers[0] == "AAA"


def test_bearish_filter() -> None:
    r = rank(_pool(), strategy=RankingStrategy.BEARISH)
    assert [p.ticker for p in r.items] == ["CCC"]


def test_min_composite_and_topn() -> None:
    r = rank(_pool(), strategy=RankingStrategy.COMPOSITE,
             params=RankingParams(top_n=2, min_composite=0.5))
    assert [p.ticker for p in r.items] == ["AAA", "BBB"]  # DDD(0.4) 被门槛剔除, 且截断到 2


def test_min_confidence_filter() -> None:
    pool = [
        _profile("HI", composite=0.9, confidence=0.8),
        _profile("LO", composite=0.95, confidence=0.1),
    ]
    r = rank(pool, strategy=RankingStrategy.COMPOSITE, params=RankingParams(min_confidence=0.5))
    assert [p.ticker for p in r.items] == ["HI"]  # LO 因 confidence 太低被剔除


def test_tie_break_by_ticker() -> None:
    pool = [_profile("ZZZ", composite=0.5), _profile("AAA", composite=0.5)]
    r = rank(pool, strategy=RankingStrategy.COMPOSITE)
    assert [p.ticker for p in r.items] == ["AAA", "ZZZ"]  # 平手时 ticker 升序
