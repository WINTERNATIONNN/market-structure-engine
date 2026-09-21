"""Scoring 层离线单测。直接构造 PhaseResult / WyckoffEvent, 隔离打分逻辑 (不联网)。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from mse.core.enums import EventType, Timeframe, WyckoffPhase
from mse.core.models import PhaseResult, TradingRange, WyckoffEvent
from mse.scoring import ScoringParams, score_candidate

_AS_OF = date(2023, 12, 15)
_TICKER = "TEST"


def _daily_dates(n: int, end: date = _AS_OF) -> list[date]:
    # n 个连续自然日, 末尾 = end (用作 bar recency 的锚)。
    return [end - timedelta(days=(n - 1 - i)) for i in range(n)]


def _weekly_dates(n: int, end: date = _AS_OF) -> list[date]:
    return [end - timedelta(weeks=(n - 1 - i)) for i in range(n)]


def _phase(label: WyckoffPhase, entered: date, prob: float, conf: float = 0.8) -> PhaseResult:
    return PhaseResult(
        label=label, ticker=_TICKER, timeframe=Timeframe.WEEKLY,
        probability=prob, confidence=conf,
        state_entered_date=entered, as_of_date=_AS_OF, reason="test",
    )


def _event(etype: EventType, when: date, prob: float) -> WyckoffEvent:
    return WyckoffEvent(event_type=etype, ticker=_TICKER, date=when, probability=prob, confidence=0.8)


def _tr(lower: float, upper: float) -> TradingRange:
    return TradingRange(
        ticker=_TICKER, timeframe=Timeframe.WEEKLY,
        start=date(2023, 6, 1), end=_AS_OF, duration=28,
        upper=upper, lower=lower, avg_volume=1e6, volume_trend=0.0,
        width_atr=5.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )


def test_bullish_springboard_scores_high() -> None:
    # 当前 ACCUMULATION + 近期高概率 Spring + 价格靠近下沿 → springboard 高。
    dd = _daily_dates(40)
    phases = [
        _phase(WyckoffPhase.UNDEFINED, dd[0], 0.0, 0.0),
        _phase(WyckoffPhase.ACCUMULATION, _weekly_dates(6)[0], 0.7),
    ]
    events = [_event(EventType.SPRING, dd[-3], 0.8)]  # 3 天前
    tr = _tr(100.0, 120.0)
    prof = score_candidate(
        _TICKER, phases, events, as_of=_AS_OF, last_close=104.0, active_tr=tr,
        daily_bar_dates=dd, weekly_bar_dates=_weekly_dates(6),
    )
    assert prof.springboard_score > 0.5, prof.springboard_score
    assert prof.composite_score >= prof.springboard_score
    assert any("Spring" in t for t in prof.tags)
    assert prof.current_phase == WyckoffPhase.ACCUMULATION
    assert prof.price_pos_in_tr == pytest.approx(0.2)  # (104-100)/20


def test_no_recent_bull_event_zero_springboard() -> None:
    # 处于 ACCUMULATION 但无任何近期看涨事件 → setup 不成立。
    dd = _daily_dates(40)
    phases = [_phase(WyckoffPhase.ACCUMULATION, dd[0], 0.7)]
    prof = score_candidate(_TICKER, phases, [], as_of=_AS_OF, daily_bar_dates=dd)
    assert prof.springboard_score == 0.0


def test_old_event_excluded_by_recency() -> None:
    # 事件远超 recency 窗口 → 不计入。
    dd = _daily_dates(400)
    phases = [_phase(WyckoffPhase.ACCUMULATION, dd[0], 0.7)]
    events = [_event(EventType.SPRING, dd[0], 0.9)]  # ~400 天前
    prof = score_candidate(
        _TICKER, phases, events, as_of=_AS_OF, daily_bar_dates=dd,
        params=ScoringParams(event_recency_bars=30),
    )
    assert prof.springboard_score == 0.0
    assert prof.recent_bull_events == []


def test_recent_bullish_transition() -> None:
    # 最近进入 MARKUP → transition_score>0, 方向 bullish。
    wd = _weekly_dates(10)
    phases = [
        _phase(WyckoffPhase.ACCUMULATION, wd[0], 0.6),
        _phase(WyckoffPhase.MARKUP, wd[-2], 0.85),  # 1 周前进入
    ]
    prof = score_candidate(_TICKER, phases, [], as_of=_AS_OF, weekly_bar_dates=wd)
    assert prof.transition_score > 0.5
    assert prof.transition_direction == "bullish"
    assert prof.last_transition is not None
    assert prof.last_transition.to_phase == WyckoffPhase.MARKUP
    assert any("markup" in t for t in prof.tags)


def test_recent_distribution_transition_is_descriptive() -> None:
    # 进入 DISTRIBUTION → 记录转折 (transition_score>0) 但**不产出方向性结论** (已降级为描述性)。
    wd = _weekly_dates(10)
    phases = [
        _phase(WyckoffPhase.MARKUP, wd[0], 0.6),
        _phase(WyckoffPhase.DISTRIBUTION, wd[-2], 0.8),
    ]
    prof = score_candidate(_TICKER, phases, [], as_of=_AS_OF, weekly_bar_dates=wd)
    assert prof.transition_direction is None  # 无方向
    assert prof.transition_score > 0.0        # 转折本身仍被记录
    assert prof.last_transition is not None
    assert prof.last_transition.to_phase == WyckoffPhase.DISTRIBUTION
    assert not any("看跌" in t for t in prof.tags)  # 标签不再看跌
    assert any("派发" in t for t in prof.tags)       # 而是描述性标签


def test_recent_markdown_transition_is_bearish() -> None:
    # 进入 MARKDOWN (真实下行段) → 方向 bearish (回归保护: 只有 distribution 被降级)。
    wd = _weekly_dates(10)
    phases = [
        _phase(WyckoffPhase.DISTRIBUTION, wd[0], 0.6),
        _phase(WyckoffPhase.MARKDOWN, wd[-2], 0.8),
    ]
    prof = score_candidate(_TICKER, phases, [], as_of=_AS_OF, weekly_bar_dates=wd)
    assert prof.transition_direction == "bearish"
    assert prof.transition_score > 0.0
    assert any("看跌" in t for t in prof.tags)


def test_stale_transition_not_scored() -> None:
    # 阶段进入远早于 transition_recency 窗口 → 转折分为 0。
    wd = _weekly_dates(30)
    phases = [
        _phase(WyckoffPhase.ACCUMULATION, wd[0], 0.6),
        _phase(WyckoffPhase.MARKUP, wd[0], 0.85),  # 很久以前
    ]
    prof = score_candidate(
        _TICKER, phases, [], as_of=_AS_OF, weekly_bar_dates=wd,
        params=ScoringParams(transition_recency_bars=8),
    )
    assert prof.transition_score == 0.0


def test_composite_is_max_of_parts() -> None:
    dd, wd = _daily_dates(40), _weekly_dates(10)
    phases = [
        _phase(WyckoffPhase.ACCUMULATION, wd[0], 0.6),
        _phase(WyckoffPhase.MARKUP, wd[-2], 0.9),
    ]
    events = [_event(EventType.SOS, dd[-2], 0.7)]
    prof = score_candidate(
        _TICKER, phases, events, as_of=_AS_OF, last_close=110.0, active_tr=_tr(100, 120),
        daily_bar_dates=dd, weekly_bar_dates=wd,
    )
    assert prof.composite_score == pytest.approx(max(prof.springboard_score, prof.transition_score))


def test_empty_phases_safe() -> None:
    prof = score_candidate(_TICKER, [], [], as_of=_AS_OF)
    assert prof.current_phase == WyckoffPhase.UNDEFINED
    assert prof.composite_score == 0.0
