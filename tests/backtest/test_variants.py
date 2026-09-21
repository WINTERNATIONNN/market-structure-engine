"""方案适配器测试 —— 以手搓输入单测适配逻辑, 另加一条端到端不变量检查。

重点单测 (不依赖检测器随机性): 陈旧守卫、周线bar计数、方向归一化、动量信号。
端到端只断言**不变量**: V1/V3 绝不产出幻影位置 (price_pos>1)。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mse.core.enums import Timeframe, WyckoffPhase
from mse.core.models import TradingRange
from mse.data.indicators import build_indicator_set
from mse.scoring import CandidateProfile

from backtest.params import BacktestParams
from backtest.variants import (
    TRContext,
    _weekly_bars_between,
    active_tr_stale_guard,
    momentum_signal,
    run_all_variants,
    signal_direction,
)
from fixtures import flat_range, trend_then_stale, uptrend

_WEEKLY_DATES = [d.date() for d in pd.date_range("2020-01-03", periods=200, freq="W-FRI")]


def _tr(start_i: int, end_i: int, lower: float, upper: float) -> TradingRange:
    return TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY,
        start=_WEEKLY_DATES[start_i], end=_WEEKLY_DATES[end_i],
        duration=end_i - start_i + 1,
        upper=upper, lower=lower, avg_volume=1e6, volume_trend=0.0,
        width_atr=5.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )


def _ctx(as_of_i: int, last_close: float, bp: BacktestParams) -> TRContext:
    return TRContext(
        as_of=_WEEKLY_DATES[as_of_i], last_close=last_close,
        weekly_atr=2.0, weekly_dates=_WEEKLY_DATES, bp=bp,
    )


def test_weekly_bars_between():
    assert _weekly_bars_between(_WEEKLY_DATES, _WEEKLY_DATES[10], _WEEKLY_DATES[18]) == 8
    assert _weekly_bars_between(_WEEKLY_DATES, _WEEKLY_DATES[5], _WEEKLY_DATES[5]) == 0


def test_stale_guard_returns_containing_tr():
    bp = BacktestParams()
    tr = _tr(10, 40, 90.0, 110.0)
    ctx = _ctx(30, 100.0, bp)  # as_of 落在区间内
    assert active_tr_stale_guard([tr], ctx) is tr


def test_stale_guard_rejects_when_too_old():
    bp = BacktestParams()  # tr_stale_max_bars=8
    tr = _tr(10, 40, 90.0, 110.0)
    ctx = _ctx(80, 100.0, bp)  # 距 end 40 个周线 bar >> 8
    assert active_tr_stale_guard([tr], ctx) is None


def test_stale_guard_rejects_when_price_too_far():
    bp = BacktestParams()  # tr_proximity_atr=3.0, atr=2.0 -> pad=6
    tr = _tr(10, 40, 90.0, 110.0)
    ctx = _ctx(45, 200.0, bp)  # 够新 (5 bars) 但价 200 >> 110+6
    assert active_tr_stale_guard([tr], ctx) is None


def test_stale_guard_accepts_fresh_and_near():
    bp = BacktestParams()
    tr = _tr(10, 40, 90.0, 110.0)
    ctx = _ctx(45, 112.0, bp)  # 5 bars 内且价在 [84,116]
    assert active_tr_stale_guard([tr], ctx) is tr


def test_stale_guard_rejects_containing_tr_when_price_ran_away():
    """单边大涨: as_of 仍落在旧区间时间跨度内, 但价远超上沿 -> 邻近度门判无活跃区间 (消除幻影)。"""
    bp = BacktestParams()  # atr=2, pad=6 -> [84,116]
    tr = _tr(10, 60, 90.0, 110.0)
    ctx = _ctx(30, 250.0, bp)  # as_of=30 落在 [10,60] 内, 但价 250 >> 116
    assert active_tr_stale_guard([tr], ctx) is None


def _profile(**over) -> CandidateProfile:
    base = dict(
        ticker="TEST", as_of=date(2024, 1, 5),
        current_phase=WyckoffPhase.UNDEFINED, phase_probability=0.5, phase_confidence=0.5,
        active_tr=None, price_pos_in_tr=None,
        springboard_score=0.0, transition_score=0.0, transition_direction=None,
        composite_score=0.0, tags=[],
    )
    base.update(over)
    return CandidateProfile(**base)


def test_signal_direction():
    bp = BacktestParams()
    assert signal_direction(_profile(transition_direction="bearish"), bp) == "bearish"
    assert signal_direction(_profile(springboard_score=0.5), bp) == "bullish"
    assert signal_direction(_profile(springboard_score=0.01), bp) is None


def test_momentum_signal_bullish_on_uptrend():
    bp = BacktestParams()
    _, daily = uptrend(n_days=300)
    dind = build_indicator_set(daily)
    score, direction = momentum_signal(dind, daily, bp)
    assert direction == "bullish"
    assert 0.0 < score <= bp.mom_score_cap


def test_momentum_signal_bearish_on_downtrend():
    bp = BacktestParams()
    _, daily = uptrend(n_days=300, start_level=200.0, daily_growth=-0.004)
    dind = build_indicator_set(daily)
    score, direction = momentum_signal(dind, daily, bp)
    assert direction == "bearish"
    assert 0.0 < score <= bp.mom_score_cap


def test_run_all_variants_shape_and_no_phantom():
    bp = BacktestParams()
    for weekly, daily in (flat_range(), uptrend(), trend_then_stale()):
        variants = run_all_variants(weekly, daily, bp=bp)
        assert set(variants) == {"V0", "V1", "V2", "V3", "V4"}
        for name, (profile, events) in variants.items():
            assert isinstance(profile, CandidateProfile)
            assert isinstance(events, list)
        # 不变量: 守卫类方案 (V1/V3) 绝不产出幻影位置
        for name in ("V1", "V3"):
            pos = variants[name][0].price_pos_in_tr
            assert pos is None or -0.01 <= pos <= 1.01


def test_v4_augments_when_v1_is_zero():
    """trend_then_stale 上, 若 V1 得 0 分则 V4 应由动量兜底 (springboard>0 且带标签)。"""
    bp = BacktestParams()
    weekly, daily = uptrend(n_days=400)
    variants = run_all_variants(weekly, daily, bp=bp)
    v1 = variants["V1"][0]
    v4 = variants["V4"][0]
    if v1.active_tr is None and v1.composite_score < bp.signal_min_score:
        assert v4.composite_score >= v1.composite_score
        if v4.composite_score > 0.0:
            assert "momentum-fallback" in v4.tags
            assert v4.composite_score <= bp.mom_score_cap
