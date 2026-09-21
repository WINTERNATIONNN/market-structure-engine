"""SOS 检测离线单元测试 (Spec §4.6)。合成日线, 完全离线。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mse.core.enums import EventType, Timeframe, WyckoffPhase
from mse.core.models import OHLCVSeries, Symbol, TradingRange
from mse.data.indicators import build_indicator_set
from mse.wyckoff import detect_sos

LOWER, UPPER = 100.0, 120.0


def _daily_with_breakout(
    break_close: float,
    *,
    break_vol: float = 2.2e6,
    hold: bool = True,
    lps_retest: bool = True,
    n_pre: int = 45,
) -> tuple[OHLCVSeries, TradingRange]:
    """在 [100,120] 区间震荡后, 于 n_pre 处向上突破 upper。

    hold=False → 突破后收盘跌回区间内 (失败突破)。
    lps_retest=True → 突破后回踩 upper 并守住 (LPS 确认)。
    """
    rng = np.random.default_rng(7)
    pre = 110 + 5 * np.sin(2 * np.pi * np.arange(n_pre) / 7) + rng.normal(0, 0.5, n_pre)
    closes = list(pre)
    highs = [c + 1.5 for c in closes]
    lows = [c - 1.5 for c in closes]
    vols = [1e6] * n_pre

    # 突破 bar
    closes.append(break_close); highs.append(break_close + 1.0)
    lows.append(UPPER - 1.0); vols.append(break_vol)

    if hold:
        if lps_retest:
            # 回踩 upper 守住 (LPS): low 触 upper, 收盘在上方。
            closes.append(UPPER + 1.5); highs.append(break_close + 1)
            lows.append(UPPER - 0.5); vols.append(1.2e6)
        # 继续在上方
        closes.append(break_close + 3); highs.append(break_close + 4)
        lows.append(break_close); vols.append(1.5e6)
    else:
        # 收盘跌回区间内 (失败突破)
        for _ in range(3):
            closes.append(UPPER - 3); highs.append(UPPER - 1); lows.append(UPPER - 5)
            vols.append(1e6)

    m = len(closes)
    idx = pd.date_range("2022-01-03", periods=m, freq="B")
    opens = [closes[0]] + closes[:-1]
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx
    )
    ohlcv = OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.DAILY, frame=frame)
    tr = TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY, start=idx[0].date(), end=idx[-1].date(),
        duration=m, upper=UPPER, lower=LOWER, avg_volume=1e6, volume_trend=0.0,
        width_atr=8.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )
    return ohlcv, tr


def test_clean_sos_detected_and_lps_confirmed() -> None:
    # 放量突破至 125, 回踩 upper 守住 → SOS + LPS 确认。
    ohlcv, tr = _daily_with_breakout(125.0, break_vol=2.5e6, hold=True, lps_retest=True)
    ind = build_indicator_set(ohlcv)

    events = detect_sos(ohlcv, ind, tr)

    assert len(events) == 1, f"应检测出 1 个 SOS, 实际 {len(events)}"
    ev = events[0]
    assert ev.event_type == EventType.SOS
    assert ev.probability > 0.0
    assert ev.confirmation_date is not None, "回踩守住应确认 LPS"
    assert ev.phase_context == WyckoffPhase.ACCUMULATION
    assert ev.meta["break_atr"] > 0 and ev.meta["held"] is True


def test_lps_confirmation_boosts_confidence() -> None:
    # 有 LPS 回踩 vs 无回踩: confidence 应更高 (Spec §4.6 确认提升 confidence)。
    # 用中等成交量 (rvol≈1.4) 让 base confidence 留出余量, boost 才可见 (满分会被 clamp)。
    with_lps, tr1 = _daily_with_breakout(125.0, break_vol=1.4e6, hold=True, lps_retest=True)
    no_lps, tr2 = _daily_with_breakout(125.0, break_vol=1.4e6, hold=True, lps_retest=False)
    ev_with = detect_sos(with_lps, build_indicator_set(with_lps), tr1)[0]
    ev_without = detect_sos(no_lps, build_indicator_set(no_lps), tr2)[0]

    assert ev_with.confirmation_date is not None
    assert ev_without.confirmation_date is None
    assert ev_with.confidence > ev_without.confidence


def test_low_volume_failed_breakout_rejected_as_ut() -> None:
    # 缩量突破且收盘跌回 → 假突破 (UT), 不作为 SOS。
    ohlcv, tr = _daily_with_breakout(122.0, break_vol=0.8e6, hold=False)
    ind = build_indicator_set(ohlcv)

    assert detect_sos(ohlcv, ind, tr) == [], "缩量假突破应被否决"


def test_no_breakout_no_event() -> None:
    # 全程在区间内 → 无 SOS。
    rng = np.random.default_rng(1)
    n = 60
    closes = 110 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.5, n)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    opens = [closes[0]] + list(closes[:-1])
    frame = pd.DataFrame(
        {"open": opens, "high": closes + 1.5, "low": closes - 1.5, "close": closes,
         "volume": [1e6] * n}, index=idx,
    )
    ohlcv = OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.DAILY, frame=frame)
    tr = TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY, start=idx[0].date(), end=idx[-1].date(),
        duration=n, upper=UPPER, lower=LOWER, avg_volume=1e6, volume_trend=0.0,
        width_atr=8.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )
    ind = build_indicator_set(ohlcv)

    assert detect_sos(ohlcv, ind, tr) == []
