"""Phase A (SC/AR/ST) 检测离线单元测试 (Spec §4.2-4.4)。合成日线, 完全离线。

设计: 22 根预热下跌 bar (位于 TR 窗口之前, 仅供 ATR/RVOL 基线) + 15 根区间内
bar 构成 SC → AR → ST 序列。SC = 极端放量长下影反弹; AR = 随后反弹至上沿;
ST = 缩量回踩下沿。
"""

from __future__ import annotations

import pandas as pd

from mse.core.enums import EventType, Timeframe, WyckoffPhase
from mse.core.models import OHLCVSeries, Symbol, TradingRange
from mse.data.indicators import build_indicator_set
from mse.wyckoff import detect_phase_a

LOWER, UPPER = 100.0, 120.0

# 区间内 15 根 bar: (high, low, close, volume)。open 取 = close (合法且不触发涨跌逻辑)。
_EVENT_BARS: list[tuple[float, float, float, float]] = [
    (106.5, 98.0, 105.0, 3.0e6),   # 0  SC  放量高潮, 长下影, 收盘回升
    (111.0, 104.0, 110.0, 1.5e6),  # 1  反弹
    (115.0, 109.0, 114.0, 1.2e6),  # 2
    (118.0, 112.0, 117.0, 1.0e6),  # 3
    (119.0, 115.0, 118.0, 0.9e6),  # 4  AR  反弹高点 (接近 upper), 缩量
    (117.0, 114.0, 116.0, 1.0e6),  # 5  回落
    (116.0, 113.0, 115.0, 1.0e6),  # 6
    (115.0, 112.0, 114.0, 1.0e6),  # 7
    (114.0, 111.0, 113.0, 1.0e6),  # 8
    (113.0, 110.0, 112.0, 1.0e6),  # 9
    (113.0, 100.5, 111.0, 0.6e6),  # 10 ST  缩量回踩下沿, 高于 SC 低点
    (114.0, 111.0, 113.0, 1.0e6),  # 11
    (115.0, 112.0, 114.0, 1.0e6),  # 12
    (114.0, 111.0, 113.0, 1.0e6),  # 13
    (116.0, 113.0, 115.0, 1.0e6),  # 14
]
_N_WARM = 22  # 预热下跌 bar 数 (TR 窗口之前)


def _build(
    event_bars: list[tuple[float, float, float, float]],
) -> tuple[OHLCVSeries, TradingRange]:
    """预热下跌 + 给定区间内 bar → (OHLCVSeries, TR)。TR 窗口只覆盖区间内 bar。"""
    warm_close = [127.0 - i for i in range(_N_WARM)]  # 127 → 106 下跌
    highs = [c + 1.2 for c in warm_close]
    lows = [c - 1.2 for c in warm_close]
    closes = list(warm_close)
    vols = [1e6] * _N_WARM

    for h, lo, c, v in event_bars:
        highs.append(h); lows.append(lo); closes.append(c); vols.append(v)

    opens = list(closes)  # open == close: 合法 OHLC, Phase A 不依赖 bar 涨跌方向
    m = len(closes)
    idx = pd.date_range("2022-01-03", periods=m, freq="B")
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx
    )
    ohlcv = OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.DAILY, frame=frame)
    tr = TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY,
        start=idx[_N_WARM].date(), end=idx[-1].date(),
        duration=len(event_bars), upper=UPPER, lower=LOWER, avg_volume=1e6, volume_trend=0.0,
        width_atr=8.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )
    return ohlcv, tr


def test_clean_phase_a_sequence() -> None:
    # SC → AR → ST, 三事件按序检出。
    ohlcv, tr = _build(_EVENT_BARS)
    events = detect_phase_a(ohlcv, build_indicator_set(ohlcv), tr)

    assert [e.event_type for e in events] == [EventType.SC, EventType.AR, EventType.ST], (
        f"应为 SC→AR→ST, 实际 {[e.event_type.value for e in events]}"
    )
    sc, ar, st = events
    # date 落在预期 bar 上 (warm 之后: SC=+0, AR=+4, ST=+10)。
    warm_dates = pd.date_range("2022-01-03", periods=_N_WARM + len(_EVENT_BARS), freq="B")
    assert sc.date == warm_dates[_N_WARM + 0].date()
    assert ar.date == warm_dates[_N_WARM + 4].date()
    assert st.date == warm_dates[_N_WARM + 10].date()

    for e in events:
        assert e.probability > 0.0
        assert e.phase_context == WyckoffPhase.ACCUMULATION
        assert e.tr_ref is tr

    # 语义校验: SC 极端放量; AR 显著反弹; ST 未跌破 SC 低点 (higher low)。
    assert sc.meta["rvol"] >= 2.5
    assert ar.meta["rally_atr"] > 1.5
    assert st.meta["undercut_atr"] == 0.0


def test_multiple_secondary_tests() -> None:
    # 第二次缩量回踩 → 输出两个 ST (§4.4 允许多次)。
    bars = list(_EVENT_BARS)
    bars[12] = (115.0, 100.8, 111.0, 0.6e6)  # 再次缩量回踩下沿, 高于 SC 低点
    ohlcv, tr = _build(bars)
    events = detect_phase_a(ohlcv, build_indicator_set(ohlcv), tr)

    sts = [e for e in events if e.event_type == EventType.ST]
    assert len(sts) == 2, f"应检出 2 个 ST, 实际 {len(sts)}"
    assert sts[0].meta["seq"] == 1 and sts[1].meta["seq"] == 2


def test_no_climax_no_phase_a() -> None:
    # 全程温和量能, 无高潮 → 无 SC → 空。
    flat = [(112.0, 108.0, 110.0, 1.0e6) for _ in range(15)]
    ohlcv, tr = _build(flat)
    assert detect_phase_a(ohlcv, build_indicator_set(ohlcv), tr) == []


def test_climax_without_recovery_rejected() -> None:
    # 放量但收盘贴近当日最低 (无回升) → 否决为高潮 (§4.2), 无 SC → 空。
    bars = list(_EVENT_BARS)
    bars[0] = (104.0, 98.0, 98.5, 3.0e6)  # 收盘 98.5 贴近最低 98
    ohlcv, tr = _build(bars)
    assert detect_phase_a(ohlcv, build_indicator_set(ohlcv), tr) == []
