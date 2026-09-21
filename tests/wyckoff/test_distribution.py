"""Distribution 顶部事件 (BC/UT/UTAD) 检测离线单元测试 (Spec §4.7-4.9)。合成日线, 完全离线。

设计: 22 根预热上涨 bar (位于 TR 窗口之前, 仅供 ATR/RVOL 基线, 模拟 Markup) +
区间内 bar 构成顶部形态。BC = 高位极端放量长上影; UT = 刺穿上沿后回落诱多;
UTAD = 刺穿创新高后快速反转跌破下沿 (需已有 BC+UT 结构)。
"""

from __future__ import annotations

import pandas as pd

from mse.core.enums import EventType, Timeframe, WyckoffPhase
from mse.core.models import OHLCVSeries, Symbol, TradingRange, WyckoffEvent
from mse.data.indicators import build_indicator_set
from mse.wyckoff import detect_bc, detect_ut, detect_utad

LOWER, UPPER = 100.0, 120.0
_N_WARM = 22  # 预热上涨 bar 数 (TR 窗口之前, 模拟 Markup 推升)


def _build(
    event_bars: list[tuple[float, float, float, float]],
) -> tuple[OHLCVSeries, TradingRange]:
    """预热上涨 + 给定区间内 bar → (OHLCVSeries, TR)。TR 窗口只覆盖区间内 bar。"""
    warm_close = [94.0 + i for i in range(_N_WARM)]  # 94 → 115 上涨
    highs = [c + 1.2 for c in warm_close]
    lows = [c - 1.2 for c in warm_close]
    closes = list(warm_close)
    vols = [1e6] * _N_WARM

    for h, lo, c, v in event_bars:
        highs.append(h); lows.append(lo); closes.append(c); vols.append(v)

    opens = list(closes)  # open == close: 合法 OHLC, 顶部事件不依赖 bar 涨跌方向
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


# ── BC + UT 共处一个派发区间 ────────────────────────────────────────
# 区间内 bar: (high, low, close, volume)。
_BC_UT_BARS: list[tuple[float, float, float, float]] = [
    (119.5, 113.0, 114.0, 3.2e6),  # 0  BC  高位极端放量, 长上影, 收盘远离最高 (未刺穿)
    (117.0, 112.0, 113.0, 1.2e6),  # 1  回落
    (116.0, 111.0, 115.0, 1.0e6),  # 2
    (119.0, 114.0, 118.0, 1.0e6),  # 3  再次拉升至上沿
    (121.5, 118.0, 119.0, 1.8e6),  # 4  UT  刺穿上沿后收回 (close≤upper), 收盘转弱
    (117.0, 112.0, 113.0, 1.3e6),  # 5  回落
    (115.0, 110.0, 111.0, 1.0e6),  # 6
    (113.0, 108.0, 109.0, 1.0e6),  # 7
]


def test_bc_detected() -> None:
    # 高位极端放量长上影 → 检出 BC (§4.7)。
    ohlcv, tr = _build(_BC_UT_BARS)
    bcs = detect_bc(ohlcv, build_indicator_set(ohlcv), tr)
    assert len(bcs) == 1, f"应检出 1 个 BC, 实际 {len(bcs)}"
    bc = bcs[0]
    all_dates = pd.date_range("2022-01-03", periods=_N_WARM + len(_BC_UT_BARS), freq="B")
    assert bc.date == all_dates[_N_WARM + 0].date()
    assert bc.event_type == EventType.BC
    assert bc.probability > 0.5
    assert bc.phase_context == WyckoffPhase.DISTRIBUTION
    assert bc.tr_ref is tr
    # 长上影: 收盘位于当日下部。
    assert bc.meta["close_pos"] < 0.4
    assert bc.meta["rvol"] >= 2.5


def test_ut_detected() -> None:
    # 刺穿上沿后快速回落 + 收盘转弱 → 检出 UT (§4.8)。
    ohlcv, tr = _build(_BC_UT_BARS)
    uts = detect_ut(ohlcv, build_indicator_set(ohlcv), tr)
    assert len(uts) == 1, f"应检出 1 个 UT, 实际 {len(uts)}"
    ut = uts[0]
    all_dates = pd.date_range("2022-01-03", periods=_N_WARM + len(_BC_UT_BARS), freq="B")
    assert ut.date == all_dates[_N_WARM + 4].date()
    assert ut.event_type == EventType.UT
    assert ut.probability > 0.5
    assert ut.phase_context == WyckoffPhase.DISTRIBUTION
    assert ut.tr_ref is tr
    assert ut.meta["depth_atr"] > 0.0
    assert ut.meta["recover_lag"] <= 2


def test_real_breakout_not_ut() -> None:
    # 刺穿上沿后站稳延续 (不回落) → 真突破 (SOS), 不应报 UT (§4.8 否决)。
    bars = [
        (121.0, 118.0, 120.5, 1.5e6),  # 刺穿, 收在上沿之上
        (123.0, 120.5, 122.0, 1.6e6),  # 继续走高
        (125.0, 122.0, 124.0, 1.7e6),
        (127.0, 124.0, 126.0, 1.8e6),
    ]
    ohlcv, tr = _build(bars)
    assert detect_ut(ohlcv, build_indicator_set(ohlcv), tr) == []


def test_no_pierce_no_ut() -> None:
    # 价格全程在区间内震荡, 从未刺穿上沿 → 无 UT。
    bars = [(118.0, 112.0, 115.0, 1.0e6) for _ in range(8)]
    ohlcv, tr = _build(bars)
    assert detect_ut(ohlcv, build_indicator_set(ohlcv), tr) == []


def test_no_climax_no_bc() -> None:
    # 高位但温和量能, 无买入高潮 → 无 BC。
    bars = [(119.0, 114.0, 115.0, 1.0e6) for _ in range(8)]
    ohlcv, tr = _build(bars)
    assert detect_bc(ohlcv, build_indicator_set(ohlcv), tr) == []


# ── UTAD: 刺穿创新高 → 快速反转 → 跌破下沿确认 ─────────────────────
_UTAD_BARS: list[tuple[float, float, float, float]] = [
    (121.0, 117.0, 118.0, 2.0e6),  # 0  UTAD 刺穿上沿, 当日反转收回 (close≤upper)
    (112.0, 106.0, 107.0, 1.6e6),  # 1  快速下跌
    (104.0, 99.0, 100.0, 1.7e6),   # 2  逼近下沿
    (101.0, 95.0, 96.0, 2.0e6),    # 3  放量跌破下沿 → Markdown 确认
    (98.0, 93.0, 94.0, 1.5e6),     # 4
]


def test_utad_with_structure_and_confirmation() -> None:
    # UTAD (§4.9): 刺穿反转 + 已有 BC/UT 结构 + 后续放量跌破下沿确认。
    ohlcv, tr = _build(_UTAD_BARS)
    ind = build_indicator_set(ohlcv)
    all_dates = pd.date_range("2022-01-03", periods=_N_WARM + len(_UTAD_BARS), freq="B")
    prior = [
        WyckoffEvent(event_type=EventType.BC, ticker="TEST",
                     date=all_dates[_N_WARM - 2].date(), probability=0.8, confidence=0.7),
        WyckoffEvent(event_type=EventType.UT, ticker="TEST",
                     date=all_dates[_N_WARM - 1].date(), probability=0.7, confidence=0.7),
    ]
    utads = detect_utad(ohlcv, ind, tr, prior_events=prior)
    assert len(utads) == 1, f"应检出 1 个 UTAD, 实际 {len(utads)}"
    utad = utads[0]
    assert utad.date == all_dates[_N_WARM + 0].date()
    assert utad.event_type == EventType.UTAD
    assert utad.probability > 0.5
    assert utad.phase_context == WyckoffPhase.DISTRIBUTION
    assert utad.tr_ref is tr
    assert utad.meta["has_bc"] and utad.meta["has_ut"]
    # 后续放量跌破下沿 → 确认日已设置 (bar +3)。
    assert utad.confirmation_date == all_dates[_N_WARM + 3].date()


def test_utad_without_prior_structure_weaker() -> None:
    # 无 BC/UT 结构证据 → UTAD 概率低于有结构时 (structure 证据缺失, probability 独立不否决)。
    ohlcv, tr = _build(_UTAD_BARS)
    ind = build_indicator_set(ohlcv)
    all_dates = pd.date_range("2022-01-03", periods=_N_WARM + len(_UTAD_BARS), freq="B")
    prior = [
        WyckoffEvent(event_type=EventType.BC, ticker="TEST",
                     date=all_dates[_N_WARM - 2].date(), probability=0.8, confidence=0.7),
        WyckoffEvent(event_type=EventType.UT, ticker="TEST",
                     date=all_dates[_N_WARM - 1].date(), probability=0.7, confidence=0.7),
    ]
    with_struct = detect_utad(ohlcv, ind, tr, prior_events=prior)[0]
    without_struct = detect_utad(ohlcv, ind, tr, prior_events=[])[0]
    assert without_struct.probability < with_struct.probability
