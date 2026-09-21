"""Range Detection 离线单元测试 (Spec §1.2)。

不联网: 用合成 OHLCV 构造已知结构 (横盘区间 / 单边趋势), 验证检测行为。
这是 Phase 2 第一个离线测试, 也填上 handover 里"补 mock 单测"的债。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mse.core.enums import Timeframe
from mse.core.models import OHLCVSeries, Symbol
from mse.core.models.structure import TradingRange
from mse.data.indicators import build_indicator_set
from mse.wyckoff import RangeParams, detect_ranges


def _make_series(closes: np.ndarray, ticker: str = "TEST") -> OHLCVSeries:
    """把一条收盘价路径包成 OHLCVSeries (周线)。

    high/low 以 close 上下小幅浮动构造; volume 恒定。索引用连续周五。
    """
    n = len(closes)
    idx = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    wiggle = 0.01 * closes  # 1% 上下影线
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes + wiggle,
            "low": closes - wiggle,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    return OHLCVSeries(
        symbol=Symbol(ticker=ticker),
        timeframe=Timeframe.WEEKLY,
        frame=frame,
    )


def _oscillation(n: int, low: float, high: float, period: int = 8) -> np.ndarray:
    """在 [low, high] 之间正弦振荡 —— 一个干净的横盘区间。"""
    mid = (low + high) / 2.0
    amp = (high - low) / 2.0
    t = np.arange(n)
    return mid + amp * np.sin(2 * np.pi * t / period)


def test_detects_clean_range() -> None:
    # 60 周在 100~120 间振荡, 振幅 20 ≈ 若干个 ATR → 应识别出一个区间。
    closes = _oscillation(60, 100.0, 120.0, period=8)
    ohlcv = _make_series(closes)
    ind = build_indicator_set(ohlcv)

    ranges = detect_ranges(ohlcv, ind)

    assert len(ranges) >= 1, "干净横盘应至少检测出一个交易区间"
    tr = ranges[0]
    assert isinstance(tr, TradingRange)
    # 上下沿应落在振荡包络附近 (中位数中枢, 允许影线/中位偏差)。
    assert 95.0 <= tr.lower <= 108.0, f"下沿越界: {tr.lower}"
    assert 112.0 <= tr.upper <= 125.0, f"上沿越界: {tr.upper}"
    assert tr.upper > tr.lower
    assert tr.duration >= RangeParams().tr_min_bars
    assert tr.num_sh >= 2 and tr.num_sl >= 2
    # mid / position 派生正确
    assert tr.lower < tr.mid < tr.upper
    assert tr.position(tr.lower) == pytest.approx(0.0)
    assert tr.position(tr.upper) == pytest.approx(1.0)
    assert tr.position(tr.lower - tr.width) < 0  # 刺穿下沿 → 负


def test_rejects_strong_uptrend() -> None:
    # 单边上涨 60 周 (100 → 220), 斜率大 → 不应识别为区间。
    closes = np.linspace(100.0, 220.0, 60)
    ohlcv = _make_series(closes)
    ind = build_indicator_set(ohlcv)

    ranges = detect_ranges(ohlcv, ind)

    assert ranges == [], f"单边趋势不应被判为交易区间, 却得到 {len(ranges)} 个"


def test_too_short_returns_empty() -> None:
    # 少于 tr_min_bars 根 bar → 直接空。
    closes = _oscillation(10, 100.0, 120.0)
    ohlcv = _make_series(closes)
    ind = build_indicator_set(ohlcv)

    assert detect_ranges(ohlcv, ind) == []


def test_ranges_are_non_overlapping_and_sorted() -> None:
    closes = _oscillation(80, 100.0, 118.0, period=8)
    ohlcv = _make_series(closes)
    ind = build_indicator_set(ohlcv)

    ranges = detect_ranges(ohlcv, ind)

    for a, b in zip(ranges, ranges[1:]):
        assert a.end < b.start, "检测出的区间应互不重叠且按时间升序"


def test_point_in_time_slice() -> None:
    # slice_until 截断后再检测 → 只应看到截断点之前的结构 (无前视)。
    closes = _oscillation(60, 100.0, 120.0, period=8)
    ohlcv = _make_series(closes)
    cutoff = ohlcv.dates[30].date()
    sliced = ohlcv.slice_until(cutoff)
    ind = build_indicator_set(sliced)

    ranges = detect_ranges(sliced, ind)
    for tr in ranges:
        assert tr.end <= cutoff, "point-in-time: 区间不得越过截断日"
