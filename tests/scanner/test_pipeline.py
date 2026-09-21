"""Scanner 管线离线单测。合成 OHLCVSeries 喂 analyze_ticker / scan (不联网)。"""

from __future__ import annotations

import pandas as pd

import mse.scanner.pipeline as pipeline
from mse.core.enums import Timeframe
from mse.core.models import OHLCVSeries, Symbol, TradingRange
from mse.scanner import analyze_ticker, scan
from mse.scoring import ScoringParams

_WEEKLY_DATES = [d.date() for d in pd.date_range("2020-01-03", periods=200, freq="W-FRI")]


def _tr(start_i: int, end_i: int, lower: float, upper: float) -> TradingRange:
    return TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY,
        start=_WEEKLY_DATES[start_i], end=_WEEKLY_DATES[end_i],
        duration=end_i - start_i + 1,
        upper=upper, lower=lower, avg_volume=1e6, volume_trend=0.0,
        width_atr=5.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )



def _series(n: int, freq: str, ticker: str, base: float = 110.0) -> OHLCVSeries:
    idx = pd.date_range("2020-01-06", periods=n, freq=freq)
    closes = [base + (2.0 if i % 2 else -2.0) for i in range(n)]  # 区间内震荡
    frame = pd.DataFrame(
        {"open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
         "close": closes, "volume": [1e6] * n},
        index=idx,
    )
    tf = Timeframe.WEEKLY if freq.startswith("W") else Timeframe.DAILY
    return OHLCVSeries(symbol=Symbol(ticker=ticker), timeframe=tf, frame=frame)


def test_analyze_ticker_returns_valid_profile() -> None:
    weekly = _series(60, "W", "AAA")
    daily = _series(300, "B", "AAA")
    prof = analyze_ticker(weekly, daily)
    assert prof.ticker == "AAA"
    assert 0.0 <= prof.composite_score <= 1.0
    assert 0.0 <= prof.springboard_score <= 1.0
    assert 0.0 <= prof.transition_score <= 1.0
    assert prof.as_of == weekly.dates[-1].date()
    assert prof.tags  # 至少含阶段标签


def test_scan_ranks_and_skips() -> None:
    data = {
        "AAA": (_series(60, "W", "AAA"), _series(300, "B", "AAA")),
        "BBB": (_series(60, "W", "BBB"), _series(300, "B", "BBB")),
    }
    profiles, skipped = scan(data)
    assert len(profiles) == 2
    assert skipped == []
    # 按 composite 降序。
    assert profiles[0].composite_score >= profiles[1].composite_score


def test_scan_skips_on_analysis_error(monkeypatch) -> None:
    # analyze_ticker 对某只抛异常 → 计入 skipped, 不中断整批。
    real = pipeline.analyze_ticker

    def flaky(weekly, daily, **kw):
        if weekly.symbol.ticker == "BAD":
            raise ValueError("boom")
        return real(weekly, daily, **kw)

    monkeypatch.setattr(pipeline, "analyze_ticker", flaky)
    data = {
        "GOOD": (_series(60, "W", "GOOD"), _series(300, "B", "GOOD")),
        "BAD": (_series(60, "W", "BAD"), _series(300, "B", "BAD")),
    }
    profiles, skipped = scan(data)
    assert [p.ticker for p in profiles] == ["GOOD"]
    assert skipped == ["BAD"]


# ── _active_tr V3 守卫 (陈旧度 + 邻近度) ─────────────────────────────
_SP = ScoringParams()  # tr_stale_max_bars=8, tr_proximity_atr=3.0


def _guard(ranges, as_of_i, last_close, atr=2.0):
    return pipeline._active_tr(
        ranges, _WEEKLY_DATES[as_of_i],
        last_close=last_close, weekly_atr=atr,
        weekly_dates=_WEEKLY_DATES, params=_SP,
    )


def test_active_tr_returns_containing_when_price_inside() -> None:
    tr = _tr(10, 40, 90.0, 110.0)
    assert _guard([tr], 30, 100.0) is tr  # as_of 落区间内, 价在区间内


def test_active_tr_rejects_stale_fallback() -> None:
    tr = _tr(10, 40, 90.0, 110.0)
    assert _guard([tr], 80, 100.0) is None  # 距 end 40 bar >> 8


def test_active_tr_rejects_containing_when_price_ran_away() -> None:
    """单边大涨: as_of 仍落在旧区间时间跨度内, 但价远超上沿 → 邻近度门判无活跃区间 (消除幻影)。"""
    tr = _tr(10, 60, 90.0, 110.0)  # atr=2, pad=6 → [84,116]
    assert _guard([tr], 30, 250.0) is None


def test_active_tr_accepts_fresh_and_near() -> None:
    tr = _tr(10, 40, 90.0, 110.0)
    assert _guard([tr], 45, 112.0) is tr  # 5 bar 内且价在 [84,116]

