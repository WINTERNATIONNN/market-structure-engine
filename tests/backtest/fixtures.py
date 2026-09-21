"""合成 OHLCVSeries 构造器 (离线单测用, 无网络、无缓存依赖)。

三种形态:
- flat_range:        横盘band -> 满足斜率门, detect_ranges 可成区间。
- uptrend:           单边匀速上涨 -> 默认斜率门拒绝, 宽松参数才成区间。
- trend_then_stale:  早期横盘 + 其后远高于旧上沿的陡涨 -> 复现盲区 (旧 TR + 幻影位置)。

价格数组同时驱动日线; 周线由日线按周重采样, 保证两周期一致。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from mse.core.enums import Timeframe
from mse.core.models import OHLCVSeries, Symbol

_TRADING_START = date(2018, 1, 1)


def _frame_from_closes(closes: list[float], index: pd.DatetimeIndex, volume: list[float] | None) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    vol = np.asarray(volume, dtype=float) if volume is not None else np.full(len(c), 1_000_000.0)
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": vol,
        },
        index=index,
    )


def make_daily(closes: list[float], *, ticker: str = "TEST", start: date = _TRADING_START,
               volume: list[float] | None = None) -> OHLCVSeries:
    idx = pd.bdate_range(start=pd.Timestamp(start), periods=len(closes))
    return OHLCVSeries(
        symbol=Symbol(ticker=ticker),
        timeframe=Timeframe.DAILY,
        frame=_frame_from_closes(closes, idx, volume),
        adjusted=True,
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def make_weekly(closes: list[float], *, ticker: str = "TEST", start: date = _TRADING_START,
                volume: list[float] | None = None) -> OHLCVSeries:
    idx = pd.date_range(start=pd.Timestamp(start), periods=len(closes), freq="W-FRI")
    return OHLCVSeries(
        symbol=Symbol(ticker=ticker),
        timeframe=Timeframe.WEEKLY,
        frame=_frame_from_closes(closes, idx, volume),
        adjusted=True,
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _resample_weekly_from_daily(daily: OHLCVSeries, ticker: str) -> OHLCVSeries:
    df = daily.frame
    wk = df.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return OHLCVSeries(
        symbol=Symbol(ticker=ticker),
        timeframe=Timeframe.WEEKLY,
        frame=wk,
        adjusted=True,
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ── 三种形态 (返回 (weekly, daily), 周线由日线重采样保持一致) ──────────
def flat_range(*, ticker: str = "FLAT", n_days: int = 400, level: float = 100.0,
               amp: float = 5.0) -> tuple[OHLCVSeries, OHLCVSeries]:
    """围绕 level 的正弦横盘 band。"""
    t = np.arange(n_days)
    closes = (level + amp * np.sin(t * 2 * np.pi / 40.0)).tolist()
    daily = make_daily(closes, ticker=ticker)
    return _resample_weekly_from_daily(daily, ticker), daily


def uptrend(*, ticker: str = "UP", n_days: int = 400, start_level: float = 50.0,
            daily_growth: float = 0.004) -> tuple[OHLCVSeries, OHLCVSeries]:
    """匀速复利上涨 (默认斜率门会拒绝成区间)。"""
    t = np.arange(n_days)
    closes = (start_level * (1.0 + daily_growth) ** t).tolist()
    daily = make_daily(closes, ticker=ticker)
    return _resample_weekly_from_daily(daily, ticker), daily


def trend_then_stale(*, ticker: str = "STALE", flat_days: int = 200, trend_days: int = 300,
                     level: float = 100.0, amp: float = 5.0,
                     daily_growth: float = 0.006) -> tuple[OHLCVSeries, OHLCVSeries]:
    """早期横盘 (成旧 TR) + 其后陡涨到远高于旧上沿 (复现盲区: 价格越界 + 旧区间)。"""
    t1 = np.arange(flat_days)
    flat = level + amp * np.sin(t1 * 2 * np.pi / 40.0)
    base = float(flat[-1])
    t2 = np.arange(1, trend_days + 1)
    trend = base * (1.0 + daily_growth) ** t2
    closes = np.concatenate([flat, trend]).tolist()
    daily = make_daily(closes, ticker=ticker)
    return _resample_weekly_from_daily(daily, ticker), daily
