"""走查日期网格 + point-in-time 切片。

build_cutoff_grid: 生成月度 (bp.rebalance) 截面日期。
slice_pit: 把周线+日线都 slice_until(as_of), 强制 point-in-time (无前视); bar 不足返回 None。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mse.core.models import OHLCVSeries

from backtest.params import BacktestParams


def build_cutoff_grid(bp: BacktestParams) -> list[date]:
    """[grid_start, grid_end] 内按 bp.rebalance 频率的截面日期 (升序)。"""
    idx = pd.date_range(start=bp.grid_start, end=bp.grid_end, freq=bp.rebalance)
    return [ts.date() for ts in idx]


def slice_pit(
    weekly: OHLCVSeries,
    daily: OHLCVSeries,
    as_of: date,
    *,
    min_weekly: int,
    min_daily: int,
) -> tuple[OHLCVSeries, OHLCVSeries] | None:
    """把两个序列截到 as_of (含); 任一序列 bar 数不足则返回 None (该截面跳过)。"""
    w = weekly.slice_until(as_of)
    d = daily.slice_until(as_of)
    if len(w) < min_weekly or len(d) < min_daily:
        return None
    return w, d
