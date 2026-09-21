"""网格 + PIT 切片测试。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from backtest.grid import build_cutoff_grid, slice_pit
from backtest.params import BacktestParams
from fixtures import flat_range


def test_grid_monotonic_month_end_and_bounds():
    bp = BacktestParams()
    grid = build_cutoff_grid(bp)
    assert grid == sorted(grid)                       # 升序
    assert len(set(grid)) == len(grid)                # 无重复
    assert all(bp.grid_start <= d <= bp.grid_end for d in grid)
    # 月末频率: 每个截面都是该月最后一日
    for d in grid:
        assert (pd.Timestamp(d) + pd.offsets.Day(1)).month != d.month


def test_grid_count_matches_pandas():
    bp = BacktestParams()
    expected = len(pd.date_range(start=bp.grid_start, end=bp.grid_end, freq=bp.rebalance))
    assert len(build_cutoff_grid(bp)) == expected


def test_slice_pit_keeps_only_past_bars():
    weekly, daily = flat_range(n_days=400)
    as_of = daily.dates[200].date()
    out = slice_pit(weekly, daily, as_of, min_weekly=5, min_daily=5)
    assert out is not None
    w, d = out
    assert all(ts.date() <= as_of for ts in w.dates)
    assert all(ts.date() <= as_of for ts in d.dates)


def test_slice_pit_returns_none_below_min():
    weekly, daily = flat_range(n_days=400)
    early = daily.dates[3].date()
    assert slice_pit(weekly, daily, early, min_weekly=30, min_daily=60) is None
