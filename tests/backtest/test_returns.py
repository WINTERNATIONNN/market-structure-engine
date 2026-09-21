"""前向收益测试 (含闭式值 + 越界 None + PIT 不前视检查)。"""

from __future__ import annotations

from datetime import date

import pytest

from backtest.params import BacktestParams
from backtest.returns import compute_fwd_returns, forward_return
from fixtures import make_daily


def test_forward_return_closed_form():
    closes = [float(x) for x in range(100, 200)]  # 100..199, 严格递增
    daily = make_daily(closes)
    as_of = daily.dates[10].date()
    got = forward_return(daily, as_of, horizon=5)
    expected = closes[15] / closes[10] - 1.0
    assert got == pytest.approx(expected)


def test_forward_return_out_of_range_none():
    closes = [float(x) for x in range(100, 130)]
    daily = make_daily(closes)
    last = daily.dates[-1].date()
    assert forward_return(daily, last, horizon=21) is None


def test_forward_return_pit_uses_only_future_bars():
    """改动 as_of 当日之后的 bar 会改变前向收益; 改动之前的 bar 不会 -> 证明只用未来。"""
    closes = [100.0] * 40
    daily = make_daily(closes)
    as_of = daily.dates[20].date()
    base = forward_return(daily, as_of, horizon=5)

    # 只改 as_of 之前的 bar
    past = list(closes)
    past[5] = 999.0
    d_past = make_daily(past)
    assert forward_return(d_past, as_of, horizon=5) == base

    # 改 as_of 之后 (exit) 的 bar
    fut = list(closes)
    fut[25] = 200.0
    d_fut = make_daily(fut)
    assert forward_return(d_fut, as_of, horizon=5) != base


def test_compute_fwd_returns_all_horizons():
    closes = [float(x) for x in range(100, 400)]
    daily = make_daily(closes)
    bp = BacktestParams()
    as_of = daily.dates[10].date()
    got = compute_fwd_returns(daily, as_of, bp)
    assert set(got.keys()) == set(bp.horizons)
    assert got[21] == pytest.approx(closes[31] / closes[10] - 1.0)
