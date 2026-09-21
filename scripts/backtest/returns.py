"""前向收益 (走查回测的"实现"侧度量)。

关键: 前向收益用**完整** (未切片) 日线序列, 锚定在 as_of / 事件日, 向后取 horizon 个交易日。
这不是前视 —— 信号本身在切片数据上产生; 这里只测量信号之后的真实走势。

前向收益只依赖 (ticker, as_of, horizon), 与方案无关 → 每个 (ticker,as_of) 只算一次。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mse.core.models import OHLCVSeries

from backtest.params import BacktestParams


def _entry_pos(dates: pd.DatetimeIndex, anchor: date) -> int:
    """最后一根日期 <= anchor 的 bar 的整数下标; 无则 -1。"""
    return int(dates.searchsorted(pd.Timestamp(anchor), side="right")) - 1


def _fwd_from_pos(closes: pd.Series, entry: int, horizon: int) -> float | None:
    exit_idx = entry + horizon
    if entry < 0 or exit_idx >= len(closes):
        return None
    c0 = float(closes.iloc[entry])
    c1 = float(closes.iloc[exit_idx])
    if c0 <= 0.0:
        return None
    return c1 / c0 - 1.0


def forward_return(daily_full: OHLCVSeries, as_of: date, horizon: int) -> float | None:
    """从 as_of 之后 horizon 个交易日的收益; 未来 bar 不足 (越界) 返回 None。"""
    return _fwd_from_pos(daily_full.close, _entry_pos(daily_full.dates, as_of), horizon)


def event_forward_move(daily_full: OHLCVSeries, event_date: date, lookahead: int) -> float | None:
    """从事件日之后 lookahead 个交易日的收益 (指标 3 用)。"""
    return _fwd_from_pos(daily_full.close, _entry_pos(daily_full.dates, event_date), lookahead)


def compute_fwd_returns(daily_full: OHLCVSeries, as_of: date, bp: BacktestParams) -> dict[int, float | None]:
    """一次算齐所有 horizon 的前向收益 (与方案无关, 供各方案记录复用)。"""
    return {h: forward_return(daily_full, as_of, h) for h in bp.horizons}


def attach_universe_means(records: list, bp: BacktestParams) -> dict[tuple[date, int], float]:
    """每个 (as_of, horizon) 的横截面平均前向收益 (跨不同 ticker 去重)。

    前向收益与方案无关, 故按 (ticker, as_of) 去重后再求均值, 避免被 5 个方案重复计权。
    """
    seen: set[tuple[str, date]] = set()
    buckets: dict[tuple[date, int], list[float]] = {}
    for r in records:
        key = (r.ticker, r.as_of)
        if key in seen:
            continue
        seen.add(key)
        for h, val in r.fwd_returns.items():
            if val is not None:
                buckets.setdefault((r.as_of, h), []).append(val)
    return {k: (sum(v) / len(v)) for k, v in buckets.items() if v}
