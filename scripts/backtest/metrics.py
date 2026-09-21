"""三项准确率指标 (用户要求"都看")。

1. metric_forward_hit_rate  —— 方向 vs 前向收益符号的命中率 (per 方案 per 周期)。
2. metric_phase_consistency —— (a) 盲区矛盾率 (KPI, 越低越好) + (b) 标签 vs 实现走势。
3. metric_event_precision   —— 各类事件事后 lookahead 天是否朝预期方向 (per 方案 per 事件类型)。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from backtest.params import BacktestParams
from backtest.records import SignalRecord

_BULL_EVENTS = {"Spring", "SOS", "PS"}
_BEAR_EVENTS = {"BC", "UT", "UTAD"}
_MARKUP = "markup"
_DISTRIBUTION = "distribution"


@dataclass(frozen=True)
class HitStats:
    n: int
    hit_rate: float
    mean_fwd_bull: float
    excess_vs_universe: float


@dataclass(frozen=True)
class ConsistencyStats:
    n_confident: int
    contradiction_rate: float
    markup_up_rate: float
    distribution_down_rate: float


@dataclass(frozen=True)
class PrecisionStats:
    n: int
    precision: float


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def metric_forward_hit_rate(
    records: list[SignalRecord],
    universe_means: dict[tuple[date, int], float],
    bp: BacktestParams,
) -> dict[str, dict[int, HitStats]]:
    """每方案每周期: 有方向且 composite>=signal_min_score 的信号命中率 + 看涨均收益 + 超额。"""
    out: dict[str, dict[int, HitStats]] = {}
    by_variant: dict[str, list[SignalRecord]] = {}
    for r in records:
        if r.direction is None or r.composite_score < bp.signal_min_score:
            continue
        by_variant.setdefault(r.variant, []).append(r)

    for variant, rows in by_variant.items():
        out[variant] = {}
        for h in bp.horizons:
            hits: list[float] = []
            bull_fwd: list[float] = []
            excess: list[float] = []
            for r in rows:
                fwd = r.fwd_returns.get(h)
                if fwd is None:
                    continue
                hit = (fwd > 0) if r.direction == "bullish" else (fwd < 0)
                hits.append(1.0 if hit else 0.0)
                if r.direction == "bullish":
                    bull_fwd.append(fwd)
                    um = universe_means.get((r.as_of, h))
                    if um is not None:
                        excess.append(fwd - um)
            out[variant][h] = HitStats(
                n=len(hits),
                hit_rate=round(_mean(hits), 4),
                mean_fwd_bull=round(_mean(bull_fwd), 4),
                excess_vs_universe=round(_mean(excess), 4),
            )
    return out


def metric_phase_consistency(
    records: list[SignalRecord], bp: BacktestParams
) -> dict[str, ConsistencyStats]:
    """每方案: 高置信 markup/distribution 的矛盾率 + 标签-实现命中率。"""
    out: dict[str, ConsistencyStats] = {}
    by_variant: dict[str, list[SignalRecord]] = {}
    for r in records:
        by_variant.setdefault(r.variant, []).append(r)

    h = bp.consistency_horizon
    for variant, rows in by_variant.items():
        confident = [
            r for r in rows
            if r.current_phase in (_MARKUP, _DISTRIBUTION)
            and r.phase_probability >= bp.confident_phase_prob
        ]
        contradictions = 0
        markup_up: list[float] = []
        dist_down: list[float] = []
        for r in confident:
            pos = r.price_pos_in_tr
            if r.composite_score == 0.0 or (pos is not None and not (0.0 <= pos <= 1.0)):
                contradictions += 1
            fwd = r.fwd_returns.get(h)
            if fwd is not None:
                if r.current_phase == _MARKUP:
                    markup_up.append(1.0 if fwd > 0 else 0.0)
                else:
                    dist_down.append(1.0 if fwd < 0 else 0.0)
        n = len(confident)
        out[variant] = ConsistencyStats(
            n_confident=n,
            contradiction_rate=round(contradictions / n, 4) if n else 0.0,
            markup_up_rate=round(_mean(markup_up), 4),
            distribution_down_rate=round(_mean(dist_down), 4),
        )
    return out


def metric_event_precision(
    records: list[SignalRecord], bp: BacktestParams
) -> dict[str, dict[str, PrecisionStats]]:
    """每方案每事件类型: 事后 lookahead 天朝预期方向的精度 (跨截面按 (ticker,类型,日期) 去重)。"""
    out: dict[str, dict[str, PrecisionStats]] = {}
    # variant -> event_type -> {(ticker, date): correct?}
    seen: dict[str, dict[str, dict[tuple[str, date], bool]]] = {}
    for r in records:
        v_bucket = seen.setdefault(r.variant, {})
        for e in r.events:
            if e.event_type in _BULL_EVENTS:
                expect_up = True
            elif e.event_type in _BEAR_EVENTS:
                expect_up = False
            else:
                continue
            if e.fwd_move is None:
                continue
            key = (r.ticker, e.date)
            correct = (e.fwd_move > 0) if expect_up else (e.fwd_move < 0)
            v_bucket.setdefault(e.event_type, {})[key] = correct

    for variant, types in seen.items():
        out[variant] = {}
        for etype, obs in types.items():
            vals = list(obs.values())
            n = len(vals)
            out[variant][etype] = PrecisionStats(
                n=n,
                precision=round(sum(1 for c in vals if c) / n, 4) if n else 0.0,
            )
    return out
