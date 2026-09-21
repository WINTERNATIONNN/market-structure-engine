"""三项指标测试 (手搓 SignalRecord, 断言闭式值)。"""

from __future__ import annotations

from datetime import date

from backtest.metrics import (
    metric_event_precision,
    metric_forward_hit_rate,
    metric_phase_consistency,
)
from backtest.params import BacktestParams
from backtest.records import EventObs, SignalRecord

_AS = date(2024, 1, 31)


def _rec(**over) -> SignalRecord:
    base = dict(
        variant="V0", ticker="AAA", as_of=_AS,
        current_phase="markup", phase_probability=0.8,
        price_pos_in_tr=0.5, has_active_tr=True,
        springboard_score=0.5, transition_score=0.0, composite_score=0.5,
        direction="bullish", fwd_returns={21: 0.1, 63: 0.1, 126: 0.1}, events=(),
    )
    base.update(over)
    return SignalRecord(**base)


def test_forward_hit_rate_closed_form():
    bp = BacktestParams()
    recs = [
        _rec(ticker="A", fwd_returns={21: 0.1, 63: 0.0, 126: 0.0}),   # bullish, +  -> hit
        _rec(ticker="B", fwd_returns={21: -0.1, 63: 0.0, 126: 0.0}),  # bullish, -  -> miss
        _rec(ticker="C", direction="bearish", fwd_returns={21: -0.1, 63: 0.0, 126: 0.0}),  # bearish,- -> hit
    ]
    out = metric_forward_hit_rate(recs, {}, bp)
    st = out["V0"][21]
    assert st.n == 3
    assert st.hit_rate == round(2 / 3, 4)


def test_forward_hit_rate_excess_vs_universe():
    bp = BacktestParams()
    recs = [_rec(ticker="A", fwd_returns={21: 0.10, 63: 0.0, 126: 0.0})]
    means = {(_AS, 21): 0.04}
    out = metric_forward_hit_rate(recs, means, bp)
    assert out["V0"][21].excess_vs_universe == round(0.10 - 0.04, 4)


def test_forward_hit_rate_filters_low_score_and_no_direction():
    bp = BacktestParams()
    recs = [
        _rec(ticker="A", composite_score=0.05),        # < signal_min_score -> 剔除
        _rec(ticker="B", direction=None, composite_score=0.5),  # 无方向 -> 剔除
    ]
    out = metric_forward_hit_rate(recs, {}, bp)
    assert out == {}


def test_phase_consistency_contradiction_rate():
    bp = BacktestParams()
    recs = [
        # 高置信 markup 但 composite==0 -> 矛盾
        _rec(current_phase="markup", phase_probability=0.9, composite_score=0.0),
        # 高置信 distribution 但 price_pos 越界 -> 矛盾
        _rec(current_phase="distribution", phase_probability=0.9, price_pos_in_tr=1.5),
        # 高置信 markup 且自洽 -> 不矛盾
        _rec(current_phase="markup", phase_probability=0.9, composite_score=0.5, price_pos_in_tr=0.5),
    ]
    out = metric_phase_consistency(recs, bp)
    st = out["V0"]
    assert st.n_confident == 3
    assert st.contradiction_rate == round(2 / 3, 4)


def test_phase_consistency_ignores_low_prob_and_other_phases():
    bp = BacktestParams()
    recs = [
        _rec(current_phase="markup", phase_probability=0.5),        # 低置信 -> 不计
        _rec(current_phase="accumulation", phase_probability=0.9),  # 非 markup/dist -> 不计
    ]
    st = metric_phase_consistency(recs, bp)["V0"]
    assert st.n_confident == 0
    assert st.contradiction_rate == 0.0


def test_phase_consistency_label_realized():
    bp = BacktestParams()  # consistency_horizon=63
    recs = [
        _rec(current_phase="markup", phase_probability=0.9, composite_score=0.5,
             fwd_returns={21: 0.0, 63: 0.2, 126: 0.0}),  # markup -> up
        _rec(current_phase="distribution", phase_probability=0.9, composite_score=0.5,
             fwd_returns={21: 0.0, 63: -0.2, 126: 0.0}),  # distribution -> down
    ]
    st = metric_phase_consistency(recs, bp)
    assert st["V0"].markup_up_rate == 1.0
    assert st["V0"].distribution_down_rate == 1.0


def test_event_precision_dedup_and_direction():
    bp = BacktestParams()
    ev_up = EventObs(event_type="Spring", date=date(2024, 1, 10), probability=0.8, fwd_move=0.05)
    ev_up_dup = EventObs(event_type="Spring", date=date(2024, 1, 10), probability=0.8, fwd_move=0.05)
    ev_down = EventObs(event_type="Spring", date=date(2024, 1, 20), probability=0.8, fwd_move=-0.05)
    ev_bear = EventObs(event_type="UT", date=date(2024, 1, 15), probability=0.8, fwd_move=-0.03)
    recs = [
        _rec(ticker="A", events=(ev_up,)),
        _rec(ticker="A", events=(ev_up_dup,)),  # 同 (ticker,date) -> 去重
        _rec(ticker="A", events=(ev_down,)),
        _rec(ticker="A", events=(ev_bear,)),
    ]
    out = metric_event_precision(recs, bp)
    # Spring 期望上涨: 两个唯一事件 (10号涨=对, 20号跌=错) -> precision 0.5
    assert out["V0"]["Spring"].n == 2
    assert out["V0"]["Spring"].precision == 0.5
    # UT 期望下跌: 一个事件跌 -> 对
    assert out["V0"]["UT"].n == 1
    assert out["V0"]["UT"].precision == 1.0


def test_event_precision_skips_none_fwd():
    bp = BacktestParams()
    ev = EventObs(event_type="SOS", date=date(2024, 1, 10), probability=0.7, fwd_move=None)
    out = metric_event_precision([_rec(events=(ev,))], bp)
    assert "SOS" not in out.get("V0", {})
