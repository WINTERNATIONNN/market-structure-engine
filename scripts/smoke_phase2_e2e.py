"""端到端冒烟测试 (Phase 2 全链路): 真实行情 → 结构 → 事件 → 阶段。

链路 (Spec §10.1 multi-timeframe):
    周线 → build_indicator_set → detect_ranges (TR) → detect_phases (Phase FSM)
    日线 → build_indicator_set → 各事件检测 (SC/AR/ST/Spring/SOS/BC/UT/UTAD), 按 TR 定域
           + detect_ps (下跌末段, 不依赖 TR)
    汇总事件 → 喂入周线 Phase FSM, 得到阶段时间线。

运行:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/smoke_phase2_e2e.py [TICKER] [START_YEAR]

依赖网络; 被限流/无数据时优雅跳过 (退出码 0), 不打印 traceback。
"""

from __future__ import annotations

import sys
from datetime import date

from mse.core.enums import EventType, Timeframe
from mse.core.models import OHLCVSeries, TradingRange, WyckoffEvent
from mse.data.indicators import build_indicator_set
from mse.data.providers import YFinanceProvider
from mse.data.providers.base import ProviderError
from mse.wyckoff import (
    detect_bc,
    detect_phase_a,
    detect_phases,
    detect_ps,
    detect_ranges,
    detect_sos,
    detect_springs,
    detect_ut,
    detect_utad,
)


def _events_for_tr(daily: OHLCVSeries, dind, tr: TradingRange) -> list[WyckoffEvent]:
    """在给定周线 TR 的时间窗内, 跑全部日线事件检测器 (底部 + 顶部)。"""
    evs: list[WyckoffEvent] = []
    evs += detect_phase_a(daily, dind, tr)      # SC / AR / ST
    evs += detect_springs(daily, dind, tr)      # Spring
    evs += detect_sos(daily, dind, tr)          # SOS
    bc = detect_bc(daily, dind, tr)             # BC
    ut = detect_ut(daily, dind, tr)             # UT
    evs += bc
    evs += ut
    # UTAD 需 "已有 BC/UT 结构" 证据 → 传入本 TR 内已检出的顶部事件。
    evs += detect_utad(daily, dind, tr, prior_events=bc + ut)
    return evs


def _check(events: list[WyckoffEvent]) -> None:
    for e in events:
        assert 0.0 <= e.probability <= 1.0, f"{e.event_type.value} probability 越界: {e.probability}"
        assert 0.0 <= e.confidence <= 1.0, f"{e.event_type.value} confidence 越界: {e.confidence}"
        assert e.reason, f"{e.event_type.value} 缺少 reason (红线: 非 bool 输出)"


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2018
    start, end = date(start_year, 1, 1), date(2024, 1, 1)

    provider = YFinanceProvider()
    try:
        weekly = provider.get_ohlcv(ticker, Timeframe.WEEKLY, start=start, end=end)
        daily = provider.get_ohlcv(ticker, Timeframe.DAILY, start=start, end=end)
    except ProviderError as exc:
        print(f"⚠️  行情不可用 (可能被限流/断网), 跳过端到端冒烟: {exc}")
        sys.exit(0)

    print(f"== {ticker}  周线 {len(weekly)} bar / 日线 {len(daily)} bar "
          f"({weekly.dates[0].date()} .. {weekly.dates[-1].date()}) ==\n")

    wind = build_indicator_set(weekly)
    dind = build_indicator_set(daily)

    # 1) 周线结构。
    ranges = detect_ranges(weekly, wind)
    print(f"== 周线交易区间: {len(ranges)} 个 ==")
    for i, tr in enumerate(ranges, 1):
        print(f"  [{i}] {tr.start} .. {tr.end}  ({tr.duration}w)  "
              f"upper={tr.upper:.2f} lower={tr.lower:.2f}")
    print()

    # 2) 日线事件 (按 TR 定域 + 全局 PS)。
    all_events: list[WyckoffEvent] = []
    for tr in ranges:
        tr_events = _events_for_tr(daily, dind, tr)
        all_events += tr_events
    ps_events = detect_ps(daily, dind)  # PS 不依赖 TR (下跌末段)
    all_events += ps_events

    # TR 定域事件应落在其窗口内。
    for tr in ranges:
        for e in _events_for_tr(daily, dind, tr):
            assert tr.start <= e.date <= tr.end, f"事件 {e.event_type.value} @ {e.date} 越出 TR 窗口"
    _check(all_events)

    by_type: dict[str, int] = {}
    for e in all_events:
        by_type[e.event_type.value] = by_type.get(e.event_type.value, 0) + 1
    print(f"== 日线事件: {len(all_events)} 个  {by_type} ==")
    for e in sorted(all_events, key=lambda x: x.date):
        conf = f"conf={e.confidence:.2f}"
        confd = f" 确认@{e.confirmation_date}" if e.confirmation_date else ""
        print(f"  {e.date}  {e.event_type.value:<5} p={e.probability:.2f} {conf}{confd}")
    print()

    # 3) 周线 Phase FSM (消费 TR + 事件 + 指标斜率/量能)。
    phases = detect_phases(weekly, wind, ranges, all_events)
    print(f"== 阶段时间线: {len(phases)} 段 ==")
    for p in phases:
        print(f"  {p.state_entered_date} .. {p.as_of_date}  {p.label.value:<13} "
              f"p={p.probability:.2f} conf={p.confidence:.2f}")

    # 阶段自洽: 时间单调、区段不回退、概率合法。
    for a, b in zip(phases, phases[1:]):
        assert a.state_entered_date < b.state_entered_date, "阶段进入时间应单调递增"
        assert a.as_of_date < b.state_entered_date or a.as_of_date <= b.state_entered_date
    for p in phases:
        assert p.state_entered_date <= p.as_of_date, "as_of 不应早于进入日"
        assert 0.0 <= p.probability <= 1.0 and 0.0 <= p.confidence <= 1.0

    print("\n✅ Phase 2 端到端冒烟测试通过 (结构 → 事件 → 阶段 全链路自洽)")


if __name__ == "__main__":
    main()
