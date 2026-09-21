"""Phase FSM 离线单元测试 (Spec §3)。合成周线 + 直接构造事件, 隔离状态机逻辑。

不跑事件检测器 —— 直接喂入 WyckoffEvent, 专测转移 / 迟滞 / min_state_bars。
"""

from __future__ import annotations

import pandas as pd

from mse.core.enums import EventType, Timeframe, WyckoffPhase
from mse.core.models import OHLCVSeries, Symbol, TradingRange, WyckoffEvent
from mse.data.indicators import build_indicator_set
from mse.wyckoff import FSMParams, detect_phases

LOWER, UPPER = 100.0, 120.0
N = 40
_DATES = pd.date_range("2021-01-03", periods=N, freq="W")


def _weekly() -> OHLCVSeries:
    # 价格在区间内小幅波动 (不跌破下沿, 避免误触 MARKDOWN), SOS 后小幅上行。
    closes = [110.0 + (2.0 if i % 2 else -2.0) for i in range(N)]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 3 for c in closes],
            "low": [c - 3 for c in closes],
            "close": closes,
            "volume": [1e6] * N,
        },
        index=_DATES,
    )
    return OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.WEEKLY, frame=frame)


def _tr() -> TradingRange:
    return TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY,
        start=_DATES[5].date(), end=_DATES[35].date(),
        duration=31, upper=UPPER, lower=LOWER, avg_volume=1e6, volume_trend=0.0,
        width_atr=8.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )


def _ev(etype: EventType, bar: int, prob: float) -> WyckoffEvent:
    return WyckoffEvent(
        event_type=etype, ticker="TEST", date=_DATES[bar].date(),
        probability=prob, confidence=0.8,
    )


def _bar_index(when) -> int:
    return list(_DATES.date).index(when)


def test_undefined_to_accumulation_to_markup() -> None:
    weekly = _weekly()
    tr = _tr()
    ind = build_indicator_set(weekly)
    events = [
        _ev(EventType.SC, 6, 0.9),
        _ev(EventType.AR, 8, 0.9),
        _ev(EventType.SOS, 25, 0.85),  # 距 ACCUM 进入 ≥ min_state_bars 后才允许
    ]
    phases = detect_phases(weekly, ind, [tr], events)

    labels = [p.label for p in phases]
    assert labels == [
        WyckoffPhase.UNDEFINED,
        WyckoffPhase.ACCUMULATION,
        WyckoffPhase.MARKUP,
    ], f"实际阶段序列 {[l.value for l in labels]}"

    accum = phases[1]
    markup = phases[2]
    assert accum.probability > 0.5 and accum.confidence > 0.0
    assert markup.probability > 0.5
    # 时间线单调; ACCUM 先于 MARKUP。
    assert accum.state_entered_date < markup.state_entered_date
    # 迟滞: ACCUM 不早于 min_state_bars 进入。
    assert _bar_index(accum.state_entered_date) >= FSMParams().min_state_bars
    # tr_ref 已锚定。
    assert accum.tr_ref is tr


def test_no_events_stays_undefined() -> None:
    # 有 TR 但无任何事件 → 底部证据缺失 → 不进入 ACCUMULATION。
    weekly = _weekly()
    phases = detect_phases(weekly, build_indicator_set(weekly), [_tr()], [])
    assert [p.label for p in phases] == [WyckoffPhase.UNDEFINED]


def test_markup_needs_sos_not_just_bottom() -> None:
    # 只有底部事件, 无 SOS → 停在 ACCUMULATION, 不进 MARKUP。
    weekly = _weekly()
    events = [_ev(EventType.SC, 6, 0.9), _ev(EventType.AR, 8, 0.9)]
    phases = detect_phases(weekly, build_indicator_set(weekly), [_tr()], events)
    labels = [p.label for p in phases]
    assert WyckoffPhase.ACCUMULATION in labels
    assert WyckoffPhase.MARKUP not in labels


def test_min_state_bars_blocks_early_transition() -> None:
    # SOS 紧跟底部事件 (bar 9), 早于 ACCUM 满足 min_state_bars → MARKUP 不应在
    # ACCUM 进入后立刻发生 (需再等 min_state_bars)。
    weekly = _weekly()
    tr = _tr()
    events = [
        _ev(EventType.SC, 6, 0.9),
        _ev(EventType.AR, 8, 0.9),
        _ev(EventType.SOS, 12, 0.85),  # 过早: ACCUM 才刚进入
    ]
    phases = detect_phases(weekly, build_indicator_set(weekly), [tr], events)
    labels = [p.label for p in phases]
    # SOS 因事件衰减 + ACCUM 最小持续期未满而无法确认 → 不进 MARKUP。
    assert WyckoffPhase.ACCUMULATION in labels
    assert WyckoffPhase.MARKUP not in labels


# ── 指标派生证据 (§3.2 前期下跌 / 跌破放量) ────────────────────────
def _weekly_rising() -> OHLCVSeries:
    # 贯穿底部事件窗口的强劲上行 → slope/ATR 明显为正。
    closes = [95.0 + 1.2 * i for i in range(N)]
    frame = pd.DataFrame(
        {"open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
         "close": closes, "volume": [1e6] * N},
        index=_DATES,
    )
    return OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.WEEKLY, frame=frame)


def test_prior_uptrend_suppresses_accumulation() -> None:
    # §3.2 "前期下跌": 若底部事件出现在强劲上行中, 前期下跌证据缺失 → 不进 ACCUM。
    # 事件置于 bar 16/18, 使 ACCUM 候选落在 ATR 已稳定的区段 (斜率证据才生效)。
    tr = _tr()
    events = [_ev(EventType.SC, 16, 0.9), _ev(EventType.AR, 18, 0.9)]

    rising = detect_phases(_weekly_rising(), build_indicator_set(_weekly_rising()), [tr], events)
    flat = detect_phases(_weekly(), build_indicator_set(_weekly()), [tr], events)

    assert WyckoffPhase.ACCUMULATION not in [p.label for p in rising], "上行中不应判为 ACCUM"
    assert WyckoffPhase.ACCUMULATION in [p.label for p in flat], "对照: 平盘应进入 ACCUM"


# ── 全周期 → MARKDOWN, 验证跌破放量证据 ───────────────────────────
N2 = 60
_DATES2 = pd.date_range("2021-01-03", periods=N2, freq="W")


def _weekly_full_cycle(breakdown_volume: float) -> OHLCVSeries:
    # 全程在区间内震荡 (slope≈0, 不干扰 ACCUM/MARKUP), 末段跌破下沿。
    closes, vols = [], []
    for i in range(N2):
        if i >= 55:
            closes.append(95.0)                      # 跌破 lower=100
            vols.append(breakdown_volume)
        else:
            closes.append(110.0 + (2.0 if i % 2 else -2.0))
            vols.append(1e6)
    frame = pd.DataFrame(
        {"open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
         "close": closes, "volume": vols},
        index=_DATES2,
    )
    return OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.WEEKLY, frame=frame)


def _tr_full() -> TradingRange:
    return TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY,
        start=_DATES2[5].date(), end=_DATES2[58].date(),
        duration=54, upper=UPPER, lower=LOWER, avg_volume=1e6, volume_trend=0.0,
        width_atr=8.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )


def _cycle_events() -> list[WyckoffEvent]:
    return [
        WyckoffEvent(event_type=EventType.SC, ticker="TEST", date=_DATES2[6].date(),
                     probability=0.9, confidence=0.8),
        WyckoffEvent(event_type=EventType.AR, ticker="TEST", date=_DATES2[8].date(),
                     probability=0.9, confidence=0.8),
        WyckoffEvent(event_type=EventType.SOS, ticker="TEST", date=_DATES2[25].date(),
                     probability=0.85, confidence=0.8),
        WyckoffEvent(event_type=EventType.BC, ticker="TEST", date=_DATES2[40].date(),
                     probability=0.85, confidence=0.8),
        WyckoffEvent(event_type=EventType.UT, ticker="TEST", date=_DATES2[42].date(),
                     probability=0.8, confidence=0.8),
        WyckoffEvent(event_type=EventType.UTAD, ticker="TEST", date=_DATES2[52].date(),
                     probability=0.8, confidence=0.8),
    ]


def test_breakdown_volume_strengthens_markdown() -> None:
    # 全周期走通 UNDEFINED→ACCUM→MARKUP→DIST→MARKDOWN; 跌破放量应强化 MARKDOWN。
    tr = _tr_full()
    events = _cycle_events()

    hi = detect_phases(_weekly_full_cycle(2.0e6), build_indicator_set(_weekly_full_cycle(2.0e6)),
                       [tr], events)
    lo = detect_phases(_weekly_full_cycle(1.0e6), build_indicator_set(_weekly_full_cycle(1.0e6)),
                       [tr], events)

    hi_labels = [p.label for p in hi]
    # 高量跌破 → 进入 MARKDOWN (全链路贯通)。
    assert WyckoffPhase.ACCUMULATION in hi_labels
    assert WyckoffPhase.MARKUP in hi_labels
    assert WyckoffPhase.DISTRIBUTION in hi_labels
    assert WyckoffPhase.MARKDOWN in hi_labels, f"实际 {[l.value for l in hi_labels]}"

    def md_prob(phases: list) -> float:
        md = [p for p in phases if p.label == WyckoffPhase.MARKDOWN]
        return md[-1].probability if md else 0.0

    # 放量使 MARKDOWN 证据分显著高于缩量情形 (缩量可能根本无法确认转移)。
    assert md_prob(hi) > md_prob(lo)


# ── 派发陷阱修复: DISTRIBUTION 向上突破 → MARKUP (而非卡死) ──────────────
def _weekly_dist_then_breakout(final_close: float = 126.0) -> OHLCVSeries:
    # 前 54 周在区间内震荡 (走通 ACCUM→MARKUP→DIST), 末段向上放量突破 upper=120。
    closes, vols = [], []
    for i in range(N2):
        if i >= 54:
            closes.append(final_close)   # 突破上沿 (>120) 或 (对照) 仍在区间内
            vols.append(2.0e6)
        else:
            closes.append(110.0 + (2.0 if i % 2 else -2.0))
            vols.append(1e6)
    frame = pd.DataFrame(
        {"open": closes, "high": [c + 3 for c in closes], "low": [c - 3 for c in closes],
         "close": closes, "volume": vols},
        index=_DATES2,
    )
    return OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.WEEKLY, frame=frame)


def _dist_cycle_events(with_escape_sos: bool) -> list[WyckoffEvent]:
    # 走通到 DISTRIBUTION 的事件; with_escape_sos 时在派发中后段追加一根 SOS (向上突破)。
    evs = [
        WyckoffEvent(event_type=EventType.SC, ticker="TEST", date=_DATES2[6].date(),
                     probability=0.9, confidence=0.8),
        WyckoffEvent(event_type=EventType.AR, ticker="TEST", date=_DATES2[8].date(),
                     probability=0.9, confidence=0.8),
        WyckoffEvent(event_type=EventType.SOS, ticker="TEST", date=_DATES2[25].date(),
                     probability=0.85, confidence=0.8),
        WyckoffEvent(event_type=EventType.BC, ticker="TEST", date=_DATES2[40].date(),
                     probability=0.85, confidence=0.8),
        WyckoffEvent(event_type=EventType.UT, ticker="TEST", date=_DATES2[42].date(),
                     probability=0.8, confidence=0.8),
    ]
    if with_escape_sos:
        # 派发进入 (~bar43) 满 min_state_bars 之后的新 SOS → 触发 DISTRIBUTION→MARKUP。
        evs.append(WyckoffEvent(event_type=EventType.SOS, ticker="TEST", date=_DATES2[54].date(),
                                probability=0.85, confidence=0.8))
    return evs


def test_distribution_escapes_up_to_markup_on_sos() -> None:
    # 核心回归: 派发区间被向上放量突破 (新 SOS) → 应从 DISTRIBUTION 转回 MARKUP, 不再卡死。
    tr = _tr_full()
    phases = detect_phases(
        _weekly_dist_then_breakout(126.0),
        build_indicator_set(_weekly_dist_then_breakout(126.0)),
        [tr],
        _dist_cycle_events(with_escape_sos=True),
    )
    labels = [p.label for p in phases]
    assert WyckoffPhase.DISTRIBUTION in labels, f"未先进入派发: {[l.value for l in labels]}"
    # 派发之后应出现 MARKUP, 且为末阶段 (逃出陷阱)。
    dist_idx = labels.index(WyckoffPhase.DISTRIBUTION)
    assert WyckoffPhase.MARKUP in labels[dist_idx + 1:], f"派发后未转回 MARKUP: {[l.value for l in labels]}"
    assert labels[-1] == WyckoffPhase.MARKUP


def test_distribution_without_escape_sos_stays_stuck() -> None:
    # 对照: 同样走到派发, 但末段无向上突破 SOS → 仍停在 DISTRIBUTION (未跌破亦不上破)。
    tr = _tr_full()
    phases = detect_phases(
        _weekly_dist_then_breakout(112.0),  # 仍在区间内
        build_indicator_set(_weekly_dist_then_breakout(112.0)),
        [tr],
        _dist_cycle_events(with_escape_sos=False),
    )
    labels = [p.label for p in phases]
    assert WyckoffPhase.DISTRIBUTION in labels
    # 无逃生 SOS → 派发之后不应出现 MARKUP。
    dist_idx = labels.index(WyckoffPhase.DISTRIBUTION)
    assert WyckoffPhase.MARKUP not in labels[dist_idx + 1:], f"不应逃出: {[l.value for l in labels]}"


