"""Phase 状态机 (Spec §3) —— 四阶段的证据驱动判定 + 迟滞防抖。

在**周线**上运行 (§10.1: 周线锚定 Phase/结构)。消费:
    * Range Detection 产出的 TradingRange 列表 (结构锚)。
    * 日线事件检测产出的 WyckoffEvent 列表 (转移证据; 按 date 映射到所在周线 bar)。
    * IndicatorSet 的斜率 (ATR 归一化的收盘回归) 与 RVOL, 补足事件本身不含、
      但 §3.2 明列的证据: ACCUM 的 "前期下跌" / MARKUP 的动量 / MARKDOWN 的 "放量"。
      这些为加权 (非必要) 项, 中性/契合时隶属度≈1 (不喧宾夺主), 明确反向才削弱。

转移 (§3.2):
    UNDEFINED/MARKDOWN → ACCUMULATION : TR 存在 + 底部事件 (PS/SC/AR) + 前期非上涨
    ACCUMULATION       → MARKUP       : SOS 突破 (事件本身已含放量/站稳) + 上行动量
    MARKUP             → DISTRIBUTION : 高位 TR + 顶部事件 (BC/UT)
    DISTRIBUTION       → MARKDOWN     : 跌破下沿 + 放量 (RVOL) + UTAD 失败
    DISTRIBUTION       → MARKUP       : 派发失败, SOS 向上突破上沿 (重归上涨)

DISTRIBUTION 有两条出边 (向下 MARKDOWN / 向上 MARKUP, 取概率最高者), 修掉旧拓扑里唯一
出边 → MARKDOWN 造成的"派发陷阱" (派发区间被向上突破涨走却卡死)。

防抖 (§3.3): 迟滞带 [exit_threshold, enter_threshold]; 每状态至少 min_state_bars;
转移需连续 confirm_bars 根 bar 的证据支持。红线: 纯计算, 零 LLM。

顶部事件 (BC/UT/UTAD, 见 events/distribution.py) 已实现: MARKUP→DISTRIBUTION 由
BC/UT 证据驱动, DISTRIBUTION→MARKDOWN 由跌破下沿 + UTAD 失败驱动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import (
    Evidence,
    IndicatorSet,
    OHLCVSeries,
    PhaseResult,
    TradingRange,
    WyckoffEvent,
)
from mse.wyckoff.evidence import ENGINE_VERSION, AggregateResult, aggregate
from mse.wyckoff.membership import ramp_down, ramp_up
from mse.wyckoff.params import AggregationParams, FSMParams

_BOTTOM_EVENTS = {EventType.PS, EventType.SC, EventType.AR}
_TOP_EVENTS = {EventType.BC, EventType.UT}


def _reg_slope(y: np.ndarray) -> float:
    """最小二乘拟合的每-bar 斜率 (窗口长度 < 2 → 0)。"""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])

# 每个状态允许的目标 (§3.2)。ACCUMULATION 可从 UNDEFINED 与 MARKDOWN 两处进入。
#
# DISTRIBUTION 有**两条**出边, 复用同一套证据构造器 —— FSM 主循环在允许目标里取概率
# 最高者, 故跌破强则走 MARKDOWN、向上突破强则走 MARKUP, 天然二选一。这修掉了旧拓扑的
# "派发陷阱": 旧图里 DISTRIBUTION 唯一出边是 → MARKDOWN (必要证据=跌破下沿), 一旦派发
# 区间被**向上**突破涨走, 既凑不齐跌破证据、又无向上逃生边 → 永久卡死。新增的
# DISTRIBUTION→MARKUP 语义上即 Wyckoff 的"派发失败/重归上涨" (必要证据=SOS 放量突破上沿,
# 复用 ACCUMULATION→MARKUP 的证据)。
#
# 注: 对称的"吸筹失败向下破位" (ACCUMULATION→MARKDOWN) 未加 —— →MARKDOWN 的证据含 UTAD
# 项 (派发末期专属), 吸筹段无 UTAD 会被几何平均拖至≈0, 直接复用不成立; 需另做 context
# 相关的证据构造, 且非本次报告的粘滞病根, 留待后续。
_TRANSITIONS: dict[WyckoffPhase, tuple[WyckoffPhase, ...]] = {
    WyckoffPhase.UNDEFINED: (WyckoffPhase.ACCUMULATION,),
    WyckoffPhase.ACCUMULATION: (WyckoffPhase.MARKUP,),
    WyckoffPhase.MARKUP: (WyckoffPhase.DISTRIBUTION,),
    WyckoffPhase.DISTRIBUTION: (WyckoffPhase.MARKDOWN, WyckoffPhase.MARKUP),
    WyckoffPhase.MARKDOWN: (WyckoffPhase.ACCUMULATION,),
}


def detect_phases(
    weekly: OHLCVSeries,
    indicators: IndicatorSet,
    ranges: list[TradingRange],
    events: list[WyckoffEvent],
    *,
    params: FSMParams = FSMParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[PhaseResult]:
    """走一遍周线, 产出阶段时间线 (每个阶段区段一个 PhaseResult, 按时间升序)。

    从 UNDEFINED 起步。point-in-time: bar t 只用 date ≤ dates[t] 的事件。
    """
    frame = weekly.frame
    n = len(frame)
    if n == 0:
        return []

    dates = frame.index
    closes = frame["close"].to_numpy(dtype=float)

    # 指标派生序列 (§3.2 前期下跌 / 跌破放量)。缺失则退化为中性。
    atr_ser = indicators.get(params.atr_key)
    rvol_ser = indicators.get(params.rvol_key)
    atr = atr_ser.to_numpy(dtype=float) if atr_ser is not None else np.full(n, np.nan)
    rvol = rvol_ser.to_numpy(dtype=float) if rvol_ser is not None else np.full(n, np.nan)

    def trend_slope_atr(t: int) -> float:
        """bar t 处、按 ATR 归一化的收盘回归斜率 (正=上行, 负=下行)。数据不足→0。"""
        atr_t = atr[t]
        if not np.isfinite(atr_t) or atr_t <= 0:
            return 0.0
        lo = max(0, t - params.trend_lookback)
        if t - lo < 2:
            return 0.0
        return _reg_slope(closes[lo : t + 1]) / atr_t

    def m_not_uptrend(t: int) -> float:
        """非 "前期上涨": 中性/下行→1, 明确上行→0 (契合 ACCUM 的 前期下跌)。"""
        up = max(0.0, trend_slope_atr(t))
        return ramp_down(up, params.trend_slope_atr * 0.5, params.trend_slope_atr)

    def m_not_downtrend(t: int) -> float:
        """非 "下跌中": 中性/上行→1, 明确下行→0 (契合 MARKUP 的动量)。"""
        down = max(0.0, -trend_slope_atr(t))
        return ramp_down(down, params.trend_slope_atr * 0.5, params.trend_slope_atr)

    # 事件 → 所在周线 bar index (最后一根 date ≤ 事件 date 的 bar)。
    date_vals = dates.values
    ev_bars: list[tuple[int, EventType, float]] = []
    for e in events:
        pos = int(np.searchsorted(date_vals, np.datetime64(pd.Timestamp(e.date)), side="right")) - 1
        if 0 <= pos < n:
            ev_bars.append((pos, e.event_type, e.probability))

    def cat_score(t: int, category: set[EventType]) -> float:
        """类别内事件对 bar t 的衰减后最强证据 (recency 越近越强)。"""
        best = 0.0
        for e_idx, etype, prob in ev_bars:
            if etype in category and e_idx <= t:
                decay = ramp_down(float(t - e_idx), 0.0, float(params.event_decay_bars))
                best = max(best, prob * decay)
        return best

    def active_tr(t: int) -> TradingRange | None:
        """bar t 日期落入的 TR; 无则取最近一个已开始的 TR。"""
        when = dates[t].date()
        containing = [r for r in ranges if r.start <= when <= r.end]
        if containing:
            return max(containing, key=lambda r: r.duration)
        started = [r for r in ranges if r.start <= when]
        return max(started, key=lambda r: r.start) if started else None

    def transition_evidence(state: WyckoffPhase, target: WyckoffPhase, t: int) -> list[Evidence]:
        tr = active_tr(t)
        tr_m = 1.0 if (tr is not None and tr.start <= dates[t].date() <= tr.end) else 0.0
        if target == WyckoffPhase.ACCUMULATION:
            return [
                Evidence(rule_id="phase.tr", weight=params.w_tr_present, membership=tr_m,
                         necessary=True, note="成形 TR"),
                Evidence(rule_id="phase.bottom", weight=params.w_bottom_event,
                         membership=round(cat_score(t, _BOTTOM_EVENTS), 4), note="底部事件 SC/AR/PS"),
                Evidence(rule_id="phase.trend", weight=params.w_prior_trend,
                         membership=round(m_not_uptrend(t), 4), note="前期下跌 (非上涨)"),
            ]
        if target == WyckoffPhase.MARKUP:
            return [
                Evidence(rule_id="phase.sos", weight=params.w_sos,
                         membership=round(cat_score(t, {EventType.SOS}), 4),
                         necessary=True, note="SOS 突破上沿"),
                Evidence(rule_id="phase.momentum", weight=params.w_prior_trend,
                         membership=round(m_not_downtrend(t), 4), note="上行动量 (非下跌中)"),
            ]
        if target == WyckoffPhase.DISTRIBUTION:
            return [
                Evidence(rule_id="phase.tr", weight=params.w_tr_present, membership=tr_m,
                         necessary=True, note="高位 TR"),
                Evidence(rule_id="phase.top", weight=params.w_top_event,
                         membership=round(cat_score(t, _TOP_EVENTS), 4), note="顶部事件 BC/UT"),
            ]
        # → MARKDOWN: 跌破 TR 下沿 (§3.2: + 放量 + UTAD 失败)。
        breakdown = 0.0
        if tr is not None and closes[t] < tr.lower:
            breakdown = 1.0
        rvol_t = rvol[t]
        m_bd_vol = ramp_up(rvol_t, 1.0, params.vol_high_mult) if np.isfinite(rvol_t) else 0.0
        return [
            Evidence(rule_id="phase.breakdown", weight=params.w_breakdown,
                     membership=round(breakdown, 4), necessary=True, note="跌破下沿"),
            Evidence(rule_id="phase.breakdown_vol", weight=params.w_breakdown_vol,
                     membership=round(m_bd_vol, 4), note="跌破放量"),
            Evidence(rule_id="phase.utad", weight=params.w_top_event,
                     membership=round(cat_score(t, {EventType.UTAD}), 4), note="UTAD 失败"),
        ]

    # ── 走 bar, 记录每次进入的阶段区段 ─────────────────────────
    entries: list[tuple[int, WyckoffPhase, AggregateResult | None]] = [
        (0, WyckoffPhase.UNDEFINED, None)
    ]
    state = WyckoffPhase.UNDEFINED
    bars_in_state = 1
    streak: dict[WyckoffPhase, int] = {}

    for t in range(1, n):
        best: tuple[float, WyckoffPhase, AggregateResult] | None = None
        for target in _TRANSITIONS[state]:
            agg = aggregate(transition_evidence(state, target, t), params=agg_params)
            if agg.probability >= params.enter_threshold:
                streak[target] = streak.get(target, 0) + 1
            else:
                streak[target] = 0
            if best is None or agg.probability > best[0]:
                best = (agg.probability, target, agg)

        fired = False
        if best is not None and bars_in_state >= params.min_state_bars:
            _, target, agg = best
            if streak.get(target, 0) >= params.confirm_bars:
                entries.append((t, target, agg))
                state = target
                bars_in_state = 1
                streak = {}
                fired = True
        if not fired:
            bars_in_state += 1

    # ── 区段 → PhaseResult ────────────────────────────────────
    results: list[PhaseResult] = []
    for k, (idx, label, agg) in enumerate(entries):
        end_idx = (entries[k + 1][0] - 1) if k + 1 < len(entries) else (n - 1)
        tr = active_tr(idx)
        if agg is None:  # 初始 UNDEFINED
            probability, confidence = 0.0, 0.0
            reason = "初始未定义 (待底部结构与事件确认)"
            evidence: list[Evidence] = []
        else:
            probability, confidence, reason = agg.probability, agg.confidence, agg.reason
            evidence = list(transition_evidence(entries[k - 1][1], label, idx))
        results.append(
            PhaseResult(
                label=label,
                ticker=weekly.symbol.ticker,
                timeframe=weekly.timeframe,
                probability=probability,
                confidence=confidence,
                state_entered_date=dates[idx].date(),
                as_of_date=dates[end_idx].date(),
                reason=reason,
                evidence=evidence,
                tr_ref=tr,
                engine_version=ENGINE_VERSION,
            )
        )
    return results
