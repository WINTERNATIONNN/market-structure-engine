"""方案适配器 —— 在**不改动引擎**的前提下测量 5 个 stale-TR 修复方案。

每个方案复用同一批底层检测器 (detect_ranges / 事件 / detect_phases / score_candidate),
只替换 (a) 传给 detect_ranges 的 RangeParams 与 (b) active-TR 选择器, 并可对 CandidateProfile
后处理 (V4 动量兜底)。引擎代码保持 pristine。

- V0 基线: 现状 (照抄 pipeline._active_tr)。
- V1 陈旧区间守卫: 旧区间离 as_of 太远 / 价格离区间太远 -> active_tr=None (消除幻影位置)。
- V2 突破重探: 价格突破上沿且无包含区间 -> 在近端窗口用宽松参数重探新区间。
- V3 放松斜率区间: 主 pass 用宽松 RangeParams, 使上升通道也成区间。
- V4 趋势动量兜底: V1 得 0 分时用动量指标合成方向信号。

run_all_variants() 复用共享中间产物 (默认参数引擎产物算一次, 宽松参数算一次), 派生全部 5 个。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pandas as pd

from mse.core.models import IndicatorSet, OHLCVSeries, PhaseResult, TradingRange, WyckoffEvent
from mse.data.indicators import build_indicator_set
from mse.scanner.pipeline import _events_for_tr
from mse.scoring import CandidateProfile, ScoringParams, score_candidate
from mse.wyckoff import detect_phases, detect_ps, detect_ranges
from mse.wyckoff.params import RangeParams

from backtest.params import BacktestParams

# V2/V3 共用的宽松 RangeParams (让上升通道也能成区间)。
def _relaxed_range_params(bp: BacktestParams) -> RangeParams:
    return RangeParams(
        tr_min_bars=bp.relaxed_tr_min_bars,
        tr_max_slope=bp.relaxed_tr_max_slope,
        tr_width_atr_max=bp.relaxed_tr_width_atr_max,
    )


# ── 中间产物 (单次引擎跑批的产物, 供同参数的多个方案共享) ──────────────
@dataclass(frozen=True)
class EngineIntermediate:
    weekly: OHLCVSeries
    daily: OHLCVSeries
    wind: IndicatorSet
    dind: IndicatorSet
    ranges: list[TradingRange]
    events: list[WyckoffEvent]
    phases: list[PhaseResult]
    as_of: date
    last_close: float | None
    weekly_atr: float | None
    weekly_dates: list[date]
    daily_dates: list[date]


@dataclass(frozen=True)
class TRContext:
    """active-TR 选择器所需的截面语境。"""

    as_of: date
    last_close: float | None
    weekly_atr: float | None
    weekly_dates: list[date]
    bp: BacktestParams


def _last_val(ind: IndicatorSet, key: str) -> float | None:
    s = ind.series.get(key)
    if s is None or len(s) == 0:
        return None
    v = s.iloc[-1]
    return None if pd.isna(v) else float(v)


def compute_intermediate(
    weekly: OHLCVSeries, daily: OHLCVSeries, *, range_params: RangeParams
) -> EngineIntermediate:
    """镜像 analyze_ticker 主体, 但用注入的 range_params; 返回全部中间产物 (含 dind, events)。"""
    wind = build_indicator_set(weekly)
    dind = build_indicator_set(daily)
    ranges = detect_ranges(weekly, wind, params=range_params)
    events: list[WyckoffEvent] = []
    for tr in ranges:
        events += _events_for_tr(daily, dind, tr)
    events += detect_ps(daily, dind)
    phases = detect_phases(weekly, wind, ranges, events)
    return EngineIntermediate(
        weekly=weekly,
        daily=daily,
        wind=wind,
        dind=dind,
        ranges=ranges,
        events=events,
        phases=phases,
        as_of=weekly.dates[-1].date(),
        last_close=float(daily.close.iloc[-1]) if len(daily) else None,
        weekly_atr=_last_val(wind, "atr_14"),
        weekly_dates=[d.date() for d in weekly.dates],
        daily_dates=[d.date() for d in daily.dates],
    )


def _ctx(inter: EngineIntermediate, bp: BacktestParams) -> TRContext:
    return TRContext(inter.as_of, inter.last_close, inter.weekly_atr, inter.weekly_dates, bp)


def _score(inter: EngineIntermediate, active_tr: TradingRange | None, sp: ScoringParams) -> CandidateProfile:
    return score_candidate(
        inter.weekly.symbol.ticker,
        inter.phases,
        inter.events,
        as_of=inter.as_of,
        last_close=inter.last_close,
        active_tr=active_tr,
        daily_bar_dates=inter.daily_dates,
        weekly_bar_dates=inter.weekly_dates,
        params=sp,
    )


# ── active-TR 选择器 ──────────────────────────────────────────────
def active_tr_baseline(ranges: list[TradingRange], ctx: TRContext) -> TradingRange | None:
    """V0: 原始基线逻辑 (as_of 落入的 TR 取最长, 否则最近已开始的 TR, 无守卫)。

    注: 引擎 pipeline._active_tr 已采纳 V3 守卫; 此处自带一份**原始**逻辑以保 V0 为纯对照组。
    """
    containing = [r for r in ranges if r.start <= ctx.as_of <= r.end]
    if containing:
        return max(containing, key=lambda r: r.duration)
    started = [r for r in ranges if r.start <= ctx.as_of]
    return max(started, key=lambda r: r.start) if started else None


def _weekly_bars_between(weekly_dates: list[date], older: date, newer: date) -> int:
    """weekly_dates 上, older 到 newer 之间的周线 bar 数 (以最后一根 <= 各自日期的 bar 定位)。"""
    def idx_le(d: date) -> int:
        lo = -1
        for i, wd in enumerate(weekly_dates):
            if wd <= d:
                lo = i
            else:
                break
        return lo
    a, b = idx_le(older), idx_le(newer)
    if a < 0 or b < 0:
        return 10**9
    return b - a


def active_tr_stale_guard(ranges: list[TradingRange], ctx: TRContext) -> TradingRange | None:
    """V1: 选出候选 TR (含 as_of 者优先, 否则最近已开始者), 再统一施加邻近度门; 陈旧度门只用于回退候选。

    关键: 邻近度门 (价格离区间太远 -> 无活跃区间) 对**两种候选都生效** —— 单边大涨里
    价格虽仍"落在"某个时间跨度很长的旧区间内, 但已远超上沿, 这类幻影位置同样应被判为无活跃区间。
    """
    containing = [r for r in ranges if r.start <= ctx.as_of <= r.end]
    if containing:
        cand = max(containing, key=lambda r: r.duration)
    else:
        started = [r for r in ranges if r.start <= ctx.as_of]
        if not started:
            return None
        cand = max(started, key=lambda r: r.start)
        # (a) 陈旧度门 (仅回退候选): as_of 距 TR 结束不超过 N 个周线 bar。
        if _weekly_bars_between(ctx.weekly_dates, cand.end, ctx.as_of) > ctx.bp.tr_stale_max_bars:
            return None
    # (b) 邻近度门 (两种候选统一): 价格在 [lower-k*ATR, upper+k*ATR] 内。
    if ctx.last_close is not None and ctx.weekly_atr:
        pad = ctx.bp.tr_proximity_atr * ctx.weekly_atr
        if not (cand.lower - pad <= ctx.last_close <= cand.upper + pad):
            return None
    return cand


# ── V2 突破重探 ───────────────────────────────────────────────────
def _redetect_recent(inter: EngineIntermediate, bp: BacktestParams) -> list[TradingRange]:
    """在近端周线窗口用宽松参数重探新区间; 仅保留起点晚于既有区间的 (避免重叠)。"""
    if len(inter.weekly) < bp.redetect_window_bars:
        return []
    tail = inter.weekly.model_copy(update={"frame": inter.weekly.frame.tail(bp.redetect_window_bars)})
    try:
        tind = build_indicator_set(tail)
        fresh = detect_ranges(tail, tind, params=_relaxed_range_params(bp))
    except Exception:  # noqa: BLE001 —— 重探失败视为无新区间, 回退干净突破
        return []
    if inter.ranges:
        last_end = max(r.end for r in inter.ranges)
        fresh = [r for r in fresh if r.start > last_end]
    return fresh


def _variant_v2(inter_default: EngineIntermediate, bp: BacktestParams, sp: ScoringParams) -> tuple[CandidateProfile, list[WyckoffEvent]]:
    ctx = _ctx(inter_default, bp)
    containing = [r for r in inter_default.ranges if r.start <= inter_default.as_of <= r.end]
    base_tr = active_tr_baseline(inter_default.ranges, ctx)
    breakout = (
        base_tr is not None
        and inter_default.last_close is not None
        and inter_default.weekly_atr
        and inter_default.last_close > base_tr.upper + bp.breakout_atr * inter_default.weekly_atr
    )
    inter = inter_default
    if not containing and (breakout or base_tr is None):
        fresh = _redetect_recent(inter_default, bp)
        if fresh:
            merged_ranges = inter_default.ranges + fresh
            fresh_events: list[WyckoffEvent] = []
            for tr in fresh:
                fresh_events += _events_for_tr(inter_default.daily, inter_default.dind, tr)
            merged_events = inter_default.events + fresh_events
            merged_phases = detect_phases(
                inter_default.weekly, inter_default.wind, merged_ranges, merged_events
            )
            inter = replace(inter_default, ranges=merged_ranges, events=merged_events, phases=merged_phases)
    active_tr = active_tr_stale_guard(inter.ranges, _ctx(inter, bp))
    return _score(inter, active_tr, sp), inter.events


# ── V4 趋势动量兜底 ───────────────────────────────────────────────
def momentum_signal(dind: IndicatorSet, daily: OHLCVSeries, bp: BacktestParams) -> tuple[float, str | None]:
    """无有效区间时的动量方向: close vs sma_50/ema_20 + rsi_14 + macd_hist, 由 rvol_20 调幅。"""
    if len(daily) == 0:
        return 0.0, None
    close = float(daily.close.iloc[-1])
    sma = _last_val(dind, "sma_50")
    ema = _last_val(dind, "ema_20")
    rsi = _last_val(dind, "rsi_14")
    hist = _last_val(dind, "macd_hist")
    rvol = _last_val(dind, "rvol_20")

    bull = sum([
        sma is not None and close > sma,
        ema is not None and close > ema,
        rsi is not None and rsi >= bp.mom_rsi_bull,
        hist is not None and hist > 0,
    ])
    bear = sum([
        sma is not None and close < sma,
        ema is not None and close < ema,
        rsi is not None and rsi <= bp.mom_rsi_bear,
        hist is not None and hist < 0,
    ])
    if bull > bear and bull >= bp.mom_min_conditions:
        direction, frac = "bullish", bull / 4.0
    elif bear > bull and bear >= bp.mom_min_conditions:
        direction, frac = "bearish", bear / 4.0
    else:
        return 0.0, None
    rvol_gate = min(1.0, (rvol or 0.0) / bp.mom_rvol_min)
    score = bp.mom_score_cap * frac * rvol_gate
    return round(score, 4), direction


def _variant_v4(v1_profile: CandidateProfile, inter_default: EngineIntermediate, bp: BacktestParams) -> CandidateProfile:
    """V1 无区间或 composite 过低时, 用动量信号增补 springboard + 方向。"""
    if v1_profile.active_tr is not None and v1_profile.composite_score >= bp.signal_min_score:
        return v1_profile
    mscore, mdir = momentum_signal(inter_default.dind, inter_default.daily, bp)
    if mdir is None or mscore <= 0.0:
        return v1_profile
    return v1_profile.model_copy(update={
        "springboard_score": max(v1_profile.springboard_score, mscore),
        "composite_score": round(max(v1_profile.composite_score, mscore), 4),
        "transition_direction": v1_profile.transition_direction or mdir,
        "tags": [*v1_profile.tags, "momentum-fallback"],
    })


def signal_direction(profile: CandidateProfile, bp: BacktestParams) -> str | None:
    """归一化方向: 优先转折方向; 否则 springboard 达标记为 bullish; 否则无方向。"""
    if profile.transition_direction:
        return profile.transition_direction
    if profile.springboard_score >= bp.signal_min_score:
        return "bullish"
    return None


# ── 顶层: 一次算齐全部方案 (共享中间产物) ─────────────────────────
def run_all_variants(
    weekly: OHLCVSeries, daily: OHLCVSeries, *, bp: BacktestParams, sp: ScoringParams = ScoringParams()
) -> dict[str, tuple[CandidateProfile, list[WyckoffEvent]]]:
    """返回 {方案名: (CandidateProfile, 该方案打分所用的 events)}。

    默认参数引擎产物算一次 (V0/V1/V2/V4 共享), 宽松参数算一次 (V3)。
    """
    inter_default = compute_intermediate(weekly, daily, range_params=RangeParams())
    ctx = _ctx(inter_default, bp)

    v0 = _score(inter_default, active_tr_baseline(inter_default.ranges, ctx), sp)
    v1 = _score(inter_default, active_tr_stale_guard(inter_default.ranges, ctx), sp)
    v2, v2_events = _variant_v2(inter_default, bp, sp)
    v4 = _variant_v4(v1, inter_default, bp)

    inter_relaxed = compute_intermediate(weekly, daily, range_params=_relaxed_range_params(bp))
    v3 = _score(inter_relaxed, active_tr_stale_guard(inter_relaxed.ranges, _ctx(inter_relaxed, bp)), sp)

    return {
        "V0": (v0, inter_default.events),
        "V1": (v1, inter_default.events),
        "V2": (v2, v2_events),
        "V3": (v3, inter_relaxed.events),
        "V4": (v4, inter_default.events),
    }


VARIANT_NAMES: tuple[str, ...] = ("V0", "V1", "V2", "V3", "V4")
