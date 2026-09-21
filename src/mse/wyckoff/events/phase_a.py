"""Accumulation Phase A 事件检测 (Spec §4.2 SC / §4.3 AR / §4.4 ST)。

一条耦合的时序链, 在给定周线 TR 内的日线上依次检测:

    SC (Selling Climax)   —— 恐慌抛售顶峰, 极端放量 + 长下影 + 收盘回升, 锚定 TR 低点。
    AR (Automatic Rally)  —— SC 后卖压枯竭的反弹, 高点接近 TR 上沿。
    ST (Secondary Test)   —— 回落测试 TR 低点, 缩量 = 卖压枯竭 (可多次)。

§4.3: "AR 高点 + SC 低点 = TR 初始上下沿"。此处 TR 已由 Range Detection 给定,
故按其边界定位这三类事件, 并输出精确 date + 证据 (统一 WyckoffEvent 契约 §9)。
日线事件层, 接收周线 TR + phase 语境 (Spec §10.1)。红线: 零 LLM, 纯计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import Evidence, IndicatorSet, OHLCVSeries, TradingRange, WyckoffEvent
from mse.wyckoff.evidence import ENGINE_VERSION, aggregate
from mse.wyckoff.membership import ramp_down, ramp_up
from mse.wyckoff.params import AggregationParams, PhaseAParams


def detect_phase_a(
    daily: OHLCVSeries,
    indicators: IndicatorSet,
    tr: TradingRange,
    *,
    phase_context: WyckoffPhase = WyckoffPhase.ACCUMULATION,
    params: PhaseAParams = PhaseAParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[WyckoffEvent]:
    """按序检测 SC → AR → ST。返回按 date 升序的 WyckoffEvent 列表。

    无 SC (缺乏高潮锚点) → 返回空。AR/ST 依赖前序事件, 缺失则相应省略。
    """
    frame = daily.frame
    mask = (frame.index >= pd.Timestamp(tr.start)) & (frame.index <= pd.Timestamp(tr.end))
    win = frame.loc[mask]
    n = len(win)
    if n == 0:
        return []

    dates = win.index
    cols = _Cols(
        opens=win["open"].to_numpy(dtype=float),
        highs=win["high"].to_numpy(dtype=float),
        lows=win["low"].to_numpy(dtype=float),
        closes=win["close"].to_numpy(dtype=float),
        atr=indicators.get(params.atr_key).reindex(dates).to_numpy(dtype=float),
        rvol=indicators.get(params.rvol_key).reindex(dates).to_numpy(dtype=float),
    )

    events: list[WyckoffEvent] = []

    # ── SC: 锚点 ───────────────────────────────────────────────
    sc = _detect_sc(daily, dates, cols, tr, phase_context, params, agg_params)
    if sc is None:
        return []
    sc_ev, sc_i = sc
    events.append(sc_ev)

    # ── AR: 紧随 SC 的反弹 ─────────────────────────────────────
    ar = _detect_ar(daily, dates, cols, tr, sc_i, phase_context, params, agg_params)
    ar_i = sc_i
    if ar is not None:
        ar_ev, ar_i = ar
        events.append(ar_ev)

    # ── ST: AR 之后回踩下沿 (可多次) ───────────────────────────
    events.extend(
        _detect_sts(daily, dates, cols, tr, sc_i, ar_i, phase_context, params, agg_params)
    )

    events.sort(key=lambda e: e.date)
    return events


class _Cols:
    """窗口内对齐的价格/指标列 (避免多参数散落)。"""

    __slots__ = ("opens", "highs", "lows", "closes", "atr", "rvol")

    def __init__(self, opens, highs, lows, closes, atr, rvol):  # noqa: ANN001
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.atr = atr
        self.rvol = rvol


def _close_pos(high: float, low: float, close: float) -> float:
    """收盘在当日区间中的位置: 0=最低, 1=最高。high==low → 0.5 (中性)。"""
    rng = high - low
    if rng <= 0:
        return 0.5
    return (close - low) / rng


# ─────────────────────────────── SC ───────────────────────────────
def _detect_sc(
    daily: OHLCVSeries,
    dates: pd.DatetimeIndex,
    c: _Cols,
    tr: TradingRange,
    phase_context: WyckoffPhase,
    params: PhaseAParams,
    agg_params: AggregationParams,
) -> tuple[WyckoffEvent, int] | None:
    """在低沿区选出概率最高的高潮 bar 作为 SC。无合格候选 → None。"""
    best: tuple[float, int, list[Evidence]] | None = None
    for i in range(len(dates)):
        atr_i = c.atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        # 候选须落在 TR 下沿区 (低点接近或跌破 lower)。
        if (c.lows[i] - tr.lower) / atr_i > params.sc_zone_atr:
            continue
        cp = _close_pos(c.highs[i], c.lows[i], c.closes[i])
        # 否决 (§4.2): 收盘贴近当日最低 (无回升) → 更可能继续下跌, 非高潮。
        if cp < params.climax_close_pos * 0.5:
            continue

        evs = _sc_evidences(c, i, atr_i, cp, params)
        agg = aggregate(evs, params=agg_params)
        if agg.vetoed:
            continue
        if best is None or agg.probability > best[0]:
            best = (agg.probability, i, evs)

    if best is None:
        return None
    _, i, evs = best
    agg = aggregate(evs, params=agg_params)
    ev = WyckoffEvent(
        event_type=EventType.SC,
        ticker=daily.symbol.ticker,
        date=dates[i].date(),
        confirmation_date=None,  # SC 的确认是 AR (作为独立事件输出), 不在此填。
        probability=agg.probability,
        confidence=agg.confidence,
        phase_context=phase_context,
        tr_ref=tr,
        reason=agg.reason,
        evidence=evs,
        meta={
            "close_pos": round(_close_pos(c.highs[i], c.lows[i], c.closes[i]), 3),
            "range_atr": round((c.highs[i] - c.lows[i]) / c.atr[i], 3),
            "rvol": round(float(c.rvol[i]), 3) if np.isfinite(c.rvol[i]) else None,
            "low": round(float(c.lows[i]), 4),
        },
        engine_version=ENGINE_VERSION,
    )
    return ev, i


def _sc_evidences(c: _Cols, i: int, atr_i: float, cp: float, params: PhaseAParams) -> list[Evidence]:
    rvol_i = c.rvol[i]
    # [N] 极端放量: 达 climax_vol_mult → 1; 低于 vol_high_mult → 0 (否决非高潮)。
    m_vol = ramp_up(rvol_i, params.vol_high_mult, params.climax_vol_mult) if np.isfinite(rvol_i) else 0.0
    # [W] 当日振幅极大。
    range_atr = (c.highs[i] - c.lows[i]) / atr_i
    m_range = ramp_up(range_atr, params.climax_range_atr * 0.5, params.climax_range_atr)
    # [W] 长下影 + 收盘回升至中上部。
    m_tail_close = ramp_up(cp, params.climax_close_pos * 0.5, params.climax_close_pos)
    # [W] 创新低后快速收回 (收盘相对最低的 ATR 幅度)。
    recover_atr = (c.closes[i] - c.lows[i]) / atr_i
    m_recover = ramp_up(recover_atr, 0.3, 1.2)
    return [
        Evidence(rule_id="sc.volume", weight=params.sc_w_volume, membership=round(m_vol, 4),
                 necessary=True, note=f"rvol={rvol_i:.2f}" if np.isfinite(rvol_i) else "rvol=nan"),
        Evidence(rule_id="sc.range", weight=params.sc_w_range, membership=round(m_range, 4),
                 note=f"range={range_atr:.2f}ATR"),
        Evidence(rule_id="sc.tail_close", weight=params.sc_w_tail_close, membership=round(m_tail_close, 4),
                 note=f"close_pos={cp:.2f}"),
        Evidence(rule_id="sc.recover", weight=params.sc_w_recover, membership=round(m_recover, 4),
                 note="创新低后收回"),
    ]


# ─────────────────────────────── AR ───────────────────────────────
def _detect_ar(
    daily: OHLCVSeries,
    dates: pd.DatetimeIndex,
    c: _Cols,
    tr: TradingRange,
    sc_i: int,
    phase_context: WyckoffPhase,
    params: PhaseAParams,
    agg_params: AggregationParams,
) -> tuple[WyckoffEvent, int] | None:
    """SC 后 ar_max_bars 窗口内的最高 high 作为 AR 高点。"""
    lo = sc_i + 1
    hi = min(sc_i + 1 + params.ar_max_bars, len(dates))
    if lo >= hi:
        return None
    ar_i = lo + int(np.argmax(c.highs[lo:hi]))
    atr_i = c.atr[ar_i]
    if not np.isfinite(atr_i) or atr_i <= 0:
        return None

    sc_low = c.lows[sc_i]
    lag = ar_i - sc_i
    rally_atr = (c.highs[ar_i] - sc_low) / atr_i
    dist_upper_atr = abs(tr.upper - c.highs[ar_i]) / atr_i

    # [N] 时间紧随 SC (lag 越小越像自动反弹)。
    m_timing = ramp_down(float(lag), 1.0, float(params.ar_max_bars))
    # [W] 自 SC 低点显著反弹。
    m_rally = ramp_up(rally_atr, params.ar_min_rally_atr * 0.5, params.ar_min_rally_atr)
    # [W] 反弹高点接近 TR.upper → 候选上沿。
    m_sh = ramp_down(dist_upper_atr, 0.0, params.ar_upper_zone_atr)
    # [W] 反弹时成交量较 SC 回落 (climax 后自然反弹)。
    rvol_ar, rvol_sc = c.rvol[ar_i], c.rvol[sc_i]
    if np.isfinite(rvol_ar) and np.isfinite(rvol_sc) and rvol_sc > 0:
        m_volrecede = ramp_down(rvol_ar / rvol_sc, 0.3, 1.0)
    else:
        m_volrecede = 0.5

    evs = [
        Evidence(rule_id="ar.timing", weight=params.ar_w_timing, membership=round(m_timing, 4),
                 necessary=True, note=f"lag={lag}bar"),
        Evidence(rule_id="ar.rally", weight=params.ar_w_rally, membership=round(m_rally, 4),
                 note=f"rally={rally_atr:.2f}ATR"),
        Evidence(rule_id="ar.sh", weight=params.ar_w_sh, membership=round(m_sh, 4),
                 note=f"dist_upper={dist_upper_atr:.2f}ATR"),
        Evidence(rule_id="ar.volrecede", weight=params.ar_w_volrecede, membership=round(m_volrecede, 4),
                 note="反弹缩量"),
    ]
    agg = aggregate(evs, params=agg_params)
    if agg.vetoed:
        return None
    ev = WyckoffEvent(
        event_type=EventType.AR,
        ticker=daily.symbol.ticker,
        date=dates[ar_i].date(),
        confirmation_date=None,
        probability=agg.probability,
        confidence=agg.confidence,
        phase_context=phase_context,
        tr_ref=tr,
        reason=agg.reason,
        evidence=evs,
        meta={
            "lag": lag,
            "rally_atr": round(rally_atr, 3),
            "dist_upper_atr": round(dist_upper_atr, 3),
            "high": round(float(c.highs[ar_i]), 4),
        },
        engine_version=ENGINE_VERSION,
    )
    return ev, ar_i


# ─────────────────────────────── ST ───────────────────────────────
def _detect_sts(
    daily: OHLCVSeries,
    dates: pd.DatetimeIndex,
    c: _Cols,
    tr: TradingRange,
    sc_i: int,
    ar_i: int,
    phase_context: WyckoffPhase,
    params: PhaseAParams,
    agg_params: AggregationParams,
) -> list[WyckoffEvent]:
    """AR 之后回踩 TR 下沿区的测试 (缩量为佳), 最多 st_max_events 个。"""
    sc_low = c.lows[sc_i]
    out: list[WyckoffEvent] = []
    i = ar_i + 1
    n = len(dates)
    while i < n and len(out) < params.st_max_events:
        atr_i = c.atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            i += 1
            continue
        dist_lower_atr = abs(c.lows[i] - tr.lower) / atr_i
        if dist_lower_atr > params.st_zone_atr:  # 未回到下沿区
            i += 1
            continue

        rvol_i = c.rvol[i]
        undercut_atr = max(0.0, (sc_low - c.lows[i]) / atr_i)
        # 否决 (§4.4): 放量创新低 (显著跌破 SC 低点) → 供给仍在, 非有效 ST。
        high_vol = np.isfinite(rvol_i) and rvol_i >= params.vol_high_mult
        if high_vol and undercut_atr > params.undercut_atr:
            i += 1
            continue

        # [N] 落在 TR.lower 区。
        m_zone = ramp_down(dist_lower_atr, 0.0, params.st_zone_atr)
        # [W] 测试缩量 (卖压枯竭的关键)。
        m_lowvol = ramp_down(rvol_i, params.vol_low_mult, params.vol_high_mult) if np.isfinite(rvol_i) else 0.0
        # [W] 未显著跌破前低 (较 SC 低点抬高或持平)。
        m_higherlow = ramp_down(undercut_atr, 0.0, params.undercut_atr)
        # [W] 测试后收回区间内 (收盘站回 lower 之上)。
        recover_atr = (c.closes[i] - tr.lower) / atr_i
        m_recover = ramp_up(recover_atr, -0.5, 0.5)

        evs = [
            Evidence(rule_id="st.zone", weight=params.st_w_zone, membership=round(m_zone, 4),
                     necessary=True, note=f"dist_lower={dist_lower_atr:.2f}ATR"),
            Evidence(rule_id="st.lowvol", weight=params.st_w_lowvol, membership=round(m_lowvol, 4),
                     note=f"rvol={rvol_i:.2f}" if np.isfinite(rvol_i) else "rvol=nan"),
            Evidence(rule_id="st.higherlow", weight=params.st_w_higherlow, membership=round(m_higherlow, 4),
                     note=f"undercut={undercut_atr:.2f}ATR"),
            Evidence(rule_id="st.recover", weight=params.st_w_recover, membership=round(m_recover, 4),
                     note="收回区间"),
        ]
        agg = aggregate(evs, params=agg_params)
        if agg.vetoed:
            i += 1
            continue
        out.append(
            WyckoffEvent(
                event_type=EventType.ST,
                ticker=daily.symbol.ticker,
                date=dates[i].date(),
                confirmation_date=None,
                probability=agg.probability,
                confidence=agg.confidence,
                phase_context=phase_context,
                tr_ref=tr,
                reason=agg.reason,
                evidence=evs,
                meta={
                    "seq": len(out) + 1,
                    "dist_lower_atr": round(dist_lower_atr, 3),
                    "undercut_atr": round(undercut_atr, 3),
                    "rvol": round(float(rvol_i), 3) if np.isfinite(rvol_i) else None,
                },
                engine_version=ENGINE_VERSION,
            )
        )
        i += 1  # 逐 bar 前进 (允许连续多次 ST)
    return out
