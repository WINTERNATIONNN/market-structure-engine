"""SOS 检测 (Spec §4.6) —— Sign of Strength (强势信号)。

需求主导的证据, 通常是放量突破 TR.upper, 标志 Accumulation → Markup。
Spring 的镜像 (向上突破 vs 向下刺穿)。日线事件, 接收周线 TR + phase 语境。

证据 (Spec §4.6):
    [N w=1.0]  突破 TR.upper (close > upper) —— 幅度越大越决定性; 过浅视为噪声跳过。
    [W 0.40]   突破放量 (RVOL ≥ sos_vol_mult) —— 真突破的关键。
    [W 0.30]   突破幅度显著 ((close-upper)/ATR)。
    [W 0.30]   突破后不跌回 TR 内 (back-up / LPS)。
后续确认 (§4.6): 回踩 TR.upper 并守住 (LPS) → confirmation_date; 确认后 confidence 提升。
否决 (§4.6): 缩量突破且迅速跌回 → 假突破 (upthrust, UT), 不作为 SOS。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import Evidence, IndicatorSet, OHLCVSeries, TradingRange, WyckoffEvent
from mse.wyckoff.evidence import ENGINE_VERSION, aggregate
from mse.wyckoff.membership import clamp01, ramp_up
from mse.wyckoff.params import AggregationParams, SOSParams


def detect_sos(
    daily: OHLCVSeries,
    indicators: IndicatorSet,
    tr: TradingRange,
    *,
    phase_context: WyckoffPhase = WyckoffPhase.ACCUMULATION,
    params: SOSParams = SOSParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[WyckoffEvent]:
    """在 TR 时间窗内的日线上检测 SOS 事件, 按时间升序返回。"""
    frame = daily.frame
    mask = (frame.index >= pd.Timestamp(tr.start)) & (frame.index <= pd.Timestamp(tr.end))
    win = frame.loc[mask]
    n = len(win)
    if n == 0:
        return []

    dates = win.index
    lows = win["low"].to_numpy(dtype=float)
    closes = win["close"].to_numpy(dtype=float)
    atr = indicators.get(params.atr_key).reindex(dates).to_numpy(dtype=float)
    rvol = indicators.get(params.rvol_key).reindex(dates).to_numpy(dtype=float)

    events: list[WyckoffEvent] = []
    i = 0
    while i < n:
        if closes[i] <= tr.upper:  # 未突破上沿
            i += 1
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            i += 1
            continue

        break_dist = closes[i] - tr.upper
        break_atr = break_dist / atr_i
        if break_atr < params.min_break_atr:  # 亚噪声级突破, 忽略
            i += 1
            continue

        rvol_i = rvol[i]
        low_vol = (not np.isfinite(rvol_i)) or rvol_i < params.sos_vol_mult

        # 突破后 holdback_bars 内是否跌回 TR (收盘 < upper)。
        after = closes[i + 1 : i + 1 + params.holdback_bars]
        fell_back = bool(len(after) and np.any(after < tr.upper))
        failed_hold = bool(len(after) and after[-1] < tr.upper)

        # 否决 (UT): 缩量突破且最终跌回 → 假突破。
        if low_vol and failed_hold:
            i += 1
            continue

        evidences = _build_evidences(params, break_atr, rvol_i, after, tr, fell_back, failed_hold)
        agg = aggregate(evidences, params=agg_params)
        if agg.vetoed:
            i += 1
            continue

        # 后续确认: LPS 回踩 —— 回踩至 upper 附近 (low ≤ upper) 但收盘守住 (close ≥ upper)。
        confirmation_date = None
        for k in range(i + 1, min(i + 1 + params.confirm_bars, n)):
            if lows[k] <= tr.upper and closes[k] >= tr.upper:
                confirmation_date = dates[k].date()
                break

        # Spec §4.6: LPS 确认提升 confidence (而非 probability)。
        confidence = agg.confidence
        if confirmation_date is not None:
            confidence = round(clamp01(confidence + params.sos_lps_confidence_boost), 4)

        events.append(
            WyckoffEvent(
                event_type=EventType.SOS,
                ticker=daily.symbol.ticker,
                date=dates[i].date(),
                confirmation_date=confirmation_date,
                probability=agg.probability,
                confidence=confidence,
                phase_context=phase_context,
                tr_ref=tr,
                reason=agg.reason,
                evidence=evidences,
                meta={
                    "break_atr": round(break_atr, 3),
                    "breakout": round(break_dist, 4),
                    "rvol": round(float(rvol_i), 3) if np.isfinite(rvol_i) else None,
                    "held": not failed_hold,
                },
                engine_version=ENGINE_VERSION,
            )
        )
        # 跳过 holdback 窗口, 避免同一突破重复触发。
        i = i + 1 + params.holdback_bars

    return events


def _build_evidences(
    params: SOSParams,
    break_atr: float,
    rvol_i: float,
    after: np.ndarray,
    tr: TradingRange,
    fell_back: bool,
    failed_hold: bool,
) -> list[Evidence]:
    # [N] 突破: 幅度达 sos_min_break_atr → 决定性 (=1)。
    m_breakout = ramp_up(break_atr, 0.0, params.sos_min_break_atr)

    # [W] 放量: RVOL 达 sos_vol_mult → 1。
    m_volume = ramp_up(rvol_i, 1.0, params.sos_vol_mult) if np.isfinite(rvol_i) else 0.0

    # [W] 突破幅度显著 (更严的曲线, 达 2× 门槛才满分)。
    m_magnitude = ramp_up(break_atr, params.sos_min_break_atr * 0.5, params.sos_min_break_atr * 2)

    # [W] back-up: 突破后是否守住 upper 之上。
    if len(after) == 0:
        m_backup = 0.5           # 窗口不足 → 中性
    elif not fell_back:
        m_backup = 1.0           # 全程守住
    elif not failed_hold:
        m_backup = 0.5           # 盘中跌回但收盘重回上方
    else:
        m_backup = 0.0           # 收盘跌回区间内

    return [
        Evidence(rule_id="sos.breakout", weight=params.w_breakout, membership=round(m_breakout, 4),
                 necessary=True, note=f"break={break_atr:.2f}ATR"),
        Evidence(rule_id="sos.volume", weight=params.w_volume, membership=round(m_volume, 4),
                 note=f"rvol={rvol_i:.2f}" if np.isfinite(rvol_i) else "rvol=nan"),
        Evidence(rule_id="sos.magnitude", weight=params.w_magnitude, membership=round(m_magnitude, 4),
                 note="突破幅度"),
        Evidence(rule_id="sos.backup", weight=params.w_backup, membership=round(m_backup, 4),
                 note="突破后站稳"),
    ]
