"""Distribution 顶部事件检测 (Spec §4.7 BC / §4.8 UT / §4.9 UTAD)。

Accumulation 主线的镜像:
    BC   ↔ SC     —— 顶部买入高潮 (极端放量 + 长上影 + 收盘远离最高)。
    UT   ↔ Spring —— 向上假突破 (刺穿 TR.upper 诱多后回落), 否决即 SOS。
    UTAD          —— UT 的派发末期强化版, 后续跌破 TR.lower 确认 → Markdown。

日线事件, 接收周线 TR + phase 语境 (Spec §10.1, 语境应为 DISTRIBUTION)。
红线: 零 LLM, 纯计算。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import Evidence, IndicatorSet, OHLCVSeries, TradingRange, WyckoffEvent
from mse.wyckoff.evidence import ENGINE_VERSION, aggregate
from mse.wyckoff.membership import ramp_down, ramp_up
from mse.wyckoff.params import AggregationParams, DistributionParams


def _close_pos(high: float, low: float, close: float) -> float:
    """收盘在当日区间的位置: 0=最低, 1=最高。high==low → 0.5。"""
    rng = high - low
    return (close - low) / rng if rng > 0 else 0.5


def _win(daily: OHLCVSeries, tr: TradingRange):
    """截取 TR 时间窗内的对齐视图。"""
    frame = daily.frame
    mask = (frame.index >= pd.Timestamp(tr.start)) & (frame.index <= pd.Timestamp(tr.end))
    return frame.loc[mask]


# ─────────────────────────────── BC ───────────────────────────────
def detect_bc(
    daily: OHLCVSeries,
    indicators: IndicatorSet,
    tr: TradingRange,
    *,
    phase_context: WyckoffPhase = WyckoffPhase.DISTRIBUTION,
    params: DistributionParams = DistributionParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[WyckoffEvent]:
    """买入高潮 (§4.7, 镜像 SC)。在上沿区选概率最高的高潮 bar; 无则空。"""
    win = _win(daily, tr)
    n = len(win)
    if n == 0:
        return []
    dates = win.index
    highs = win["high"].to_numpy(dtype=float)
    lows = win["low"].to_numpy(dtype=float)
    closes = win["close"].to_numpy(dtype=float)
    atr = indicators.get(params.atr_key).reindex(dates).to_numpy(dtype=float)
    rvol = indicators.get(params.rvol_key).reindex(dates).to_numpy(dtype=float)

    best: tuple[float, int, list[Evidence]] | None = None
    for i in range(n):
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        # 候选须落在 TR 上沿区 (高点接近或突破 upper)。
        if (tr.upper - highs[i]) / atr_i > params.bc_zone_atr:
            continue
        cp = _close_pos(highs[i], lows[i], closes[i])
        # 否决 (§4.7): 收盘强力收在最高 → 可能仍是 Markup 延续。
        if cp > 1.0 - params.climax_close_pos * 0.5:
            continue

        rvol_i = rvol[i]
        m_vol = ramp_up(rvol_i, params.vol_high_mult, params.climax_vol_mult) if np.isfinite(rvol_i) else 0.0
        range_atr = (highs[i] - lows[i]) / atr_i
        m_range = ramp_up(range_atr, params.climax_range_atr * 0.5, params.climax_range_atr)
        # 长上影 + 收盘远离最高: close_pos ≤ 1 - climax_close_pos → 强。
        m_tail = ramp_down(cp, 1.0 - params.climax_close_pos, 1.0 - params.climax_close_pos * 0.5)
        # 创新高后快速回落。
        m_recover = ramp_up((highs[i] - closes[i]) / atr_i, 0.3, 1.2)
        evs = [
            Evidence(rule_id="bc.volume", weight=params.bc_w_volume, membership=round(m_vol, 4),
                     necessary=True, note=f"rvol={rvol_i:.2f}" if np.isfinite(rvol_i) else "rvol=nan"),
            Evidence(rule_id="bc.range", weight=params.bc_w_range, membership=round(m_range, 4),
                     note=f"range={range_atr:.2f}ATR"),
            Evidence(rule_id="bc.tail_close", weight=params.bc_w_tail_close, membership=round(m_tail, 4),
                     note=f"close_pos={cp:.2f}"),
            Evidence(rule_id="bc.recover", weight=params.bc_w_recover, membership=round(m_recover, 4),
                     note="创新高后回落"),
        ]
        agg = aggregate(evs, params=agg_params)
        if agg.vetoed:
            continue
        if best is None or agg.probability > best[0]:
            best = (agg.probability, i, evs)

    if best is None:
        return []
    _, i, evs = best
    agg = aggregate(evs, params=agg_params)
    return [
        WyckoffEvent(
            event_type=EventType.BC, ticker=daily.symbol.ticker, date=dates[i].date(),
            probability=agg.probability, confidence=agg.confidence, phase_context=phase_context,
            tr_ref=tr, reason=agg.reason, evidence=evs,
            meta={
                "close_pos": round(_close_pos(highs[i], lows[i], closes[i]), 3),
                "range_atr": round((highs[i] - lows[i]) / atr[i], 3),
                "rvol": round(float(rvol[i]), 3) if np.isfinite(rvol[i]) else None,
                "high": round(float(highs[i]), 4),
            },
            engine_version=ENGINE_VERSION,
        )
    ]


# ─────────────────────────────── UT ───────────────────────────────
def detect_ut(
    daily: OHLCVSeries,
    indicators: IndicatorSet,
    tr: TradingRange,
    *,
    phase_context: WyckoffPhase = WyckoffPhase.DISTRIBUTION,
    params: DistributionParams = DistributionParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[WyckoffEvent]:
    """向上假突破 (§4.8, 镜像 Spring)。刺穿 TR.upper 后回落; 站稳延续则否决为 SOS。"""
    win = _win(daily, tr)
    n = len(win)
    if n == 0:
        return []
    dates = win.index
    highs = win["high"].to_numpy(dtype=float)
    lows = win["low"].to_numpy(dtype=float)
    closes = win["close"].to_numpy(dtype=float)
    atr = indicators.get(params.atr_key).reindex(dates).to_numpy(dtype=float)
    rvol = indicators.get(params.rvol_key).reindex(dates).to_numpy(dtype=float)

    # TR 内 daily swing high 价格 (前高比较基准)。
    sh_prices = [
        (pd.Timestamp(s.date), s.price)
        for s in indicators.swings
        if s.type.value == "high"
        and pd.Timestamp(tr.start) <= pd.Timestamp(s.date) <= pd.Timestamp(tr.end)
    ]

    events: list[WyckoffEvent] = []
    i = 0
    while i < n:
        if highs[i] <= tr.upper:
            i += 1
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            i += 1
            continue
        penetration = highs[i] - tr.upper
        depth_atr = penetration / atr_i
        if depth_atr < params.min_penetration_atr:
            i += 1
            continue

        # 回落: ut_recover_bars 内首个收盘回到 TR.upper 之下。
        recover_lag: int | None = None
        recover_idx = i
        for j in range(i, min(i + params.ut_recover_bars + 1, n)):
            if closes[j] <= tr.upper:
                recover_lag = j - i
                recover_idx = j
                break
        if recover_lag is None:  # 未回落 → 真突破 (SOS), 非 UT (否决语义)
            i += 1
            continue

        evs = _ut_evidences(params, depth_atr, recover_lag, rvol, i, recover_idx,
                            highs, closes, lows, sh_prices, dates, atr_i, tr)
        agg = aggregate(evs, params=agg_params)
        if agg.vetoed:
            i = recover_idx + 1
            continue
        events.append(
            WyckoffEvent(
                event_type=EventType.UT, ticker=daily.symbol.ticker, date=dates[i].date(),
                probability=agg.probability, confidence=agg.confidence, phase_context=phase_context,
                tr_ref=tr, reason=agg.reason, evidence=evs,
                meta={"depth_atr": round(depth_atr, 3), "penetration": round(penetration, 4),
                      "recover_lag": recover_lag},
                engine_version=ENGINE_VERSION,
            )
        )
        i = recover_idx + 1
    return events


def _ut_evidences(params, depth_atr, recover_lag, rvol, pierce_i, recover_idx,
                  highs, closes, lows, sh_prices, dates, atr_i, tr) -> list[Evidence]:  # noqa: ANN001
    # [N] 刺穿: 浅→干净诱多 (=1); 深于 deep_atr → 0 (否决=真突破/SOS)。
    m_pierce = ramp_down(depth_atr, params.ut_max_penetration_atr, params.ut_deep_atr)
    # [W] 快速回落。
    m_recover = ramp_down(float(recover_lag), 0.0, float(params.ut_recover_bars))
    # [W] 放量但无法守住 (刺穿放量 或 回落放量)。
    rvol_p, rvol_r = rvol[pierce_i], rvol[recover_idx]
    m_vp = ramp_up(rvol_p, 1.0, params.vol_high_mult) if np.isfinite(rvol_p) else 0.0
    m_vr = ramp_up(rvol_r, 1.0, params.vol_high_mult) if np.isfinite(rvol_r) else 0.0
    m_volume = max(m_vp, m_vr)
    # [W] 收盘转弱 (收在当日下部)。
    cp = _close_pos(highs[pierce_i], lows[pierce_i], closes[pierce_i])
    m_weakclose = ramp_down(cp, 0.3, 0.6)
    # [W] 未创有效持续新高 (相对此前 swing high 的超出幅度小 → 假突破)。
    pierce_date = dates[pierce_i]
    prior_highs = [p for (d, p) in sh_prices if d < pierce_date]
    if prior_highs:
        overshoot = max(0.0, (highs[pierce_i] - max(prior_highs)) / atr_i)
        m_no_new_high = ramp_down(overshoot, 0.0, params.ut_max_penetration_atr)
    else:
        m_no_new_high = 0.5
    return [
        Evidence(rule_id="ut.pierce", weight=params.ut_w_pierce, membership=round(m_pierce, 4),
                 necessary=True, note=f"depth={depth_atr:.2f}ATR"),
        Evidence(rule_id="ut.recover", weight=params.ut_w_recover, membership=round(m_recover, 4),
                 note=f"lag={recover_lag}bar"),
        Evidence(rule_id="ut.volume", weight=params.ut_w_volume, membership=round(m_volume, 4),
                 note="放量无法守住"),
        Evidence(rule_id="ut.weakclose", weight=params.ut_w_weakclose, membership=round(m_weakclose, 4),
                 note=f"close_pos={cp:.2f}"),
        Evidence(rule_id="ut.no_new_high", weight=params.ut_w_no_new_high,
                 membership=round(m_no_new_high, 4), note="未创有效新高"),
    ]


# ────────────────────────────── UTAD ──────────────────────────────
def detect_utad(
    daily: OHLCVSeries,
    indicators: IndicatorSet,
    tr: TradingRange,
    *,
    prior_events: Sequence[WyckoffEvent] = (),
    phase_context: WyckoffPhase = WyckoffPhase.DISTRIBUTION,
    params: DistributionParams = DistributionParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[WyckoffEvent]:
    """派发后向上假突破 (§4.9)。刺穿创区间新高后快速反转跌回, 后续跌破下沿确认。

    prior_events: 此前已检出的顶部事件 (用于 "已有 BC+UT 结构" 证据)。
    """
    win = _win(daily, tr)
    n = len(win)
    if n == 0:
        return []
    dates = win.index
    highs = win["high"].to_numpy(dtype=float)
    lows = win["low"].to_numpy(dtype=float)
    closes = win["close"].to_numpy(dtype=float)
    atr = indicators.get(params.atr_key).reindex(dates).to_numpy(dtype=float)
    rvol = indicators.get(params.rvol_key).reindex(dates).to_numpy(dtype=float)

    has_bc = any(e.event_type == EventType.BC for e in prior_events)
    has_ut = any(e.event_type == EventType.UT for e in prior_events)
    m_structure = 1.0 if (has_bc and has_ut) else (0.5 if (has_bc or has_ut) else 0.0)

    events: list[WyckoffEvent] = []
    i = 0
    while i < n:
        if highs[i] <= tr.upper:
            i += 1
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            i += 1
            continue
        penetration = highs[i] - tr.upper
        if penetration / atr_i < params.min_penetration_atr:
            i += 1
            continue

        # 快速反转跌回 TR 内 (收盘 ≤ upper) 且随后跌势延续。
        reversal_lag: int | None = None
        rev_idx = i
        for j in range(i, min(i + params.utad_recover_bars + 1, n)):
            if closes[j] <= tr.upper:
                reversal_lag = j - i
                rev_idx = j
                break
        if reversal_lag is None:  # 未反转 → 持续走高, 非 UTAD (否决)
            i += 1
            continue

        # 反转后是否逼近/跌破 TR.mid (取反转后 confirm 窗口内的最低收盘)。
        post = closes[rev_idx : min(rev_idx + params.confirm_bars + 1, n)]
        min_post_close = float(np.min(post)) if len(post) else closes[rev_idx]
        m_belowmid = ramp_down((min_post_close - tr.lower) / (tr.mid - tr.lower), 0.0, 1.0) \
            if tr.mid > tr.lower else 0.0

        rvol_r = rvol[rev_idx]
        m_reversal = ramp_down(float(reversal_lag), 0.0, float(params.utad_recover_bars))
        m_volume = ramp_up(rvol_r, 1.0, params.vol_high_mult) if np.isfinite(rvol_r) else 0.0

        evs = [
            Evidence(rule_id="utad.pierce", weight=params.utad_w_pierce, membership=1.0,
                     necessary=True, note=f"pen={penetration / atr_i:.2f}ATR 创区间新高"),
            Evidence(rule_id="utad.reversal", weight=params.utad_w_reversal, membership=round(m_reversal, 4),
                     note=f"lag={reversal_lag}bar"),
            Evidence(rule_id="utad.volume", weight=params.utad_w_volume, membership=round(m_volume, 4),
                     note="反转放量"),
            Evidence(rule_id="utad.structure", weight=params.utad_w_structure, membership=round(m_structure, 4),
                     note=f"BC={has_bc} UT={has_ut}"),
            Evidence(rule_id="utad.belowmid", weight=params.utad_w_belowmid, membership=round(m_belowmid, 4),
                     note="跌回逼近下沿"),
        ]
        agg = aggregate(evs, params=agg_params)
        if agg.vetoed:
            i = rev_idx + 1
            continue

        # 后续确认: confirm_bars 内放量跌破 TR.lower → 进入 Markdown。
        confirmation_date = None
        for k in range(rev_idx, min(rev_idx + params.confirm_bars + 1, n)):
            if closes[k] < tr.lower and np.isfinite(rvol[k]) and rvol[k] >= params.vol_high_mult:
                confirmation_date = dates[k].date()
                break

        events.append(
            WyckoffEvent(
                event_type=EventType.UTAD, ticker=daily.symbol.ticker, date=dates[i].date(),
                confirmation_date=confirmation_date,
                probability=agg.probability, confidence=agg.confidence, phase_context=phase_context,
                tr_ref=tr, reason=agg.reason, evidence=evs,
                meta={"penetration": round(penetration, 4), "reversal_lag": reversal_lag,
                      "has_bc": has_bc, "has_ut": has_ut},
                engine_version=ENGINE_VERSION,
            )
        )
        i = rev_idx + 1
    return events
