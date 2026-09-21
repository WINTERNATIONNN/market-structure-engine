"""Spring 检测 (Spec §4.5) —— 弹簧 / 假跌破。

Accumulation 末期, 价格盘中刺穿 TR.lower 诱空后快速收回。最重要的进场结构之一。
日线事件, 接收周线 TR + phase 语境 (Spec §10.1)。

证据 (Spec §4.5):
    [N w=1.0]  刺穿 TR.lower —— 深度越浅越"干净"; 过深 (> deep_atr) 视为真跌破而否决。
    [W 0.35]   快速收回 TR 内 (收回越快隶属越高)。
    [W 0.25]   刺穿时缩量 或 收回时放量 (供给枯竭 / 需求进场), 取二者较强。
    [W 0.20]   刺穿深度浅 (干净) → 越浅越高。
    [W 0.20]   刺穿低点未显著低于此前低点 (前低 = TR 内 daily swing low 聚类)。
后续确认 (§1.4): 收回后 confirm_bars 内出现放量上涨 bar → confirmation_date;
    未确认则 probability × spring_unconfirmed_factor。
否决 (§4.5): 刺穿后在 recover_bars 内未收回 → 真跌破, 不作为 Spring 候选 (跳过)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import Evidence, IndicatorSet, OHLCVSeries, TradingRange, WyckoffEvent
from mse.wyckoff.evidence import ENGINE_VERSION, aggregate
from mse.wyckoff.membership import ramp_down, ramp_up
from mse.wyckoff.params import AggregationParams, SpringParams


def detect_springs(
    daily: OHLCVSeries,
    indicators: IndicatorSet,
    tr: TradingRange,
    *,
    phase_context: WyckoffPhase = WyckoffPhase.ACCUMULATION,
    params: SpringParams = SpringParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[WyckoffEvent]:
    """在 TR 时间窗内的日线上检测 Spring 事件, 按时间升序返回。"""
    frame = daily.frame
    # 限定在 TR 时间跨度内 (Spring 属该区间末段)。
    mask = (frame.index >= pd.Timestamp(tr.start)) & (frame.index <= pd.Timestamp(tr.end))
    win = frame.loc[mask]
    n = len(win)
    if n == 0:
        return []

    dates = win.index
    lows = win["low"].to_numpy(dtype=float)
    closes = win["close"].to_numpy(dtype=float)
    opens = win["open"].to_numpy(dtype=float)
    atr = indicators.get(params.atr_key).reindex(dates).to_numpy(dtype=float)
    rvol = indicators.get(params.rvol_key).reindex(dates).to_numpy(dtype=float)

    # TR 内 daily swing low 价格 (前低比较基准)。日期 → 价格。
    sl_prices: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(s.date), s.price)
        for s in indicators.swings
        if s.type.value == "low" and pd.Timestamp(tr.start) <= pd.Timestamp(s.date) <= pd.Timestamp(tr.end)
    ]

    events: list[WyckoffEvent] = []
    i = 0
    while i < n:
        if lows[i] >= tr.lower:  # 未刺穿下沿
            i += 1
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            i += 1
            continue

        penetration = tr.lower - lows[i]
        depth_atr = penetration / atr_i
        if depth_atr < params.min_penetration_atr:  # 亚噪声级刺穿, 忽略
            i += 1
            continue

        # 收回: recover_bars 内首个收盘回到 TR.lower 之上。
        recover_lag: int | None = None
        for j in range(i, min(i + params.spring_recover_bars + 1, n)):
            if closes[j] >= tr.lower:
                recover_lag = j - i
                recover_idx = j
                break
        if recover_lag is None:  # 未收回 → 真跌破, 非 Spring (否决语义)
            i += 1
            continue

        evidences = _build_evidences(
            params, tr, depth_atr, recover_lag, rvol, i, recover_idx, lows, sl_prices, dates, atr_i
        )
        agg = aggregate(evidences, params=agg_params)
        if agg.vetoed:  # 刺穿过深等 → 判定不成立
            i = recover_idx + 1
            continue

        # 后续确认: 收回后 confirm_bars 内的放量上涨 bar。
        confirmation_date = None
        for k in range(recover_idx, min(recover_idx + params.confirm_bars + 1, n)):
            if closes[k] > opens[k] and np.isfinite(rvol[k]) and rvol[k] >= params.vol_high_mult:
                confirmation_date = dates[k].date()
                break

        probability = agg.probability
        if confirmation_date is None:
            probability = round(probability * params.spring_unconfirmed_factor, 4)

        events.append(
            WyckoffEvent(
                event_type=EventType.SPRING,
                ticker=daily.symbol.ticker,
                date=dates[i].date(),
                confirmation_date=confirmation_date,
                probability=probability,
                confidence=agg.confidence,
                phase_context=phase_context,
                tr_ref=tr,
                reason=agg.reason,
                evidence=evidences,
                meta=_build_meta(depth_atr, penetration, recover_lag, params),
                engine_version=ENGINE_VERSION,
            )
        )
        i = recover_idx + 1  # 跳过本次 spring 覆盖的 bar

    return events


def _build_evidences(
    params: SpringParams,
    tr: TradingRange,
    depth_atr: float,
    recover_lag: int,
    rvol: np.ndarray,
    pierce_i: int,
    recover_idx: int,
    lows: np.ndarray,
    sl_prices: list[tuple[pd.Timestamp, float]],
    dates: pd.DatetimeIndex,
    atr_i: float,
) -> list[Evidence]:
    # [N] 刺穿: 浅→干净 (=1); 深于 deep_atr → 0 (否决真跌破)。
    m_pierce = ramp_down(depth_atr, params.spring_max_depth_atr, params.spring_deep_atr)

    # [W] 快速收回: lag 0 → 1, 到时限 → 递减。
    m_recover = ramp_down(float(recover_lag), 0.0, float(params.spring_recover_bars))

    # [W] 成交量: 刺穿缩量 或 收回放量, 取较强。
    rvol_pierce = rvol[pierce_i]
    rvol_recover = rvol[recover_idx]
    m_low = ramp_down(rvol_pierce, params.vol_low_mult, 1.0) if np.isfinite(rvol_pierce) else 0.0
    m_high = ramp_up(rvol_recover, 1.0, params.vol_high_mult) if np.isfinite(rvol_recover) else 0.0
    m_volume = max(m_low, m_high)

    # [W] 刺穿浅 (干净): depth 0 → 1, 达 max_depth → 0。
    m_depth = ramp_down(depth_atr, 0.0, params.spring_max_depth_atr)

    # [W] 未显著低于此前 daily swing low (前低比较, ST 未实现时的代理)。
    pierce_date = dates[pierce_i]
    prior_lows = [p for (d, p) in sl_prices if d < pierce_date]
    if prior_lows:
        prior_low = min(prior_lows)
        undercut_atr = max(0.0, (prior_low - lows[pierce_i]) / atr_i)
        m_not_undercut = ramp_down(undercut_atr, 0.0, params.spring_max_depth_atr)
    else:
        m_not_undercut = 0.5  # 无前低参照 → 中性

    return [
        Evidence(rule_id="spring.pierce", weight=params.w_pierce, membership=round(m_pierce, 4),
                 necessary=True, note=f"depth={depth_atr:.2f}ATR"),
        Evidence(rule_id="spring.recover", weight=params.w_recover, membership=round(m_recover, 4),
                 note=f"lag={recover_lag}bar"),
        Evidence(rule_id="spring.volume", weight=params.w_volume, membership=round(m_volume, 4),
                 note=f"rvol_pierce={rvol_pierce:.2f} rvol_recover={rvol_recover:.2f}"),
        Evidence(rule_id="spring.depth", weight=params.w_depth, membership=round(m_depth, 4),
                 note="越浅越干净"),
        Evidence(rule_id="spring.not_undercut", weight=params.w_not_undercut,
                 membership=round(m_not_undercut, 4), note="未显著破前低"),
    ]


def _build_meta(depth_atr: float, penetration: float, recover_lag: int, params: SpringParams) -> dict:
    # 亚型: 越浅越接近 terminal shakeout (Spec §4.5 亚型标注)。
    if depth_atr <= params.spring_max_depth_atr:
        spring_type = 1  # 干净浅刺穿
    elif depth_atr <= (params.spring_max_depth_atr + params.spring_deep_atr) / 2:
        spring_type = 2
    else:
        spring_type = 3  # 深刺穿 (临界真跌破)
    return {
        "spring_type": spring_type,
        "terminal_shakeout": spring_type == 1 and recover_lag == 0,
        "depth_atr": round(depth_atr, 3),
        "penetration": round(penetration, 4),
        "recover_lag": recover_lag,
    }
