"""PS (Preliminary Support) 检测 (Spec §4.1)。

Accumulation 序列的**首个**事件: 下跌趋势中首次出现的显著承接, 预示卖压开始被吸收。
与 SC/AR/ST 不同, PS 出现在 **尚无成形 TR** 的 MARKDOWN 末段, 故不接收 TradingRange,
自身从收盘回归斜率判定 "前期明确下跌" 这一必要条件。

红线: 零 LLM, 纯计算。
"""

from __future__ import annotations

import numpy as np

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import Evidence, IndicatorSet, OHLCVSeries, WyckoffEvent
from mse.wyckoff.evidence import ENGINE_VERSION, aggregate
from mse.wyckoff.membership import ramp_down, ramp_up
from mse.wyckoff.params import AggregationParams, PSParams


def _close_pos(high: float, low: float, close: float) -> float:
    rng = high - low
    return (close - low) / rng if rng > 0 else 0.5


def _reg_slope(y: np.ndarray) -> float:
    """最小二乘拟合的每-bar 斜率 (窗口长度 < 2 → 0)。"""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def detect_ps(
    daily: OHLCVSeries,
    indicators: IndicatorSet,
    *,
    phase_context: WyckoffPhase = WyckoffPhase.MARKDOWN,
    params: PSParams = PSParams(),
    agg_params: AggregationParams = AggregationParams(),
) -> list[WyckoffEvent]:
    """初步支撑 (§4.1)。返回概率最高的单个 PS; 无合格候选则空。

    候选须: 前期明确下跌 (必要) + 放量承接 + 长下影 + 跌速放缓。
    否决 (前视 veto_bars): 之后未再创新低却单边拉升 → 趋势反转而非 PS。
    """
    frame = daily.frame
    n = len(frame)
    dates = frame.index
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    atr = indicators.get(params.atr_key).to_numpy(dtype=float)
    rvol = indicators.get(params.rvol_key).to_numpy(dtype=float)

    # 需要足够的左侧历史来估计趋势斜率。
    start = params.trend_lookback
    best: tuple[float, int, list[Evidence]] | None = None
    for i in range(start, n):
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # [N] 前期明确下跌: 收盘回归斜率 (归一化 ATR) 越负越强。
        window = closes[i - params.trend_lookback : i + 1]
        slope_atr = _reg_slope(window) / atr_i
        strength = -slope_atr  # 下跌 → 正
        thr = -params.downtrend_slope
        m_trend = ramp_up(strength, thr * 0.5, thr)

        # [W] 放量承接。
        rvol_i = rvol[i]
        m_vol = ramp_up(rvol_i, 1.0, params.vol_high_mult) if np.isfinite(rvol_i) else 0.0

        # [W] 长下影 / 收盘远离最低 (承接痕迹)。
        cp = _close_pos(highs[i], lows[i], closes[i])
        m_tail = ramp_up(cp, params.close_pos_lo, params.close_pos_hi)

        # [W] 跌速放缓 (相对前 decel_lookback 根的平均单-bar 跌幅)。
        p0 = max(0, i - params.decel_lookback)
        prior_drops = closes[p0 : i] - closes[p0 + 1 : i + 1]  # 正 = 下跌
        prior_drop = float(np.mean(prior_drops)) if len(prior_drops) else 0.0
        cur_drop = float(closes[i - 1] - closes[i])
        if prior_drop > 0:
            ratio = cur_drop / prior_drop  # <1 = 减速, ≤0 = 反向
            m_decel = ramp_down(ratio, 0.3, 1.0)
        else:
            m_decel = 0.0  # 前期无净下跌 → 无从谈减速

        evs = [
            Evidence(rule_id="ps.trend", weight=params.ps_w_trend, membership=round(m_trend, 4),
                     necessary=True, note=f"slope/ATR={slope_atr:.3f}"),
            Evidence(rule_id="ps.support_vol", weight=params.ps_w_support_vol,
                     membership=round(m_vol, 4),
                     note=f"rvol={rvol_i:.2f}" if np.isfinite(rvol_i) else "rvol=nan"),
            Evidence(rule_id="ps.tail", weight=params.ps_w_tail, membership=round(m_tail, 4),
                     note=f"close_pos={cp:.2f}"),
            Evidence(rule_id="ps.decel", weight=params.ps_w_decel, membership=round(m_decel, 4),
                     note="跌速放缓"),
        ]
        agg = aggregate(evs, params=agg_params)
        if agg.vetoed:
            continue

        # 前视否决: 未再创新低却单边拉升 → 趋势反转, 非 PS。
        hi = min(i + params.veto_bars + 1, n)
        made_new_low = bool(np.any(lows[i + 1 : hi] < lows[i])) if hi > i + 1 else False
        max_rally = float(np.max(highs[i + 1 : hi]) - closes[i]) if hi > i + 1 else 0.0
        if not made_new_low and max_rally / atr_i > params.veto_rally_atr:
            continue

        if best is None or agg.probability > best[0]:
            best = (agg.probability, i, evs)

    if best is None:
        return []
    _, i, evs = best
    agg = aggregate(evs, params=agg_params)
    return [
        WyckoffEvent(
            event_type=EventType.PS, ticker=daily.symbol.ticker, date=dates[i].date(),
            probability=agg.probability, confidence=agg.confidence, phase_context=phase_context,
            tr_ref=None, reason=agg.reason, evidence=evs,
            meta={
                "close_pos": round(_close_pos(highs[i], lows[i], closes[i]), 3),
                "slope_atr": round(_reg_slope(closes[i - params.trend_lookback : i + 1]) / atr[i], 4),
                "rvol": round(float(rvol[i]), 3) if np.isfinite(rvol[i]) else None,
            },
            engine_version=ENGINE_VERSION,
        )
    ]
