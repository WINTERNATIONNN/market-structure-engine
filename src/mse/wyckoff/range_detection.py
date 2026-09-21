"""Range Detection — 交易区间检测 (Spec §1.2)。

从 (周线) 行情 + IndicatorSet 中识别横盘交易区间 (TradingRange)。
这是 Phase 2 三层管线的第一层, 所有区间类事件都锚定在其输出的上下沿上。

算法 (贪心 + swing 聚类):
    1. 从每个起点 i 出发, 向右生长窗口, 记录满足全部 TR 条件的**最长**窗口。
    2. TR 条件 (Spec §1.2):
         a. 时长   duration ≥ tr_min_bars
         b. 振幅   width_atr = (max_high - min_low) / ATR ∈ [min, max]
         c. 横盘   slope_atr = |收盘线性回归斜率/bar| / ATR < tr_max_slope
         d. 触碰   每侧 ≥ min_touches 个 swing (区间被反复测试)
    3. 取到最长窗口后跳到其右侧, 得到互不重叠的区间序列。

边界定义 (Spec §1.2 "SH/SL 聚类"):
    upper = 窗口内 swing high 价格的中位数 (阻力中枢)
    lower = 窗口内 swing low  价格的中位数 (支撑中枢)
    —— 用中枢而非绝对极值, 使 Spring (跌破 lower) / UT (突破 upper) 成为可度量刺穿。

point-in-time: 本函数处理传入的整段 frame。如需 as-of 检测, 调用方先用
`OHLCVSeries.slice_until(as_of)` 截断再传入 (与 swings 一致)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mse.core.models import IndicatorSet, OHLCVSeries, TradingRange
from mse.wyckoff.params import RangeParams


def _slope_per_bar(y: np.ndarray) -> float:
    """普通最小二乘拟合 y ~ x (x=0..n-1), 返回每 bar 斜率。"""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y.mean())).sum() / denom)


def detect_ranges(
    ohlcv: OHLCVSeries,
    indicators: IndicatorSet,
    *,
    params: RangeParams = RangeParams(),
) -> list[TradingRange]:
    """检测行情序列中的全部交易区间, 按时间升序返回 (互不重叠)。"""
    frame = ohlcv.frame
    n = len(frame)
    if n < params.tr_min_bars:
        return []

    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    vols = frame["volume"].to_numpy(dtype=float)
    atr_arr = indicators.get(params.atr_key).to_numpy(dtype=float)
    dates = frame.index

    # swing 日期 → 位置索引 (窗口成员判断)。只保留能对上 frame 的。
    pos_of: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(dates)}
    sh_pos: list[int] = []
    sl_pos: list[int] = []
    for sw in indicators.swings:
        ts = pd.Timestamp(sw.date)
        i = pos_of.get(ts)
        if i is None:
            continue
        (sh_pos if sw.type.value == "high" else sl_pos).append(i)
    sh_pos.sort()
    sl_pos.sort()

    def touches(lo: int, hi: int, positions: list[int]) -> int:
        """positions 中落在 [lo, hi] 内的个数 (positions 已升序)。"""
        left = np.searchsorted(positions, lo, side="left")
        right = np.searchsorted(positions, hi, side="right")
        return int(right - left)

    ranges: list[TradingRange] = []
    start = 0
    min_len = params.tr_min_bars
    while start <= n - min_len:
        best_end = -1  # 满足条件的最长窗口右端
        end = start + min_len - 1
        while end < n:
            atr_ref = float(np.nanmedian(atr_arr[start : end + 1]))
            if not np.isfinite(atr_ref) or atr_ref <= 0:
                end += 1
                continue

            span = float(highs[start : end + 1].max() - lows[start : end + 1].min())
            width_atr = span / atr_ref
            # span 随窗口生长单调不减 → 一旦过宽, 继续生长只会更宽, 停止。
            if width_atr > params.tr_width_atr_max:
                break

            slope_atr = abs(_slope_per_bar(closes[start : end + 1])) / atr_ref
            n_sh = touches(start, end, sh_pos)
            n_sl = touches(start, end, sl_pos)

            if (
                width_atr >= params.tr_width_atr_min
                and slope_atr < params.tr_max_slope
                and n_sh >= params.min_touches
                and n_sl >= params.min_touches
            ):
                best_end = end
            end += 1

        if best_end < 0:
            start += 1
            continue

        ranges.append(_build_range(ohlcv, params, start, best_end, highs, lows, vols, atr_arr, dates, sh_pos, sl_pos))
        start = best_end + 1  # 非重叠: 跳到已接受区间右侧

    return ranges


def _build_range(
    ohlcv: OHLCVSeries,
    params: RangeParams,
    start: int,
    end: int,
    highs: np.ndarray,
    lows: np.ndarray,
    vols: np.ndarray,
    atr_arr: np.ndarray,
    dates: pd.DatetimeIndex,
    sh_pos: list[int],
    sl_pos: list[int],
) -> TradingRange:
    """从已确定的窗口 [start, end] 组装 TradingRange。"""
    sh_in = [p for p in sh_pos if start <= p <= end]
    sl_in = [p for p in sl_pos if start <= p <= end]

    # 上/下沿 = swing 聚类中枢 (中位数, 抗单点极值/刺穿干扰)。
    upper = float(np.median([highs[p] for p in sh_in]))
    lower = float(np.median([lows[p] for p in sl_in]))
    # 极端情况兜底: 中位数塌陷时退回窗口极值, 保证 upper > lower。
    if upper <= lower:
        upper = float(highs[start : end + 1].max())
        lower = float(lows[start : end + 1].min())

    window_vol = vols[start : end + 1]
    avg_volume = float(window_vol.mean())
    vol_slope = _slope_per_bar(window_vol)
    volume_trend = vol_slope / avg_volume if avg_volume > 0 else 0.0

    atr_ref = float(np.nanmedian(atr_arr[start : end + 1]))
    span = float(highs[start : end + 1].max() - lows[start : end + 1].min())
    width_atr = span / atr_ref if atr_ref > 0 else 0.0
    slope_atr = abs(_slope_per_bar(ohlcv.frame["close"].to_numpy(float)[start : end + 1])) / atr_ref if atr_ref > 0 else 0.0

    return TradingRange(
        ticker=ohlcv.symbol.ticker,
        timeframe=ohlcv.timeframe,
        start=dates[start].date(),
        end=dates[end].date(),
        duration=end - start + 1,
        upper=upper,
        lower=lower,
        avg_volume=avg_volume,
        volume_trend=round(volume_trend, 6),
        width_atr=round(width_atr, 3),
        slope_atr=round(slope_atr, 4),
        num_sh=len(sh_in),
        num_sl=len(sl_in),
    )
