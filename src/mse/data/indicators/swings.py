"""Swing / Pivot 检测 (Spec §1.1)。

Swing High/Low: 局部极值, 左右各 k 根 bar 比它低/高 (k = swing_lookback)。
Pivot: 经 ATR 幅度过滤的显著 Swing (强度 = 与邻近对称摆动的幅度 / ATR)。

point-in-time 说明: 判定一个 swing 需要其右侧 k 根 bar, 故一个 swing 在其
发生 k 根之后才能被"确认"。检测函数返回全部 swing (含日期), 下游按需做
point-in-time 过滤 (只用已确认的)。
"""

from __future__ import annotations

import pandas as pd

from mse.core.models.indicators import Pivot, Swing, SwingType


def detect_swings(frame: pd.DataFrame, lookback: int = 5) -> list[Swing]:
    """检测 Swing High/Low。

    Swing High @ i: high[i] 严格大于左右各 lookback 根的 high。
    Swing Low  @ i: low[i]  严格小于左右各 lookback 根的 low。
    """
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    dates = frame.index
    n = len(frame)
    swings: list[Swing] = []

    for i in range(lookback, n - lookback):
        left = slice(i - lookback, i)
        right = slice(i + 1, i + 1 + lookback)

        if highs[i] > highs[left].max() and highs[i] > highs[right].max():
            swings.append(
                Swing(date=dates[i].date(), price=float(highs[i]), type=SwingType.HIGH)
            )
        if lows[i] < lows[left].min() and lows[i] < lows[right].min():
            swings.append(
                Swing(date=dates[i].date(), price=float(lows[i]), type=SwingType.LOW)
            )

    swings.sort(key=lambda s: s.date)
    return swings


def detect_pivots(
    frame: pd.DataFrame,
    atr: pd.Series,
    lookback: int = 5,
    min_atr: float = 1.0,
) -> list[Pivot]:
    """从 swings 中过滤出显著 pivot (幅度 / ATR ≥ min_atr)。

    强度定义: 该 swing 相对前一个反向 swing 的价格落差 / 当时 ATR。
    """
    swings = detect_swings(frame, lookback=lookback)
    if not swings:
        return []

    pivots: list[Pivot] = []
    prev: Swing | None = None
    for sw in swings:
        if prev is not None and prev.type != sw.type:
            ts = pd.Timestamp(sw.date)
            atr_val = float(atr.loc[ts]) if ts in atr.index and pd.notna(atr.loc[ts]) else None
            if atr_val and atr_val > 0:
                strength = abs(sw.price - prev.price) / atr_val
                if strength >= min_atr:
                    pivots.append(
                        Pivot(
                            date=sw.date,
                            price=sw.price,
                            type=sw.type,
                            strength=round(strength, 3),
                        )
                    )
        prev = sw
    return pivots
