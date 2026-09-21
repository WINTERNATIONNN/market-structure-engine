"""内置指标 (Phase 1) — MA/EMA/ATR/RSI/MACD/RVOL。

全部为纯函数、point-in-time 安全 (仅用当前及历史 bar, 无未来数据)。
通过 @registry.register 注册到全局 registry。

命名约定: 指标名含参数, 如 'ema_20', 'atr_14', 'rvol_20' — 便于同一指标多参数并存。
但注册名用基名 (如 'ema'), 参数在 compute 时传入; pipeline 负责拼最终 key。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mse.data.indicators.registry import registry


@registry.register("sma")
def sma(frame: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """简单移动平均 (MA)。"""
    return frame[column].rolling(window=period, min_periods=period).mean()


@registry.register("ema")
def ema(frame: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """指数移动平均。"""
    return frame[column].ewm(span=period, adjust=False, min_periods=period).mean()


@registry.register("atr")
def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真实波幅 (Wilder)。用于波动率归一化 (Spec 中大量 ×ATR 阈值)。"""
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    # Wilder 平滑 = EMA(alpha=1/period)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


@registry.register("rsi")
def rsi(frame: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """相对强弱指标 (Wilder 平滑)。"""
    delta = frame[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # avg_loss=0 (全涨) → RSI=100
    out = out.where(avg_loss != 0, 100.0)
    return out


@registry.register("macd")
def macd(
    frame: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
    line: str = "macd",
) -> pd.Series:
    """MACD。line 参数选择返回哪条线: 'macd' | 'signal' | 'hist'。

    (拆成单 Series 返回以适配 registry 的 Series 契约; pipeline 会分别请求三条。)
    """
    price = frame[column]
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    if line == "macd":
        return macd_line
    if line == "signal":
        return signal_line
    if line == "hist":
        return macd_line - signal_line
    raise ValueError(f"macd line 参数非法: {line!r} (应为 macd/signal/hist)")


@registry.register("rvol")
def rvol(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """相对成交量 RVOL = volume / rolling_mean(volume) (Spec §1.3)。

    放量/缩量判断的基础。用 shift(1) 的均值? — 否: 用含当日的均值是常规做法,
    但为严格 point-in-time 且避免当日自身稀释, 这里用截至前一日的均值。
    """
    vol = frame["volume"]
    baseline = vol.rolling(window=period, min_periods=period).mean().shift(1)
    return vol / baseline
