"""指标管线 — 组装 IndicatorSet (Phase 1)。

按一份 spec (指标名 + 参数) 批量计算, 填入 IndicatorSet.series。
spec 可配置化 (未来从 yaml 注入), 目前给出默认集。

命名: 最终 series key 由基名 + 关键参数拼成, 如 'ema_20'、'atr_14'、'macd_hist'。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mse.core.enums import Timeframe
from mse.core.models import IndicatorSet, OHLCVSeries
from mse.data.indicators import builtin  # noqa: F401  触发注册
from mse.data.indicators.registry import registry
from mse.data.indicators.swings import detect_pivots, detect_swings


@dataclass(frozen=True)
class IndicatorSpec:
    """单个指标的计算规格。key = 最终存入 IndicatorSet.series 的名字。"""

    key: str
    name: str                      # registry 中的基名
    params: dict = field(default_factory=dict)


# 默认指标集 (对应 Spec §1: MA/EMA/ATR/RSI/MACD/Volume)。
DEFAULT_SPECS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec("sma_50", "sma", {"period": 50}),
    IndicatorSpec("ema_20", "ema", {"period": 20}),
    IndicatorSpec("atr_14", "atr", {"period": 14}),
    IndicatorSpec("rsi_14", "rsi", {"period": 14}),
    IndicatorSpec("macd", "macd", {"line": "macd"}),
    IndicatorSpec("macd_signal", "macd", {"line": "signal"}),
    IndicatorSpec("macd_hist", "macd", {"line": "hist"}),
    IndicatorSpec("rvol_20", "rvol", {"period": 20}),
)


def build_indicator_set(
    ohlcv: OHLCVSeries,
    specs: tuple[IndicatorSpec, ...] = DEFAULT_SPECS,
    *,
    swing_lookback: int = 5,
    pivot_min_atr: float = 1.0,
    atr_key: str = "atr_14",
) -> IndicatorSet:
    """计算全部指标 + swings/pivots, 返回 IndicatorSet。"""
    frame = ohlcv.frame
    series: dict = {}
    for spec in specs:
        series[spec.key] = registry.compute(spec.name, frame, **spec.params)

    # swings 与 pivots (pivots 需 ATR 归一化)。
    swings = detect_swings(frame, lookback=swing_lookback)
    atr = series.get(atr_key)
    pivots = (
        detect_pivots(frame, atr, lookback=swing_lookback, min_atr=pivot_min_atr)
        if atr is not None
        else []
    )

    return IndicatorSet(
        ticker=ohlcv.symbol.ticker,
        timeframe=ohlcv.timeframe,
        series=series,
        swings=swings,
        pivots=pivots,
    )
