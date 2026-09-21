"""指标模型 — Swing / Pivot / IndicatorSet (Spec §1.1–§1.3, Phase 1)。

IndicatorSet 是指标计算的统一产物, 挂载于某个 (symbol, timeframe)。
指标值以 pandas Series 存储 (与 OHLCVSeries 索引对齐), 便于 point-in-time 切片。
"""

from __future__ import annotations

from datetime import date
from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mse.core.enums import Timeframe


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class Swing(BaseModel):
    """摆动高/低点 (Spec §1.1)。"""

    model_config = ConfigDict(frozen=True)

    date: date
    price: float
    type: SwingType


class Pivot(BaseModel):
    """显著摆动点 — 经 ATR 幅度过滤的 Swing (Spec §1.1)。"""

    model_config = ConfigDict(frozen=True)

    date: date
    price: float
    type: SwingType
    strength: float = Field(..., ge=0, description="幅度 / ATR, 越大越显著")


class IndicatorSet(BaseModel):
    """某 (symbol, timeframe) 上的全部指标产物 (Spec §1: IndicatorSet)。

    series: 逐 bar 的指标, key=指标名 (如 'ema_20', 'atr_14', 'rvol')。
            由 IndicatorRegistry 填充 (Phase 1 指标层), 可插拔扩展。
    swings/pivots: 结构点, 供 Range Detection 使用。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    ticker: str
    timeframe: Timeframe
    series: dict[str, pd.Series] = Field(default_factory=dict, repr=False)
    swings: list[Swing] = Field(default_factory=list)
    pivots: list[Pivot] = Field(default_factory=list)

    def get(self, name: str) -> pd.Series:
        """取某指标序列; 不存在则报错 (指标名拼写错误应尽早暴露)。"""
        if name not in self.series:
            raise KeyError(
                f"指标 '{name}' 未计算。已有: {sorted(self.series)}"
            )
        return self.series[name]

    def has(self, name: str) -> bool:
        return name in self.series

    def value_at(self, name: str, when: date) -> float:
        """point-in-time 取值。"""
        return float(self.get(name).loc[pd.Timestamp(when)])
