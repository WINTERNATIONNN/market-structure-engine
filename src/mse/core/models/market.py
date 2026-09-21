"""行情数据模型 — Symbol / Bar / OHLCVSeries (Spec §1.2, Phase 1 Data Model)。

设计要点:
    * 强类型 + 校验 (pydantic v2), 在边界处拦截脏数据。
    * OHLCVSeries 内部持有 pandas DataFrame (计算高效), 但对外暴露受控接口。
    * 一切带 timeframe (multi-timeframe 决策, Spec §10.1)。
    * 美股: 数据为已复权 (yfinance auto_adjust); 无涨跌停字段。
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mse.core.enums import DataQualityFlag, Timeframe

# OHLCVSeries.frame 的规范列 (统一契约, 下游依赖此约定)。
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class Symbol(BaseModel):
    """标的元数据 (Spec §1: Symbol)。行业/市值供 Scanner 与第二层分析使用。"""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., min_length=1, description="如 AAPL")
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = Field(default=None, ge=0)
    float_shares: float | None = Field(default=None, ge=0)
    exchange: str | None = None
    currency: str = "USD"

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.strip().upper()


class Bar(BaseModel):
    """单根 K 线 (Spec §1: Bar)。用于逐事件引用与序列化; 批量计算走 OHLCVSeries。"""

    model_config = ConfigDict(frozen=True)

    date: date
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    quality: DataQualityFlag = DataQualityFlag.OK

    @model_validator(mode="after")
    def _check_ohlc_consistency(self) -> Bar:
        # high 必须是当日上界, low 必须是下界; 否则数据错乱。
        hi = max(self.open, self.close, self.low)
        lo = min(self.open, self.close, self.high)
        if self.high < hi or self.low > lo:
            raise ValueError(
                f"OHLC 不一致 @ {self.date}: "
                f"O={self.open} H={self.high} L={self.low} C={self.close}"
            )
        return self


class OHLCVSeries(BaseModel):
    """一只标的在单一周期上的完整行情序列。

    内部以 pandas DataFrame 存储 (计算友好), 对外通过类型化接口访问。
    这是 Data Layer 输出、Wyckoff Engine 输入的核心载体。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    symbol: Symbol
    timeframe: Timeframe
    frame: pd.DataFrame = Field(..., repr=False)
    adjusted: bool = Field(default=True, description="是否已复权 (美股默认 True)")
    fetched_at: datetime | None = None

    @field_validator("frame")
    @classmethod
    def _validate_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCVSeries.frame 缺少列: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OHLCVSeries.frame 索引必须是 DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            raise ValueError("OHLCVSeries.frame 索引必须按时间升序 (point-in-time 前提)")
        if df.index.has_duplicates:
            raise ValueError("OHLCVSeries.frame 索引存在重复日期")
        return df

    # ── 受控访问接口 ──────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.frame)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.frame.index

    @property
    def close(self) -> pd.Series:
        return self.frame["close"]

    @property
    def volume(self) -> pd.Series:
        return self.frame["volume"]

    def slice_until(self, as_of: date) -> OHLCVSeries:
        """返回 as_of (含) 之前的子序列 — 强制 point-in-time, 防前视偏差 (Spec §8)。"""
        sub = self.frame.loc[self.frame.index <= pd.Timestamp(as_of)]
        return self.model_copy(update={"frame": sub})

    def bar_at(self, when: date) -> Bar:
        """取某日的类型化 Bar。"""
        row = self.frame.loc[pd.Timestamp(when)]
        return Bar(
            date=when,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
