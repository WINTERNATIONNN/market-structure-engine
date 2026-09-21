"""结构模型 — TradingRange (交易区间, Spec §1.2)。

TR 是 Wyckoff 分析的核心结构对象: Phase FSM 与所有区间类事件
(Spring / SOS / UT / ST ...) 都锚定在它的上下沿上。

按 Spec §10.1 (multi-timeframe): TR 在**周线**锚定, 故带 `timeframe` 字段。
边界 (upper/lower) 是 swing 的**聚类中枢**而非绝对极值 —— 这样"刺穿"
(Spring 跌破 lower / UT 突破 upper) 才是可度量的事件。
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mse.core.enums import Timeframe


class TradingRange(BaseModel):
    """交易区间 (Spec §1.2)。

    属性对应 Spec: {start, end, upper, lower, mid, duration, avg_volume, volume_trend}。
    额外记录 width_atr / slope_atr / 触碰数, 供下游解释与调参; mid/width 为派生属性。
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    timeframe: Timeframe

    start: date
    end: date
    duration: int = Field(..., gt=0, description="区间跨越的 bar 数")

    upper: float = Field(..., gt=0, description="上沿 (阻力): 区间内 SH 聚类中枢")
    lower: float = Field(..., gt=0, description="下沿 (支撑): 区间内 SL 聚类中枢")

    avg_volume: float = Field(..., ge=0, description="区间内平均成交量")
    volume_trend: float = Field(
        ..., description="成交量回归斜率 / 均量 (每 bar 相对变化, 供/需趋势)"
    )

    width_atr: float = Field(..., ge=0, description="振幅 (max_high-min_low) / ATR")
    slope_atr: float = Field(..., ge=0, description="|收盘回归斜率/bar| / ATR (横盘度, 越小越平)")
    num_sh: int = Field(..., ge=0, description="定义上沿的 swing high 触碰数")
    num_sl: int = Field(..., ge=0, description="定义下沿的 swing low 触碰数")

    @model_validator(mode="after")
    def _check_bounds(self) -> TradingRange:
        if self.upper <= self.lower:
            raise ValueError(f"TR 上沿必须高于下沿: upper={self.upper} lower={self.lower}")
        if self.end < self.start:
            raise ValueError(f"TR end 早于 start: {self.start}..{self.end}")
        return self

    # ── 派生属性 ──────────────────────────────────────────────
    @property
    def mid(self) -> float:
        return (self.upper + self.lower) / 2.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    # ── 事件检测辅助 (供 Phase 2 事件层) ──────────────────────
    def contains_date(self, when: date) -> bool:
        """when 是否落在区间时间跨度内。"""
        return self.start <= when <= self.end

    def position(self, price: float) -> float:
        """价格在区间内的相对位置: 0=下沿, 1=上沿。

        <0 表示刺穿下沿 (Spring 候选), >1 表示刺穿上沿 (UT/SOS 候选)。
        供事件层度量刺穿深度用。
        """
        w = self.width
        return (price - self.lower) / w if w > 0 else 0.5
