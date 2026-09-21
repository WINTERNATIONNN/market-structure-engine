"""CandidateProfile —— 扫描器对单只股票的打分结果 (Scoring 层输出)。

红线一致: 输出**非布尔** —— 带 composite/分项分数 + 方向 + 可读 tags,
供下游排序与人工审阅, 而非简单的"选中/未选中"。
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import TradingRange


class EventRef(BaseModel):
    """候选相关的一个近期事件的轻量引用 (给榜单展示/解释)。"""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    date: date
    probability: float = Field(..., ge=0.0, le=1.0)


class TransitionRef(BaseModel):
    """最近一次阶段转折 (FSM 进入新阶段) 的引用。"""

    model_config = ConfigDict(frozen=True)

    from_phase: WyckoffPhase
    to_phase: WyckoffPhase
    entered_date: date
    bars_ago: int = Field(..., ge=0, description="距今多少个 (周线) bar")
    probability: float = Field(..., ge=0.0, le=1.0)


class CandidateProfile(BaseModel):
    """单只股票的扫描打分档案。"""

    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date = Field(..., description="评估基准日 (最新周线 bar)")

    # 当前阶段快照
    current_phase: WyckoffPhase
    phase_probability: float = Field(..., ge=0.0, le=1.0)
    phase_confidence: float = Field(..., ge=0.0, le=1.0)

    # 结构 / 位置
    active_tr: TradingRange | None = None
    price_pos_in_tr: float | None = Field(
        default=None, description="最新收盘价在 TR 内相对位置 (0=下沿,1=上沿); 无 TR→None"
    )

    # 近期事件 (已按 recency 过滤)
    recent_bull_events: list[EventRef] = Field(default_factory=list)
    recent_bear_events: list[EventRef] = Field(default_factory=list)
    last_transition: TransitionRef | None = None

    # 打分 (均 0..1)
    springboard_score: float = Field(..., ge=0.0, le=1.0, description="看涨吸筹 setup 强度")
    transition_score: float = Field(..., ge=0.0, le=1.0, description="近期阶段转折强度")
    transition_direction: str | None = Field(
        default=None, description="'bullish' / 'bearish' / None"
    )
    composite_score: float = Field(..., ge=0.0, le=1.0, description="排序主键 = max(两分项)")

    tags: list[str] = Field(default_factory=list, description="可读理由标签")
