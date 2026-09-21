"""证据 / 事件契约模型 (Spec §2, §9)。

这是取代 if-else 的核心数据结构: 事件 = 一组带权证据的聚合。
本文件只定义**数据契约** (pydantic 强类型); 聚合算法在 wyckoff/evidence.py。

红线 (Spec §9): 没有 bool 字段表示"是不是"。是与不是由 probability + confidence
+ 下游阈值共同决定。
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models.structure import TradingRange


class Evidence(BaseModel):
    """单条证据 = 原子条件的命中结果 (Spec §2.1)。

    membership 是**模糊隶属度** (0..1), 不是 bool —— 避免阈值悬崖 (§2.2)。
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(..., min_length=1, description="规则标识, 如 'spring.penetration'")
    weight: float = Field(..., ge=0, description="证据权重 (相对重要性)")
    membership: float = Field(..., ge=0, le=1, description="隶属度 0..1")
    note: str = Field(default="", description="人类可读的命中说明")
    necessary: bool = Field(
        default=False, description="是否必要条件 (低于 floor 触发一票否决, §2.4)"
    )


class WyckoffEvent(BaseModel):
    """Wyckoff 事件输出契约 (Spec §9, 所有事件统一 schema)。"""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    ticker: str
    date: datetime.date = Field(..., description="事件确认发生日")
    confirmation_date: datetime.date | None = Field(
        default=None, description="需后续确认的事件才有 (Spec §1.4)"
    )

    probability: float = Field(..., ge=0, le=1, description="'是该事件'的可能性 (证据加权聚合)")
    confidence: float = Field(..., ge=0, le=1, description="对判断的确定程度 (完整性/一致性/数据质量)")

    phase_context: WyckoffPhase = Field(
        default=WyckoffPhase.UNDEFINED, description="事件发生时的 FSM 阶段语境"
    )
    tr_ref: TradingRange | None = Field(default=None, description="关联交易区间")

    reason: str = Field(default="", description="模板化摘要 (给人看)")
    evidence: list[Evidence] = Field(default_factory=list, description="结构化证据 (给下游/LLM)")
    meta: dict = Field(default_factory=dict, description="亚型/深度等细节")
    engine_version: str = Field(default="unset", description="可复现的引擎/规则版本")
