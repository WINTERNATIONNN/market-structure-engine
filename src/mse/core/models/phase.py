"""Phase 判定输出 — PhaseResult (Spec §3.4)。

Phase FSM (周线) 每进入一个阶段产出一个 PhaseResult, 描述该阶段区段:
标签 / 概率 / 置信度 / 结构化证据 / 起止时间。与 WyckoffEvent 一致,
probability (是该阶段的可能性) 与 confidence (对判断的确定程度) 独立输出 (§2.3)。
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from mse.core.enums import Timeframe, WyckoffPhase
from mse.core.models.evidence import Evidence
from mse.core.models.structure import TradingRange


class PhaseResult(BaseModel):
    """单个阶段区段的判定 (Spec §3.4)。FSM 输出一串 PhaseResult 构成阶段时间线。"""

    model_config = ConfigDict(frozen=True)

    label: WyckoffPhase
    ticker: str
    timeframe: Timeframe

    probability: float = Field(..., ge=0.0, le=1.0, description="处于该阶段的可能性")
    confidence: float = Field(..., ge=0.0, le=1.0, description="对该判定的确定程度 (独立于 probability)")

    state_entered_date: date = Field(..., description="进入该阶段的 bar 日期 (§3.4)")
    as_of_date: date = Field(..., description="该区段最后评估的 bar 日期 (point-in-time)")

    reason: str = Field(..., description="模板化结构说明 (非自然语言, 供 LLM 二次生成)")
    evidence: list[Evidence] = Field(default_factory=list)
    tr_ref: TradingRange | None = Field(default=None, description="锚定该阶段的 TR (若有)")

    engine_version: str = ""
