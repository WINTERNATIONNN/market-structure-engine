"""Agent 层 —— LLM 叙述: 把结构化证据翻成人话。

红线: 本层是唯一允许出现 LLM 的地方, 且**零计算** (只格式化+转发, 不打分)。
默认 TemplateNarrator 可离线运行; AnthropicNarrator 接入真正的 Claude。

分层: agent → reporting → ranking → analysis → scanner → scoring → wyckoff → data → core。
"""

from mse.agent.explain import explain_candidate, explain_market
from mse.agent.narrator import (
    AnthropicNarrator,
    Narrator,
    TemplateNarrator,
    build_candidate_prompt,
    build_market_prompt,
)
from mse.agent.params import AgentParams

__all__ = [
    "AgentParams",
    "Narrator",
    "TemplateNarrator",
    "AnthropicNarrator",
    "build_candidate_prompt",
    "build_market_prompt",
    "explain_candidate",
    "explain_market",
]
