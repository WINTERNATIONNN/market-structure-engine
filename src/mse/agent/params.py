"""Agent 层参数 (无魔法数字)。控制 LLM 调用的模型/长度/温度/语言。

红线: Agent 层**零计算** —— 只把引擎算好的结构化证据翻译成人话, 不重新打分。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentParams:
    """LLM 叙述参数。model 默认取当前最强且性价比合适的 Claude (可覆盖)。"""

    model: str = "claude-sonnet-latest"   # 用 -latest 别名: 兼容网关 (数字型 ID 可能被拒), 直连亦有效
    max_tokens: int = 1024
    temperature: float = 0.2         # 偏低: 解释类任务要稳定、少发散
    language: str = "zh"             # 输出语言 (zh / en)
    max_retries: int = 3             # 偶发 5xx/网络抖动时的重试次数 (含首次共 max_retries 次)
    retry_backoff_s: float = 1.0     # 重试线性退避基数 (第 n 次等待 n × 该值秒)

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens 必须 ≥ 1")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature 须在 [0,1]: {self.temperature}")
        if not self.model:
            raise ValueError("model 不能为空")
        if self.max_retries < 1:
            raise ValueError("max_retries 必须 ≥ 1")
        if self.retry_backoff_s < 0:
            raise ValueError("retry_backoff_s 不能为负")
