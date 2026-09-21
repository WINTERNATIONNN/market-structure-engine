"""Ranking 层参数 (无魔法数字)。控制过滤门槛与榜单容量。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingParams:
    """排名过滤 / 容量参数。"""

    top_n: int = 20                  # 榜单最大长度
    min_composite: float = 0.0       # 低于此综合分的候选剔除
    min_confidence: float = 0.0      # 低于此阶段 confidence 的候选剔除

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n 必须 ≥ 1")
        if not 0.0 <= self.min_composite <= 1.0:
            raise ValueError(f"min_composite 须在 [0,1]: {self.min_composite}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence 须在 [0,1]: {self.min_confidence}")
