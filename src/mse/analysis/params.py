"""Analysis 层参数 (无魔法数字, 集中外置)。

控制市场级派生视图的分类阈值: 何为"强候选"、方向如何判定、"近期转折"的时间窗。
仿 wyckoff/params.py 的 frozen dataclass + __post_init__ 校验范式。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisParams:
    """市场分析分类阈值。"""

    # composite 分档: ≥strong 记"强", ≥watch 记"关注", 其余忽略。
    strong_composite: float = 0.70
    watch_composite: float = 0.50

    # springboard ≥ 此值且无看跌转折 → 归入看涨 setup。
    bullish_springboard: float = 0.50

    # 近期转折窗口: last_transition.bars_ago ≤ 此值才算"新鲜转折"。
    fresh_transition_bars: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.watch_composite <= self.strong_composite <= 1.0:
            raise ValueError(
                f"须满足 0 ≤ watch ({self.watch_composite}) ≤ strong "
                f"({self.strong_composite}) ≤ 1"
            )
        if not 0.0 <= self.bullish_springboard <= 1.0:
            raise ValueError(f"bullish_springboard 须在 [0,1]: {self.bullish_springboard}")
        if self.fresh_transition_bars < 0:
            raise ValueError("fresh_transition_bars 不能为负")
