"""Reporting 层参数 (无魔法数字)。控制渲染的列宽/小数位/榜单容量。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportParams:
    """渲染参数。"""

    top_n: int = 25              # 表格/榜单最大行数
    decimals: int = 2            # 分数小数位
    bucket_n: int = 10           # markdown 报告里每个分桶展示的条数

    def __post_init__(self) -> None:
        if self.top_n < 1 or self.bucket_n < 1:
            raise ValueError("top_n / bucket_n 必须 ≥ 1")
        if self.decimals < 0:
            raise ValueError("decimals 不能为负")
