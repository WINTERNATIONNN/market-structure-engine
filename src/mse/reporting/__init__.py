"""Reporting 层 —— 渲染候选/榜单/市场汇总为终端表 / Markdown / JSON。纯计算, 零 LLM。

分层: reporting → ranking → analysis → scanner → scoring → wyckoff → data → core。
"""

from mse.reporting.params import ReportParams
from mse.reporting.render import (
    render_json,
    render_markdown_report,
    render_table,
)

__all__ = [
    "ReportParams",
    "render_table",
    "render_markdown_report",
    "render_json",
]
