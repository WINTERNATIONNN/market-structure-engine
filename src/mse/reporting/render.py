"""Reporting 层 —— 把候选/榜单/市场汇总渲染成人类可读产物 (终端表格 / Markdown / JSON)。

红线: 纯计算 (字符串格式化), 零 LLM。把原先散在 CLI 里的表格逻辑收敛到此, 供 CLI 与
未来 API/agent 复用。

    * render_table  —— 等宽终端表 (原 scan_market.py 的排名表)。
    * render_markdown_report —— 完整 Markdown 报告 (广度 + 多空分桶 + 榜单)。
    * render_json   —— 结构化 JSON (给程序/前端消费)。

分层: reporting → ranking → analysis → scanner → scoring → ...
"""

from __future__ import annotations

import json

from mse.analysis import MarketSummary
from mse.ranking import Ranking
from mse.reporting.params import ReportParams
from mse.scoring import CandidateProfile

_DIRECTION_CN = {"bullish": "看涨", "bearish": "看跌"}


def _dir_cn(direction: str | None) -> str:
    return _DIRECTION_CN.get(direction or "", "-")


def render_table(
    profiles: list[CandidateProfile],
    *,
    params: ReportParams = ReportParams(),
    title: str | None = None,
) -> str:
    """等宽终端表 (# ticker 阶段 综合 吸筹 转折 方向 标签)。"""
    d = params.decimals
    rows = profiles[: params.top_n]
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(
        f"{'#':>2}  {'ticker':<7} {'阶段':<13} {'综合':>5} {'吸筹':>5} "
        f"{'转折':>5} {'方向':<5} 标签"
    )
    for i, p in enumerate(rows, 1):
        tags = " | ".join(p.tags)
        lines.append(
            f"{i:>2}  {p.ticker:<7} {p.current_phase.value:<13} "
            f"{p.composite_score:>5.{d}f} {p.springboard_score:>5.{d}f} "
            f"{p.transition_score:>5.{d}f} {_dir_cn(p.transition_direction):<5} {tags}"
        )
    return "\n".join(lines)


def _md_table(profiles: list[CandidateProfile], params: ReportParams) -> str:
    d = params.decimals
    head = "| # | 代码 | 阶段 | 综合 | 吸筹 | 转折 | 方向 | 标签 |\n" \
           "|---|------|------|------|------|------|------|------|"
    body = []
    for i, p in enumerate(profiles[: params.bucket_n], 1):
        tags = " ".join(p.tags).replace("|", "/")
        body.append(
            f"| {i} | {p.ticker} | {p.current_phase.value} | "
            f"{p.composite_score:.{d}f} | {p.springboard_score:.{d}f} | "
            f"{p.transition_score:.{d}f} | {_dir_cn(p.transition_direction)} | {tags} |"
        )
    return head + ("\n" + "\n".join(body) if body else "\n| — | — | — | — | — | — | — | — |")


def render_markdown_report(
    summary: MarketSummary,
    *,
    params: ReportParams = ReportParams(),
    universe_label: str = "",
) -> str:
    """完整 Markdown 报告: 市场广度 + 看涨/看跌/新鲜转折分桶 + 综合榜单。"""
    b = summary.breadth
    parts: list[str] = []
    label = f" — {universe_label}" if universe_label else ""
    parts.append(f"# Wyckoff 市场扫描报告{label}")
    if b.as_of:
        parts.append(f"*基准日: {b.as_of} | 分析 {b.analyzed} 只*")

    # 市场广度
    parts.append("\n## 市场广度")
    parts.append(f"- 多空: 看涨 **{b.bullish}** / 看跌 **{b.bearish}** / 中性 {b.neutral}")
    if b.phase_counts:
        phase_line = ", ".join(
            f"{ph} {n} ({100 * b.phase_pct(ph):.0f}%)"
            for ph, n in sorted(b.phase_counts.items(), key=lambda kv: kv[1], reverse=True)
        )
        parts.append(f"- 阶段分布: {phase_line}")

    parts.append("\n## 🟢 看涨 setup / 转折")
    parts.append(_md_table(summary.bullish_setups, params))
    parts.append("\n## 🔴 看跌预警")
    parts.append(_md_table(summary.bearish_warnings, params))
    parts.append("\n## ⚡ 新鲜转折 (近期刚切换阶段)")
    parts.append(_md_table(summary.fresh_transitions, params))
    parts.append("\n## 综合榜单")
    parts.append(_md_table(summary.top_overall, params))
    return "\n".join(parts)


def render_json(
    obj: MarketSummary | Ranking | list[CandidateProfile],
    *,
    indent: int = 2,
) -> str:
    """结构化 JSON (日期序列化为 ISO 字符串)。接受 MarketSummary / Ranking / 候选列表。"""
    if isinstance(obj, list):
        payload = [p.model_dump(mode="json") for p in obj]
    else:
        payload = obj.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=indent)
