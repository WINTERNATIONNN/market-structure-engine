"""Agent 层入口 —— explain_candidate / explain_market。

把结构化证据 (Scoring/Analysis 层产物) 经 Narrator 翻译成自然语言解读。
默认用 TemplateNarrator (确定性、离线); 传入 AnthropicNarrator 即接入真正的 Claude。

红线: 本层零计算 —— 只构造 prompt (格式化已算好的字段) 并转发给 Narrator。
"""

from __future__ import annotations

from mse.agent.narrator import (
    SYSTEM_CANDIDATE,
    SYSTEM_MARKET,
    Narrator,
    TemplateNarrator,
    build_candidate_prompt,
    build_market_prompt,
)
from mse.analysis import MarketSummary
from mse.scoring import CandidateProfile


def explain_candidate(
    profile: CandidateProfile,
    *,
    narrator: Narrator | None = None,
) -> str:
    """对单只候选生成自然语言解读。narrator 缺省用 TemplateNarrator (离线兜底)。"""
    narrator = narrator or TemplateNarrator()
    return narrator.narrate(build_candidate_prompt(profile), system=SYSTEM_CANDIDATE)


def explain_market(
    summary: MarketSummary,
    *,
    narrator: Narrator | None = None,
    bucket_n: int = 8,
) -> str:
    """对整次市场扫描生成自然语言概览。"""
    narrator = narrator or TemplateNarrator()
    return narrator.narrate(
        build_market_prompt(summary, bucket_n=bucket_n), system=SYSTEM_MARKET
    )
