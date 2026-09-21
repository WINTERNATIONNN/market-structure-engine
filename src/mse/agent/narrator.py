"""Agent 层叙述器 —— 把结构化证据 (CandidateProfile / MarketSummary) 交给 LLM 翻成人话。

红线落地:
    * **零计算**: 本层不做任何打分/派生数值。prompt 构造只是把引擎已算好的字段格式化;
      LLM 只被要求"依据给定证据解释", 系统提示明令禁止臆造或重算数字。
    * **可离线/可测**: Narrator 是一个协议 (seam)。
        - TemplateNarrator —— 确定性、无 LLM、无网络的兜底 (直接回吐结构化摘要), 单测用它。
        - AnthropicNarrator —— 真正调用 Claude (anthropic SDK 惰性导入; 缺库/缺 key 时给出
          清晰报错)。不在单测中联网。

分层: agent → reporting → ranking → analysis → scanner → scoring → ...
"""

from __future__ import annotations

import os
import time
from typing import Protocol

from mse.agent.params import AgentParams
from mse.analysis import MarketSummary
from mse.scoring import CandidateProfile

# 系统提示: 约束 LLM 只做"翻译/解释", 不越界重算 —— 对应"引擎零 LLM / agent 零计算"红线。
SYSTEM_CANDIDATE = (
    "你是 Wyckoff 量化分析助手。下面给出的是引擎已经计算好的结构化证据 (阶段、概率、"
    "各项分数、事件、转折)。请**只依据这些证据**, 用简洁中文写一段面向交易员的解读: "
    "这只股票当前处于什么结构、看涨还是看跌、关键依据、需要注意的风险。"
    "严禁臆造任何数字、严禁自行重新计算分数、严禁编造证据里没有的事件。"
    "阶段标签可能因引擎局限而不完全可信, 若证据自相矛盾请明确指出。"
)

SYSTEM_MARKET = (
    "你是 Wyckoff 量化分析助手。下面是一次全市场扫描的结构化汇总 (广度、多空计数、"
    "看涨/看跌/新鲜转折分桶)。请**只依据这些数据**, 用简洁中文写一段市场概览: "
    "当前市场整体结构 (广度)、值得关注的看涨与看跌候选、以及解读时的注意事项。"
    "严禁臆造数字或编造未列出的标的。"
)


class Narrator(Protocol):
    """叙述器协议 —— 输入 (prompt, system), 输出自然语言。"""

    def narrate(self, prompt: str, *, system: str) -> str: ...


# ── prompt 构造 (纯格式化, 零计算) ─────────────────────────────────
def _fmt_events(events) -> str:
    if not events:
        return "无"
    return ", ".join(f"{e.event_type.value}({e.date}, p={e.probability:.2f})" for e in events)


def build_candidate_prompt(profile: CandidateProfile) -> str:
    """把单只候选的结构化证据格式化成可读 brief (不做任何计算)。"""
    p = profile
    lines = [
        f"标的: {p.ticker}   基准日: {p.as_of}",
        f"当前阶段: {p.current_phase.value} (p={p.phase_probability:.2f}, "
        f"conf={p.phase_confidence:.2f})",
        f"综合分: {p.composite_score:.2f} | 吸筹 {p.springboard_score:.2f} | "
        f"转折 {p.transition_score:.2f} (方向: {p.transition_direction or '无'})",
    ]
    if p.active_tr is not None:
        pos = f"{p.price_pos_in_tr:.2f}" if p.price_pos_in_tr is not None else "?"
        lines.append(
            f"活跃交易区间: [{p.active_tr.lower:.2f}, {p.active_tr.upper:.2f}], "
            f"价格相对位置: {pos} (0=下沿,1=上沿)"
        )
    lines.append(f"近期看涨事件: {_fmt_events(p.recent_bull_events)}")
    lines.append(f"近期看跌事件: {_fmt_events(p.recent_bear_events)}")
    if p.last_transition is not None:
        lt = p.last_transition
        lines.append(
            f"最近转折: {lt.from_phase.value} → {lt.to_phase.value}, "
            f"{lt.bars_ago} 周前 (p={lt.probability:.2f})"
        )
    if p.tags:
        lines.append(f"标签: {' | '.join(p.tags)}")
    return "\n".join(lines)


def build_market_prompt(summary: MarketSummary, *, bucket_n: int = 8) -> str:
    """把市场汇总格式化成可读 brief (不做任何计算)。"""
    b = summary.breadth
    lines = [
        f"分析标的数: {b.analyzed}   基准日: {b.as_of}",
        f"多空计数: 看涨 {b.bullish} / 看跌 {b.bearish} / 中性 {b.neutral}",
    ]
    if b.phase_counts:
        lines.append(
            "阶段分布: "
            + ", ".join(f"{ph} {n}" for ph, n in sorted(
                b.phase_counts.items(), key=lambda kv: kv[1], reverse=True))
        )

    def bucket(name: str, seq) -> None:
        top = seq[:bucket_n]
        if top:
            lines.append(f"\n{name}:")
            for p in top:
                lines.append(
                    f"  - {p.ticker} [{p.current_phase.value}] 综合 "
                    f"{p.composite_score:.2f} | {'; '.join(p.tags)}"
                )

    bucket("看涨 setup/转折", summary.bullish_setups)
    bucket("看跌预警", summary.bearish_warnings)
    bucket("新鲜转折", summary.fresh_transitions)
    return "\n".join(lines)


# ── Narrator 实现 ─────────────────────────────────────────────────
class TemplateNarrator:
    """确定性兜底叙述器 —— 不调用 LLM, 直接回吐结构化摘要 (离线/单测用)。"""

    def narrate(self, prompt: str, *, system: str) -> str:  # noqa: ARG002 —— 无需 system
        return "【结构化摘要 · 未接入 LLM】\n" + prompt


class AnthropicNarrator:
    """真正调用 Claude 的叙述器 (anthropic SDK 惰性导入)。

    需要环境变量 ANTHROPIC_API_KEY (或构造时传入 api_key)。SDK 未安装 / key 缺失 → 抛出
    清晰的 RuntimeError, 提示改用 TemplateNarrator 或补齐依赖。
    """

    def __init__(self, *, params: AgentParams = AgentParams(), api_key: str | None = None) -> None:
        self.params = params
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def narrate(self, prompt: str, *, system: str) -> str:
        try:
            import anthropic  # 惰性导入: 未装 SDK 也不影响引擎/离线路径
        except ImportError as exc:  # pragma: no cover —— 依赖缺失路径
            raise RuntimeError(
                "未安装 anthropic SDK。请 `pip install anthropic`, 或改用 TemplateNarrator。"
            ) from exc
        if not self._api_key:
            raise RuntimeError(
                "缺少 ANTHROPIC_API_KEY 环境变量 (或构造时传入 api_key)。"
            )
        client = anthropic.Anthropic(api_key=self._api_key)
        # 偶发 5xx / 网络抖动重试: 对瞬时错误线性退避重试, 其余错误 (400/401 等) 立即上抛。
        transient = (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.params.max_retries + 1):
            try:
                resp = client.messages.create(
                    model=self.params.model,
                    max_tokens=self.params.max_tokens,
                    temperature=self.params.temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                # 拼接所有文本块。
                return "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
            except transient as exc:
                last_exc = exc
                if attempt < self.params.max_retries:
                    time.sleep(attempt * self.params.retry_backoff_s)
        raise RuntimeError(
            f"调用 Claude 失败 (已重试 {self.params.max_retries} 次): {last_exc}"
        ) from last_exc
