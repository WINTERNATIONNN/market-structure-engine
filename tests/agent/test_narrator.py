"""Agent 层离线单测 —— prompt 构造 + TemplateNarrator + explain_*。不联网。

AnthropicNarrator 仅测"缺 key/缺库时给出清晰报错", 不发起真实调用。
"""

from __future__ import annotations

from datetime import date

import pytest

from mse.agent import (
    AgentParams,
    AnthropicNarrator,
    TemplateNarrator,
    build_candidate_prompt,
    build_market_prompt,
    explain_candidate,
    explain_market,
)
from mse.analysis import analyze_market
from mse.core.enums import EventType, WyckoffPhase
from mse.scoring import CandidateProfile, EventRef, TransitionRef

_AS_OF = date(2026, 8, 8)


def _profile() -> CandidateProfile:
    return CandidateProfile(
        ticker="AAPL", as_of=_AS_OF, current_phase=WyckoffPhase.MARKUP,
        phase_probability=0.72, phase_confidence=0.55,
        springboard_score=0.54, transition_score=0.69,
        transition_direction="bullish", composite_score=0.69,
        recent_bull_events=[EventRef(event_type=EventType.SOS, date=_AS_OF, probability=1.0)],
        last_transition=TransitionRef(
            from_phase=WyckoffPhase.ACCUMULATION, to_phase=WyckoffPhase.MARKUP,
            entered_date=_AS_OF, bars_ago=2, probability=0.85,
        ),
        tags=["阶段=markup", "近期SOS p=1.00"],
    )


def test_build_candidate_prompt_has_facts() -> None:
    txt = build_candidate_prompt(_profile())
    assert "AAPL" in txt
    assert "markup" in txt
    assert "0.69" in txt          # 综合分
    assert "SOS" in txt           # 近期事件
    assert "bullish" in txt       # 方向
    assert "2 周前" in txt        # 转折


def test_template_narrator_echoes_facts() -> None:
    tn = TemplateNarrator()
    out = tn.narrate("标的: AAPL", system="sys")
    assert "AAPL" in out
    assert "未接入 LLM" in out


def test_explain_candidate_default_offline() -> None:
    out = explain_candidate(_profile())  # 默认 TemplateNarrator
    assert "AAPL" in out and "markup" in out


def test_explain_uses_injected_narrator() -> None:
    # 自定义 narrator 记录收到的 prompt/system, 验证 explain 只转发不改写。
    seen = {}

    class Spy:
        def narrate(self, prompt: str, *, system: str) -> str:
            seen["prompt"] = prompt
            seen["system"] = system
            return "OK"

    assert explain_candidate(_profile(), narrator=Spy()) == "OK"
    assert "AAPL" in seen["prompt"]
    assert "Wyckoff" in seen["system"]  # 系统提示已注入


def test_build_market_prompt_and_explain() -> None:
    summary = analyze_market([_profile()])
    prompt = build_market_prompt(summary)
    assert "分析标的数: 1" in prompt
    assert "AAPL" in prompt
    out = explain_market(summary)
    assert "AAPL" in out


def test_anthropic_narrator_errors_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    narrator = AnthropicNarrator(params=AgentParams(), api_key=None)
    # 缺 SDK 或缺 key 都应是清晰的 RuntimeError, 而非静默失败。
    with pytest.raises(RuntimeError):
        narrator.narrate("hi", system="sys")


def test_agent_params_validation() -> None:
    with pytest.raises(ValueError):
        AgentParams(temperature=2.0)
    with pytest.raises(ValueError):
        AgentParams(max_tokens=0)
