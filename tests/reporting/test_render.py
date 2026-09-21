"""Reporting 层离线单测 —— render_table / render_markdown_report / render_json。"""

from __future__ import annotations

import json
from datetime import date

from mse.analysis import analyze_market
from mse.core.enums import WyckoffPhase
from mse.reporting import render_json, render_markdown_report, render_table
from mse.scoring import CandidateProfile

_AS_OF = date(2026, 8, 8)


def _profile(ticker: str, phase: WyckoffPhase, *, composite: float,
             direction: str | None = None) -> CandidateProfile:
    return CandidateProfile(
        ticker=ticker, as_of=_AS_OF, current_phase=phase,
        phase_probability=0.6, phase_confidence=0.5,
        springboard_score=composite, transition_score=0.0,
        transition_direction=direction, composite_score=composite,
        tags=[f"阶段={phase.value}", "近期SOS p=1.00"],
    )


def _pool() -> list[CandidateProfile]:
    return [
        _profile("AAA", WyckoffPhase.MARKUP, composite=0.9, direction="bullish"),
        _profile("BBB", WyckoffPhase.DISTRIBUTION, composite=0.7, direction="bearish"),
    ]


def test_render_table() -> None:
    out = render_table(_pool(), title="== Top ==")
    assert "== Top ==" in out
    assert "ticker" in out
    assert "AAA" in out and "BBB" in out
    assert "看涨" in out and "看跌" in out
    # 表头 + 标题 + 2 行数据。
    assert len(out.splitlines()) == 4


def test_render_markdown_report() -> None:
    summary = analyze_market(_pool())
    md = render_markdown_report(summary, universe_label="测试池")
    assert md.startswith("# Wyckoff 市场扫描报告")
    assert "测试池" in md
    assert "市场广度" in md
    assert "🟢 看涨" in md and "🔴 看跌" in md
    assert "AAA" in md and "BBB" in md
    assert "| # |" in md  # 含 markdown 表


def test_render_json_list_roundtrip() -> None:
    out = render_json(_pool())
    data = json.loads(out)
    assert isinstance(data, list) and len(data) == 2
    assert data[0]["ticker"] == "AAA"
    assert data[0]["as_of"] == "2026-08-08"  # 日期已 ISO 序列化
    assert data[0]["current_phase"] == "markup"


def test_render_json_market_summary() -> None:
    summary = analyze_market(_pool())
    data = json.loads(render_json(summary))
    assert data["breadth"]["analyzed"] == 2
    assert "bullish_setups" in data
