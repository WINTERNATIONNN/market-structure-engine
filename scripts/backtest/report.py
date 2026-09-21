"""汇总报告 —— 读全部 DONE 分片, 算三项指标, 产出 report.md + report.json。

每项指标一张「方案(V0–V4) × 字段」表; 表下标注该指标的胜者。
report.json 存结构化数字 (便于程序化对比 / 二次分析), report.md 供人读。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from backtest.checkpoint import read_all
from backtest.metrics import (
    metric_event_precision,
    metric_forward_hit_rate,
    metric_phase_consistency,
)
from backtest.params import BacktestParams
from backtest.returns import attach_universe_means
from backtest.variants import VARIANT_NAMES


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def _hit_rate_section(hit: dict, bp: BacktestParams) -> tuple[str, dict]:
    """指标 1: 前向收益命中率, 按周期分块; 每块标注最高命中率的方案。"""
    blocks: list[str] = []
    winners: dict[str, str] = {}
    for h in bp.horizons:
        headers = ["方案", "n", "hit_rate", "mean_fwd_bull", "excess_vs_universe"]
        rows: list[list[str]] = []
        best_v, best_rate = None, -1.0
        for v in VARIANT_NAMES:
            st = hit.get(v, {}).get(h)
            if st is None:
                rows.append([v, "0", "-", "-", "-"])
                continue
            rows.append([v, str(st.n), _fmt(st.hit_rate), _fmt(st.mean_fwd_bull), _fmt(st.excess_vs_universe)])
            if st.n > 0 and st.hit_rate > best_rate:
                best_v, best_rate = v, st.hit_rate
        winners[f"h{h}"] = best_v or "-"
        note = f"\n\n> 周期 {h}d 命中率最高: **{best_v or '-'}**" + (f" ({_fmt(best_rate)})" if best_v else "")
        blocks.append(f"#### 前向收益命中率 — {h} 交易日\n\n" + _md_table(headers, rows) + note)
    return "\n\n".join(blocks), winners


def _consistency_section(cons: dict) -> tuple[str, str]:
    """指标 2: 阶段一致性; 胜者 = 矛盾率最低 (盲区 KPI)。"""
    headers = ["方案", "n_confident", "contradiction_rate↓", "markup_up_rate", "distribution_down_rate"]
    rows: list[list[str]] = []
    best_v, best_rate = None, 2.0
    for v in VARIANT_NAMES:
        st = cons.get(v)
        if st is None:
            rows.append([v, "0", "-", "-", "-"])
            continue
        rows.append([v, str(st.n_confident), _fmt(st.contradiction_rate), _fmt(st.markup_up_rate), _fmt(st.distribution_down_rate)])
        if st.n_confident > 0 and st.contradiction_rate < best_rate:
            best_v, best_rate = v, st.contradiction_rate
    note = f"\n\n> 矛盾率最低 (盲区修复最佳): **{best_v or '-'}**" + (f" ({_fmt(best_rate)})" if best_v else "")
    return "#### 阶段一致性 (矛盾率越低越好)\n\n" + _md_table(headers, rows) + note, best_v or "-"


def _precision_section(prec: dict) -> str:
    """指标 3: 事件事后精度, 每方案每事件类型一行。"""
    event_types = sorted({et for v in prec.values() for et in v})
    if not event_types:
        return "#### 事件事后精度\n\n> (无事件记录)"
    headers = ["方案"] + [f"{et} (n/prec)" for et in event_types]
    rows: list[list[str]] = []
    for v in VARIANT_NAMES:
        row = [v]
        for et in event_types:
            st = prec.get(v, {}).get(et)
            row.append(f"{st.n}/{_fmt(st.precision)}" if st else "-")
        rows.append(row)
    return "#### 事件事后精度 (lookahead 内朝预期方向的比率)\n\n" + _md_table(headers, rows)


def build_report(out_dir: Path, bp: BacktestParams) -> tuple[Path, Path]:
    """读全部 DONE 分片, 算三项指标, 写 report.md + report.json; 返回两文件路径。"""
    out_dir = Path(out_dir)
    records = read_all(out_dir)
    universe_means = attach_universe_means(records, bp)

    hit = metric_forward_hit_rate(records, universe_means, bp)
    cons = metric_phase_consistency(records, bp)
    prec = metric_event_precision(records, bp)

    hit_md, hit_winners = _hit_rate_section(hit, bp)
    cons_md, cons_winner = _consistency_section(cons)
    prec_md = _precision_section(prec)

    n_variants = len({r.variant for r in records})
    n_tickers = len({r.ticker for r in records})
    n_cuts = len({r.as_of for r in records})

    md = "\n\n".join([
        "# Stale active-TR 修复方案回测报告",
        f"- 记录数: {len(records)}  |  方案数: {n_variants}  |  ticker 数: {n_tickers}  |  截面数: {n_cuts}",
        f"- 数据窗口: {bp.data_start} → {bp.data_end}  |  网格: {bp.grid_start} → {bp.grid_end} ({bp.rebalance})",
        "## 指标 1 — 前向收益命中率",
        hit_md,
        "## 指标 2 — 阶段一致性",
        cons_md,
        "## 指标 3 — 事件事后精度",
        prec_md,
        "## 优胜提示",
        f"- 命中率优胜 (各周期): {hit_winners}\n"
        f"- 盲区矛盾率优胜: **{cons_winner}**\n"
        f"- 综合选型: 看重「消除盲区」选矛盾率最低者; 看重「实盘方向」参考命中率与 excess; "
        f"注意某些方案可能提升一致性却拉低命中率 —— 结合三项权衡。",
    ])

    report_md = out_dir / "report.md"
    report_json = out_dir / "report.json"
    report_md.write_text(md, encoding="utf-8")

    payload = {
        "summary": {"n_records": len(records), "n_variants": n_variants, "n_tickers": n_tickers, "n_cuts": n_cuts},
        "forward_hit_rate": {v: {h: asdict(st) for h, st in hs.items()} for v, hs in hit.items()},
        "phase_consistency": {v: asdict(st) for v, st in cons.items()},
        "event_precision": {v: {et: asdict(st) for et, st in ets.items()} for v, ets in prec.items()},
        "winners": {"forward_hit_rate": hit_winners, "phase_consistency": cons_winner},
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_md, report_json
