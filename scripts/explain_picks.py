"""逐票 Opus 详解 —— 扫描后对「看涨 top-N」+ 指定 ticker 跑 explain_candidate。

用法 (需 SAP 网关 env 已配好; 模型走 -latest 别名):
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 ANTHROPIC_API_KEY=... \
        python scripts/explain_picks.py --top-bullish 10 --tickers STX,SNDK,GOOG,INT,LLY,BE,MSFT \
        --model claude-opus-latest

流程: sp500 股票池 + 额外 ticker → 批量抓周线/日线(缓存)→ scan 打分 →
analyze_market 取看涨前 N → 对这批 + 额外 ticker 逐只 explain_candidate。
Agent 层零计算: 数字全来自引擎, Opus 只解读。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

from mse.agent import AgentParams, AnthropicNarrator, TemplateNarrator, explain_candidate
from mse.analysis import analyze_market
from mse.core.enums import Timeframe
from mse.data.cache import load_or_fetch
from mse.scanner import load_universe, parse_tickers, scan


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="逐票 Opus 详解")
    p.add_argument("--universe", default="sp500", help="打底股票池 (默认 sp500)")
    p.add_argument("--tickers", default="", help="额外/指定 ticker (逗号分隔), 无论是否在股票池都会抓取")
    p.add_argument("--top-bullish", type=int, default=10, help="额外解释看涨榜前 N (默认 10)")
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2027)
    p.add_argument("--model", default="claude-opus-latest", help="Claude 模型 (默认 claude-opus-latest)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    extra = parse_tickers(args.tickers) if args.tickers else []
    universe = load_universe(args.universe)
    # 额外 ticker 并入抓取列表(去重, 保留额外的在前以确保一定抓到)。
    tickers = list(dict.fromkeys(extra + universe))
    start, end = date(args.start_year, 1, 1), date(args.end_year, 1, 1)

    print(f"== 抓取 {len(tickers)} 只 (含指定 {len(extra)}) | {start}..{end} ==")
    weekly, _ = load_or_fetch(tickers, Timeframe.WEEKLY, start, end)
    daily, _ = load_or_fetch(tickers, Timeframe.DAILY, start, end)
    data = {t: (weekly[t], daily[t]) for t in weekly.keys() & daily.keys()}
    profiles, skipped = scan(data)
    by_ticker = {p.ticker: p for p in profiles}
    print(f"  分析 {len(profiles)} 只, 跳过 {len(skipped)}")

    summary = analyze_market(profiles)
    top_bull = summary.bullish_setups[: args.top_bullish]

    # 选叙述器: 有 key 走 Opus, 否则离线模板。
    if os.environ.get("ANTHROPIC_API_KEY"):
        narrator = AnthropicNarrator(params=AgentParams(model=args.model))
        print(f"  叙述器: Claude {args.model}\n")
    else:
        narrator = TemplateNarrator()
        print("  叙述器: 离线模板 (未设 ANTHROPIC_API_KEY)\n")

    # ── 看涨 top-N ─────────────────────────────────────────────
    print("#" * 70)
    print(f"# 看涨榜前 {len(top_bull)}")
    print("#" * 70)
    for i, prof in enumerate(top_bull, 1):
        print(f"\n───── [{i}] {prof.ticker}  (综合 {prof.composite_score:.2f}, "
              f"阶段 {prof.current_phase.value}) ─────")
        print(explain_candidate(prof, narrator=narrator))

    # ── 指定 ticker ────────────────────────────────────────────
    if extra:
        print("\n" + "#" * 70)
        print(f"# 指定 ticker: {', '.join(extra)}")
        print("#" * 70)
        missing = []
        for t in extra:
            prof = by_ticker.get(t)
            if prof is None:
                missing.append(t)
                continue
            print(f"\n───── {t}  (综合 {prof.composite_score:.2f}, "
                  f"阶段 {prof.current_phase.value}, "
                  f"方向 {prof.transition_direction or '-'}) ─────")
            print(explain_candidate(prof, narrator=narrator))
        if missing:
            print(f"\n⚠️  以下 ticker 无行情/无法分析 (可能退市/代码有误): {', '.join(missing)}")


if __name__ == "__main__":
    main()
