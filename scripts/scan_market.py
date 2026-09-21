"""美股市场扫描器 CLI —— 扫股票池, 按 Wyckoff setup/转折点排名, 打印 top-N。

链路: 圈定股票池 → 批量抓周线+日线 (带本地缓存) → 逐只全链路分析打分 → 按 composite 排名。

用户诉求「看涨 setup 和转折点都要」: composite = max(springboard, transition);
榜单同时浮现两类候选, direction 标签区分看涨/看跌。

运行 (Windows 需 UTF-8 前缀, 否则中文/emoji 报 UnicodeEncodeError):
    # 小样快速验证管线
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/scan_market.py --tickers AAPL,MSFT,NVDA,TSLA,AMD --top 5
    # 全量 S&P 500 (首跑较慢, 缓存后二跑秒回; 被限流的标的优雅跳过)
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/scan_market.py --universe sp500 --top 20
    # 额外产出: Markdown 报告 + JSON + LLM 市场综述
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/scan_market.py --universe sp500 --report scan.md --json scan.json --explain

依赖网络; 被限流/断网的标的计入"跳过", 不中断整批。
--explain 若检测到 ANTHROPIC_API_KEY 走真正的 Claude, 否则退化为离线模板 (仍零计算)。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

from mse.agent import AgentParams, AnthropicNarrator, TemplateNarrator, explain_market
from mse.analysis import analyze_market
from mse.core.enums import Timeframe
from mse.data.cache import load_or_fetch
from mse.reporting import render_json, render_markdown_report, render_table
from mse.scanner import load_universe, parse_tickers, scan


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wyckoff 美股市场扫描器")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--universe", default="sp500", help="命名股票池 (默认 sp500)")
    g.add_argument("--tickers", help="自定义逗号分隔列表, 如 AAPL,MSFT,NVDA (覆盖 --universe)")
    p.add_argument("--top", type=int, default=20, help="打印前 N 名 (默认 20)")
    p.add_argument("--start-year", type=int, default=2018, help="行情起始年 (默认 2018)")
    p.add_argument("--end-year", type=int, default=2024, help="行情结束年 (默认 2024)")
    p.add_argument("--limit", type=int, default=None, help="仅扫描股票池前 N 只 (调试用)")
    p.add_argument("--no-cache", action="store_true", help="禁用本地缓存, 强制重新抓取")
    p.add_argument("--report", metavar="PATH", help="额外写出 Markdown 市场报告到该路径")
    p.add_argument("--json", metavar="PATH", help="额外写出 JSON (MarketSummary) 到该路径")
    p.add_argument("--explain", action="store_true", help="打印 LLM 市场综述 (无 key 时用离线模板)")
    p.add_argument("--model", default=AgentParams().model,
                   help=f"--explain 用的 Claude 模型 (默认 {AgentParams().model}; opus 最新用 claude-opus-5)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.tickers:
        tickers = parse_tickers(args.tickers)
        label = f"自定义 {len(tickers)} 只"
    else:
        tickers = load_universe(args.universe)
        label = f"股票池 {args.universe}"
    if args.limit:
        tickers = tickers[: args.limit]
    if not tickers:
        print("⚠️  股票池为空, 退出。")
        sys.exit(0)

    start, end = date(args.start_year, 1, 1), date(args.end_year, 1, 1)
    print(f"== 扫描 {label} —— {len(tickers)} 只 | {start} .. {end} ==")

    # 批量抓周线 + 日线 (带缓存)。
    print("拉取周线 ...")
    weekly, sk_w = load_or_fetch(tickers, Timeframe.WEEKLY, start, end, use_cache=not args.no_cache)
    print(f"  周线到手 {len(weekly)} 只, 跳过 {len(sk_w)}")
    print("拉取日线 ...")
    daily, sk_d = load_or_fetch(tickers, Timeframe.DAILY, start, end, use_cache=not args.no_cache)
    print(f"  日线到手 {len(daily)} 只, 跳过 {len(sk_d)}")

    # 两周期都齐的标的才能分析。
    data = {t: (weekly[t], daily[t]) for t in weekly.keys() & daily.keys()}
    missing = [t for t in tickers if t not in data]
    if not data:
        print("⚠️  无任何标的同时取到周线+日线 (可能全被限流), 跳过扫描。")
        sys.exit(0)

    profiles, skipped_analyze = scan(data)
    skipped_total = sorted(set(missing) | set(skipped_analyze))

    # ── 排名表 (reporting 层渲染) ──────────────────────────────
    top = profiles[: args.top]
    print(f"\n== Top {len(top)} 候选 (按 composite 降序; 共分析 {len(profiles)} 只) ==")
    print(render_table(top))

    # ── 市场聚合 (analysis 层) ────────────────────────────────
    summary = analyze_market(profiles)

    # ── 可选产出: Markdown 报告 / JSON / LLM 综述 ──────────────
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(render_markdown_report(summary, universe_label=label))
        print(f"📄 Markdown 报告已写出: {args.report}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(render_json(summary))
        print(f"🧾 JSON 已写出: {args.json}")
    if args.explain:
        if os.environ.get("ANTHROPIC_API_KEY"):
            narrator = AnthropicNarrator(params=AgentParams(model=args.model))
            print(f"\n== 🤖 Claude 市场综述 ({args.model}) ==")
        else:
            narrator = TemplateNarrator()
            print("\n== 🤖 市场综述 (离线模板; 设 ANTHROPIC_API_KEY 可启用 Claude) ==")
        print(explain_market(summary, narrator=narrator))

    print(
        f"\n✅ 扫描完成: 分析 {len(profiles)} 只, 跳过 {len(skipped_total)} 只"
        + (f" (示例: {', '.join(skipped_total[:8])} ...)" if skipped_total else "")
    )


if __name__ == "__main__":
    main()
