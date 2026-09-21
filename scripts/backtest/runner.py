"""走查回测执行器 —— 并行分片跑批 + 断点续跑 + CLI 入口。

并行轴 = ticker 分片 (ProcessPoolExecutor); 每个 worker 只加载本分片 ticker 的
全窗口日线+周线 (命中 .cache/ohlcv 即读 pkl 无网络), 内存内按 截面×方案 迭代。

用法:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/backtest/runner.py \
      --universe sp500 --shards 16 --workers 8 --out bt_out/full
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/backtest/runner.py \
      --tickers AAPL,MSFT,NVDA,INTC,GOOG --shards 2 --workers 2 --out bt_out/smoke
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

# 让 `python scripts/backtest/runner.py` 直接运行时能 import backtest.* 与 mse.*
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for p in (_REPO / "scripts", _REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mse.core.enums import Timeframe  # noqa: E402
from mse.core.models import OHLCVSeries, Symbol  # noqa: E402
from mse.scanner.universe import load_universe, parse_tickers  # noqa: E402
from mse.scoring import ScoringParams  # noqa: E402

from backtest.checkpoint import shard_done, write_checkpoint  # noqa: E402
from backtest.grid import build_cutoff_grid, slice_pit  # noqa: E402
from backtest.params import BacktestParams  # noqa: E402
from backtest.records import EventObs, SignalRecord  # noqa: E402
from backtest.report import build_report  # noqa: E402
from backtest.returns import compute_fwd_returns, event_forward_move  # noqa: E402
from backtest.variants import run_all_variants, signal_direction  # noqa: E402

_CACHE_DIR = str((_REPO / ".cache" / "ohlcv").resolve())


def _load_cached_series(ticker: str, timeframe: Timeframe) -> OHLCVSeries | None:
    """纯离线加载: 无视缓存日期键, 取该 ticker 覆盖最广 (数据最新、行数最多) 的本地 pkl。

    .cache/ohlcv 的键含 (start,end), 不同批次写入的键不一 -> 精确匹配会漏掉半数 ticker。
    这里按 ticker+timeframe 直接 glob, 选数据跨度最优者, 保证全程零联网、确定性。
    """
    import pandas as pd

    hits = sorted(Path(_CACHE_DIR).glob(f"{ticker.upper()}_{timeframe.value}_*.pkl"))
    if not hits:
        return None
    best_frame: pd.DataFrame | None = None
    best_key: tuple = (pd.Timestamp.min, -1)
    for f in hits:
        try:
            df = pd.read_pickle(f)
        except Exception:  # noqa: BLE001 —— 损坏的 pkl 跳过
            continue
        if df is None or len(df) == 0:
            continue
        key = (df.index.max(), len(df))  # 优先数据最新, 再比行数
        if key > best_key:
            best_key, best_frame = key, df
    if best_frame is None:
        return None
    return OHLCVSeries(
        symbol=Symbol(ticker=ticker),
        timeframe=timeframe,
        frame=best_frame,
        adjusted=True,
        fetched_at=None,
    )


def _load_shard_data(tickers: list[str], timeframe: Timeframe) -> dict[str, OHLCVSeries]:
    out: dict[str, OHLCVSeries] = {}
    for t in tickers:
        s = _load_cached_series(t, timeframe)
        if s is not None:
            out[t] = s
    return out



def _shards(tickers: list[str], n_shards: int) -> list[list[str]]:
    """把 ticker 轮转分配到 n_shards 个分片 (轮转 => 各分片负载更均衡)。"""
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for i, t in enumerate(tickers):
        buckets[i % n_shards].append(t)
    return [b for b in buckets if b]


def _recent_events(events, as_of: date, daily_full: OHLCVSeries, bp: BacktestParams) -> tuple[EventObs, ...]:
    """近期 (recency 窗口内且 <= as_of) 事件 + 其事后 lookahead 走势。"""
    lo = as_of - timedelta(days=bp.event_log_recency_days)
    out: list[EventObs] = []
    for e in events:
        if lo <= e.date <= as_of:
            out.append(EventObs(
                event_type=e.event_type.value,
                date=e.date,
                probability=round(float(e.probability), 4),
                fwd_move=event_forward_move(daily_full, e.date, bp.event_lookahead_days),
            ))
    return tuple(out)


def _records_for_cut(
    ticker: str,
    as_of: date,
    weekly_full: OHLCVSeries,
    daily_full: OHLCVSeries,
    bp: BacktestParams,
    sp: ScoringParams,
) -> list[SignalRecord]:
    """单 (ticker, 截面): PIT 切片 → 全方案打分 → 组装每方案一条 SignalRecord。"""
    sliced = slice_pit(
        weekly_full, daily_full, as_of,
        min_weekly=bp.min_weekly_bars, min_daily=bp.min_daily_bars,
    )
    if sliced is None:
        return []
    weekly_s, daily_s = sliced
    variants = run_all_variants(weekly_s, daily_s, bp=bp, sp=sp)
    fwd = compute_fwd_returns(daily_full, as_of, bp)  # 方案无关, 只算一次

    records: list[SignalRecord] = []
    for variant, (profile, events) in variants.items():
        records.append(SignalRecord(
            variant=variant,
            ticker=ticker,
            as_of=as_of,
            current_phase=profile.current_phase.value,
            phase_probability=round(float(profile.phase_probability), 4),
            price_pos_in_tr=(None if profile.price_pos_in_tr is None else round(float(profile.price_pos_in_tr), 4)),
            has_active_tr=profile.active_tr is not None,
            springboard_score=round(float(profile.springboard_score), 4),
            transition_score=round(float(profile.transition_score), 4),
            composite_score=round(float(profile.composite_score), 4),
            direction=signal_direction(profile, bp),
            fwd_returns=fwd,
            events=_recent_events(events, as_of, daily_full, bp),
        ))
    return records


def run_shard(shard_id: int, tickers: list[str], out_dir: str, bp: BacktestParams, sp: ScoringParams) -> tuple[int, int, int]:
    """一个分片的 worker 入口: 加载数据 → 全 (ticker×截面×方案) 迭代 → 原子落盘。

    返回 (shard_id, 记录数, 跳过的 ticker 数)。已 DONE 分片直接跳过 (续跑)。
    """
    out = Path(out_dir)
    if shard_done(out, shard_id):
        return shard_id, -1, 0  # -1 = 已完成, 跳过

    weekly_map = _load_shard_data(tickers, Timeframe.WEEKLY)
    daily_map = _load_shard_data(tickers, Timeframe.DAILY)
    grid = build_cutoff_grid(bp)

    records: list[SignalRecord] = []
    skipped = 0
    for t in tickers:
        weekly_full = weekly_map.get(t)
        daily_full = daily_map.get(t)
        if weekly_full is None or daily_full is None:
            skipped += 1
            continue
        for as_of in grid:
            try:
                records += _records_for_cut(t, as_of, weekly_full, daily_full, bp, sp)
            except Exception:  # noqa: BLE001 —— 单 (ticker,截面) 异常不中断整分片
                continue

    write_checkpoint(out, shard_id, records)
    return shard_id, len(records), skipped


def run_backtest(
    *,
    universe: list[str],
    out_dir: str | Path,
    bp: BacktestParams,
    sp: ScoringParams,
    resume: bool = True,
) -> None:
    """跨分片并行跑全量回测; resume=True 时跳过已 DONE 分片。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = _shards(universe, bp.ticker_shards)
    print(f"[backtest] {len(universe)} tickers -> {len(shards)} shards, {bp.n_workers} workers", flush=True)

    pending = list(enumerate(shards))
    if resume:
        pending = [(sid, ts) for sid, ts in pending if not shard_done(out, sid)]
        print(f"[backtest] resume: {len(pending)}/{len(shards)} shards pending", flush=True)

    total_records = 0
    with ProcessPoolExecutor(max_workers=bp.n_workers) as ex:
        futs = {ex.submit(run_shard, sid, ts, str(out), bp, sp): sid for sid, ts in pending}
        for fut in as_completed(futs):
            sid, n, skip = fut.result()
            if n < 0:
                print(f"[backtest] shard {sid:03d} already done, skipped", flush=True)
            else:
                total_records += n
                print(f"[backtest] shard {sid:03d} done: {n} records, {skip} tickers skipped", flush=True)
    print(f"[backtest] all shards done: {total_records} new records", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stale active-TR 修复方案走查回测")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--universe", help="universe 名 (如 sp500)")
    group.add_argument("--tickers", help="逗号分隔的 ticker 列表 (如 AAPL,MSFT)")
    ap.add_argument("--out", required=True, help="输出目录 (shards/ + report.*)")
    ap.add_argument("--shards", type=int, default=None, help="分片数 (默认取 BacktestParams.ticker_shards)")
    ap.add_argument("--workers", type=int, default=None, help="并行 worker 数")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有分片, 全部重算")
    ap.add_argument("--report", default=None, help="报告输出路径 (默认 <out>/report.md); 传 skip 则只跑不出报告")
    ap.add_argument("--report-only", action="store_true", help="不跑回测, 只从已有分片生成报告")
    args = ap.parse_args()

    overrides: dict = {}
    if args.shards is not None:
        overrides["ticker_shards"] = args.shards
    if args.workers is not None:
        overrides["n_workers"] = args.workers
    bp = BacktestParams(**overrides)
    sp = ScoringParams()

    if not args.report_only:
        universe = load_universe(args.universe) if args.universe else parse_tickers(args.tickers)
        run_backtest(universe=universe, out_dir=args.out, bp=bp, sp=sp, resume=not args.no_resume)

    if args.report != "skip":
        md, js = build_report(Path(args.out), bp)
        print(f"[backtest] report written: {md} , {js}", flush=True)


if __name__ == "__main__":
    main()
