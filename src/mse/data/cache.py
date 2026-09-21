"""本地行情缓存 —— 避免重复抓取, 迭代打分逻辑时不再反复联网/撞限流。

按 (ticker, timeframe, start, end) 缓存 OHLCV frame 到 .cache/ohlcv/*.pkl。
选用 pickle 而非 parquet: 零额外依赖 (parquet 需 pyarrow/fastparquet 引擎, 未必已装)。
仅作本地开发缓存, 不追求跨版本可移植。

load_or_fetch: 命中的直接读缓存, 未命中的才走 download_batch 并回填缓存。
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from mse.core.enums import Timeframe
from mse.core.models import OHLCVSeries, Symbol
from mse.data.providers.batch import download_batch

_DEFAULT_CACHE_DIR = Path(".cache") / "ohlcv"


def _key(ticker: str, timeframe: Timeframe, start: date | None, end: date | None) -> str:
    raw = f"{ticker.upper()}|{timeframe.value}|{start}|{end}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _path(cache_dir: Path, ticker: str, timeframe: Timeframe, start, end) -> Path:
    return cache_dir / f"{ticker.upper()}_{timeframe.value}_{_key(ticker, timeframe, start, end)}.pkl"


def load_or_fetch(
    tickers: list[str],
    timeframe: Timeframe,
    start: date | None = None,
    end: date | None = None,
    *,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    auto_adjust: bool = True,
    **batch_kwargs,
) -> tuple[dict[str, OHLCVSeries], list[str]]:
    """命中缓存的直接读, 未命中的批量抓取并回填。返回 (ticker → OHLCVSeries, 跳过列表)。"""
    cache_dir = Path(cache_dir)
    result: dict[str, OHLCVSeries] = {}
    misses: list[str] = []

    if use_cache:
        for t in tickers:
            frame = _read(_path(cache_dir, t, timeframe, start, end))
            if frame is not None:
                result[t] = _to_series(t, timeframe, frame, auto_adjust)
            else:
                misses.append(t)
    else:
        misses = list(tickers)

    skipped: list[str] = []
    if misses:
        fetched, skipped = download_batch(
            misses, timeframe, start, end, auto_adjust=auto_adjust, **batch_kwargs
        )
        for t, series in fetched.items():
            result[t] = series
            if use_cache:
                _write(_path(cache_dir, t, timeframe, start, end), series.frame)

    return result, skipped


def _to_series(ticker: str, timeframe: Timeframe, frame: pd.DataFrame, adjusted: bool) -> OHLCVSeries:
    return OHLCVSeries(
        symbol=Symbol(ticker=ticker),
        timeframe=timeframe,
        frame=frame,
        adjusted=adjusted,
        fetched_at=datetime.now(timezone.utc),
    )


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:  # noqa: BLE001 —— 缓存损坏视为未命中
        return None


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_pickle(path)
    except Exception:  # noqa: BLE001 —— 缓存写失败不应影响主流程
        pass
