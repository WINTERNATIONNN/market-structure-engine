"""批量行情抓取 —— 为多标的扫描解决 yfinance 限流。

现有 YFinanceProvider.get_ohlcv 是单标的、串行、无批量。扫 500 只会被限流打死。
本模块用 yf.download 的**批量接口** (一次请求多只) + 分块 + 块间 sleep, 大幅减少请求数。

设计:
    * yf.download(tickers="AAPL MSFT ...", group_by="ticker") → 列为 MultiIndex,
      level 0 = ticker, level 1 = 字段。逐 ticker 取子表, 复用 _normalize 规范化。
    * 单只 (空数据/缺列/异常) → 跳过并计入 skipped, 不中断整批。
    * 整块失败 (网络) → 该块全部计入 skipped, 继续下一块。
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf

from mse.core.enums import Timeframe
from mse.core.models import OHLCVSeries, Symbol
from mse.data.providers.yfinance_provider import YFinanceProvider, _INTERVAL


def download_batch(
    tickers: list[str],
    timeframe: Timeframe,
    start: date | None = None,
    end: date | None = None,
    *,
    chunk_size: int = 50,
    pause_s: float = 1.0,
    auto_adjust: bool = True,
    timeout: int = 60,
) -> tuple[dict[str, OHLCVSeries], list[str]]:
    """分块批量抓取。返回 (ticker → OHLCVSeries, 跳过的 ticker 列表)。"""
    result: dict[str, OHLCVSeries] = {}
    skipped: list[str] = []

    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for ci, chunk in enumerate(chunks):
        try:
            raw = yf.download(
                tickers=" ".join(chunk),
                interval=_INTERVAL[timeframe],
                start=start.isoformat() if start else None,
                end=end.isoformat() if end else None,
                auto_adjust=auto_adjust,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=timeout,
                period=None if start else "max",
            )
        except Exception:  # noqa: BLE001 —— 整块网络失败: 全部跳过, 继续
            skipped.extend(chunk)
            if ci < len(chunks) - 1:
                time.sleep(pause_s)
            continue

        for ticker in chunk:
            sub = _extract_ticker_frame(raw, ticker)
            if sub is None:
                skipped.append(ticker)
                continue
            try:
                frame = YFinanceProvider._normalize(sub, ticker)
            except Exception:  # noqa: BLE001 —— 单只规范化失败 (空/缺列): 跳过
                skipped.append(ticker)
                continue
            result[ticker] = OHLCVSeries(
                symbol=Symbol(ticker=ticker),
                timeframe=timeframe,
                frame=frame,
                adjusted=auto_adjust,
                fetched_at=datetime.now(timezone.utc),
            )

        if ci < len(chunks) - 1:
            time.sleep(pause_s)  # 块间节流, 降低限流概率

    return result, skipped


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """从批量下载结果中取出单只 ticker 的字段子表。取不到→None。"""
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by='ticker': level 0 = ticker。
        if ticker in raw.columns.get_level_values(0):
            sub = raw[ticker]
            return sub if not sub.dropna(how="all").empty else None
        return None
    # 单只标的时 yfinance 可能返回扁平列 (无 MultiIndex)。
    return raw if not raw.dropna(how="all").empty else None
