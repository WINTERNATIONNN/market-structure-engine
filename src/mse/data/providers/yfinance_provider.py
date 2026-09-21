"""YFinanceProvider — 基于 yfinance 的美股数据源 (Phase 1)。

要点 (对应已锁定决策):
    * 美股: auto_adjust=True → 返回已复权 OHLC (Spec §10 决策 1/2)。
    * Multi-timeframe: 支持日线 + 周线 (Spec §10.1)。
    * 规范化 yfinance 的输出 (列名大写、可能的 MultiIndex) 为统一契约。
    * 网络/空数据 → ProviderError, 不静默返回空。

注意: yfinance 依赖网络; 单元测试应 mock, 冒烟测试才真正联网。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf

from mse.core.enums import Timeframe
from mse.core.models import OHLCVSeries, Symbol
from mse.core.models.market import OHLCV_COLUMNS
from mse.data.providers.base import ProviderError

# yfinance interval 映射。
_INTERVAL = {
    Timeframe.DAILY: "1d",
    Timeframe.WEEKLY: "1wk",
}


class YFinanceProvider:
    """实现 DataProvider 协议 (结构化子类型, 无需显式继承)。"""

    def __init__(self, *, auto_adjust: bool = True, timeout: int = 30) -> None:
        # auto_adjust=True: 复权口径固定 (美股决策)。改动会影响所有历史结果, 勿轻易切换。
        self._auto_adjust = auto_adjust
        self._timeout = timeout

    # ── Symbol 元数据 ─────────────────────────────────────────
    def get_symbol(self, ticker: str) -> Symbol:
        tk = yf.Ticker(ticker)
        try:
            info = tk.info or {}
        except Exception as exc:  # yfinance 内部异常类型不稳定, 统一包装
            raise ProviderError(f"获取 {ticker} 元数据失败: {exc}") from exc

        return Symbol(
            ticker=ticker,
            name=info.get("shortName") or info.get("longName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("marketCap"),
            float_shares=info.get("floatShares"),
            exchange=info.get("exchange"),
            currency=info.get("currency") or "USD",
        )

    # ── 单周期行情 ────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: date | None = None,
        end: date | None = None,
        *,
        symbol: Symbol | None = None,
    ) -> OHLCVSeries:
        raw = self._download(ticker, timeframe, start, end)
        frame = self._normalize(raw, ticker)
        return OHLCVSeries(
            symbol=symbol or Symbol(ticker=ticker),
            timeframe=timeframe,
            frame=frame,
            adjusted=self._auto_adjust,
            fetched_at=datetime.now(timezone.utc),
        )

    # ── 多周期 ────────────────────────────────────────────────
    def get_multi_timeframe(
        self,
        ticker: str,
        timeframes: list[Timeframe],
        start: date | None = None,
        end: date | None = None,
    ) -> dict[Timeframe, OHLCVSeries]:
        # 元数据只拉一次, 复用到各周期, 省去重复网络请求。
        symbol = self.get_symbol(ticker)
        return {
            tf: self.get_ohlcv(ticker, tf, start, end, symbol=symbol)
            for tf in timeframes
        }

    # ── 内部: 下载 + 规范化 ───────────────────────────────────
    def _download(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: date | None,
        end: date | None,
    ) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=ticker,
                interval=_INTERVAL[timeframe],
                start=start.isoformat() if start else None,
                end=end.isoformat() if end else None,
                auto_adjust=self._auto_adjust,
                progress=False,
                threads=False,
                timeout=self._timeout,
                # period 缺省时 yfinance 默认拉全部可用历史 (start=None)
                period=None if start else "max",
            )
        except Exception as exc:
            raise ProviderError(f"下载 {ticker} ({timeframe.value}) 失败: {exc}") from exc

        if raw is None or raw.empty:
            raise ProviderError(f"{ticker} ({timeframe.value}) 无数据返回")
        return raw

    @staticmethod
    def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """把 yfinance 输出规范化为统一契约: 小写列 + DatetimeIndex。"""
        df = raw.copy()

        # yfinance 单标的有时返回 MultiIndex 列 (('Close','AAPL')...); 压平。
        if isinstance(df.columns, pd.MultiIndex):
            # 取第 0 层字段名 (Open/High/...), 丢弃 ticker 层。
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]

        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ProviderError(
                f"{ticker} 返回缺少列 {missing}; 实际列: {list(df.columns)}"
            )

        df = df[list(OHLCV_COLUMNS)].copy()

        # 索引规范化为无时区 DatetimeIndex (日/周线只需日期)。
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"

        # 丢弃含 NaN 的行 (停牌/缺失), 保证下游指标计算干净。
        df = df.dropna(subset=list(OHLCV_COLUMNS))
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if df.empty:
            raise ProviderError(f"{ticker} 规范化后无有效行 (全为 NaN?)")
        return df
