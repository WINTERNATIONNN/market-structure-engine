"""Provider 抽象接口 (Phase 1)。

定义数据源契约。上层 (loaders / engine) 只依赖此协议, 不依赖具体实现。
未来加 Provider (IB / Polygon / 本地 CSV) 只需实现此接口。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from mse.core.enums import Timeframe
from mse.core.models import OHLCVSeries, Symbol


class ProviderError(RuntimeError):
    """数据源相关错误 (网络失败、标的不存在、数据为空等)。"""


@runtime_checkable
class DataProvider(Protocol):
    """行情数据源契约。

    实现须保证:
        * 返回的 OHLCVSeries 已复权 (美股口径), 索引升序、无重复。
        * 空数据或标的无效时抛 ProviderError, 不返回空对象。
    """

    def get_symbol(self, ticker: str) -> Symbol:
        """获取标的元数据 (行业/市值等)。"""
        ...

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: date | None = None,
        end: date | None = None,
    ) -> OHLCVSeries:
        """获取单周期行情序列。"""
        ...

    def get_multi_timeframe(
        self,
        ticker: str,
        timeframes: list[Timeframe],
        start: date | None = None,
        end: date | None = None,
    ) -> dict[Timeframe, OHLCVSeries]:
        """一次获取多周期 (multi-timeframe 决策, Spec §10.1)。"""
        ...
