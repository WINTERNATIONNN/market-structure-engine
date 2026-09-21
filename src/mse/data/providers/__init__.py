"""数据源适配层 (Phase 1)。

Provider 抽象接口 + 具体实现 (yfinance)。
上层只依赖抽象, 换数据源不改调用方 (依赖倒置)。
"""

from mse.data.providers.base import DataProvider, ProviderError
from mse.data.providers.yfinance_provider import YFinanceProvider

__all__ = ["DataProvider", "ProviderError", "YFinanceProvider"]
