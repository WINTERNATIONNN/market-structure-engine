"""Scanner 层 —— 把单标的 Wyckoff 引擎升级为多标的市场扫描。

红线: 纯计算, 零 LLM。网络抓取放在 CLI (scripts/scan_market.py), 本层保持可离线单测。
分层: scanner → scoring → wyckoff → data → core (见 pyproject.toml importlinter)。
"""

from mse.scanner.pipeline import analyze_ticker, scan
from mse.scanner.universe import load_universe, parse_tickers

__all__ = [
    "analyze_ticker",
    "scan",
    "load_universe",
    "parse_tickers",
]
