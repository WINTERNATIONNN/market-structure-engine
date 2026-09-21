"""股票池加载 —— 命名股票池 (如 sp500) 或自定义列表。

S&P 500 成分以**打包静态文件** (data/sp500.txt) 提供, 避免运行时依赖 Wikipedia。
文件为每行一个 ticker (# 开头为注释)。需刷新时重新生成该文件即可。
"""

from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"

_UNIVERSE_FILES = {
    "sp500": "sp500.txt",
}


def load_universe(name: str) -> list[str]:
    """按名称加载股票池 (当前支持 'sp500')。返回去重、保序的 ticker 列表。"""
    key = name.strip().lower()
    if key not in _UNIVERSE_FILES:
        raise ValueError(f"未知股票池 {name!r}; 可选: {sorted(_UNIVERSE_FILES)}")
    path = _DATA_DIR / _UNIVERSE_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"股票池文件缺失: {path}")
    return _read_ticker_file(path)


def parse_tickers(csv: str) -> list[str]:
    """解析逗号分隔的自定义 ticker 串 (如 'AAPL,MSFT,NVDA')。去重保序、大写。"""
    return _dedup([t.strip().upper() for t in csv.split(",") if t.strip()])


def _read_ticker_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return _dedup(out)


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
