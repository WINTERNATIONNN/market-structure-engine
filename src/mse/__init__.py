"""市场结构引擎 — 顶层包。

分层 (依赖只能自上而下):
    agent → reporting → ranking → analysis → scanner → scoring → wyckoff → data → core

红线:
    * core/data/wyckoff/scoring/scanner 禁止 import mse.services (LLM/news)。
    * Engine 层不做 LLM 调用; LLM 只在 services 层。
由 import-linter 强制 (见 pyproject.toml)。
"""

__version__ = "0.1.0"
