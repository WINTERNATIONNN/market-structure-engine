"""mse.data — Data Layer (Phase 1)。

职责: 行情接入、清洗、复权、指标计算, 输出统一 Data Model。
禁止 import mse.services (LLM/news) — 由 import-linter 强制。
"""
