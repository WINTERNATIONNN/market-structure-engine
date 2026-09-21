"""指标层 (Phase 1)。

可插拔架构 (Spec 强调): 指标以 registry 注册, 新增指标不改调用方。
    registry.py — IndicatorRegistry + Indicator 协议
    builtin.py  — MA/EMA/ATR/RSI/MACD/RVOL
    swings.py   — Swing/Pivot 检测
    pipeline.py — 组装 IndicatorSet
"""

from mse.data.indicators.pipeline import build_indicator_set
from mse.data.indicators.registry import Indicator, IndicatorRegistry, registry

__all__ = ["Indicator", "IndicatorRegistry", "registry", "build_indicator_set"]
