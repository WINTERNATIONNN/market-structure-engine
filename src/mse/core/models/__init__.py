"""数据模型 (Phase 0 契约的强类型实现, pydantic v2)。

分组:
    market.py     — Symbol / Bar / OHLCVSeries (原始行情)
    indicators.py — Swing / Pivot / IndicatorSet (计算产物)
    structure.py  — TradingRange (Wyckoff 结构, Phase 2)
    evidence.py   — Evidence / WyckoffEvent (证据与事件契约, Phase 2)
    phase.py      — PhaseResult (Phase FSM 输出契约, Phase 2)

评分模型将在 Phase 3 继续加入。
"""

from mse.core.models.evidence import Evidence, WyckoffEvent
from mse.core.models.indicators import IndicatorSet, Pivot, Swing, SwingType
from mse.core.models.market import Bar, OHLCVSeries, Symbol
from mse.core.models.phase import PhaseResult
from mse.core.models.structure import TradingRange

__all__ = [
    "Symbol",
    "Bar",
    "OHLCVSeries",
    "IndicatorSet",
    "Swing",
    "SwingType",
    "Pivot",
    "TradingRange",
    "Evidence",
    "WyckoffEvent",
    "PhaseResult",
]
