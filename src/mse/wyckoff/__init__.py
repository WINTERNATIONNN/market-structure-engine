"""Wyckoff 引擎层 (Phase 2)。

分工 (Spec §10.1, multi-timeframe):
    * 周线 → Range Detection (本层 range_detection) + Phase FSM。
    * 日线 → 事件检测 (SC/AR/ST/Spring/SOS ...), 接收周线的 TR 与 phase 语境。

红线: 本层**零 LLM 调用** (纯计算)。参数一律外置 (见 params.py), 禁魔法数字。
"""

from mse.wyckoff.evidence import ENGINE_VERSION, AggregateResult, aggregate
from mse.wyckoff.events import (
    detect_bc,
    detect_phase_a,
    detect_ps,
    detect_sos,
    detect_springs,
    detect_ut,
    detect_utad,
)
from mse.wyckoff.params import (
    AggregationParams,
    DistributionParams,
    FSMParams,
    PhaseAParams,
    PSParams,
    RangeParams,
    SOSParams,
    SpringParams,
)
from mse.wyckoff.phase_fsm import detect_phases
from mse.wyckoff.range_detection import detect_ranges

__all__ = [
    "RangeParams",
    "AggregationParams",
    "SpringParams",
    "SOSParams",
    "PhaseAParams",
    "FSMParams",
    "DistributionParams",
    "PSParams",
    "detect_ranges",
    "detect_springs",
    "detect_sos",
    "detect_phase_a",
    "detect_ps",
    "detect_bc",
    "detect_ut",
    "detect_utad",
    "detect_phases",
    "aggregate",
    "AggregateResult",
    "ENGINE_VERSION",
]
