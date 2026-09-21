"""Wyckoff 事件检测 (Spec §4)。

日线事件层: 在周线给定的 Phase 语境 + TradingRange 下, 检测具体事件的
精确 date 与证据 (Spec §10.1)。每个事件输出统一的 WyckoffEvent 契约 (§9)。
"""

from mse.wyckoff.events.distribution import detect_bc, detect_ut, detect_utad
from mse.wyckoff.events.phase_a import detect_phase_a
from mse.wyckoff.events.ps import detect_ps
from mse.wyckoff.events.sos import detect_sos
from mse.wyckoff.events.spring import detect_springs

__all__ = [
    "detect_springs",
    "detect_sos",
    "detect_phase_a",
    "detect_ps",
    "detect_bc",
    "detect_ut",
    "detect_utad",
]
