"""枚举 — 系统共享词汇表 (对应 Spec §1, §3, §4)。

集中定义所有类型常量, 避免散落的魔法字符串。
"""

from __future__ import annotations

from enum import Enum


class Timeframe(str, Enum):
    """周期。Multi-timeframe 决策: 周线定 Phase, 日线定事件 (Spec §10.1)。"""

    DAILY = "1d"
    WEEKLY = "1wk"

    @property
    def is_higher(self) -> bool:
        """是否为更高周期 (用于结构 vs 事件的分工判断)。"""
        return self is Timeframe.WEEKLY


class WyckoffPhase(str, Enum):
    """Wyckoff 四阶段 + 未定义 (Spec §3.1)。FSM 的状态集合。"""

    UNDEFINED = "undefined"
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"


class EventType(str, Enum):
    """9 个核心 Wyckoff 事件 (Spec §4)。"""

    PS = "PS"          # Preliminary Support 初步支撑
    SC = "SC"          # Selling Climax 卖出高潮
    AR = "AR"          # Automatic Rally 自动反弹
    ST = "ST"          # Secondary Test 二次测试
    SPRING = "Spring"  # 弹簧 / 假跌破
    SOS = "SOS"        # Sign of Strength 强势信号
    BC = "BC"          # Buying Climax 买入高潮
    UT = "UT"          # Upthrust 向上假突破
    UTAD = "UTAD"      # Upthrust After Distribution 派发后向上假突破

    @property
    def phase(self) -> WyckoffPhase:
        """事件所属的典型阶段 (Spec §5 关系矩阵)。"""
        return _EVENT_PHASE[self]


_EVENT_PHASE: dict[EventType, WyckoffPhase] = {
    EventType.PS: WyckoffPhase.ACCUMULATION,
    EventType.SC: WyckoffPhase.ACCUMULATION,
    EventType.AR: WyckoffPhase.ACCUMULATION,
    EventType.ST: WyckoffPhase.ACCUMULATION,
    EventType.SPRING: WyckoffPhase.ACCUMULATION,
    EventType.SOS: WyckoffPhase.ACCUMULATION,  # Accum→Markup 过渡
    EventType.BC: WyckoffPhase.DISTRIBUTION,
    EventType.UT: WyckoffPhase.DISTRIBUTION,
    EventType.UTAD: WyckoffPhase.DISTRIBUTION,  # Distrib→Markdown 过渡
}


class LabelVerdict(str, Enum):
    """Golden dataset 人工判定 (Spec §7.1)。"""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


class DataQualityFlag(str, Enum):
    """数据质量标记 (Spec §8 数据质量 gate)。降低相关窗口的 confidence。"""

    OK = "ok"
    GAP = "gap"                # 缺失 bar
    HALT = "halt"              # 停牌
    ANOMALY = "anomaly"        # 异常值 (如价格跳变)
    LOW_LIQUIDITY = "low_liquidity"
