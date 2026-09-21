"""Scoring 层参数 (外置阈值)。

红线一致: 禁魔法数字 —— 所有打分阈值/权重集中在此 frozen dataclass,
运行时注入。仿 `mse.wyckoff.params` 的范式 (frozen + __post_init__ 校验)。

默认值为**待标定的合理起点**, 非最终标定。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringParams:
    """候选打分参数。

    两条打分主线 (对应用户诉求「看涨 setup 和转折点都要」):
      * springboard —— 看涨吸筹 setup: 近期 Spring/SOS + 处于/接近 Accumulation。
      * transition  —— 近期阶段转折点 (看涨/看跌方向由目标阶段决定)。
    composite = max(springboard, transition), 两类候选同榜。
    """

    # --- 时间窗 (recency) ---
    event_recency_bars: int = 30         # 日线: 事件"近期"窗口 (约 6 周交易日)
    transition_recency_bars: int = 8     # 周线: 阶段转折"近期"窗口

    # --- 事件门槛 ---
    min_event_prob: float = 0.5          # 低于此概率的事件不计入 setup (噪声)

    # --- recency 衰减: 距今越远权重越低 (线性衰减到 0) ---
    # 衰减因子 = max(0, 1 - age_bars / decay_bars)
    event_decay_bars: int = 30           # 日线事件衰减跨度
    transition_decay_bars: int = 8       # 周线转折衰减跨度

    # --- springboard 分项权重 (加权平均, 内部归一化) ---
    sb_w_event: float = 0.6              # 最强近期看涨事件 (Spring/SOS) 强度
    sb_w_phase: float = 0.3              # 阶段因子 (在 Accumulation 最佳)
    sb_w_location: float = 0.1           # 价格位置因子 (靠近 TR 下沿=吸筹区)

    # 阶段因子映射 (springboard 视角: Accumulation 最佳)
    phase_factor_accumulation: float = 1.0
    phase_factor_markup: float = 0.7     # 刚进入 markup 仍可搭车
    phase_factor_other: float = 0.2      # distribution/markdown/undefined

    # 位置因子: 最新收盘价在 TR 内的相对位置 (0=下沿,1=上沿);
    # 越低越像吸筹区 → location = 1 - clamp(pos)。无 TR → 中性 0.5。
    location_neutral: float = 0.5

    # --- active-TR 选择守卫 (方案 V3 第二半, 由 5 年全量 S&P500 走查回测验证) ---
    # 单边大涨里 detect_ranges 可能不成新区间 → 回退陈旧旧区间 → price_pos 越界 (幻影位置)。
    # 两道门消除之: 陈旧度门 (仅约束回退候选) + 邻近度门 (约束所有候选)。
    tr_stale_max_bars: int = 8            # 回退候选: as_of 距 TR 结束不超过此周线 bar 数
    tr_proximity_atr: float = 3.0         # 邻近度: 最新价须在 [lower-k*ATR, upper+k*ATR] 内

    # --- 看涨 / 看跌事件与阶段的归类 ---
    # (事件/阶段的具体成员在 score.py 用枚举判断; 此处只放可调阈值)

    def __post_init__(self) -> None:
        if self.event_recency_bars < 1 or self.transition_recency_bars < 1:
            raise ValueError("recency 窗口必须 ≥ 1")
        if self.event_decay_bars < 1 or self.transition_decay_bars < 1:
            raise ValueError("decay 跨度必须 ≥ 1")
        if not 0.0 <= self.min_event_prob <= 1.0:
            raise ValueError("min_event_prob 须在 [0,1]")
        if self.tr_stale_max_bars < 0:
            raise ValueError("tr_stale_max_bars 不能为负")
        if self.tr_proximity_atr <= 0.0:
            raise ValueError("tr_proximity_atr 必须为正")
        if self.sb_w_event + self.sb_w_phase + self.sb_w_location <= 0:
            raise ValueError("springboard 权重之和必须 > 0")
        for name, v in (
            ("phase_factor_accumulation", self.phase_factor_accumulation),
            ("phase_factor_markup", self.phase_factor_markup),
            ("phase_factor_other", self.phase_factor_other),
            ("location_neutral", self.location_neutral),
        ):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} 须在 [0,1]: {v}")
