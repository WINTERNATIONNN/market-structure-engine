"""回测参数 (外置阈值, 无魔法数字)。

仿 `mse.wyckoff.params` / `mse.scoring.params` 的范式: frozen dataclass + __post_init__ 校验。
所有阈值 (数据窗口 / 走查网格 / 各方案门槛 / 指标口径 / 运行时) 集中于此, 便于调参与复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class BacktestParams:
    """走查回测的全部可调参数。默认值为待标定的合理起点。"""

    # --- 数据加载窗口: 必须匹配 .cache/ohlcv 的缓存键, 否则未命中会尝试联网抓取 ---
    data_start: date = date(2018, 1, 1)
    data_end: date = date(2026, 8, 13)

    # --- 走查截面网格 (月度再平衡) ---
    grid_start: date = date(2020, 8, 31)   # 首个截面 (留 >2y 历史给周线 TR/FSM)
    grid_end: date = date(2026, 2, 27)     # 末截面: 之后仍留 >=max(horizon) 交易日到 data_end
    rebalance: str = "M"                   # pandas 月末频率 (月度再平衡; pandas<2.2 用 "M")

    # --- 截面切片的最小 bar 要求 (不足则跳过该截面) ---
    min_weekly_bars: int = 30              # 周线检测器 + FSM 所需的最小历史
    min_daily_bars: int = 60               # 日线事件检测所需的最小历史

    # --- 前向收益周期 (日线交易日) ---
    horizons: tuple[int, ...] = (21, 63, 126)

    # --- V1: 陈旧区间守卫 + 位置裁剪 ---
    tr_stale_max_bars: int = 8             # 周线: TR "活跃" 仅当 (as_of - r.end) <= N 个周线 bar
    tr_proximity_atr: float = 3.0          # 且 last_close 在 [lower-k*ATR, upper+k*ATR] 内 (周线 ATR)

    # --- V2: 突破重探 ---
    breakout_atr: float = 3.0              # last_close > upper + k*ATR => 视为突破, 触发重探
    redetect_window_bars: int = 30         # 近端周线滚动窗口长度 (喂给宽松 detect_ranges)

    # --- V2 / V3: 宽松 RangeParams (须守约束 tr_min_bars >= 2*swing_lookback+1 = 11) ---
    relaxed_tr_max_slope: float = 0.45
    relaxed_tr_width_atr_max: float = 30.0
    relaxed_tr_min_bars: int = 11

    # --- V4: 趋势动量兜底打分 ---
    mom_rsi_bull: float = 55.0
    mom_rsi_bear: float = 45.0
    mom_rvol_min: float = 1.0
    mom_score_cap: float = 0.6             # 合成 springboard 上限 (压在真 TR setup 之下)
    mom_min_conditions: int = 3            # 4 个动量条件里至少命中几个才给方向

    # --- 指标口径 ---
    confident_phase_prob: float = 0.7      # 指标 2: markup/distribution "高置信" 门槛
    signal_min_score: float = 0.1          # 指标 1: composite 达此值才算可操作方向信号
    event_lookahead_days: int = 21         # 指标 3: 事件后前视窗口 (交易日)
    event_log_recency_days: int = 63       # 只记录 as_of 前该窗口内新近的事件 (控制记录体积)
    consistency_horizon: int = 63          # 指标 2b: 标签-实现所用周期 (须 in horizons)

    # --- 运行时 ---
    ticker_shards: int = 16
    n_workers: int = 8

    def __post_init__(self) -> None:
        if self.data_end <= self.data_start:
            raise ValueError("data_end 必须晚于 data_start")
        if self.grid_end < self.grid_start:
            raise ValueError("grid_end 不得早于 grid_start")
        if not (self.data_start <= self.grid_start and self.grid_end <= self.data_end):
            raise ValueError("走查网格须落在数据窗口内")
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("horizons 必须全为正整数")
        if self.consistency_horizon not in self.horizons:
            raise ValueError("consistency_horizon 必须是 horizons 之一")
        if self.relaxed_tr_min_bars < 11:
            raise ValueError("relaxed_tr_min_bars 须 >= 2*swing_lookback+1 (=11)")
        if self.relaxed_tr_width_atr_max <= 3.0:
            raise ValueError("relaxed_tr_width_atr_max 须 > tr_width_atr_min(3.0)")
        if not 0.0 <= self.signal_min_score <= 1.0:
            raise ValueError("signal_min_score 须在 [0,1]")
        if not 0.0 <= self.confident_phase_prob <= 1.0:
            raise ValueError("confident_phase_prob 须在 [0,1]")
        if not 0.0 <= self.mom_score_cap <= 1.0:
            raise ValueError("mom_score_cap 须在 [0,1]")
        if not (0.0 <= self.mom_rsi_bear <= self.mom_rsi_bull <= 100.0):
            raise ValueError("须满足 0 <= mom_rsi_bear <= mom_rsi_bull <= 100")
        if self.mom_rvol_min <= 0.0:
            raise ValueError("mom_rvol_min 必须为正")
        if not 1 <= self.mom_min_conditions <= 4:
            raise ValueError("mom_min_conditions 须在 [1,4]")
        if self.tr_stale_max_bars < 0 or self.redetect_window_bars < 11:
            raise ValueError("tr_stale_max_bars>=0 且 redetect_window_bars>=11")
        if self.breakout_atr < 0 or self.tr_proximity_atr < 0:
            raise ValueError("breakout_atr / tr_proximity_atr 不能为负")
        if self.event_lookahead_days < 1 or self.event_log_recency_days < 1:
            raise ValueError("event_lookahead_days / event_log_recency_days 须 >= 1")
        if self.ticker_shards < 1 or self.n_workers < 1:
            raise ValueError("ticker_shards / n_workers 须 >= 1")
