"""Wyckoff 引擎参数 (外置阈值, 对应 Spec §6)。

Spec 红线: "禁止在事件逻辑中出现魔法数字"。所有阈值集中在此,
以 frozen dataclass 承载, 运行时注入。未来 Phase 0 从 yaml 加载覆盖默认值。

当前只含 Range Detection 所需的结构参数 (Spec §6.1); FSM / 事件参数
将随对应模块加入。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RangeParams:
    """交易区间检测参数 (Spec §1.2, §6.1)。默认值取自 Spec §6.1 结构参数表。"""

    # swing / pivot (与 Phase 1 indicator 层一致)
    swing_lookback: int = 5          # swing 左右比较窗口
    pivot_min_atr: float = 1.0       # pivot 最小幅度 (×ATR)

    # TR 成立条件 (Spec §1.2)
    # 注: 以下三项由 5 年全量 S&P500 走查回测重标定 (方案 V3)。原 Spec §6.1 起点
    # (tr_min_bars=15 / tr_width_atr_max=15 / tr_max_slope=0.15) 在单边上涨中会拒绝
    # 每个窗口 -> 不成新区间 -> 回退陈旧旧区间 -> 位置越界 + 评分归零的自相矛盾盲区。
    # 放松后上升通道也能成真区间: 命中率各周期均升、阶段矛盾率 0.883->0.575。
    tr_min_bars: int = 11            # TR 最小时长 (bar 数); 下限 = 2*swing_lookback+1
    tr_width_atr_min: float = 3.0    # 振幅/ATR 下限 (太窄不算区间, 见 §6.1 [3,15])
    tr_width_atr_max: float = 30.0   # 振幅/ATR 上限 (放宽: 趋势通道也纳入)
    tr_max_slope: float = 0.45       # |收盘回归斜率/bar| / ATR 上限 (放宽横盘度门)

    # 结构确认: 每侧最少 swing 触碰数 (TR 被反复测试才成立)
    min_touches: int = 2

    # 成交量
    vol_window: int = 20             # RVOL / 均量窗口

    # 依赖的指标 series key (须由 Phase 1 IndicatorSet 提供)
    atr_key: str = "atr_14"

    def __post_init__(self) -> None:
        if self.tr_width_atr_min >= self.tr_width_atr_max:
            raise ValueError(
                f"tr_width_atr_min ({self.tr_width_atr_min}) 必须小于 "
                f"tr_width_atr_max ({self.tr_width_atr_max})"
            )
        if self.tr_min_bars < 2 * self.swing_lookback + 1:
            raise ValueError(
                f"tr_min_bars ({self.tr_min_bars}) 太小, 无法容纳 swing_lookback "
                f"({self.swing_lookback}) 定义的摆动点"
            )


@dataclass(frozen=True)
class AggregationParams:
    """证据聚合参数 (Spec §2.4)。控制 probability / confidence 的计算。"""

    # 必要条件一票否决 (Spec §2.4): 必要证据隶属度低于此值 → probability=0
    necessary_floor: float = 0.3

    # 命中阈值: 隶属度 ≥ 此值才算"命中"一条证据 (用于 completeness)
    hit_threshold: float = 0.1

    # confidence 三分量权重 (Spec §2.4: 完整性 / 一致性 / 数据质量)。内部归一化。
    conf_w_completeness: float = 0.4
    conf_w_agreement: float = 0.3
    conf_w_quality: float = 0.3

    # 几何平均数值下限 (防 ln(0) = -inf)
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.necessary_floor <= 1.0:
            raise ValueError(f"necessary_floor 须在 [0,1]: {self.necessary_floor}")
        total = self.conf_w_completeness + self.conf_w_agreement + self.conf_w_quality
        if total <= 0:
            raise ValueError("confidence 三分量权重之和必须 > 0")


@dataclass(frozen=True)
class SpringParams:
    """Spring 事件参数 (Spec §4.5, §6.2)。默认值取自 Spec §6.2。"""

    # 刺穿深度 (×ATR)
    spring_max_depth_atr: float = 0.75   # 干净刺穿上限 → 越浅越像 terminal shakeout
    spring_deep_atr: float = 2.0         # 超过此深度视为真跌破 (必要条件隶属度→0, 否决)
    min_penetration_atr: float = 0.05    # 最小刺穿门: 更浅视为噪声, 不作为候选

    # 收回 / 确认
    spring_recover_bars: int = 2         # 收回时限 (bar)
    confirm_bars: int = 5                # 收回后寻找放量上涨确认的窗口
    spring_unconfirmed_factor: float = 0.7  # 未确认 Spring 的概率折扣

    # 成交量 (RVOL)
    vol_low_mult: float = 0.7            # 缩量阈值 (刺穿时供给枯竭)
    vol_high_mult: float = 1.5           # 放量阈值 (收回/确认时需求进场)

    # 证据权重 (Spec §4.5); 必要条件 (刺穿) 权重 w_pierce
    w_pierce: float = 1.0                # [N] 刺穿 TR.lower
    w_recover: float = 0.35              # [W] 快速收回
    w_volume: float = 0.25               # [W] 刺穿缩量 / 收回放量
    w_depth: float = 0.20                # [W] 刺穿浅 (干净)
    w_not_undercut: float = 0.20         # [W] 未显著低于前低

    # 依赖的指标 series key
    atr_key: str = "atr_14"
    rvol_key: str = "rvol_20"

    def __post_init__(self) -> None:
        if self.spring_deep_atr <= self.spring_max_depth_atr:
            raise ValueError("spring_deep_atr 必须大于 spring_max_depth_atr")
        if not 0.0 < self.spring_unconfirmed_factor <= 1.0:
            raise ValueError("spring_unconfirmed_factor 须在 (0,1]")


@dataclass(frozen=True)
class SOSParams:
    """SOS (Sign of Strength) 事件参数 (Spec §4.6, §6.2)。默认值取自 Spec §6.2。

    SOS 是 Spring 的镜像: 放量突破 TR.upper 并站稳, 标志 Accumulation→Markup。
    """

    # 突破 (×ATR)
    sos_min_break_atr: float = 0.5       # 显著突破幅度 (close-upper)/ATR
    min_break_atr: float = 0.05          # 最小突破门: 更浅视为噪声, 不作为候选

    # 成交量 (RVOL)
    sos_vol_mult: float = 1.8            # 真突破放量阈值

    # 站稳 / 确认
    holdback_bars: int = 3               # 突破后检查是否跌回 TR 的窗口
    confirm_bars: int = 5                # 寻找 LPS 回踩确认的窗口
    sos_lps_confidence_boost: float = 0.15  # LPS 确认后 confidence 提升 (Spec §4.6)

    # 证据权重 (Spec §4.6); 必要条件 (突破) 权重 w_breakout
    w_breakout: float = 1.0              # [N] 突破 TR.upper
    w_volume: float = 0.4                # [W] 突破放量
    w_magnitude: float = 0.3             # [W] 突破幅度显著
    w_backup: float = 0.3                # [W] 突破后不跌回 (back-up/LPS)

    # 依赖的指标 series key
    atr_key: str = "atr_14"
    rvol_key: str = "rvol_20"

    def __post_init__(self) -> None:
        if self.min_break_atr >= self.sos_min_break_atr:
            raise ValueError("min_break_atr 必须小于 sos_min_break_atr")
        if not 0.0 <= self.sos_lps_confidence_boost <= 1.0:
            raise ValueError("sos_lps_confidence_boost 须在 [0,1]")


@dataclass(frozen=True)
class PhaseAParams:
    """Accumulation Phase A 事件参数 (Spec §4.2 SC / §4.3 AR / §4.4 ST)。

    SC/AR/ST 是一条耦合的序列 (AR 依赖 SC, ST 依赖 AR), 共同界定 TR 初始边界
    (§4.3), 故合于一个参数类。取自 §6.2 者标注; §4.3/§4.4 提及但 §6.2 未列
    默认值者 (ar_max_bars / ar_min_rally_atr / st_zone_atr ...), 给出待 Phase 0
    标定的合理起点。
    """

    # --- SC 卖出高潮 (§4.2, 参数取自 §6.2) ---
    climax_vol_mult: float = 2.5         # [N] 极端放量 (RVOL)
    climax_range_atr: float = 2.0        # [W] 当日振幅 (range/ATR)
    climax_close_pos: float = 0.6        # [W] 收盘位置下限 (0=最低,1=最高)
    sc_zone_atr: float = 2.0             # SC 低点须落在 TR.lower 附近 (×ATR); §6.2 未列, 默认

    # --- AR 自动反弹 (§4.3; §6.2 未列, 默认待标定) ---
    ar_max_bars: int = 15                # [N] AR 高点须在 SC 后此窗口内
    ar_min_rally_atr: float = 1.5        # [W] 自 SC 低点反弹幅度 (×ATR)
    ar_upper_zone_atr: float = 2.0       # [W] AR 高点接近 TR.upper 的容差 (×ATR)

    # --- ST 二次测试 (§4.4; st_zone_atr §6.2 未列, 默认待标定) ---
    st_zone_atr: float = 1.0             # [N] 回落至 TR.lower 附近的容差 (×ATR)
    st_max_events: int = 3               # 最多输出的 ST 数 (§4.4 允许多次)
    undercut_atr: float = 0.5            # ST 低点低于 SC 低点超过此值 → 视为跌破前低

    # --- 成交量 (RVOL, §6.2) ---
    vol_high_mult: float = 1.5           # 放量阈值
    vol_low_mult: float = 0.7            # 缩量阈值 (ST 卖压枯竭的关键)

    # --- 证据权重 ---
    # SC (§4.2): [N] 极端放量 w=1.0
    sc_w_volume: float = 1.0
    sc_w_range: float = 0.4              # 当日振幅极大
    sc_w_tail_close: float = 0.4         # 长下影 + 收盘回升
    sc_w_recover: float = 0.2            # 创新低后快速收回
    # AR (§4.3): [N] 时间紧随 SC w=1.0
    ar_w_timing: float = 1.0
    ar_w_rally: float = 0.5              # 显著反弹
    ar_w_sh: float = 0.3                 # 反弹高点接近 TR.upper
    ar_w_volrecede: float = 0.2         # 反弹缩量
    # ST (§4.4): [N] 落在 TR.lower 区 w=1.0
    st_w_zone: float = 1.0
    st_w_lowvol: float = 0.5            # 测试缩量
    st_w_higherlow: float = 0.3        # 未跌破前低
    st_w_recover: float = 0.2          # 测试后收回区间

    # 依赖的指标 series key
    atr_key: str = "atr_14"
    rvol_key: str = "rvol_20"

    def __post_init__(self) -> None:
        if self.ar_max_bars < 1:
            raise ValueError("ar_max_bars 必须 ≥ 1")
        if self.st_max_events < 1:
            raise ValueError("st_max_events 必须 ≥ 1")
        if not 0.0 <= self.climax_close_pos <= 1.0:
            raise ValueError("climax_close_pos 须在 [0,1]")


@dataclass(frozen=True)
class FSMParams:
    """Phase 状态机参数 (Spec §3)。控制四阶段判定的迟滞与防抖。

    min_state_bars 取自 §6.1 结构参数表; enter/exit 阈值与累积支持窗口 §6 未列,
    给出待 Phase 0 标定的合理起点。FSM 在周线上运行 (§10.1)。
    """

    # --- 防抖动 (§3.3) ---
    enter_threshold: float = 0.55        # 进入某状态所需证据分下限 (须 > exit_threshold)
    exit_threshold: float = 0.35         # 退出当前状态的证据分上限 (迟滞带 = [exit, enter])
    min_state_bars: int = 10             # 每个状态最小持续 (bar), 取自 §6.1
    confirm_bars: int = 2                # 转移需连续 n 根 bar 证据支持 (单根异动不触发)

    # --- 转移证据权重 (§3.2 各转移的证据构成) ---
    # 每个转移分 = 加权几何平均(该转移相关证据的 membership)。
    # 底部锚定 (UNDEFINED/MARKDOWN → ACCUMULATION): TR 存在 + 底部事件 (PS/SC/AR)。
    w_tr_present: float = 1.0            # [N] 存在成形 TR
    w_bottom_event: float = 0.8          # 底部事件 (SC/AR/PS) 证据强度
    # ACCUMULATION → MARKUP: SOS 突破。
    w_sos: float = 1.0                   # [N] SOS 事件强度
    # MARKUP → DISTRIBUTION: 高位 TR + 顶部事件 (BC/UT)。
    w_top_event: float = 0.8             # 顶部事件 (BC/UT) 证据强度
    # DISTRIBUTION → MARKDOWN: 跌破下沿 + UTAD 失败。
    w_breakdown: float = 1.0             # [N] 向下跌破 + UTAD

    # 事件影响的时间半衰 (bar): 事件对 recency 越近影响越大。
    event_decay_bars: int = 8

    # --- 指标派生证据 (§3.2 明列但事件本身不含: 前期下跌 / 跌破放量) ---
    # 这些为**加权** (非必要) 证据, 且以 "反向趋势才削弱" 的方式构造:
    # 中性/契合 → 隶属度≈1 (不破坏事件驱动的主转移), 明确反向趋势 → 隶属度→0。
    trend_lookback: int = 8              # 收盘回归斜率的回看窗口 (周线 bar)
    trend_slope_atr: float = 0.10        # 斜率/bar 归一化 ATR 的 "显著趋势" 幅度门槛
    w_prior_trend: float = 0.5           # ACCUM: 非前期上涨 / MARKUP: 非下跌中
    w_breakdown_vol: float = 0.5         # MARKDOWN: 跌破放量
    vol_high_mult: float = 1.5           # 放量阈值 (RVOL)
    atr_key: str = "atr_14"
    rvol_key: str = "rvol_20"

    def __post_init__(self) -> None:
        if not self.enter_threshold > self.exit_threshold:
            raise ValueError(
                f"enter_threshold ({self.enter_threshold}) 必须 > exit_threshold "
                f"({self.exit_threshold}) —— 迟滞带不成立"
            )
        if not 0.0 <= self.exit_threshold <= self.enter_threshold <= 1.0:
            raise ValueError("阈值须满足 0 ≤ exit ≤ enter ≤ 1")
        if self.min_state_bars < 1 or self.confirm_bars < 1:
            raise ValueError("min_state_bars / confirm_bars 必须 ≥ 1")
        if self.trend_lookback < 2:
            raise ValueError("trend_lookback 必须 ≥ 2")
        if self.trend_slope_atr <= 0.0:
            raise ValueError("trend_slope_atr 必须为正 (幅度门槛)")


@dataclass(frozen=True)
class DistributionParams:
    """Distribution 顶部事件参数 (Spec §4.7 BC / §4.8 UT / §4.9 UTAD)。

    Accumulation 的镜像: BC↔SC (顶部高潮), UT↔Spring (向上假突破)。UTAD 为 UT
    的派发末期强化版, 标志 Distribution→Markdown。取自 §6.2 者标注; §4.8/§4.9
    提及但 §6.2 未列默认值者给出待 Phase 0 标定的起点。三者同属顶部, 合于一类。
    """

    # --- BC 买入高潮 (§4.7, 镜像 SC; 参数取自 §6.2) ---
    climax_vol_mult: float = 2.5         # [N] 极端放量 (RVOL)
    climax_range_atr: float = 2.0        # [W] 当日振幅 (range/ATR)
    climax_close_pos: float = 0.6        # 与 SC 共用; BC 用 (1-此值) 作收盘上限 (长上影)
    bc_zone_atr: float = 2.0             # BC 高点须接近 TR.upper (×ATR)

    # --- UT 向上假突破 (§4.8, 镜像 Spring; §6.2 未列, 默认待标定) ---
    ut_max_penetration_atr: float = 0.75  # 干净假突破深度上限 (越浅越像诱多)
    ut_deep_atr: float = 2.0             # 超过则视为真突破 (必要条件→0, 否决=SOS)
    min_penetration_atr: float = 0.05    # 最小刺穿门 (更浅视为噪声)
    ut_recover_bars: int = 2             # 回落 TR 内的时限 (bar)

    # --- UTAD (§4.9; §6.2 未列, 默认待标定) ---
    utad_recover_bars: int = 2           # 快速反转跌回的时限 (bar)
    confirm_bars: int = 5                # 后续跌破 TR.lower 确认 Markdown 的窗口

    # --- 成交量 (RVOL, §6.2) ---
    vol_high_mult: float = 1.5           # 放量阈值
    vol_low_mult: float = 0.7            # 缩量阈值

    # --- 证据权重 ---
    # BC (§4.7): [N] 极端放量 w=1.0
    bc_w_volume: float = 1.0
    bc_w_range: float = 0.4              # 当日振幅极大
    bc_w_tail_close: float = 0.4         # 长上影 + 收盘远离最高 (抛压)
    bc_w_recover: float = 0.2            # 创新高后快速回落
    # UT (§4.8): [N] 刺穿 TR.upper w=1.0
    ut_w_pierce: float = 1.0
    ut_w_recover: float = 0.35           # 快速回落 TR 内
    ut_w_volume: float = 0.25            # 放量但无法守住
    ut_w_weakclose: float = 0.20         # 收盘转弱 (收在当日下部)
    ut_w_no_new_high: float = 0.20       # 未创有效持续新高
    # UTAD (§4.9): [N] 刺穿并短暂创区间新高 w=1.0
    utad_w_pierce: float = 1.0
    utad_w_reversal: float = 0.35        # 快速反转跌回且跌势延续
    utad_w_volume: float = 0.30          # 反转放量 (派发痕迹)
    utad_w_structure: float = 0.20       # 此前已有 BC+UT 结构
    utad_w_belowmid: float = 0.15        # 跌回后跌破 TR.mid / 逼近 lower

    # 依赖的指标 series key
    atr_key: str = "atr_14"
    rvol_key: str = "rvol_20"

    def __post_init__(self) -> None:
        if self.ut_deep_atr <= self.ut_max_penetration_atr:
            raise ValueError("ut_deep_atr 必须大于 ut_max_penetration_atr")
        if not 0.0 <= self.climax_close_pos <= 1.0:
            raise ValueError("climax_close_pos 须在 [0,1]")
        if self.ut_recover_bars < 0 or self.utad_recover_bars < 0:
            raise ValueError("recover_bars 不能为负")


@dataclass(frozen=True)
class PSParams:
    """PS (Preliminary Support) 事件参数 (Spec §4.1)。

    PS 是 Accumulation 序列的**首个**事件, 出现在 MARKDOWN/下跌末段, **尚无成形 TR**
    (故 detect_ps 不接收 TradingRange)。§6.2 未单列 PS 默认值, 给出待 Phase 0 标定的
    合理起点。
    """

    # --- 前期下跌趋势 (§4.1 [N]) ---
    trend_lookback: int = 15             # 回归斜率的回看窗口 (bar)
    downtrend_slope: float = -0.10       # 收盘回归斜率/bar 归一化 ATR 的下跌门槛 (≤ 此值)
    decel_lookback: int = 5              # 跌速放缓比较的前窗口 (bar)

    # --- 承接 SL (§4.1 [W]) ---
    close_pos_lo: float = 0.3            # 长下影/收盘回升的位置下限
    close_pos_hi: float = 0.6            # 位置饱和点 (≥ 此值 → 满隶属)

    # --- 成交量 (RVOL, §6.2) ---
    vol_high_mult: float = 1.5           # 放量承接阈值

    # --- 否决: 之后未再创新低而单边拉升 → 趋势反转非 PS (§4.1) ---
    veto_bars: int = 5                   # 前视窗口 (bar)
    veto_rally_atr: float = 2.0          # 无新低却拉升超过此幅度 (×ATR) → 否决

    # --- 证据权重 (§4.1) ---
    ps_w_trend: float = 1.0              # [N] 前期明确下跌
    ps_w_support_vol: float = 0.4        # 放量下跌后的显著承接
    ps_w_tail: float = 0.3               # 长下影 / 收盘远离最低
    ps_w_decel: float = 0.3              # 相对前几根 bar 跌速放缓

    # 依赖的指标 series key
    atr_key: str = "atr_14"
    rvol_key: str = "rvol_20"

    def __post_init__(self) -> None:
        if self.downtrend_slope >= 0.0:
            raise ValueError("downtrend_slope 必须为负 (下跌)")
        if self.trend_lookback < 2 or self.decel_lookback < 1:
            raise ValueError("trend_lookback ≥ 2 且 decel_lookback ≥ 1")
        if not 0.0 <= self.close_pos_lo < self.close_pos_hi <= 1.0:
            raise ValueError("须满足 0 ≤ close_pos_lo < close_pos_hi ≤ 1")
