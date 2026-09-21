"""候选打分 (Scoring 层核心)。

纯函数: 输入是**已算好的** phases / events + 最新收盘价 + 可选 TR, 输出 CandidateProfile。
不触碰网络、不跑检测器 —— 因此可完全离线单测。

两条打分主线 (用户诉求「看涨 setup 和转折点都要」):
  * springboard_score —— 看涨吸筹: 近期高概率 Spring/SOS × 阶段因子 × 位置因子。
  * transition_score  —— 近期阶段转折: 最近一次 FSM 进入的目标阶段概率 × 时间衰减,
                          方向由目标阶段决定 (进 ACCUM/MARKUP=bullish, 进 MARKDOWN=bearish;
                          distribution 为描述性阶段, 不产出方向性结论 —— 5 年全量回测证实其
                          无下行预测力)。
composite = max(两者), 使两类候选同榜; direction 让看跌转折也可见。

设计取舍 (见 README「已知限制」): FSM 会长期"粘"在 distribution, 故打分**主要依赖
敏感的日线事件**, transition 只作并列信号, 不作唯一依据。
"""

from __future__ import annotations

from datetime import date

from mse.core.enums import EventType, WyckoffPhase
from mse.core.models import PhaseResult, TradingRange, WyckoffEvent
from mse.scoring.candidate import CandidateProfile, EventRef, TransitionRef
from mse.scoring.params import ScoringParams

# 看涨 setup 事件 (吸筹/走强) 与看跌事件 (派发/走弱)。
_BULL_EVENTS = {EventType.SPRING, EventType.SOS, EventType.PS}
_BEAR_EVENTS = {EventType.BC, EventType.UT, EventType.UTAD}

# 转折方向归类 (按目标阶段)。
_BULLISH_PHASES = {WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP}
_BEARISH_PHASES = {WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN}  # 语义上的顶部/下行阶段
# 其中真正产出「看跌」方向信号的阶段。distribution 已降级为描述性 (5 年全量回测: 高置信
# distribution 的未来 63d 下跌率相对基线 lift≈0, 无下行预测力), 仅 markdown (真实下行段)
# 产出方向性结论。distribution 仍作阶段标签显示, 但不再据此输出 transition_direction。
_DIRECTIONAL_BEARISH_PHASES = {WyckoffPhase.MARKDOWN}


def _recency_factor(age_bars: int, decay_bars: int) -> float:
    """线性 recency 衰减: age=0 → 1.0, age≥decay → 0.0。"""
    if age_bars <= 0:
        return 1.0
    return max(0.0, 1.0 - age_bars / float(decay_bars))


def _phase_factor(phase: WyckoffPhase, params: ScoringParams) -> float:
    if phase == WyckoffPhase.ACCUMULATION:
        return params.phase_factor_accumulation
    if phase == WyckoffPhase.MARKUP:
        return params.phase_factor_markup
    return params.phase_factor_other


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """加权平均, 权重之和为 0 时返回 0。pairs = [(weight, value), ...]。"""
    wsum = sum(w for w, _ in pairs)
    if wsum <= 0:
        return 0.0
    return sum(w * v for w, v in pairs) / wsum


def score_candidate(
    ticker: str,
    phases: list[PhaseResult],
    events: list[WyckoffEvent],
    *,
    as_of: date,
    last_close: float | None = None,
    active_tr: TradingRange | None = None,
    daily_bar_dates: list[date] | None = None,
    weekly_bar_dates: list[date] | None = None,
    params: ScoringParams = ScoringParams(),
) -> CandidateProfile:
    """把单只股票的引擎产物打分成 CandidateProfile。

    * as_of: 评估基准日 (通常最新周线 bar 日期)。
    * daily_bar_dates / weekly_bar_dates: 用于把事件/转折日期换算成"距今多少 bar"
      (recency)。缺省时退化为按自然日估算。
    """
    # 当前阶段 = 最后一段 (时间升序)。
    current = phases[-1] if phases else None
    current_phase = current.label if current else WyckoffPhase.UNDEFINED
    phase_prob = current.probability if current else 0.0
    phase_conf = current.confidence if current else 0.0

    price_pos = active_tr.position(last_close) if (active_tr and last_close is not None) else None

    # ── 事件分类 + recency 过滤 ──────────────────────────────
    def age_daily(when: date) -> int:
        if daily_bar_dates:
            # 最后一根 ≤ when 的日线 bar 到末尾的距离。
            idx = _bars_ago(daily_bar_dates, when)
            return idx if idx is not None else (as_of - when).days
        return (as_of - when).days

    bull_refs: list[EventRef] = []
    bear_refs: list[EventRef] = []
    best_bull_strength = 0.0  # 最强近期看涨事件 (prob × recency)
    for e in events:
        age = age_daily(e.date)
        if age > params.event_recency_bars or e.date > as_of:
            continue
        recency = _recency_factor(age, params.event_decay_bars)
        ref = EventRef(event_type=e.event_type, date=e.date, probability=e.probability)
        if e.event_type in _BULL_EVENTS:
            bull_refs.append(ref)
            if e.probability >= params.min_event_prob:
                best_bull_strength = max(best_bull_strength, e.probability * recency)
        elif e.event_type in _BEAR_EVENTS:
            bear_refs.append(ref)

    # ── springboard_score ───────────────────────────────────
    phase_f = _phase_factor(current_phase, params)
    location_f = params.location_neutral if price_pos is None else max(0.0, min(1.0, 1.0 - price_pos))
    springboard = _weighted_mean(
        [
            (params.sb_w_event, best_bull_strength),
            (params.sb_w_phase, phase_f),
            (params.sb_w_location, location_f),
        ]
    )
    # 无任何近期看涨事件 → setup 不成立 (阶段/位置不足以独撑)。
    if best_bull_strength <= 0.0:
        springboard = 0.0

    # ── transition_score ────────────────────────────────────
    transition_score = 0.0
    transition_direction: str | None = None
    last_transition: TransitionRef | None = None
    if current is not None and len(phases) >= 2 and current_phase != WyckoffPhase.UNDEFINED:
        entered = current.state_entered_date
        age_w = _weekly_age(entered, as_of, weekly_bar_dates)
        if age_w <= params.transition_recency_bars:
            recency = _recency_factor(age_w, params.transition_decay_bars)
            transition_score = current.probability * recency
            if current_phase in _BULLISH_PHASES:
                transition_direction = "bullish"
            elif current_phase in _DIRECTIONAL_BEARISH_PHASES:
                transition_direction = "bearish"
            # distribution 为描述性阶段: 记录转折 (transition_score) 但不产出方向性结论
            # (transition_direction 保持 None)。
            last_transition = TransitionRef(
                from_phase=phases[-2].label,
                to_phase=current_phase,
                entered_date=entered,
                bars_ago=age_w,
                probability=current.probability,
            )

    composite = max(springboard, transition_score)

    tags = _build_tags(
        current_phase, bull_refs, bear_refs, last_transition, springboard, transition_score, params
    )

    return CandidateProfile(
        ticker=ticker,
        as_of=as_of,
        current_phase=current_phase,
        phase_probability=phase_prob,
        phase_confidence=phase_conf,
        active_tr=active_tr,
        price_pos_in_tr=price_pos,
        recent_bull_events=bull_refs,
        recent_bear_events=bear_refs,
        last_transition=last_transition,
        springboard_score=round(springboard, 4),
        transition_score=round(transition_score, 4),
        transition_direction=transition_direction,
        composite_score=round(composite, 4),
        tags=tags,
    )


def _bars_ago(bar_dates: list[date], when: date) -> int | None:
    """when 之后 (含) 还有多少根 bar。找不到 (when 晚于所有 bar) → None。

    用最后一根 ≤ when 的 bar 作为锚: bars_ago = len - 1 - anchor_idx。
    """
    anchor = None
    for i, d in enumerate(bar_dates):
        if d <= when:
            anchor = i
        else:
            break
    if anchor is None:
        return None
    return len(bar_dates) - 1 - anchor


def _weekly_age(entered: date, as_of: date, weekly_bar_dates: list[date] | None) -> int:
    if weekly_bar_dates:
        ba = _bars_ago(weekly_bar_dates, entered)
        if ba is not None:
            return ba
    # 退化: 按自然周估算。
    return max(0, (as_of - entered).days // 7)


def _build_tags(
    phase: WyckoffPhase,
    bull: list[EventRef],
    bear: list[EventRef],
    transition: TransitionRef | None,
    springboard: float,
    transition_score: float,
    params: ScoringParams,
) -> list[str]:
    tags: list[str] = [f"阶段={phase.value}"]
    strong_bull = [e for e in bull if e.probability >= params.min_event_prob]
    if strong_bull:
        best = max(strong_bull, key=lambda e: e.probability)
        tags.append(f"近期{best.event_type.value} p={best.probability:.2f}")
    strong_bear = [e for e in bear if e.probability >= params.min_event_prob]
    if strong_bear:
        best = max(strong_bear, key=lambda e: e.probability)
        tags.append(f"近期{best.event_type.value} p={best.probability:.2f}")
    if transition is not None:
        to = transition.to_phase
        if to in _BULLISH_PHASES:
            arrow = "看涨"
        elif to in _DIRECTIONAL_BEARISH_PHASES:
            arrow = "看跌"
        elif to == WyckoffPhase.DISTRIBUTION:
            arrow = "顶部结构(派发·描述性)"  # 无下行预测力: 只描述, 不给方向
        else:
            arrow = ""
        tags.append(f"→{to.value}({transition.bars_ago}周前){arrow}")
    return tags
