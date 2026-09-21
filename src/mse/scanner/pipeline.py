"""Scanner 管线 —— 单标的全链路分析 + 多标的排序。

本模块是**纯计算** (不联网): 输入已抓好的 OHLCVSeries, 输出打分候选。
网络抓取 (批量/缓存/限流处理) 放在 CLI (scripts/scan_market.py) 里, 使本层可离线单测。

analyze_ticker 复用 scripts/smoke_phase2_e2e.py 验证过的单标的链路:
    周线 → 指标 → detect_ranges → 逐 TR 日线事件 + detect_ps → detect_phases → 打分。
"""

from __future__ import annotations

from datetime import date

from mse.core.models import OHLCVSeries, TradingRange, WyckoffEvent
from mse.data.indicators import build_indicator_set
from mse.scoring import CandidateProfile, ScoringParams, score_candidate
from mse.wyckoff import (
    detect_bc,
    detect_phase_a,
    detect_phases,
    detect_ps,
    detect_ranges,
    detect_sos,
    detect_springs,
    detect_ut,
    detect_utad,
)


def _events_for_tr(daily: OHLCVSeries, dind, tr: TradingRange) -> list[WyckoffEvent]:
    """在给定周线 TR 的时间窗内跑全部日线事件检测器 (底部 + 顶部)。"""
    evs: list[WyckoffEvent] = []
    evs += detect_phase_a(daily, dind, tr)   # SC / AR / ST
    evs += detect_springs(daily, dind, tr)   # Spring
    evs += detect_sos(daily, dind, tr)       # SOS
    bc = detect_bc(daily, dind, tr)          # BC
    ut = detect_ut(daily, dind, tr)          # UT
    evs += bc
    evs += ut
    evs += detect_utad(daily, dind, tr, prior_events=bc + ut)  # UTAD 需 BC/UT 结构证据
    return evs


def _weekly_bars_between(weekly_dates: list[date], older: date, newer: date) -> int:
    """weekly_dates 上 older→newer 之间的周线 bar 数 (各自定位到最后一根 ≤ 该日期的 bar)。"""
    def idx_le(d: date) -> int:
        lo = -1
        for i, wd in enumerate(weekly_dates):
            if wd <= d:
                lo = i
            else:
                break
        return lo
    a, b = idx_le(older), idx_le(newer)
    if a < 0 or b < 0:
        return 10**9
    return b - a


def _active_tr(
    ranges: list[TradingRange],
    as_of: date,
    *,
    last_close: float | None,
    weekly_atr: float | None,
    weekly_dates: list[date],
    params: ScoringParams,
) -> TradingRange | None:
    """as_of 落入的 TR (取最长); 否则最近一个已开始的 TR —— 但施加陈旧度/邻近度守卫。

    守卫 (方案 V3 第二半, 已由 5 年全量 S&P500 走查回测验证):
      * 陈旧度门 (仅回退候选): as_of 距 TR 结束不超过 tr_stale_max_bars 个周线 bar;
      * 邻近度门 (两种候选统一): 最新价须在 [lower-k*ATR, upper+k*ATR] 内。单边大涨里价格
        虽仍"落在"某个时间跨度很长的旧区间内, 却已远超上沿, 这类幻影位置应判为无活跃区间。
    """
    containing = [r for r in ranges if r.start <= as_of <= r.end]
    if containing:
        cand = max(containing, key=lambda r: r.duration)
    else:
        started = [r for r in ranges if r.start <= as_of]
        if not started:
            return None
        cand = max(started, key=lambda r: r.start)
        if _weekly_bars_between(weekly_dates, cand.end, as_of) > params.tr_stale_max_bars:
            return None
    if last_close is not None and weekly_atr:
        pad = params.tr_proximity_atr * weekly_atr
        if not (cand.lower - pad <= last_close <= cand.upper + pad):
            return None
    return cand


def _last_weekly_atr(wind, key: str = "atr_14") -> float | None:
    """周线最新 ATR (最后一根 bar); NaN/缺失 → None。避免在本层引入 pandas 依赖。"""
    s = wind.series.get(key)
    if s is None or len(s) == 0:
        return None
    v = float(s.iloc[-1])
    return None if v != v else v  # v != v 即 NaN


def analyze_ticker(
    weekly: OHLCVSeries,
    daily: OHLCVSeries,
    *,
    params: ScoringParams = ScoringParams(),
) -> CandidateProfile:
    """对单只股票跑完整 Wyckoff 链路并打分。"""
    wind = build_indicator_set(weekly)
    dind = build_indicator_set(daily)

    ranges = detect_ranges(weekly, wind)

    events: list[WyckoffEvent] = []
    for tr in ranges:
        events += _events_for_tr(daily, dind, tr)
    events += detect_ps(daily, dind)  # PS 不依赖 TR (下跌末段)

    phases = detect_phases(weekly, wind, ranges, events)

    as_of = weekly.dates[-1].date()
    last_close = float(daily.close.iloc[-1]) if len(daily) else None
    weekly_bar_dates = [d.date() for d in weekly.dates]
    active_tr = _active_tr(
        ranges,
        as_of,
        last_close=last_close,
        weekly_atr=_last_weekly_atr(wind),
        weekly_dates=weekly_bar_dates,
        params=params,
    )

    return score_candidate(
        weekly.symbol.ticker,
        phases,
        events,
        as_of=as_of,
        last_close=last_close,
        active_tr=active_tr,
        daily_bar_dates=[d.date() for d in daily.dates],
        weekly_bar_dates=weekly_bar_dates,
        params=params,
    )


def scan(
    data_by_ticker: dict[str, tuple[OHLCVSeries, OHLCVSeries]],
    *,
    params: ScoringParams = ScoringParams(),
) -> tuple[list[CandidateProfile], list[str]]:
    """对已抓取的一批 (ticker → (weekly, daily)) 逐只分析并按 composite 降序排名。

    返回 (排好序的候选列表, 分析失败被跳过的 ticker 列表)。单只异常不影响整批。
    """
    profiles: list[CandidateProfile] = []
    skipped: list[str] = []
    for ticker, (weekly, daily) in data_by_ticker.items():
        try:
            profiles.append(analyze_ticker(weekly, daily, params=params))
        except Exception:  # noqa: BLE001 —— 单只标的任何计算异常都不应中断整批扫描
            skipped.append(ticker)
    profiles.sort(key=lambda p: p.composite_score, reverse=True)
    return profiles, skipped
