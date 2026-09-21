"""Spring 检测离线单元测试 (Spec §4.5)。合成日线, 完全离线。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mse.core.enums import EventType, Timeframe, WyckoffPhase
from mse.core.models import OHLCVSeries, Symbol, TradingRange
from mse.data.indicators import build_indicator_set
from mse.wyckoff import detect_springs

LOWER, UPPER = 100.0, 120.0


def _daily_with_pierce(
    pierce_low: float,
    recover_close: float | None,
    *,
    pierce_vol: float = 0.5e6,
    confirm: bool = True,
    n_pre: int = 45,
) -> tuple[OHLCVSeries, TradingRange]:
    """构造一段在 [100,120] 区间震荡的日线, 在 n_pre 处制造一次刺穿。

    recover_close=None 表示刺穿后不收回 (持续走低)。
    """
    rng = np.random.default_rng(42)
    # 区间内震荡, 刻意抬离下沿 (mid=112, 振幅6, 低噪声) → lows 稳定 >103, 无非预期刺穿。
    pre = 112 + 6 * np.sin(2 * np.pi * np.arange(n_pre) / 7) + rng.normal(0, 0.5, n_pre)
    closes = list(pre)
    highs = [c + 1.5 for c in closes]
    lows = [c - 1.5 for c in closes]
    vols = [1e6] * n_pre

    # 刺穿 bar
    closes.append(pierce_low + 1.0 if recover_close is not None else pierce_low - 1.0)
    highs.append(LOWER + 1.0)
    lows.append(pierce_low)
    vols.append(pierce_vol)

    # 收回 / 持续走低
    if recover_close is not None:
        closes.append(recover_close); highs.append(recover_close + 1.5); lows.append(LOWER - 0.5)
        vols.append(1e6)
        # 确认 bar (放量上涨)
        if confirm:
            closes.append(recover_close + 4); highs.append(recover_close + 5)
            lows.append(recover_close); vols.append(2.2e6)
        else:
            closes.append(recover_close + 0.5); highs.append(recover_close + 1)
            lows.append(recover_close - 1); vols.append(0.9e6)
    else:
        for _ in range(3):  # 持续走低放量 → 真跌破
            closes.append(closes[-1] - 3); highs.append(closes[-1] + 1)
            lows.append(closes[-1] - 2); vols.append(2e6)

    m = len(closes)
    idx = pd.date_range("2022-01-03", periods=m, freq="B")
    # open = 前一根 close (gap-less) → 上涨日 close>open 成立, 确认逻辑才能触发。
    opens = [closes[0]] + closes[:-1]
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx
    )
    ohlcv = OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.DAILY, frame=frame)
    tr = TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY,
        start=idx[0].date(), end=idx[-1].date(), duration=m,
        upper=UPPER, lower=LOWER, avg_volume=1e6, volume_trend=0.0,
        width_atr=8.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )
    return ohlcv, tr


def test_clean_spring_detected_and_confirmed() -> None:
    # 浅刺穿 (low=97, 深度3 ≈ 亚ATR), 次日收回, 放量上涨确认。
    ohlcv, tr = _daily_with_pierce(97.0, recover_close=104.0, confirm=True)
    ind = build_indicator_set(ohlcv)

    events = detect_springs(ohlcv, ind, tr)

    assert len(events) == 1, f"应检测出 1 个 Spring, 实际 {len(events)}"
    ev = events[0]
    assert ev.event_type == EventType.SPRING
    assert ev.probability > 0.0
    assert ev.confirmation_date is not None, "放量上涨应确认 Spring"
    assert ev.phase_context == WyckoffPhase.ACCUMULATION
    assert ev.tr_ref is not None and ev.tr_ref.lower == LOWER
    assert any(e.rule_id == "spring.pierce" and e.necessary for e in ev.evidence)
    assert ev.meta["depth_atr"] > 0


def test_unconfirmed_spring_discounted() -> None:
    # 同样刺穿+收回, 但无放量上涨确认 → probability 打折, confirmation_date=None。
    conf, tr1 = _daily_with_pierce(97.0, recover_close=104.0, confirm=True)
    unconf, tr2 = _daily_with_pierce(97.0, recover_close=104.0, confirm=False)
    ind_c = build_indicator_set(conf)
    ind_u = build_indicator_set(unconf)

    ev_c = detect_springs(conf, ind_c, tr1)[0]
    ev_u = detect_springs(unconf, ind_u, tr2)[0]

    assert ev_u.confirmation_date is None
    assert ev_u.probability < ev_c.probability, "未确认应比已确认概率低"


def test_deep_pierce_rejected() -> None:
    # 深刺穿 (low=80, 深度20 → 远超 deep_atr) → 真跌破, 一票否决, 不产出。
    ohlcv, tr = _daily_with_pierce(80.0, recover_close=101.0, confirm=True)
    ind = build_indicator_set(ohlcv)

    events = detect_springs(ohlcv, ind, tr)

    assert events == [], "深刺穿应被否决, 不作为 Spring"


def test_no_recover_rejected() -> None:
    # 刺穿后持续走低不收回 → 非 Spring。
    ohlcv, tr = _daily_with_pierce(97.0, recover_close=None)
    ind = build_indicator_set(ohlcv)

    events = detect_springs(ohlcv, ind, tr)

    assert events == [], "未收回的刺穿是真跌破, 不应判为 Spring"


def test_no_pierce_no_event() -> None:
    # 全程在区间内震荡, 无刺穿 → 无事件。
    rng = np.random.default_rng(0)
    n = 60
    closes = 112 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.5, n)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    frame = pd.DataFrame(
        {"open": closes, "high": closes + 1.5, "low": closes - 1.5, "close": closes,
         "volume": [1e6] * n}, index=idx,
    )
    ohlcv = OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.DAILY, frame=frame)
    tr = TradingRange(
        ticker="TEST", timeframe=Timeframe.WEEKLY, start=idx[0].date(), end=idx[-1].date(),
        duration=n, upper=UPPER, lower=LOWER, avg_volume=1e6, volume_trend=0.0,
        width_atr=8.0, slope_atr=0.05, num_sh=3, num_sl=3,
    )
    ind = build_indicator_set(ohlcv)

    assert detect_springs(ohlcv, ind, tr) == []
