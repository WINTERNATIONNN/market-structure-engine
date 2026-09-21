"""PS (Preliminary Support) 检测离线单元测试 (Spec §4.1)。合成日线, 完全离线。

PS 出现在下跌末段 (尚无 TR): 明确下跌 + 放量承接 + 长下影 + 跌速放缓。
构造: 20 根稳定下跌 bar → 1 根缩幅放量长下影承接 bar (PS) → 后续走势决定是否否决。
"""

from __future__ import annotations

import pandas as pd

from mse.core.enums import EventType, Timeframe, WyckoffPhase
from mse.core.models import OHLCVSeries, Symbol
from mse.data.indicators import build_indicator_set
from mse.wyckoff import detect_ps

_PS_I = 20  # PS 承接 bar 的位置


def _series(closes, highs, lows, vols):
    idx = pd.date_range("2022-01-03", periods=len(closes), freq="B")
    frame = pd.DataFrame(
        {"open": list(closes), "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )
    return OHLCVSeries(symbol=Symbol(ticker="TEST"), timeframe=Timeframe.DAILY, frame=frame), idx


def _downtrend_then_support(*, rally: bool):
    """20 根下跌 + PS 承接 bar + 后续 (rally=True → 无新低单边拉升; False → 再创新低)。"""
    closes, highs, lows, vols = [], [], [], []
    # 0..19: 稳定下跌 (每 bar 跌 1.2)
    for i in range(_PS_I):
        c = 130.0 - 1.2 * i
        closes.append(c); highs.append(c + 1.0); lows.append(c - 1.0); vols.append(1.0e6)
    # 20: PS —— 跌幅骤缩 (0.2) + 放量 + 深下影 (low 102, 收盘回升至 107)
    closes.append(107.0); highs.append(107.5); lows.append(102.0); vols.append(1.9e6)
    # 21..: 后续
    if rally:
        seq = [(110.0, 111.0, 108.0), (113.0, 114.0, 111.0), (115.0, 116.0, 113.0),
               (116.0, 117.0, 114.0), (117.0, 118.0, 115.0)]
    else:
        seq = [(103.0, 104.0, 101.0), (100.0, 101.0, 98.0), (98.0, 99.0, 96.0),
               (96.5, 97.5, 94.5), (95.0, 96.0, 93.0)]
    for c, h, lo in seq:
        closes.append(c); highs.append(h); lows.append(lo); vols.append(1.0e6)
    return _series(closes, highs, lows, vols)


def test_ps_detected_when_downtrend_continues() -> None:
    # 承接后再创新低 (下跌延续) → PS 有效。
    ohlcv, idx = _downtrend_then_support(rally=False)
    events = detect_ps(ohlcv, build_indicator_set(ohlcv))
    assert len(events) == 1, f"应检出 1 个 PS, 实际 {len(events)}"
    ps = events[0]
    assert ps.event_type == EventType.PS
    assert ps.date == idx[_PS_I].date()
    assert ps.probability > 0.5
    assert ps.phase_context == WyckoffPhase.MARKDOWN
    assert ps.tr_ref is None  # PS 阶段尚无成形 TR
    assert ps.meta["slope_atr"] < 0  # 前期确为下跌
    assert ps.meta["rvol"] >= 1.5
    assert ps.meta["close_pos"] > 0.6  # 长下影 / 收盘回升


def test_ps_vetoed_when_reverses() -> None:
    # 承接后未再创新低而单边拉升 → 趋势反转, 否决为 PS (§4.1)。
    ohlcv, idx = _downtrend_then_support(rally=True)
    events = detect_ps(ohlcv, build_indicator_set(ohlcv))
    ps_date = idx[_PS_I].date()
    assert all(e.date != ps_date for e in events), "反转情形不应在承接 bar 报 PS"


def test_no_downtrend_no_ps() -> None:
    # 全程横盘 (无下跌趋势) → 必要条件缺失 → 无 PS。
    n = 30
    closes = [110.0] * n
    highs = [111.0] * n
    lows = [109.0] * n
    vols = [1.0e6] * n
    lows[_PS_I] = 105.0; vols[_PS_I] = 2.0e6  # 即便有放量长下影, 无下跌趋势仍否决
    ohlcv, _ = _series(closes, highs, lows, vols)
    assert detect_ps(ohlcv, build_indicator_set(ohlcv)) == []
