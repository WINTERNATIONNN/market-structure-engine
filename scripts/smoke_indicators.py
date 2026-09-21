"""冒烟测试 (Phase 1 指标层): 在真实美股数据上计算指标 + swings/pivots。

运行: python scripts/smoke_indicators.py
"""

from __future__ import annotations

from datetime import date

from mse.core.enums import Timeframe
from mse.data.indicators import build_indicator_set, registry
from mse.data.providers import YFinanceProvider


def main() -> None:
    print(f"== registry 已注册指标: {registry.available()} ==\n")

    provider = YFinanceProvider()
    ohlcv = provider.get_ohlcv(
        "AAPL", Timeframe.DAILY, start=date(2023, 1, 1), end=date(2024, 1, 1)
    )

    ind = build_indicator_set(ohlcv)

    print(f"== IndicatorSet ({ind.ticker}, {ind.timeframe.name}) ==")
    print(f"  series keys: {sorted(ind.series)}")
    print(f"  swings: {len(ind.swings)}  pivots: {len(ind.pivots)}\n")

    last = ohlcv.dates[-1].date()
    print(f"== 末日 {last} 指标值 ==")
    for key in ("sma_50", "ema_20", "atr_14", "rsi_14", "macd", "macd_hist", "rvol_20"):
        val = ind.value_at(key, last)
        print(f"  {key:14s} = {val:.4f}")

    print(f"\n== 最近 5 个 pivot ==")
    for p in ind.pivots[-5:]:
        print(f"  {p.date} {p.type.value:5s} price={p.price:.2f} strength={p.strength}")

    # 校验: RSI 落在 0..100, ATR > 0
    rsi_last = ind.value_at("rsi_14", last)
    atr_last = ind.value_at("atr_14", last)
    assert 0 <= rsi_last <= 100, f"RSI 越界: {rsi_last}"
    assert atr_last > 0, f"ATR 非正: {atr_last}"

    print("\n✅ Phase 1 指标层冒烟测试通过")


if __name__ == "__main__":
    main()
