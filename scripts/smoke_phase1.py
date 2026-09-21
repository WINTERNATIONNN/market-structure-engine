"""冒烟测试 (Phase 1): 真实联网拉取美股日线+周线, 验证 Data Layer 端到端。

运行: python scripts/smoke_phase1.py
非单元测试 (需网络)。CI 里的单元测试应 mock yfinance。
"""

from __future__ import annotations

from datetime import date

from mse.core.enums import Timeframe
from mse.data.providers import YFinanceProvider


def main() -> None:
    provider = YFinanceProvider()
    ticker = "AAPL"

    print(f"== 拉取 {ticker} 多周期 (2023-01-01 起) ==")
    series = provider.get_multi_timeframe(
        ticker,
        [Timeframe.DAILY, Timeframe.WEEKLY],
        start=date(2023, 1, 1),
        end=date(2024, 1, 1),
    )

    for tf, s in series.items():
        print(f"\n--- {tf.name} ({tf.value}) ---")
        print(f"  bars: {len(s)}")
        print(f"  range: {s.dates[0].date()} .. {s.dates[-1].date()}")
        print(f"  adjusted: {s.adjusted}")
        print(f"  columns: {list(s.frame.columns)}")
        last = s.bar_at(s.dates[-1].date())
        print(f"  last bar: O={last.open:.2f} H={last.high:.2f} "
              f"L={last.low:.2f} C={last.close:.2f} V={last.volume:,.0f}")

    # point-in-time 切片验证 (防前视偏差)
    daily = series[Timeframe.DAILY]
    cut = daily.slice_until(date(2023, 6, 30))
    print(f"\n== point-in-time 切片到 2023-06-30: {len(cut)} bars "
          f"(末日 {cut.dates[-1].date()}) ==")

    # 元数据验证
    sym = daily.symbol
    print(f"\n== Symbol 元数据 ==")
    print(f"  {sym.ticker} | {sym.name} | {sym.sector} / {sym.industry}")
    print(f"  market_cap={sym.market_cap} exchange={sym.exchange}")

    print("\n✅ Phase 1 Data Layer 冒烟测试通过")


if __name__ == "__main__":
    main()
