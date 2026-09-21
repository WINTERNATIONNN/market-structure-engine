"""冒烟测试 (Phase 2 Range Detection): 在真实周线数据上检测交易区间。

运行: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/smoke_ranges.py
"""

from __future__ import annotations

from datetime import date

from mse.core.enums import Timeframe
from mse.data.indicators import build_indicator_set
from mse.data.providers import YFinanceProvider
from mse.wyckoff import RangeParams, detect_ranges


def main() -> None:
    provider = YFinanceProvider()
    # 周线锚定结构 (Spec §10.1)。取一段够长的历史以容纳多个区间。
    ohlcv = provider.get_ohlcv(
        "AAPL", Timeframe.WEEKLY, start=date(2018, 1, 1), end=date(2024, 1, 1)
    )
    print(f"== 周线 bar 数: {len(ohlcv)} ({ohlcv.dates[0].date()} .. {ohlcv.dates[-1].date()}) ==")

    ind = build_indicator_set(ohlcv)
    print(f"   swings: {len(ind.swings)}  pivots: {len(ind.pivots)}\n")

    ranges = detect_ranges(ohlcv, ind, params=RangeParams())
    print(f"== 检测到 {len(ranges)} 个交易区间 ==")
    for i, tr in enumerate(ranges, 1):
        print(
            f"  [{i}] {tr.start} .. {tr.end}  ({tr.duration} 周)\n"
            f"      upper={tr.upper:.2f} lower={tr.lower:.2f} mid={tr.mid:.2f} "
            f"width_atr={tr.width_atr} slope_atr={tr.slope_atr}\n"
            f"      触碰 SH={tr.num_sh} SL={tr.num_sl}  "
            f"avg_vol={tr.avg_volume:,.0f} vol_trend={tr.volume_trend:+.4f}"
        )

    # 基本自洽校验
    for tr in ranges:
        assert tr.upper > tr.lower, "上沿必须高于下沿"
        assert tr.duration >= RangeParams().tr_min_bars, "区间时长不足"
        assert RangeParams().tr_width_atr_min <= tr.width_atr <= RangeParams().tr_width_atr_max
    for a, b in zip(ranges, ranges[1:]):
        assert a.end < b.start, "区间应互不重叠"

    print("\n✅ Phase 2 Range Detection 冒烟测试通过")


if __name__ == "__main__":
    main()
