"""断点/落盘测试 (往返相等 + DONE 标记语义 + read_all 只并合 DONE)。"""

from __future__ import annotations

from datetime import date

from backtest.checkpoint import (
    read_all,
    read_checkpoint,
    shard_done,
    shard_path,
    write_checkpoint,
)
from backtest.records import EventObs, SignalRecord


def _rec(ticker: str, variant: str = "V0") -> SignalRecord:
    return SignalRecord(
        variant=variant, ticker=ticker, as_of=date(2024, 1, 31),
        current_phase="markup", phase_probability=0.8,
        price_pos_in_tr=0.5, has_active_tr=True,
        springboard_score=0.5, transition_score=0.1, composite_score=0.5,
        direction="bullish", fwd_returns={21: 0.1, 63: None, 126: 0.2},
        events=(EventObs(event_type="Spring", date=date(2024, 1, 10), probability=0.8, fwd_move=0.05),),
    )


def test_write_read_roundtrip(tmp_path):
    recs = [_rec("AAA"), _rec("BBB", "V1")]
    write_checkpoint(tmp_path, 0, recs)
    back = read_checkpoint(tmp_path, 0)
    assert back == recs  # frozen dataclass 逐字段相等 (含 None fwd 与嵌套 events)


def test_shard_done_semantics(tmp_path):
    assert shard_done(tmp_path, 5) is False           # 不存在
    write_checkpoint(tmp_path, 5, [_rec("AAA")])
    assert shard_done(tmp_path, 5) is True             # 有 DONE 标记

    # 手动写一个无标记的半截文件 -> 未完成
    p = shard_path(tmp_path, 6)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"partial": true}\n', encoding="utf-8")
    assert shard_done(tmp_path, 6) is False


def test_read_all_merges_only_done_shards(tmp_path):
    write_checkpoint(tmp_path, 0, [_rec("AAA")])
    write_checkpoint(tmp_path, 1, [_rec("BBB"), _rec("CCC")])
    # 一个未完成分片不应被并入
    p = shard_path(tmp_path, 2)
    p.write_text('{"partial": true}\n', encoding="utf-8")

    merged = read_all(tmp_path)
    tickers = sorted(r.ticker for r in merged)
    assert tickers == ["AAA", "BBB", "CCC"]


def test_read_all_empty_when_no_shards(tmp_path):
    assert read_all(tmp_path) == []
