"""回测记录模型 + jsonl 序列化。

一个 (方案 × ticker × 截面) 产出一条 SignalRecord; 每条内嵌若干 EventObs (近期事件 + 其事后走势)。
以 jsonl (每行一个 JSON 对象) 落盘, 便于分片写入、断点续跑与流式读取。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass(frozen=True)
class EventObs:
    """一个被检测到的事件 + 其事后 lookahead 天走势 (指标 3 用)。"""

    event_type: str          # EventType.value
    date: date
    probability: float
    fwd_move: float | None   # 事件日后 bp.event_lookahead_days 交易日的收益


@dataclass(frozen=True)
class SignalRecord:
    """单条走查信号记录 (某方案在某截面对某 ticker 的打分快照 + 前向收益 + 近期事件)。"""

    variant: str
    ticker: str
    as_of: date
    current_phase: str
    phase_probability: float
    price_pos_in_tr: float | None
    has_active_tr: bool
    springboard_score: float
    transition_score: float
    composite_score: float
    direction: str | None                    # 'bullish'/'bearish'/None (signal_direction 归一化后)
    fwd_returns: dict[int, float | None]      # horizon -> 收益
    events: tuple[EventObs, ...] = field(default_factory=tuple)


def record_to_json(r: SignalRecord) -> str:
    """序列化为单行 JSON。date -> ISO 字符串; fwd_returns 键 int -> str (JSON 要求)。"""
    d = asdict(r)
    d["as_of"] = r.as_of.isoformat()
    d["fwd_returns"] = {str(h): v for h, v in r.fwd_returns.items()}
    d["events"] = [
        {**asdict(e), "date": e.date.isoformat()} for e in r.events
    ]
    return json.dumps(d, ensure_ascii=False)


def record_from_json(line: str) -> SignalRecord:
    d = json.loads(line)
    events = tuple(
        EventObs(
            event_type=e["event_type"],
            date=date.fromisoformat(e["date"]),
            probability=e["probability"],
            fwd_move=e["fwd_move"],
        )
        for e in d.get("events", [])
    )
    return SignalRecord(
        variant=d["variant"],
        ticker=d["ticker"],
        as_of=date.fromisoformat(d["as_of"]),
        current_phase=d["current_phase"],
        phase_probability=d["phase_probability"],
        price_pos_in_tr=d["price_pos_in_tr"],
        has_active_tr=d["has_active_tr"],
        springboard_score=d["springboard_score"],
        transition_score=d["transition_score"],
        composite_score=d["composite_score"],
        direction=d["direction"],
        fwd_returns={int(h): v for h, v in d["fwd_returns"].items()},
        events=events,
    )
