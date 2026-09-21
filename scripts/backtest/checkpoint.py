"""分片断点续跑 —— 原子落盘 + 完成标记, 让长跑可中断可恢复。

一个分片 = 一批 ticker 的全部 (截面 × 方案) 记录, 写成一个 jsonl 文件, 末行加 "# DONE" 标记。
写入走 tmp + os.replace 原子替换, 崩溃不会留下半截的 DONE 文件。
续跑时跳过已 DONE 的分片; 未完成 (无标记 / 只有 .tmp) 的分片重算。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from backtest.records import SignalRecord, record_from_json, record_to_json

_DONE_MARKER = "# DONE"


def shards_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "shards"


def shard_path(out_dir: Path, shard_id: int) -> Path:
    return shards_dir(out_dir) / f"shard_{shard_id:03d}.jsonl"


def shard_done(out_dir: Path, shard_id: int) -> bool:
    """文件存在且末行为 DONE 标记 → 视为完成。"""
    path = shard_path(out_dir, shard_id)
    if not path.exists():
        return False
    last = ""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s:
                last = s
    return last == _DONE_MARKER


def write_checkpoint(out_dir: Path, shard_id: int, records: Iterable[SignalRecord]) -> Path:
    """原子写入分片: 先写 .tmp, 追加 DONE 标记, 再 os.replace 到正式路径。"""
    path = shard_path(out_dir, shard_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(record_to_json(r))
            fh.write("\n")
        fh.write(_DONE_MARKER + "\n")
    os.replace(tmp, path)
    return path


def read_checkpoint(out_dir: Path, shard_id: int) -> list[SignalRecord]:
    path = shard_path(out_dir, shard_id)
    if not path.exists():
        return []
    return _read_records(path)


def read_all(out_dir: Path) -> list[SignalRecord]:
    """并合全部**已完成** (DONE) 分片的记录。"""
    d = shards_dir(out_dir)
    if not d.exists():
        return []
    out: list[SignalRecord] = []
    for path in sorted(d.glob("shard_*.jsonl")):
        # 只读 DONE 分片, 忽略未完成的半截文件。
        sid = int(path.stem.split("_")[1])
        if shard_done(out_dir, sid):
            out.extend(_read_records(path))
    return out


def _read_records(path: Path) -> list[SignalRecord]:
    out: list[SignalRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if not s or s == _DONE_MARKER:
                continue
            out.append(record_from_json(s))
    return out
