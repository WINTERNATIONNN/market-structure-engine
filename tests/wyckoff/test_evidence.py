"""证据聚合引擎单元测试 (Spec §2.4)。纯计算, 完全离线。"""

from __future__ import annotations

import math

import pytest

from mse.core.models import Evidence
from mse.wyckoff.evidence import aggregate
from mse.wyckoff.params import AggregationParams


def _ev(rule: str, w: float, m: float, necessary: bool = False) -> Evidence:
    return Evidence(rule_id=rule, weight=w, membership=m, necessary=necessary)


def test_empty_evidence() -> None:
    r = aggregate([])
    assert r.probability == 0.0
    assert r.confidence == 0.0


def test_weighted_geometric_mean() -> None:
    # 两条等权证据 m=0.5, 0.8 → 几何平均 = sqrt(0.4) ≈ 0.6325
    r = aggregate([_ev("a", 1.0, 0.5), _ev("b", 1.0, 0.8)])
    assert r.probability == pytest.approx(math.sqrt(0.5 * 0.8), abs=1e-3)
    assert not r.vetoed


def test_geometric_mean_punishes_weak_link() -> None:
    # 几何平均: 一条接近 0 应显著拉低总分 (对比算术平均 0.63)
    r = aggregate([_ev("a", 1.0, 0.9), _ev("b", 1.0, 0.05)])
    assert r.probability < 0.3, "缺一不可: 弱证据应大幅拉低几何平均"


def test_necessary_veto() -> None:
    # 必要证据隶属度低于 floor(默认0.3) → 一票否决 probability=0
    params = AggregationParams()
    r = aggregate(
        [_ev("must", 1.0, 0.1, necessary=True), _ev("other", 1.0, 0.95)],
        params=params,
    )
    assert r.vetoed is True
    assert r.probability == 0.0
    assert "否决" in r.reason


def test_necessary_pass() -> None:
    # 必要证据达标 → 不否决, 正常聚合
    r = aggregate([_ev("must", 1.0, 0.8, necessary=True), _ev("other", 1.0, 0.7)])
    assert r.vetoed is False
    assert r.probability > 0.0


def test_confidence_completeness() -> None:
    # 全部命中 vs 半数命中: completeness 不同 → confidence 不同
    full = aggregate([_ev("a", 1, 0.8), _ev("b", 1, 0.8), _ev("c", 1, 0.8)])
    partial = aggregate([_ev("a", 1, 0.8), _ev("b", 1, 0.0), _ev("c", 1, 0.0)])
    assert full.confidence > partial.confidence


def test_confidence_data_quality() -> None:
    good = aggregate([_ev("a", 1, 0.8), _ev("b", 1, 0.8)], data_quality=1.0)
    bad = aggregate([_ev("a", 1, 0.8), _ev("b", 1, 0.8)], data_quality=0.2)
    assert good.confidence > bad.confidence


def test_confidence_agreement() -> None:
    # 一致的证据 (都 0.8) 比离散的 (0.95/0.1) agreement 高 → confidence 更高
    agree = aggregate([_ev("a", 1, 0.8), _ev("b", 1, 0.8)])
    disagree = aggregate([_ev("a", 1, 0.95), _ev("b", 1, 0.1)])
    assert agree.confidence > disagree.confidence


def test_probability_confidence_independent() -> None:
    # Spec §2.3 示例: 命中强证据但不完整/数据差 → p 高而 c 低
    r = aggregate(
        [_ev("strong1", 1, 0.9), _ev("strong2", 1, 0.85), _ev("missing", 1, 0.0)],
        data_quality=0.4,
    )
    assert r.probability > 0.0  # 有强证据 (虽被弱的拉低)
    assert r.confidence < 0.7   # 不完整 + 数据差
    assert 0.0 <= r.probability <= 1.0 and 0.0 <= r.confidence <= 1.0


def test_reason_lists_top_evidence() -> None:
    r = aggregate([_ev("top", 1.0, 0.9), _ev("weak", 0.2, 0.3)])
    assert "top" in r.reason
