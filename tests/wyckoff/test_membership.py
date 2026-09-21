"""模糊隶属函数单元测试 (Spec §2.2)。纯函数, 完全离线。"""

from __future__ import annotations

import pytest

from mse.wyckoff.membership import band, clamp01, ramp_down, ramp_up, sigmoid


def test_clamp01() -> None:
    assert clamp01(-0.5) == 0.0
    assert clamp01(0.3) == 0.3
    assert clamp01(1.7) == 1.0


def test_ramp_up() -> None:
    assert ramp_up(0.0, 1.0, 3.0) == 0.0     # 低于 lo
    assert ramp_up(3.0, 1.0, 3.0) == 1.0     # 高于 hi
    assert ramp_up(2.0, 1.0, 3.0) == pytest.approx(0.5)
    # 退化: hi<=lo
    assert ramp_up(5.0, 3.0, 3.0) == 1.0


def test_ramp_down() -> None:
    assert ramp_down(0.5, 1.0, 3.0) == 1.0   # 低于 lo → 满隶属
    assert ramp_down(3.0, 1.0, 3.0) == 0.0   # 高于 hi
    assert ramp_down(2.0, 1.0, 3.0) == pytest.approx(0.5)


def test_sigmoid_monotonic_and_centered() -> None:
    assert sigmoid(0.0, 0.0, 1.0) == pytest.approx(0.5)
    lo, mid, hi = sigmoid(-5, 0, 1), sigmoid(0, 0, 1), sigmoid(5, 0, 1)
    assert lo < mid < hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    # 极端值不溢出
    assert sigmoid(1e6, 0, 1) == pytest.approx(1.0)
    assert sigmoid(-1e6, 0, 1) == pytest.approx(0.0)


def test_band() -> None:
    assert band(5.0, 3.0, 15.0) == 1.0       # 区间内
    assert band(2.0, 3.0, 15.0) == 0.0       # 区间外, 硬边界
    assert band(20.0, 3.0, 15.0) == 0.0
    # 软边界: 落在过渡带中点
    assert band(2.0, 3.0, 15.0, soft=2.0) == pytest.approx(0.5)
    assert band(16.0, 3.0, 15.0, soft=2.0) == pytest.approx(0.5)
