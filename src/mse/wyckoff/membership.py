"""模糊隶属函数 (Spec §2.2)。

把原始度量 (刺穿深度、RVOL、幅度/ATR ...) 映射为 0..1 隶属度, 取代 bool 阈值,
避免"阈值悬崖"。全部纯函数、无副作用, 是事件规则的基础构件。

约定: 返回值恒在 [0, 1]。
"""

from __future__ import annotations

import math


def clamp01(x: float) -> float:
    """裁剪到 [0, 1]。"""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def ramp_up(x: float, lo: float, hi: float) -> float:
    """上升斜坡: x≤lo → 0, x≥hi → 1, 中间线性。

    用于"越大越强"的度量 (如放量 RVOL、突破幅度)。
    """
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return clamp01((x - lo) / (hi - lo))


def ramp_down(x: float, lo: float, hi: float) -> float:
    """下降斜坡: x≤lo → 1, x≥hi → 0, 中间线性。

    用于"越小越强"的度量 (如缩量、刺穿越浅越干净)。
    """
    if hi <= lo:
        return 1.0 if x <= lo else 0.0
    return clamp01((hi - x) / (hi - lo))


def sigmoid(x: float, center: float, scale: float) -> float:
    """逻辑斯蒂 sigmoid, 中心 center, 斜率由 scale 控制 (scale 越小越陡)。

    平滑版的"越大越强", 无硬边界。scale>0。
    """
    if scale <= 0:
        return 1.0 if x >= center else 0.0
    z = (x - center) / scale
    # 防溢出
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def band(x: float, lo: float, hi: float, soft: float = 0.0) -> float:
    """带通: x∈[lo,hi] → 1, 两侧按 soft 宽度线性衰减到 0。

    用于"落在合理区间内"的度量 (如 TR 宽度/ATR 落在 [3,15])。soft=0 为硬带通。
    """
    if x < lo:
        return ramp_up(x, lo - soft, lo) if soft > 0 else 0.0
    if x > hi:
        return ramp_down(x, hi, hi + soft) if soft > 0 else 0.0
    return 1.0
