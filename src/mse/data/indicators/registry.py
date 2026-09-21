"""指标注册表 — 可插拔指标架构核心 (Phase 1)。

设计目标 (Spec: 未来加 Volume Profile / VWAP 只是注册新计算器, 不改调用方):
    * 每个指标 = 一个纯函数 (OHLCV frame + 参数) → pd.Series。
    * 通过 @registry.register(name) 装饰器注册。
    * pipeline 按需批量计算, 结果填入 IndicatorSet.series。

指标是纯函数、无副作用、point-in-time 安全 (只用当前及历史 bar)。
"""

from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

# 指标计算函数签名: (frame, **params) -> Series (与 frame 索引对齐)。
IndicatorFn = Callable[..., pd.Series]


class Indicator(Protocol):
    """指标计算协议 (函数即可满足)。"""

    def __call__(self, frame: pd.DataFrame, **params: object) -> pd.Series: ...


class IndicatorRegistry:
    """指标注册表。支持装饰器注册与按名计算。"""

    def __init__(self) -> None:
        self._fns: dict[str, IndicatorFn] = {}

    def register(self, name: str) -> Callable[[IndicatorFn], IndicatorFn]:
        """装饰器: 注册一个指标计算函数。"""

        def deco(fn: IndicatorFn) -> IndicatorFn:
            if name in self._fns:
                raise ValueError(f"指标 '{name}' 已注册")
            self._fns[name] = fn
            return fn

        return deco

    def compute(self, name: str, frame: pd.DataFrame, **params: object) -> pd.Series:
        """计算单个指标。"""
        if name not in self._fns:
            raise KeyError(f"未注册指标 '{name}'。已有: {sorted(self._fns)}")
        series = self._fns[name](frame, **params)
        return series

    def available(self) -> list[str]:
        return sorted(self._fns)


# 全局单例 registry (builtin 指标在导入时注册到此)。
registry = IndicatorRegistry()
