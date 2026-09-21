"""走查回测测量框架 (walk-forward backtest harness)。

对「陈旧活跃区间 (stale active-TR)」盲区的多个修复方案做 5 年全市场回测,
并排对比三项准确率指标 (前向收益命中率 / 阶段一致性 / 事件事后精度)。

红线: 本框架**不改动引擎**, 通过策略适配器 (注入 RangeParams + active-TR 选择器 +
可选后处理) 复用同一批底层检测器; 引擎保持 pristine, 选出优胜方案后再另起改动。
所有阈值集中在 BacktestParams (frozen dataclass), 无魔法数字。
"""

from __future__ import annotations
