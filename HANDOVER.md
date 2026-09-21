# HANDOVER — Market Structure Engine

> 交接文档。记录当前进度、已验证状态、下一步。
> 最后更新: 2026-08-05 (下班交接)

---

## 一句话状态

**Phase 0 (事件 Spec) + Phase 1 (Data Layer) 已完成并在真实美股数据上验证通过。下一步进入 Phase 2 (Wyckoff Engine),建议从 Range Detection 切入。**

---

## 已锁定的核心决策 (不要轻易改)

| # | 项目 | 决策 |
|---|---|---|
| 1 | 市场 | **美股** (无涨跌停, 处理 split/dividend 复权) |
| 2 | 数据源 | **yfinance** (auto_adjust=True, 复权口径固定) |
| 3 | 周期 | **Multi-timeframe**: 周线定 Phase/大结构, 日线定事件 |
| 4 | Golden dataset | **文献案例起步** (20–30 教科书案例, 尚未开始录入) |
| 5 | 首个事件 | **Accumulation 主线**: SC → AR → ST → Spring → SOS |
| 6 | 依赖管理 | pyproject.toml (已 `pip install -e .`) |
| 7 | 数据模型 | pydantic v2 |

**三条架构红线** (贯穿全系统):
1. 计算与推理分离 — Engine 层零 LLM 调用。
2. 扫描与调度分离 — Agent 层零计算。
3. 事件输出 `probability + confidence + reason`, **禁止 bool**。

---

## 环境

- Python 3.11.9 (Windows, WindowsApps python)
- 已装: pandas 2.1.2, numpy 1.26.4, pydantic 2.10.6, yfinance 1.5.2
- **Windows 控制台跑脚本需加编码前缀**: `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python ...`
  (否则 cp1252 无法打印中文, 会 UnicodeEncodeError)
- Shell 是 Git Bash (POSIX)

---

## 已完成 (已运行验证 ✅)

### Phase 0 — 事件定义 Spec
- 文件: `PHASE0_WYCKOFF_EVENT_SPEC.md` (v0.2, 决策已锁定)
- 内容: 证据模型 (Condition→Evidence→Event)、probability vs confidence 区分、
  FSM 四阶段 + 防抖动、9 事件规则化定义、外置参数表、标签体系、回测口径、输出契约。

### Phase 1 — Data Layer
```
src/mse/
├── __init__.py                          分层说明
├── core/
│   ├── enums.py                         Timeframe/WyckoffPhase/EventType/...
│   └── models/
│       ├── market.py                    Symbol / Bar / OHLCVSeries (带 point-in-time slice)
│       └── indicators.py                Swing / Pivot / IndicatorSet
└── data/
    ├── providers/
    │   ├── base.py                      DataProvider 协议 + ProviderError
    │   └── yfinance_provider.py         YFinanceProvider (多周期, 复权规范化)
    └── indicators/
        ├── registry.py                  IndicatorRegistry (可插拔)
        ├── builtin.py                   sma/ema/atr/rsi/macd/rvol (纯函数, PIT 安全)
        ├── swings.py                    detect_swings / detect_pivots
        └── pipeline.py                  build_indicator_set (配置驱动)
```

**验证**: 两个冒烟脚本都通过 (真实 AAPL 数据):
- `scripts/smoke_phase1.py` — 拉日线+周线, point-in-time 切片, 元数据
- `scripts/smoke_indicators.py` — 全套指标 + swings/pivots, 数值合理

重跑命令:
```bash
cd "c:/Users/I741185/Documents/vscode_playground"
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/smoke_indicators.py
```

**关键实现决策**:
- 指标 = 纯函数 + `@registry.register` 注册 → 加新指标不改调用方。
- RVOL 用 `shift(1)` 历史均值 → 严格 point-in-time, 当日不参与自身基准。
- 分层依赖已在 pyproject.toml 用 import-linter 锁死 (data/wyckoff/scoring 禁 import services)。
- yfinance `info` 返回的是**实时** market_cap, 非历史 → Scanner 阶段要注意。

---

## 下一步 — Phase 2 Wyckoff Engine (未开始)

三层管线:
```
1. Range Detection (周线)  → 交易区间 TR {upper, lower, mid, duration, ...}
2. Phase FSM (周线)         → Accumulation/Markup/Distribution/Markdown, 带迟滞防抖
3. Event Detection (日线)   → 证据模型 + 规则引擎: SC→AR→ST→Spring→SOS
```

**未决问题 (下次开工第一件事)**: 从哪层切入?
- **建议 B → A**: 先做 Range Detection (所有事件都依赖 TR 边界), 再做证据模型/规则引擎框架, 然后填第一个事件 (Spring)。
- 理由: 证据模型里几乎每个 Condition 都引用 TR.upper/lower, 先有 TR 才能写真实规则验证框架。

参考 Spec 章节: §1.2 (TR 定义)、§2 (证据模型)、§3 (FSM)、§4.5 (Spring)、§6 (参数表)。

---

## 待办清单 (中期)

- [ ] Phase 2: Range Detection → Phase FSM → 事件检测
- [ ] Phase 0 运行时: Golden dataset 表 + 录入工具 + 前 3–5 个文献案例
- [ ] 补单元测试 (mock yfinance, 不联网), 目前只有联网冒烟脚本
- [ ] 建 README.md (pyproject 引用了但还没建)
- [ ] 之后: Phase 3 评分 → 4 扫描 → 5 二层 → 6 排序 → 7 报告 → 8 Agent

---

## 给接手者的提示

- 整体架构蓝图在对话里 (8 个 Phase), 已浓缩进 Spec 和本文档。
- 严格按 Phase 依赖顺序推进, 先把契约夯实再写实现。
- 每步都跑冒烟验证, 不堆代码。
