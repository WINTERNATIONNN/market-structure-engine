# Market Structure Engine (MSE)

> 一个模块化、可解释的 **Wyckoff / 市场结构分析引擎**，用于股票筛选。

MSE 把 Wyckoff 量价分析拆解成一条**纯计算**的流水线：从 OHLCV 行情出发，逐层推导出
交易区间（Trading Range）、9 个核心 Wyckoff 事件、以及四阶段（Accumulation / Markup /
Distribution / Markdown）的状态机时间线。每一层的输出都是 **概率 + 置信度 + 可读理由**，
而非简单的布尔判断，方便下游排序、复核与人工审阅。

---

## 设计红线（架构约束）

引擎的可信度来自三条不可逾越的红线，由 `pyproject.toml` 里的 import-linter 契约在 CI 中强制：

1. **引擎层零 LLM**：`mse.data` / `mse.wyckoff` / `mse.scoring` / `mse.scanner` 全是确定性计算，
   禁止 import 任何 LLM / 新闻服务（`mse.services`）。
2. **Agent 层零计算**：数值计算只发生在引擎层，Agent 层只做编排与叙述。
3. **输出非布尔**：事件与阶段一律输出 `probability + confidence + reason`，
   概率与置信度**相互独立**（一个讲"像不像"，一个讲"证据够不够、质量高不高"）。

此外：**禁止魔法数字** —— 所有阈值集中在 [params.py](src/mse/wyckoff/params.py) 的 frozen dataclass 中，
运行时注入，未来可由 yaml 覆盖。

---

## 架构分层

依赖只能自上而下，由 import-linter 的 `layers` 契约保证不反向：

```
mse.agent        ← 编排 / 叙述（无计算）
mse.reporting
mse.ranking
mse.analysis
mse.scanner
mse.scoring
mse.wyckoff      ← Wyckoff 引擎（本仓库当前重点，Phase 2）
mse.data         ← 数据接入 + 指标计算
mse.core         ← 共享词汇：枚举 + pydantic 模型
```

### 目录速览

| 路径 | 职责 |
|------|------|
| [src/mse/core/](src/mse/core/) | 枚举（`Timeframe` / `WyckoffPhase` / `EventType`）与 pydantic v2 frozen 模型（`OHLCVSeries` / `TradingRange` / `WyckoffEvent` / `PhaseResult` / `Evidence`） |
| [src/mse/data/indicators/](src/mse/data/indicators/) | `build_indicator_set`：ATR/RSI/MACD/RVOL/均线 + swing/pivot 检测 |
| [src/mse/data/providers/](src/mse/data/providers/) | 行情源；`YFinanceProvider` 支持日线/周线与多周期拉取 |
| [src/mse/wyckoff/](src/mse/wyckoff/) | 引擎核心：证据聚合、区间检测、事件检测、阶段状态机 |
| [src/mse/scoring/](src/mse/scoring/) | 打分：把阶段/事件聚成 `CandidateProfile`（springboard 吸筹分 + transition 转折分 + composite + 方向 + 可读标签） |
| [src/mse/scanner/](src/mse/scanner/) | 编排：`load_universe` 圈池、`analyze_ticker` 单标的全链路、`scan` 批量扫描并按 composite 降序 |
| [src/mse/analysis/](src/mse/analysis/) | 聚合：`analyze_market` 产出 `MarketBreadth`（阶段占比/多空计数）+ 看涨/看跌/新鲜转折分桶（纯统计，不重打分） |
| [src/mse/ranking/](src/mse/ranking/) | 排名：`rank` 按 `RankingStrategy`（COMPOSITE/SPRINGBOARD/TRANSITION/BULLISH/BEARISH）过滤+稳定排序 |
| [src/mse/reporting/](src/mse/reporting/) | 渲染：`render_table`（终端表）/ `render_markdown_report`（报告）/ `render_json`（date→ISO），纯格式化 |
| [src/mse/agent/](src/mse/agent/) | 叙述（**唯一允许 LLM、零计算**）：`Narrator` 接口 + 离线 `TemplateNarrator` / `AnthropicNarrator`；`explain_candidate` / `explain_market` |

---

## 核心概念

### 证据模型（Spec §2）

每条规则先算出一个 **模糊隶属度**（0..1，见 [membership.py](src/mse/wyckoff/membership.py) 的
`ramp_up` / `ramp_down` / `sigmoid` / `band`），包装成 `Evidence(rule_id, weight, membership, necessary)`，
再由 [evidence.py](src/mse/wyckoff/evidence.py) 的 `aggregate` 汇总：

- **probability** = 加权几何平均：`exp(Σ wᵢ·ln(max(mᵢ, eps)) / Σ wᵢ)`
  → 任何一条权重不低的证据隶属度≈0，都会把整体概率压向 0（几何平均的"近一票否决"特性）。
- **必要条件一票否决**：标了 `necessary=True` 的证据若隶属度 < `necessary_floor`(0.3)，probability 直接归零。
- **confidence** = 完整性 / 一致性 / 数据质量 的加权平均，与 probability 独立。

### 多周期分工（Spec §10.1）

- **周线**：锚定结构 —— 交易区间（`detect_ranges`）与阶段状态机（`detect_phases`）。
- **日线**：检测事件 —— 接收周线的 TR 与阶段语境。

### 9 个核心事件（Spec §4）与镜像关系（Spec §5）

| 事件 | 含义 | 阶段 | 镜像 |
|------|------|------|------|
| PS   | Preliminary Support 初步支撑 | Accumulation 起点 | — |
| SC   | Selling Climax 卖出高潮 | Accumulation | ↔ BC |
| AR   | Automatic Rally 自动反弹 | Accumulation | — |
| ST   | Secondary Test 二次测试 | Accumulation | — |
| Spring | 弹簧 / 假跌破 | Accumulation | ↔ UT |
| SOS  | Sign of Strength 强势信号 | Accum→Markup 过渡 | ↔ (向下跌破) |
| BC   | Buying Climax 买入高潮 | Distribution | ↔ SC |
| UT   | Upthrust 向上假突破 | Distribution | ↔ Spring |
| UTAD | Upthrust After Distribution 派发后假突破 | Distrib→Markdown 过渡 | 强化版 UT |

### 阶段状态机（Spec §3）

`detect_phases` 在周线上走一遍，带**迟滞防抖**（`enter/exit` 阈值 + `min_state_bars` +
连续 `confirm_bars` 确认），转移如下：

```
UNDEFINED/MARKDOWN → ACCUMULATION : 成形 TR + 底部事件(PS/SC/AR) + 前期非上涨
ACCUMULATION       → MARKUP       : SOS 突破 + 上行动量
MARKUP             → DISTRIBUTION : 高位 TR + 顶部事件(BC/UT)
DISTRIBUTION       → MARKDOWN     : 跌破下沿 + 放量 + UTAD 失败
```

---

## 安装

需要 Python ≥ 3.11。

```bash
python -m pip install -e ".[dev]"
```

> **Windows 提示**：终端为非 UTF-8 时，运行脚本请加前缀
> `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`，避免中文/emoji 输出乱码。

---

## 快速上手

完整流水线（真实行情 → 结构 → 事件 → 阶段）：

```python
from datetime import date

from mse.core.enums import Timeframe
from mse.data.indicators import build_indicator_set
from mse.data.providers import YFinanceProvider
from mse.wyckoff import detect_ranges, detect_phases, detect_springs, detect_sos

provider = YFinanceProvider()
weekly = provider.get_ohlcv("AAPL", Timeframe.WEEKLY, start=date(2018, 1, 1), end=date(2024, 1, 1))
daily  = provider.get_ohlcv("AAPL", Timeframe.DAILY,  start=date(2018, 1, 1), end=date(2024, 1, 1))

wind, dind = build_indicator_set(weekly), build_indicator_set(daily)

# 1) 周线结构
ranges = detect_ranges(weekly, wind)

# 2) 日线事件（按 TR 定域）
events = []
for tr in ranges:
    events += detect_springs(daily, dind, tr)
    events += detect_sos(daily, dind, tr)
    # ... detect_phase_a / detect_bc / detect_ut / detect_utad ...

# 3) 周线阶段时间线
phases = detect_phases(weekly, wind, ranges, events)
for p in phases:
    print(p.state_entered_date, p.label.value, f"p={p.probability:.2f} conf={p.confidence:.2f}")
```

---

## 冒烟脚本

`scripts/` 下的脚本用真实/合成数据端到端跑通各阶段，网络不可用时优雅跳过（退出码 0）：

```bash
# Phase 2 全链路：结构 → 事件 → 阶段（可传 TICKER 与起始年份）
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/smoke_phase2_e2e.py AAPL 2018
```

其它：[smoke_indicators.py](scripts/smoke_indicators.py)、[smoke_phase1.py](scripts/smoke_phase1.py)、
[smoke_ranges.py](scripts/smoke_ranges.py)。

### 市场扫描器 CLI（Phase 3–5 全链路）

[scripts/scan_market.py](scripts/scan_market.py) 把单标的引擎升级成全市场发现工具：圈池 →
批量抓周线+日线（本地缓存，二跑秒回）→ 逐只打分 → 排名，并可产出 Markdown 报告 / JSON /
LLM 市场综述。`composite = max(springboard, transition)`，看涨 setup 与近期转折都会浮上榜，
方向标签区分看涨/看跌。

```bash
# 小样快速验证管线
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/scan_market.py --tickers AAPL,MSFT,NVDA,TSLA,AMD --top 5
# 全量 S&P 500 + Markdown 报告 + JSON + LLM 综述（无 ANTHROPIC_API_KEY 时退化为离线模板）
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/scan_market.py --universe sp500 --top 20 \
    --report scan.md --json scan.json --explain
```
1
---

## 测试

全部为**离线**单元测试（合成数据，不依赖网络）：

```bash
python -m pytest            # 全量
python -m pytest tests/wyckoff/test_phase_fsm.py   # 单文件
```

| 测试文件 | 覆盖 |
|----------|------|
| [test_membership.py](tests/wyckoff/test_membership.py) | 模糊隶属度函数 |
| [test_evidence.py](tests/wyckoff/test_evidence.py) | 证据聚合 / 一票否决 |
| [test_range_detection.py](tests/wyckoff/test_range_detection.py) | 交易区间检测 |
| [test_spring.py](tests/wyckoff/test_spring.py) / [test_sos.py](tests/wyckoff/test_sos.py) | Spring / SOS |
| [test_phase_a.py](tests/wyckoff/test_phase_a.py) | SC / AR / ST |
| [test_distribution.py](tests/wyckoff/test_distribution.py) | BC / UT / UTAD |
| [test_ps.py](tests/wyckoff/test_ps.py) | PS |
| [test_phase_fsm.py](tests/wyckoff/test_phase_fsm.py) | 阶段状态机（含指标斜率/量能证据、派发逃生边回归） |
| [tests/scoring/](tests/scoring/) / [tests/scanner/](tests/scanner/) | 打分逻辑（springboard/transition/composite/tag）、单标的管线与批量扫描排序/跳过 |
| [tests/analysis/test_market.py](tests/analysis/test_market.py) | 方向分类、市场广度计数/占比、分桶排序过滤 |
| [tests/ranking/test_strategy.py](tests/ranking/test_strategy.py) | 五种排名策略、门槛过滤、ticker 升序平手 |
| [tests/reporting/test_render.py](tests/reporting/test_render.py) | 终端表 / Markdown 报告 / JSON 往返 |
| [tests/agent/test_narrator.py](tests/agent/test_narrator.py) | prompt 构造、离线 TemplateNarrator、explain_*、缺 key 清晰报错 |

依赖契约校验：

```bash
lint-imports    # 违反红线即失败
```

---

## 现状与已知限制

- **已完成（Phase 2）**：9 个核心事件全部实现并离线测试通过；交易区间检测；阶段状态机
  （已把指标斜率/量能证据接入 FSM）。全链路端到端冒烟在真实 AAPL 数据上跑通。
- **已完成（Phase 3–5）**：`scoring`（`CandidateProfile` 打分）、`scanner`（批量扫描 + 缓存 +
  S&P 500 股票池）、`analysis`（市场广度/分桶聚合）、`ranking`（多策略排名）、`reporting`
  （终端表 / Markdown / JSON）、`agent`（LLM 叙述层）六层全部实现并离线测试通过。全市场扫描器
  CLI [scripts/scan_market.py](scripts/scan_market.py) 打通"圈池→打分→排名→报告→叙述"。
  Agent 层严守红线：**唯一允许 LLM、零计算**，无 key 时退化为离线 `TemplateNarrator`。
- **FSM 派发陷阱已修复**：旧拓扑里 `DISTRIBUTION` 唯一出边是 `→ MARKDOWN`（必要证据=跌破
  下沿），派发区间一旦被**向上**放量突破涨走，既凑不齐跌破证据又无向上逃生边 → 永久卡死
  在 `distribution`。已新增 `DISTRIBUTION → MARKUP`（"派发失败/重归上涨"，必要证据=SOS
  向上突破，复用 `ACCUMULATION→MARKUP` 证据）。全量 S&P 500 扫描显示：修复后 61% 成分股在
  `markup`、37% 在 `distribution`（此前几乎全员冻结在 `distribution`），阶段随突破/破位实时
  切换。见 [tests/wyckoff/test_phase_fsm.py](tests/wyckoff/test_phase_fsm.py)。
- **残留已知限制**：本次只补了向上逃生边，未补对称的"吸筹失败 `ACCUMULATION → MARKDOWN`"
  （`→MARKDOWN` 证据含 UTAD 项，吸筹段无 UTAD 会被几何平均拖至≈0，需另做 context 相关证据）。
  故底部侧（accumulation/markdown）灵敏度仍偏低。参数权重均为"合理起点"，尚待 **Phase 0** 用
  golden dataset 回测标定。
- **陈旧活跃区间（stale active-TR）—— 已修复**：原先当价格单边大涨、远离最近一个已检测交易区间且
  未形成新区间时，`_active_tr` 会锚定旧区间，导致 `price_pos_in_tr` 越界（可 >1，甚至 >10），且该
  区间内检测不到新事件 → `composite=0`、事件清单为空，却仍带高概率的 `markup`/`distribution`
  FSM 标签（如全量扫描中的 INTC/GOOG/MSFT），"阶段标签"与"评分/事件"自相矛盾。根因是斜率门
  （`tr_max_slope=0.15`）在单边上涨中拒绝每个窗口 → 不成新区间 → 回退陈旧旧区间。经 **5 年全量
  S&P 500 × 月度**走查回测对比 5 个方案（`scripts/backtest/`）后，采纳 **V3（放松斜率区间）**：
  将 `RangeParams` 默认改为 `tr_min_bars=11 / tr_width_atr_max=30 / tr_max_slope=0.45`，使上升通道
  也能成真区间。V3 是唯一同时改善**各周期前向命中率**（21/63/126d 均升）与**阶段矛盾率
  0.883→0.575** 的方案。回测报告见 [bt_out/full/report.md](bt_out/full/report.md)。
- **`distribution` 标签缺乏向下预测力 —— 已降级为描述性**：回测显示 confident `distribution`
  （prob≥0.7）之后 63 日下跌率仅 ≈0.40，与全体基础下跌率（2020–2026 牛市里 ≈0.40）几乎无差异
  ——即该标签**无方向性 edge**（对比 `accumulation` 上涨率 0.75、`markdown` 上涨率 0.65 则明显有效）。
  且提高 distribution 置信度也不改善（P(down) 0.378→0.393→0.411）。我们进一步做过 PIT 复核，尝试
  用市场 regime / 广度过滤器抢救其方向性，但 PIT 价格广度（%>SMA200）与未来 63d 下跌率相关
  **r≈−0.02（≈0）**，数据不支持任何过滤器。根因：`distribution` 由"价格高位 + 反复出现的顶部
  事件（BC/UT）"在整段上涨中被持续触发 → 系统性偏早 → 不优于牛市里本就偏低的基础下跌率。
  **修复**：引擎已把 `distribution` 降级为**描述性阶段**——仍作阶段标签检测与显示，但**不再据此
  输出方向性结论**（`transition_direction` 不再因进入 distribution 而置 `bearish`；标签改为
  "顶部结构(派发·描述性)"）。故它不再进入「看跌预警」桶、不被 `BEARISH` 排序策略拾取。只有
  `markdown`（真实下行段）产出看跌方向。见 [score.py](src/mse/scoring/score.py) 的
  `_DIRECTIONAL_BEARISH_PHASES`。（注：`markdown` 在同批回测里下行力亦偏弱，列为待观察项，本次未改。）

---

## 参考

- [PHASE0_WYCKOFF_EVENT_SPEC.md](PHASE0_WYCKOFF_EVENT_SPEC.md) —— 事件/阶段/证据模型的权威规格。
- 许可证：MIT。
