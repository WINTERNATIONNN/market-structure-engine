# Phase 0 — Wyckoff 事件定义 Spec (v0.1)

> 本文件是 **Market Structure Engine 的地基契约**。
> 它定义"什么是一个 Wyckoff 事件"、"用什么证据判定"、"如何输出概率与置信度"。
> Phase 2 (Wyckoff Engine) 的每一行逻辑都必须可追溯到本文件。
> **本文件不含代码,只含规则契约。** 阈值全部外置为参数,不写死。

---

## 0. Spec 的目的与范围

| 项目 | 说明 |
|---|---|
| 目的 | 把 Wyckoff 领域知识固化为**可计算、可回测、可解释**的规则 |
| 范围 | 4 个 Phase + 9 个核心事件 (PS/SC/AR/ST/Spring/SOS/BC/UT/UTAD) |
| 输出契约 | 每个事件输出 `probability / confidence / date / reason / evidence`,**禁止 bool** |
| 不做什么 | 不预测价格、不给买卖信号、不调用 LLM |
| 消费者 | Phase 2 引擎实现、Phase 0 回测校准、Phase 3 打分 |

---

## 1. 通用概念定义 (词汇表)

所有事件建立在以下基础对象之上。这些必须由 Phase 1 (Data Layer) 提供。

### 1.1 Swing / Pivot
- **Swing High (SH)**: 局部高点,满足左右各 `k` 根 bar 的高点均低于它 (`k = swing_lookback`)。
- **Swing Low (SL)**: 对称定义。
- **Pivot**: 显著的 SH/SL,由 ATR 归一化的幅度过滤 (`pivot_min_atr`)。

### 1.2 Trading Range (TR, 交易区间)
一段横盘结构,由以下界定:
- **上沿 (Resistance)**: 区间内 SH 的聚类上界。
- **下沿 (Support)**: 区间内 SL 的聚类下界。
- **成立条件**: 时长 ≥ `tr_min_bars`,且价格振幅 / ATR 落在 `tr_width_atr_range` 内,且未形成持续单边趋势 (趋势斜率 < `tr_max_slope`)。
- **属性**: `{start, end, upper, lower, mid, duration, avg_volume, volume_trend}`。

### 1.3 Volume 基准
- **相对成交量 RVOL** = `volume / rolling_mean(volume, vol_window)`。
- **放量 (high volume)**: `RVOL ≥ vol_high_mult`。
- **缩量 (low volume)**: `RVOL ≤ vol_low_mult`。
- **成交量趋势**: 在 range 内 volume 的回归斜率 (用于 Wyckoff "供给/需求"判断)。

### 1.4 时间锚点
- 所有事件带 `date` = 事件确认发生的那根 bar。
- 严格 **point-in-time**: 判定只能用 `date` 及之前的数据 (防前视偏差)。
- 有些事件需要**后续确认** (如 Spring 需次日反弹),此时 `date` = 刺穿日,`confirmation_date` = 确认日,`probability` 在确认后才升高。

---

## 2. 证据模型 (核心机制)

这是取代 if-else 的关键设计。**事件 = 一组带权证据的聚合**。

### 2.1 三层结构

```
Condition (原子条件, 返回 0..1 的隶属度, 不是 bool)
    │  例: "价格刺穿 TR 下沿的程度" → 归一化为 0..1
    ▼
Evidence (证据 = Condition + 权重 + 命中值)
    │  { rule_id, weight, membership, note }
    ▼
Event (证据聚合 → probability + confidence + reason)
```

### 2.2 为什么 Condition 返回隶属度而非 bool
- 避免阈值"悬崖":价格刺穿下沿 0.1% 和 3% 不该是同样的 True。
- 用 **模糊隶属函数** (fuzzy membership): 例如刺穿深度用 sigmoid/线性斜坡映射到 0..1。
- 这让 `probability` 天然连续,可校准。

### 2.3 probability 与 confidence 的区别 (重要)

| 指标 | 含义 | 计算来源 |
|---|---|---|
| **probability** | "这**是**该事件的可能性有多大" | 各证据隶属度的**加权聚合** |
| **confidence** | "我对这个判断有**多确定**" | 证据的**完整性 / 一致性 / 数据质量** |

- 举例: 一个 Spring 只命中了 3 个证据中的 2 个,且成交量数据缺失 →
  - `probability` 可能 0.65 (命中的证据挺强),
  - 但 `confidence` 只有 0.4 (证据不完整 + 数据缺失)。
- 两者**独立输出**,下游 (Phase 3 打分) 可分别使用。

### 2.4 聚合公式 (可配置,默认加权几何平均)

```
probability = weighted_geometric_mean(memberships, weights)
            = exp( Σ(wᵢ · ln(mᵢ)) / Σwᵢ )
```
- 用几何平均而非算术平均 → **任一关键证据接近 0 会显著拉低总分** (符合"缺一不可"的强规则)。
- 支持 **necessary conditions (必要条件)**: 若必要证据隶属度 < `necessary_floor`,直接判定该事件不成立 (一票否决)。

```
confidence = f(evidence_completeness, evidence_agreement, data_quality)
```
- `evidence_completeness` = 命中证据数 / 期望证据数。
- `evidence_agreement` = 证据方向一致性 (无相互矛盾)。
- `data_quality` = 该窗口数据是否有停牌/缺失/异常。

### 2.5 reason 的生成
- `reason` = 由命中的高权重 evidence 自动拼装的**结构化说明**(非自然语言,自然语言留给 LLM)。
- 格式: `[{rule_id, note, membership, weight}]` + 一句模板化摘要。
- 目的: 让 Phase 5 的 LLM 拿到结构化证据后再生成人话,而不是自己判断。

---

## 3. Phase 检测 (状态机 FSM)

四个阶段用**状态机**判定,而非独立分类。

### 3.1 状态集合
```
{ UNDEFINED, ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN }
```

### 3.2 状态转移 (证据驱动 + 迟滞)

```
UNDEFINED ──(检测到 TR + 前期下跌 + 底部事件 PS/SC/AR)──► ACCUMULATION
ACCUMULATION ──(SOS 突破上沿 + 放量 + 站稳)──► MARKUP
MARKUP ──(检测到高位 TR + 顶部事件 BC/UT)──► DISTRIBUTION
DISTRIBUTION ──(跌破下沿 + 放量 + UTAD 失败)──► MARKDOWN
MARKDOWN ──(检测到底部 TR + PS/SC)──► ACCUMULATION   (循环)
```

### 3.3 防抖动机制 (关键)
- **迟滞 (hysteresis)**: 进入某状态需要证据分 ≥ `enter_threshold`,退出需要 ≤ `exit_threshold` 且 `enter > exit`。
- **最小持续期**: 每个状态至少维持 `min_state_bars` 根 bar 才允许转移。
- **累积证据**: 转移需要连续 `n` 根 bar 的证据支持,单根异动不触发。

### 3.4 输出
```
PhaseResult {
  label: ACCUMULATION,
  probability: 0.78,
  confidence: 0.65,
  reason: [...],
  as_of_date,
  state_entered_date        # 进入该阶段的时间
}
```

---

## 4. 事件规则化定义 (9 个核心事件)

> 每个事件给出: **上下文前提 / 必要证据 / 加权证据 / 后续确认 / 反例(否决条件)**。
> 阈值用参数名表示 (见第 6 节参数表),不写死数值。

### 记号约定
- `TR` = 当前交易区间; `TR.lower` / `TR.upper` = 区间上下沿。
- `[N]` = 必要条件 (necessary); `[W:x]` = 加权证据,权重 x。
- membership 均为 0..1,由对应隶属函数计算。

---

### 4.1 PS — Preliminary Support (初步支撑)
> 下跌趋势中首次出现的显著承接,预示卖压开始被吸收。

- **上下文**: 处于 MARKDOWN / 下跌趋势末段,尚无成形 TR。
- 证据:
  - `[N]` 前期存在明确下跌趋势 (下跌斜率 ≤ `downtrend_slope`)。
  - `[W:0.4]` 出现放量下跌后的显著承接 SL (RVOL ≥ `vol_high_mult`)。
  - `[W:0.3]` 该 SL 处出现长下影线 / 收盘远离最低 (承接痕迹)。
  - `[W:0.3]` 相对前几根 bar 跌速放缓 (动量背离)。
- **否决**: 若之后未再创新低而是直接单边拉升 → 更可能是趋势反转而非 PS。
- **输出**: `date` = 承接 SL 那根 bar。

---

### 4.2 SC — Selling Climax (卖出高潮)
> 恐慌抛售的顶峰,通常伴随极端放量和大幅下影。

- **上下文**: PS 之后 / 下跌加速段。
- 证据:
  - `[N]` 极端放量 (RVOL ≥ `climax_vol_mult`,显著高于 `vol_high_mult`)。
  - `[W:0.4]` 当日振幅极大 (range/ATR ≥ `climax_range_atr`)。
  - `[W:0.4]` 长下影线 + 收盘回升至当日中上部 (`close_position ≥ climax_close_pos`)。
  - `[W:0.2]` 创近期新低后快速收回。
- **后续确认**: SC 之后应出现 AR (见 4.3),二者共同界定 TR 初始边界。
- **否决**: 放量但收盘在最低 (无回升) → 更可能是继续下跌而非高潮。

---

### 4.3 AR — Automatic Rally (自动反弹)
> SC 之后卖压枯竭引发的反弹,定义 TR 上沿。

- **上下文**: 紧随 SC。
- 证据:
  - `[N]` 时间上位于 SC 之后 `ar_max_bars` 根 bar 内。
  - `[W:0.5]` 从 SC 低点显著反弹 (幅度/ATR ≥ `ar_min_rally_atr`)。
  - `[W:0.3]` 反弹高点形成初步 SH → 候选 TR.upper。
  - `[W:0.2]` 反弹时成交量较 SC 回落 (climax 后自然反弹特征)。
- **作用**: AR 高点 + SC 低点 = **TR 初始上下沿**,供 Range Detection 锚定。

---

### 4.4 ST — Secondary Test (二次测试)
> 价格回落测试 SC/支撑区,验证卖压是否真的枯竭。

- **上下文**: AR 之后,TR 已初步成形。
- 证据:
  - `[N]` 价格回落至 TR.lower 附近 (`|price - TR.lower| / ATR ≤ st_zone_atr`)。
  - `[W:0.5]` 测试时**缩量** (RVOL ≤ `vol_low_mult`) → 卖压枯竭的关键信号。
  - `[W:0.3]` 未显著跌破前低 (较 SC 低点抬高或持平)。
  - `[W:0.2]` 测试后收回区间内。
- **多次 ST**: 允许输出多个 ST 事件,后续 ST 缩量越明显 → Accumulation 概率越高。
- **否决**: 测试时放量创新低 → 供给仍在,可能非有效 ST。

---

### 4.5 Spring — (弹簧 / 假跌破)
> Accumulation 末期,价格短暂刺穿 TR.lower 诱空后快速收回。最重要的进场结构之一。

- **上下文**: ACCUMULATION 中后段,TR 成熟 (duration ≥ `tr_min_bars`)。
- 证据:
  - `[N]` 价格盘中刺穿 TR.lower (`penetration_depth > 0`),映射为隶属度 (刺穿越浅越"干净")。
  - `[W:0.35]` **快速收回** TR 内 (收盘 ≥ TR.lower,收回速度 ≤ `spring_recover_bars`)。
  - `[W:0.25]` 刺穿时**缩量**或收回时放量 (供给枯竭 / 需求进场)。
  - `[W:0.20]` 刺穿深度浅 (`penetration / ATR ≤ spring_max_depth_atr`) → Spring #1;深刺穿为不同亚型。
  - `[W:0.20]` 刺穿低点未显著低于此前 ST 低点太多。
- **后续确认**: `confirmation_date` = 收回后出现放量上涨 bar; 确认前 `probability` 打折 (`× spring_unconfirmed_factor`)。
- **否决**: 刺穿后**未收回、持续走低放量** → 是真跌破 (Markdown),Spring 不成立。
- **亚型标注**: `{terminal_shakeout, spring_type_1/2/3}` 记入 meta,供回测细分。

---

### 4.6 SOS — Sign of Strength (强势信号)
> 需求主导的证据,通常是放量突破 TR.upper,标志 Accumulation → Markup。

- **上下文**: ACCUMULATION 后段 (常在 Spring / LPS 之后)。
- 证据:
  - `[N]` 价格突破 TR.upper (`close > TR.upper`)。
  - `[W:0.4]` 突破时**放量** (RVOL ≥ `sos_vol_mult`) → 真突破关键。
  - `[W:0.3]` 突破幅度显著 (`(close-TR.upper)/ATR ≥ sos_min_break_atr`)。
  - `[W:0.3]` 突破后回踩不跌回 TR 内 (形成 LPS, back-up)。
- **后续确认**: 回踩确认 (Last Point of Support) 后 `confidence` 升高。
- **否决**: 突破缩量且迅速跌回 → 假突破 (upthrust,见 UT)。

---

### 4.7 BC — Buying Climax (买入高潮)
> Distribution 顶部,极端放量上涨后需求耗尽 (SC 的镜像)。

- **上下文**: MARKUP 末段 / 高位。
- 证据:
  - `[N]` 极端放量 (RVOL ≥ `climax_vol_mult`)。
  - `[W:0.4]` 大幅上涨后当日振幅极大 (range/ATR ≥ `climax_range_atr`)。
  - `[W:0.4]` 长上影线 / 收盘远离最高 (`close_position ≤ 1 - climax_close_pos`) → 抛压出现。
  - `[W:0.2]` 创新高后快速回落。
- **后续**: BC 之后常伴 AR (向下的自动反应) → 界定高位 TR。
- **否决**: 放量新高且强力收在最高 → 可能仍是 Markup 延续。

---

### 4.8 UT — Upthrust (向上假突破)
> Distribution 中价格短暂突破 TR.upper 诱多后回落 (Spring 的镜像)。

- **上下文**: DISTRIBUTION 中段,高位 TR 成熟。
- 证据:
  - `[N]` 价格刺穿 TR.upper (`penetration_up > 0`)。
  - `[W:0.35]` **快速回落** TR 内 (收盘 ≤ TR.upper)。
  - `[W:0.25]` 突破时放量但无法守住 / 回落放量 (需求虚假)。
  - `[W:0.20]` 突破后收盘转弱 (收在当日下部)。
  - `[W:0.20]` 未能创造有效的持续新高。
- **否决**: 突破后放量站稳并延续 → 是 SOS 而非 UT。

---

### 4.9 UTAD — Upthrust After Distribution (派发后向上假突破)
> Distribution 末期的终极诱多,UT 的强化版,标志 Distribution → Markdown。

- **上下文**: DISTRIBUTION 后段 (TR 已成熟,已出现 BC/UT/ST 等)。
- 证据:
  - `[N]` 价格刺穿 TR.upper 并短暂创出区间新高。
  - `[W:0.35]` **快速反转跌回** TR 内且跌势延续 (`reversal within utad_recover_bars`)。
  - `[W:0.30]` 反转时放量 (派发痕迹)。
  - `[W:0.20]` 此前区间已有明确 Distribution 结构 (BC + UT 前置存在)。
  - `[W:0.15]` 跌回后跌破 TR.mid / 逼近 TR.lower。
- **后续确认**: 后续跌破 TR.lower 放量 → 确认进入 Markdown。
- **否决**: 突破后持续放量走高 → 非 UTAD (可能是二次 Markup)。

---

## 5. 事件间关系矩阵 (供 FSM 与回测使用)

| 事件 | 所属 Phase | 典型前置 | 典型后继 | 镜像事件 |
|---|---|---|---|---|
| PS | Accumulation 起 | 下跌趋势 | SC | (BC) |
| SC | Accumulation | PS | AR | BC |
| AR | Accumulation | SC | ST | AR(down) |
| ST | Accumulation | AR | Spring / SOS | ST(up) |
| Spring | Accumulation 末 | ST | SOS | UT |
| SOS | Accum→Markup | Spring/LPS | Markup | (跌破 down) |
| BC | Distribution 起 | Markup | AR(down)/UT | SC |
| UT | Distribution | BC | UTAD | Spring |
| UTAD | Distrib→Markdown | UT/ST | Markdown | Spring |

> **约束**: 事件识别应参考所属 Phase (来自 FSM),避免"在 Markup 里报 Spring"这类语境错误。事件与 Phase **互相约束但不循环依赖**: Phase 用底层事件初判,事件用 Phase 语境加权。

---

## 6. 市场参数表 (外置,不写死)

> 同一套事件定义,换参数表即可适配不同市场。默认给出通用起点,由 Phase 0 回测标定。

### 6.1 结构参数 (市场无关,默认)
| 参数 | 默认 | 说明 |
|---|---|---|
| `swing_lookback` | 5 | swing 左右比较窗口 |
| `pivot_min_atr` | 1.0 | pivot 最小幅度 (×ATR) |
| `tr_min_bars` | 15 | TR 最小时长 |
| `tr_width_atr_range` | [3, 15] | TR 合理宽度区间 (×ATR) |
| `tr_max_slope` | 0.15 | TR 内最大趋势斜率 |
| `vol_window` | 20 | RVOL 均值窗口 |
| `min_state_bars` | 10 | FSM 最小状态持续 |

### 6.2 成交量 / 事件参数 (默认)
| 参数 | 默认 | 说明 |
|---|---|---|
| `vol_high_mult` | 1.5 | 放量阈值 (RVOL) |
| `vol_low_mult` | 0.7 | 缩量阈值 (RVOL) |
| `climax_vol_mult` | 2.5 | 高潮极端放量 |
| `climax_range_atr` | 2.0 | 高潮日振幅 (×ATR) |
| `climax_close_pos` | 0.6 | SC 收盘位置下限 (0=最低,1=最高) |
| `spring_max_depth_atr` | 0.75 | Spring 干净刺穿深度上限 |
| `spring_recover_bars` | 2 | Spring 收回时限 |
| `spring_unconfirmed_factor` | 0.7 | 未确认 Spring 的概率折扣 |
| `sos_vol_mult` | 1.8 | SOS 突破放量 |
| `sos_min_break_atr` | 0.5 | SOS 突破幅度 |

### 6.3 市场特定覆盖 (示例)
| 市场 | 覆盖项 |
|---|---|
| **A股** | 加 `price_limit = 0.10` (涨跌停约束: 涨跌停日成交量/振幅需特殊处理);交易日历;T+1 |
| **美股** | 无涨跌停;盘前盘后成交量单独处理;split/dividend 复权 |
| **港股** | 无涨跌停;仙股流动性过滤 (`min_liquidity`) |

> **实现约束**: 参数表以配置文件 (yaml) 形式存在,归 Phase 0 管理,引擎运行时注入。禁止在事件逻辑中出现魔法数字。

---

## 7. 标签体系 (Labeling Schema, 供回测与校准)

为验证引擎正确性,需要 **golden dataset** — 人工标注的标准案例。

### 7.1 标签结构
```
Label {
  ticker,
  date,                 # 事件发生日
  event_type,           # PS/SC/.../UTAD, 或 phase 标签
  verdict,              # confirmed / rejected / ambiguous
  annotator,            # 标注人 / 来源
  quality,              # 该案例的教科书程度 1..5
  notes,
  source                # 手工 / 半自动 / 文献案例
}
```

### 7.2 用途
- **概率校准**: 把引擎输出的 `probability` 分桶,对比该桶内 golden label 的真实命中率 → 画 calibration curve。目标: 输出 0.7 的事件,实际约 70% 是真事件。
- **规则调参**: 权重与阈值通过在 golden set 上的表现来标定,而非拍脑袋。
- **回归测试**: 规则升级后,在 golden set 上跑一遍,防止退化。

---

## 8. 回测与校准口径 (防造假)

| 口径 | 要求 |
|---|---|
| **Point-in-time** | 事件判定只用当日及之前数据; 需确认的事件明确区分 `date` 与 `confirmation_date` |
| **样本外验证** | golden set 分 train/test; 参数只在 train 标定,test 只评估 |
| **Walk-forward** | 按时间滚动验证,模拟真实上线场景 |
| **版本化** | 每次规则/参数变更产生新 `ruleset_version`; 所有产出带版本,可对比 |
| **校准指标** | Brier score / calibration error / 分事件的 precision-recall |
| **数据质量 gate** | 停牌、缺失、异常 bar 需在 Data Layer 标记; 低质量窗口降低 `confidence` |

---

## 9. 事件输出契约 (最终 schema,所有事件统一)

```
WyckoffEvent {
  event_type:   EventType          # 枚举 PS/SC/AR/ST/Spring/SOS/BC/UT/UTAD
  ticker:       str
  date:         date               # 事件确认发生日
  confirmation_date: date | null   # 需后续确认的事件才有
  probability:  float  # 0..1      # "是该事件"的可能性
  confidence:   float  # 0..1      # 对判断的确定程度
  phase_context: PhaseType         # 事件发生时的 FSM 阶段
  tr_ref:       TR | null          # 关联的交易区间
  reason:       str                # 模板化摘要 (给人看)
  evidence: [                      # 结构化证据 (给下游/LLM)
    { rule_id, weight, membership, note }
  ]
  meta:         dict               # 亚型/深度/等细节
  engine_version: str              # 可复现
}
```

> **红线复述**: 本 schema **没有 bool 字段表示"是不是"**。是与不是,由 `probability` + `confidence` + 下游阈值共同决定。引擎只负责给证据和概率,不替下游拍板。

---

## 10. 已确认决策 (v0.2 锁定)

| # | 项目 | 决策 | 对实现的约束 |
|---|---|---|---|
| 1 | **市场** | **美股** | 无涨跌停 (删除 A股 `price_limit`);处理 split/dividend 复权;盘前盘后成交量单独处理 |
| 2 | **数据源** | **yfinance** | 首个 Provider = `YFinanceProvider`;注意其 `auto_adjust` 返回已复权数据,需固定口径 |
| 3 | **周期** | **Multi-timeframe** | **周线定 Phase/大结构,日线定事件**;TR 在周线锚定,Spring/SOS 等在日线检测 |
| 4 | **Golden dataset** | **文献案例起步** | 先收集 20–30 个教科书级公认案例 (高 quality),手工录入;后续逐步补充 |
| 5 | **首个落地事件** | **Accumulation 主线** | SC → AR → ST → Spring → SOS,跑通端到端 + 校准后再扩展顶部事件 |

### 10.1 Multi-timeframe 的分工契约 (新增,因周期决策)

```
周线 (Weekly)  →  Phase Detection (FSM):  Accumulation / Markup / Distribution / Markdown
              →  Trading Range 锚定 (TR.upper / TR.lower 用周线 swing)

日线 (Daily)   →  Event Detection:  在周线给定的 Phase + TR 语境下,检测具体事件
              →  Spring / SOS / SC / AR / ST 等的精确 date 与证据
```

- **约束**: 日线事件检测**必须**接收周线传入的 `phase_context` 与 `tr_ref`;不允许日线独立臆断大结构。
- **对齐**: 周线 bar 的归属需明确 (周五收盘为准);日线事件的 `date` 落在对应周线区间内。
- **数据模型影响**: `IndicatorSet` 与 `TR` 需带 `timeframe` 字段;引擎接口需支持双周期输入。

---

## 11. 下一步实现路径 (已就绪)

1. **Phase 1 — Data Model + YFinanceProvider**
   - 定义 `Symbol / Bar / IndicatorSet` (带 `timeframe`);
   - `YFinanceProvider` 拉取日线 + 周线,统一复权口径;
   - 指标 registry (MA/EMA/ATR/RSI/MACD/Swing/Pivot/RVOL)。
2. **Phase 0 运行时 — Golden dataset 骨架**
   - `Label` 表 + 录入工具;先填 3–5 个文献 Spring/SOS 案例作冒烟测试。
3. **Phase 2 — Accumulation 主线**
   - Range Detection (周线) → Phase FSM → 日线事件 (SC/AR/ST/Spring/SOS);
   - 输出 §9 契约;在 golden 案例上校准。

---

*版本: v0.2 — 决策已锁定 (美股 / yfinance / multi-timeframe / 文献案例 / Accumulation 主线)。下一步进入 Phase 1 代码实现。*
