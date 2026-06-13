# Claude 代码评审待修项

2026-06-12 评审产出。全部 6 项已于同日修复,每项的"状态"行记录落地方式;
确认无回归后本文件可删除。

---

## 1. 混合 lookback 字段推断回退到 1,导致取数天数不足(高优先级)

状态:已修复。lookback 合并改为 max,实现收敛进
`FrameInfo.merged_source_context()` / `merge_source_contexts()`(execution.py)。
修复中发现并一并修掉的根因:column op 和行级 op 的 info 构建丢弃父 frame
其余字段的元数据(source 全部变 None),旧的 metrics 启发式靠"忽略 None-source
字段"掩盖了它;现在两处都保留 `parent.fields`。
测试:`test_merged_source_context_uses_max_lookback`、
`test_mixed_lookback_operands_inherit_max_lookback`。

### 问题

`draco_model/layers/operators.py` 的 `_single_field_lookback`:当一个 frame 内各字段
`lookback_days` 不一致时返回 `1`,而不是 `max`。

复现(已验证):对含 `lookback_days=5` 和 `lookback_days=3` 两个字段的 FrameInfo 调用
`_source_context_from_schema`,返回 `('trades_tbar', 1, ())`。

### 后果链

`Join` 两个不同 lookback 的指标 → 行级 `Col("a") + Col("b")` 派生字段
`lookback_days=1` → 下游 `Grid` 只铺 1 天网格、`FillNull('state')` 只取 1 天 close
历史 → 数据被静默截断,不报错。

### 修复方案

- lookback 合并语义统一为 `max`(它表达"需要多少历史",取小是错的方向)。
- source / grain_path 不一致时回退 `None` / `()` 的现有行为保留。
- 顺带收敛三处重复实现(见第 5 节"source-context 启发式收敛")。

### 验证

- 回归测试:构造 mixed-lookback frame,断言派生字段 `lookback_days == max(...)`。
- 端到端测试:Join(lookback=5, lookback=3) 后接行级 op 再接 Grid,断言网格日期数为 5。

---

## 2. rolling 算子操作数类型校验滞后

状态:已修复。`Op()` 构造期校验 WINDOW_OPS 必须是恰好两个 Node 操作数
(operators.py);README 与 docs/user-guide/rolling.md 同步更新。
测试:`test_window_op_rejects_non_frame_operands`。

### 问题

`draco_model/layers/operators.py` 的 `Op()`:

- `Op("rolling_corr", Col("a"), Col("b"), window=5)` 构造期正常返回 `OpExpr`,
  执行期才在 `_combine_expr` 报 `Unsupported arithmetic operator 'rolling_corr'`
  (报错信息与用户错误无关,已验证)。
- frame 级 rolling op 接 literal 操作数(如 `Op("rolling_beta", node, 2.0, window=5)`)
  能通过 `len(operands) == 2` 检查,对 `pl.lit` 做 rolling_mean,静默产出无意义结果。

### 修复方案

在 `Op()` 构造期校验:WINDOW_OPS 只接受恰好两个 Node(frame 级)操作数,
拒绝 Col / literal 操作数,报错信息直接说明约束。

### 验证

- 测试:上述两种误用在构造期抛 ValueError,信息包含算子名和约束说明。

---

## 3. rolling_corr / rolling_beta 数值稳定性

状态:已修复。guard 改为 `<= 0`,`rolling_corr` 结果 clip 到 `[-1, 1]`
(operators.py `_window_op`);demean 增强未做(需 benchmark,暂缓)。
测试:`test_rolling_corr_guards_float_negative_variance`
(fixture 同时覆盖旧实现的 NaN 和 corr>1 两种失败模式)。

### 问题

`draco_model/layers/operators.py` 的 `_window_op` 用 `E[x²] − E[x]²` 计算方差,
浮点误差可产生微小负值,`x_var.sqrt()` 得 NaN;现有 guard 只挡 `== 0`。
对价格类大数值、小波动序列,灾难性消去风险偏高。

### 修复方案

- 最小修复:guard 从 `== 0` 改为 `<= 0`(或对方差 clip 到 0 再比较)。
- 可选增强:窗口内先 demean 再算协方差/方差,规避大数消去;需 benchmark 确认开销。

### 验证

- 测试:构造均值很大、波动极小的序列,断言 rolling_corr 结果在 [-1, 1] 内且无 NaN
  (除窗口不足的前导 null)。

---

## 4. 跨 grain 误用时报错滞后且难读

状态:已修复。`_require_minute_columns` 校验加在 `_fill_null` / `_fill_null_info`
(仅 `value='state'`)与 `_aggregate` / `_aggregate_info`(仅分钟频率)两侧,
info 推断与执行路径都会触发。
测试:`test_aggregate_minute_frequency_requires_minute_input`、
`test_fillnull_state_requires_minute_input`。

### 问题

- `FillNull('state')` 作用在 daily-grain frame 上:`draco_model/layers/transforms.py`
  无条件 `select(KEY_COLUMNS)`,缺 `minute` 时抛 polars 原生缺列错误。
- `Aggregate("5m", ...)` 作用在 daily frame 上同理。

### 修复方案

两处都已有 FrameInfo 在手,在 info 推断阶段(`_fill_null_info` / `_aggregate_info`)
校验输入 identity 含 `minute`,给出"该层要求 minute 粒度输入"的明确错误。

### 验证

- 测试:daily frame 接 `FillNull('state')` / `Aggregate("5m", ...)`,
  断言在 info 推断阶段抛出含层名与粒度要求的 ValueError。

---

## 5. source-context 启发式收敛(随第 1 项一起做)

状态:已修复。`_source_context_from_schema` / `_common_source_context` /
`_single_field_lookback` / `_common_lookback`(operators.py)与
`_field_source_context`(metrics.py)全部删除,统一指向
`FrameInfo.merged_source_context()` 与 `merge_source_contexts()`。
语义统一为:source/grain_path 归一不了 → `None`/`()`;lookback → max。

### 问题

派生字段继承 (source, lookback_days, grain_path) 的归一逻辑有三份实现,语义已漂移:

- `operators.py` `_source_context_from_schema`(行级 op 用,lookback 不一致回退 1 ← bug)
- `operators.py` `_common_source_context`(frame 级 op 用)
- `metrics.py` `_field_source_context`(metric_reserved 用,lookback 取 max,不处理 grain_path)

### 修复方案

收敛为 `FrameInfo` 上的单一方法(如 `merged_source_context()`),
统一语义:source/grain_path 不一致 → None/();lookback → max。三处调用方改用它。

### 验证

- 现有 66 个测试全过;第 1 项的新增测试覆盖统一后的语义。

---

## 6. 同结构不同名节点静默丢失 name

状态:已修复。`_topological_nodes`(core.py)按对象遍历、按 structural id
去重输出;冲突显式命名抛 ValueError,匿名在前有名在后时保留显式名。
测试:`test_conflicting_names_on_identical_structure_raise`、
`test_duplicate_structure_keeps_explicit_name`。

### 问题

`Node.name` 不参与 structural id(这是对的,name 不应影响缓存命中)。但
`core.py` 的 `_topological_nodes` 按 id 去重时直接跳过第二个同 id 对象,
导致它的 name、以及它子树深处的显式 name 全部被静默吞掉——用户在 trace /
mermaid 里找不到自己命名的步骤,且无任何提示。

### 修复方案

改 `_topological_nodes` 的遍历,name 仍不进 id:

- 按对象(`id(obj)`)遍历、按 structural id 去重输出,即使 structural id 已见过
  也下钻 children,保证重复子树深处的显式 name 能被看到。
- 每个 structural id 维护一个代表节点:
  - 两边都有显式 name 且不同 → 抛 ValueError(静默丢失变响亮报错);
  - 先见匿名、后见有名 → 用有名对象替换代表,显式 name 不因遍历顺序丢失。
- 拓扑顺序不变;`resolve_node_names` 现有的重名兜底保留。

### 验证

- 测试:同结构冲突命名抛 ValueError,信息含两个名字。
- 测试:匿名节点在前、同结构命名节点在后,`Model.nodes()` / trace 仍显示显式名。

---

## 7. 行为级兜底改为显式报错(2026-06-13 决定)

状态:已修复。三处静默兜底全部改为 ValueError:

- `TradingCalendar.from_data_root`:日历文件列名不再回退到第一列,
  必须含 `date` 或 `trading_day`。
- `Engine._infer_info`:op 未注册 info builder 时不再"求值 + collect_schema 反推",
  直接报错要求 `register_info`。
- `SourceCatalog.identity_keys`:推断不出 identity 的 source 不再返回 `()`,
  必须有固定注册键或标准键列(`date`/`secu_code`[/`minute`])。

同批删除的死兜底(`from_columns` 不变量保证不可达):
`aggregate_value_columns` 的全列扫描分支、`FrameInfo.field_for` /
`_renamed_field` / passthrough payload 的裸 FieldInfo 默认值。

测试:`test_trading_calendar_requires_known_date_column`、
`test_infer_info_requires_registered_info_builder`、
`test_unknown_source_without_key_columns_rejects_identity`。
文档:docs/user-guide/data-sources.md 补充 identity 契约与日历列要求。
