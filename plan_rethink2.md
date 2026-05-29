# Draco Model 架构级重构候选

收录哲学上一致、但落地成本高、影响面大的**架构级**改造想法。只有当 plan.md / plan_rethink.md 的小项都跑通、且工程价值得到验证后，才考虑动这类项。

---

## #7A — `_close_state_frame` 改为显式 Node 子树（已落地）

### 背景

`Fill("state")` 当前的实现里，close_state 是一段**手写重放**：`_close_state_frame`（`layers/transforms.py:308-334`）遍历主链路上记录下来的 `lineage.transforms`，自己调 `_auction_frame` / `_resample_frame` 把 close field 一步步算出来，最后 forward_fill + daily preclose 兜底。

这段重放跟主图的 transform 执行**逻辑等价但代码分离**——主图走 executor 体系，close_state 走自己的命令式调用。两边一旦 dispatch 规则不一致就会出 bug，#16（Auction merge 时 close 应该用 `last` 而不是原始 agg）就是这类 bug 的典型表现。

### 哲学定位

这是 [[#17]] 思路在**单一字段（close）级别**的预演：把"跟着 frame 流动的隐式状态"重写为**显式 Node 子树**，让 close_state 跟主图共享同一套 executor、同一套 schema 推断、同一套缓存命中。

| 层级 | 显式化对象 | 对应 plan |
|---|---|---|
| close_state（单字段子树） | `_close_state_frame` 内部隐式构造的 LazyFrame | **#7A（本项）** |
| ratio payload（双字段子树） | `__ratio_*_num` / `__ratio_*_den` 列 | [[#17]] |

#7A 工程量小（不动 Layer 用户 API）、回报立竿见影（CSE 命中 + bug 面收窄）、且是 #17 的可行性验证——证明"把隐式状态搬成 Node 子树"在 Draco Model 现有 executor / schema / 缓存体系里跑得通。

### 现状

```python
# layers/transforms.py:308-334
def _close_state_frame(lineage: _StateFillLineage, context: EvalContext) -> pl.LazyFrame:
    dates = context.trading_calendar.previous_sessions(context.eval_date, lineage.lookback_days)
    raw = context.sources.scan(lineage.source, dates)
    close = get_field_builder("close")(raw, raw.collect_schema().names())
    close_columns = [*KEY_COLUMNS, "close"]
    for transform in lineage.transforms:
        if transform.op == "auction":
            mode = str(transform.params["mode"])
            close = _auction_frame(close, mode, "last" if mode == "merge" else None, close_columns)
        elif transform.op == "resample":
            close = _resample_frame(close, str(transform.params["frequency"]), "last", context, close_columns)
        else:
            raise ValueError(f"Fill('state') cannot replay transform {transform.op!r}.")

    daily_preclose = _daily_preclose_frame(context, dates)
    return (
        close.join(daily_preclose, on=list(DAILY_KEY_COLUMNS), how="left")
        .sort(list(KEY_COLUMNS))
        .with_columns(
            pl.col("close").forward_fill().over(list(DAILY_KEY_COLUMNS))
            .fill_null(pl.col("__daily_preclose"))
            .alias("__close_state")
        )
        .select([*KEY_COLUMNS, "__close_state", "__daily_preclose"])
    )
```

**症状**：
1. 重放代码跟主图 executor **平行存在**，dispatch 表两份。新增 transform op（比如未来的 `Rolling` / `Shift`）必须同时在两边加分支。
2. 主图如果已经独立构造了等价的 `Field("close")(Input(source=X, lookback_days=N))` 子链（比如 `Fill("state")(...) + DailyAgg(close)` 并存于一个 model），close_state 还是**从头算一遍**——`Engine._memory` 帮不上忙，因为这段 LazyFrame 根本没经过 executor 体系。
3. `_auction_frame` / `_resample_frame` 这两个 internal helper 被**两类调用者**绑死：自己 executor + close_state 重放。改任何一个 helper 都要回归两条路径。

### 设计

`Fill("state")` executor 在入口**构造一棵等价的 close_state Node 子树**，然后 `context.evaluate(close_state_node)` 拿 LazyFrame，剩下的 forward_fill + daily preclose fallback 不变。

#### 改造前后对比

**改造前**：
```python
@register_executor("fill")
def _fill_executor(node, context):
    ...
    lineage = _parse_state_fill_lineage(parent)
    state = _close_state_frame(lineage, context)   # 手写重放
    ...
```

**改造后**：
```python
@register_executor("fill")
def _fill_executor(node, context):
    ...
    lineage = _parse_state_fill_lineage(parent)
    close_node = _build_close_state_subtree(lineage)   # 构造 Node 子树
    close_frame = context.evaluate(close_node)         # 走主 executor 体系
    state = _ffill_and_preclose(close_frame, lineage, context)
    ...


def _build_close_state_subtree(lineage: _StateFillLineage) -> Node:
    """Build a Node sub-DAG that computes close with last-agg semantics."""
    node = Input(source=lineage.source, lookback_days=lineage.lookback_days)
    node = Field("close")(node)
    for transform in lineage.transforms:
        if transform.op == "auction":
            mode = str(transform.params["mode"])
            agg = "last" if mode == "merge" else None
            node = Auction(mode, agg)(node)
        elif transform.op == "resample":
            node = Resample(str(transform.params["frequency"]), "last")(node)
        else:
            raise ValueError(f"Fill('state') cannot replay transform {transform.op!r}.")
    return node
```

关键变化：
- **dispatch 规则集中在 `_build_close_state_subtree` 一处**——而且这一处只做 Layer 构造，不做 frame 计算
- close_state 子树的每个 Node 都有结构性 id（[[#8]] 已落地）→ 跟主图等价子链**自动 CSE**
- `_auction_frame` / `_resample_frame` 这些 helper 只服务于 executor，不再被外部调用

### 待解决的设计问题

#### 1. preclose fallback 放在哪

当前 `_close_state_frame` 末尾的 `daily_k.preclose` join + forward_fill + fill_null 不是纯 Layer 能表达的（没有 "join 外部 daily 表" 的通用 layer）。两种放法：

| 方案 | 描述 | 优劣 |
|---|---|---|
| **A. 留在 Fill executor 内** | close_state 子树只算到「close + transforms」，executor 拿到 LazyFrame 后**自己**做 ffill + preclose fallback | ✅ Node 体系不需要为 close_state 专门开新 op；❌ Fill executor 自己仍持有一段非 Layer 逻辑 |
| **B. 新增 op="close_state" executor** | 把 ffill + preclose fallback 也包装成 Node，`_build_close_state_subtree` 末尾再套一层 `CloseState()(...)` | ✅ Fill executor 彻底干净；❌ 多出一个仅服务于 Fill 的 op，CSE 收益不明显（一个 model 通常只有一个 close_state 调用点） |

**倾向 A**。preclose fallback 是 Fill executor 的**职责**，不是独立的"close 状态"概念。子树只负责"close 经过等价 transforms 后是什么"，fallback 在 executor 内部完成。

#### 2. transform 克隆规则集中在哪

`_build_close_state_subtree` 里的 "Auction merge 永远用 last / Resample 永远用 last / drop 不传 agg" 规则——

| 方案 | 描述 |
|---|---|
| **A. inline 在 builder 里** | 就写在 `_build_close_state_subtree` 的 if/elif 分支里。规则跟主图 Layer 同步更新——主图加新 transform op，这里也加 |
| **B. 抽成 Layer-level helper** | 给每个 transform Layer 定义一个 `for_close_state(self) -> Layer` 类方法，返回等价的 last-agg 版本 |

倾向 **A**。规则集中在一处（builder 内部）反而比分散到各 Layer 更易读。Layer 类的 OO 抽象目前是「Node 工厂」，再塞 `for_close_state` 会模糊职责。

#### 3. RatioField 路径

当前 `_parse_state_fill_lineage` 在 `ratio_field` op 上特判，把 `output_column = alias`。改造后：
- **#17 未落地时**：`_build_close_state_subtree` 起点仍是 `Field("close")`，跟 lineage.field 是 "close" 还是 "vwap" 无关——close_state 永远算的是 close，不算 ratio 本身。这跟当前行为一致
- **[[#17]] 落地后**：ratio_combine 子树下两条 num/den 分支，`Fill("state")` 作用在 ratio_combine 上时仍然按选项 B（plan_rethink2 #17 第 6 点的结论）填 ratio output 列，close_state 子树构造方式不变

所以 #7A 落地不需要预判 #17 的语义。

#### 4. CSE 命中的可观察性

证明 #7A 真的工作 = 证明"主图 close 子链跟 close_state 子树共享一个 `_memory` 槽"。验证手段：
- 构造一个 `Model` 同时含 `Fill("state")(Auction("merge","sum")(Field("amount")(raw)))` 和 `DailyAgg("close","last")(Auction("merge","last")(Field("close")(raw)))`
- 跑 `engine.evaluate(...)` 后，inspect `engine._memory`，断言两个等价的 close + Auction("merge","last") 节点共享同一 key

### 影响范围

- `draco_model/layers/transforms.py`
  - 删除 `_close_state_frame`
  - 新增 `_build_close_state_subtree(lineage) -> Node`
  - 新增 `_ffill_and_preclose(close_frame, lineage, context) -> pl.LazyFrame`（原 `_close_state_frame` 末尾那段）
  - `_fill_executor` state 分支改成「构造子树 + context.evaluate + ffill_and_preclose」
  - `_parse_state_fill_lineage` 不变
  - `_auction_frame` / `_resample_frame` 的 `columns` 参数现在只有 executor 调用，签名可以简化（保留向后兼容也行，影响很小）
- `draco_model/layers/inputs/field.py` — 不变；`_close_field` 已经是合法 Field builder
- `draco_model/layers/transforms.py` 顶部 import — 不再需要 `_daily_preclose_frame` 在 hot path 之外被调用，保持当前位置即可
- 测试
  - 现有 `test_fill_state_replays_auction_merge_with_close_last` / `test_high_fill_state_uses_matching_close_state_after_transforms` 这些 happy-path 保留
  - 新增 CSE 命中验证：双子链共享 `_memory` 槽
  - 新增等价性回归：随机构造若干 transform 链，跑改造前/后的结果对比（一次性比对脚本，不进 CI）

### 落地前置条件

1. [[#8]]（结构性 node id）✓ 已落地——CSE 命中的前提
2. [[#16]]（close_state Auction agg dispatch 规则）✓ 已落地——`_build_close_state_subtree` 直接编进的规则就是 #16 的结论
3. 决定 preclose fallback 放法（上面第 1 点）
4. 决定 transform 克隆规则放法（上面第 2 点）

### 关联项

- **依赖 [[#8]]**（已落地）
- **建议在 [[#16]] 之后**（已落地）——#16 的 dispatch 规则直接成为 #7A builder 的 if/elif 逻辑
- **解锁 [[#17]]**——#17 的"payload 拆 num/den 子树"是 #7A 的最大范围应用；先用 #7A 在单字段（close）上验证「Node 子树 + context.evaluate + CSE」工程可行，再考虑 #17

### 落地结论

- 已选择 preclose fallback 方案 A：close 子树只负责计算 transformed close，`forward_fill + daily_k.preclose` 仍留在 `Fill("state")` executor 内。
- transform 克隆规则 inline 放在 close_state subtree builder 中：`Auction("merge")` 固定用 `last`，`Auction("drop")` 不传 agg，`Resample` 固定用 `last`。
- `Fill("state")` 构造出的 fill node 现在有显式 `close_state` input，因此 Mermaid / trace / `Model.nodes()` 都能看到这条依赖。
- 已增加 CSE 验证测试，证明主图 close 子链路和 close_state 子树可以共享同一个结构性 node id / `_memory` 槽。
- 已跑全量 `tests/` 回归。

### 原决策清单

- [x] 决定 preclose fallback 放法（方案 A vs B）
- [x] 决定 transform 克隆规则放法（方案 A vs B）
- [ ] 落地后跑一次小规模 profiling：「子树 evaluate + ffill_and_preclose」vs「现状 `_close_state_frame` 重放」的耗时对比，作为 #17 决策依据
- [x] 写 CSE 命中验证测试，证明主图 close 子链跟 close_state 子树共享 `_memory` 槽
- [x] 跑全量 `tests/` 回归

---

## #17 — 把 RatioField 拆成 num / den / combine 三节点，消除 payload 体系

### 背景

当前 `RatioField` 用 payload 列（`__ratio_*_num` / `__ratio_*_den`）在一个 frame 内携带 numerator / denominator，让 Auction/Resample 等 transform 能正确地「先聚合 num/den，再重算 ratio」。这套机制是**为 ratio 一种场景定制**的 special case，限制在「一个 value 列 + 两个隐藏列」的结构。

如果把 ratio 改成**图节点的组合**，payload 机制就可以整个删掉，所有「跟着 frame 流动的隐式列状态」统一变成显式的 graph node。

### 哲学定位

这是 **#7A 思路的最大范围应用**：

| 层级 | 显式化对象 | 对应 plan |
|---|---|---|
| close_state | `_close_state_frame` 内部隐式构造的 LazyFrame | #7A |
| ratio payload | `__ratio_*_num` / `__ratio_*_den` 列 | **#17（本项）** |
| 未来可能的状态 | （比如 cumulative volume、rolling std payload 等） | 同一模式自然外推 |

**「所有状态都是图节点」是更干净的设计**。当前 payload 体系是一个 special case 的工具；如果未来想加更多「需要分别聚合再组合」的字段（covariance、quantile spread 等），payload 机制会越扩展越复杂。node 方式天然外推。

### 设计

#### 用户 API 不变

```python
vwap = Resample("5m", "sum")(
    Auction("merge", "sum")(
        RatioField("amount", "volume", alias="vwap")(raw)
    )
)
```

#### 内部图结构改变

**改造前**（payload）：
```
RatioField → Node(op="ratio_field", params={num, den, alias})
执行后 frame = {date, secu_code, minute, vwap, __ratio_vwap_num, __ratio_vwap_den}
```

**改造后**（node）：
```
RatioField → Node(op="ratio_combine", params={alias}, inputs={
    "num": Node(op="field", params={"name": "amount"}, inputs={"input": raw}),
    "den": Node(op="field", params={"name": "volume"}, inputs={"input": raw}),
})
执行后 frame = {date, secu_code, minute, vwap}
```

#### Transform 分发：两种方案

**方案 A（推荐）：构造时分发（eager rewrite）**

`Auction("merge", "sum")(ratio_combine_node)` 在 Layer 构造阶段就改写为：

```python
Node(op="ratio_combine", params={alias: "vwap"}, inputs={
    "num": Node(op="auction", params={mode, agg}, inputs={"input": <num field>}),
    "den": Node(op="auction", params={mode, agg}, inputs={"input": <den field>}),
})
```

ratio_combine 始终在树顶，所有 transform 被「推下」到 num/den 子树。

- ✅ 图结构清晰，每个 transform node 只处理单 value 列
- ✅ CSE 自然命中（#8 之后）：amount field 上的 auction 跨 model 复用
- ❌ Layer 必须识别「我作用的是 ratio_combine」并重写，违反「Layer 只是 Node 工厂」的简洁性

**方案 B：执行时分发（lazy distribution）**

构造时保持自然树（Auction 在 ratio_combine 之上），`_auction_executor` 检测到 input 是 ratio_combine 时生成两条独立子计划再 join。

- ✅ 构造逻辑简单
- ❌ 每个 executor 都要做 ratio detection；本质上跟现在 payload 机制差不多

**倾向方案 A**。

### Payload 体系可以删除的代码

| 现有代码 | 处理 |
|---|---|
| `_ratio_payloads()` | 删除 |
| `_ratio_payload_columns()` | 删除 |
| `_value_and_payload_columns()` | 删除 |
| `_ordered_agg_expr` 里 ratio 分支 | 删除 |
| `_aggregate_frame` 里 num/den 分别聚合 + 重算 ratio | 删除 |
| `_fill_executor` 末尾 `drop(_ratio_payload_columns(...))` | 删除 |
| `_fill_forward` 里 ratio 分支 | 删除 |
| `_fill_literal` 里 ratio 分支 | 删除（但见下面 Fill(0) 语义讨论） |
| `__ratio_*_num` / `__ratio_*_den` 列约定 | 整个消失 |

剩下的「隐式状态」只有 close_state——而 close_state 在 #7A 之后也变成显式 Node 子树。**payload 概念整个消失。**

### 待解决的设计问题

#### 1. 性能：plan tree 变胖

每个 transform 现在要构造**两条独立子计划**（num + den），最后 join。

**保守估计**：`Auction("merge", "sum")` 在 ratio 上 = 两次 group_by + 一次 join，比 payload 的「一次 group_by」慢。
**乐观估计**：num 和 den 都来自同一个 `raw` 节点（CSE 命中），Polars 可能识别并 fuse。但赌优化器不靠谱。

**必须 profiling 验证**。如果性能不行，要么放弃 #17，要么给 `ratio_combine` 写特殊 executor 把两个 group_by fuse 成一个（这等于回到 payload，只是从 executor 内部实现）。

#### 2. num/den 同构 transform 链的保证

payload 方式有强保证：num 和 den 必然经历完全相同的 transform 链（在同一个 frame 里同步流动）。

node 方式下，由 RatioField 的 Layer 构造逻辑保证「num 和 den 子树同构」。**这个保证更脆弱**——如果未来有人手动构造 ratio_combine 并传两条结构不同的子树，结果就错了。

缓解：把 ratio_combine 标记为 private op，禁止用户直接构造，只能通过 `RatioField` Layer 创建。

#### 3. ratio_combine 的 join 语义

```python
@register_executor("ratio_combine")
def _ratio_combine(node, context):
    num_frame = context.evaluate(node.inputs["num"])
    den_frame = context.evaluate(node.inputs["den"])
    return num_frame.join(den_frame, on=KEY_COLUMNS, how="?").with_columns(
        _ratio_expr(num_col, den_col).alias(alias)
    )
```

待决定：
- **how**：inner（任一缺失整行丢）/ left（保留 num 侧）/ outer（保留所有）。当前 payload 方式下 num/den 同步存在/不存在，等价于 inner——保持这个语义最安全。
- **num/den 同名字段**：`RatioField("price", "price")` 这种边角案例下列名冲突，join 前需要 rename。

#### 4. `Fill(0)` 在 ratio 上的语义变化（**重要**）

当前 `_fill_literal` 直接把 `vwap` 列 fill_null(0)。

node 方式下，`Fill(0)(ratio_combine)` 如果走分发，变成 `num.fill(0) / den.fill(0)`——**结果不一样**：
- 当前：`null vwap → 0`
- 分发后：`null/null → 0/0 → null`（den 被 fill 成 0 触发 ratio_expr 的 zero guard）

需要决定：
- **保留当前语义**：`Fill(数值)` 不分发，直接作用在 ratio_combine 的 output value 列上。需要在 Fill executor 里做这个特殊处理。
- **改变语义**：承认「填 ratio 本身」和「填 num/den 后重算」是两件事，让用户自己想清楚要哪个。

倾向**保留当前语义**——`Fill(0)` 是个稳定的 API，不应该被内部重构破坏。

#### 5. `Fill("ffill")` 的处理

当前 `_fill_forward` 对 ratio 走 num/den 各自 ffill 然后重算。

node 方式分发后语义一致：`num.ffill / den.ffill → 重算 ratio`。**这个 case 行为不变**，可以放心分发。

#### 6. `Fill("state")` 的处理

当前 `_fill_executor` 的 state 分支对 ratio 输入做了特判（`_parse_state_fill_lineage` 里识别 `ratio_field` op）。

node 方式下，ratio 已经变成 ratio_combine + 两个 field 子树。`Fill("state")(...)` 作用在 ratio_combine 上时，**正确语义是什么**？

- 选项 A：分发到 num/den 各自 fill state，然后重算 ratio。但 num/den 各自的 close_state 是什么？它们本来就不是 price field。
- 选项 B：在 ratio_combine 的 output value 上做 state fill，直接用 close_state 填 vwap 的 null。

倾向**选项 B**——跟当前行为一致，且 close_state 本来就是为 price field 设计的填充策略。

### 影响范围

- `draco_model/layers/inputs/field.py` — `RatioField` Layer 改造成「构造 num/den/combine 三节点」的工厂；删除 `_ratio_field_executor`
- `draco_model/runtime/execution.py` 或新文件 — 注册 `ratio_combine` executor
- `draco_model/layers/transforms.py` — 删除所有 payload 相关辅助函数；`_fill_executor` 对 ratio_combine 做特判（Fill(0) / Fill("state")）；`_auction_executor` / `_resample_executor` 在方案 A 下变干净（只处理单 value）
- `draco_model/layers/combine.py` — 检查 `Concat` 对 ratio_combine 输出的处理
- `draco_model/core.py` — `Layer.__call__` 在方案 A 下需要支持「检测 input 是 ratio_combine 时做 transform 下推」的能力（或者把这逻辑放在每个相关 Layer 类的 `__call__` 里）
- 测试：所有 ratio 相关测试要重新跑 + 验证语义

### 落地前置条件

1. **#8 必须落地**（结构性 node id）——CSE 是 #17 性能可接受的前提
2. **#7A 必须落地并验证**——证明「Node 子树 + context.evaluate + CSE」模式工程可行
3. **Profiling 通过**——在真实 model workload 下验证 num/den 双计划 + join 的性能不比 payload 差太多（容忍阈值 < 30% 慢可接受？需要团队定）
4. **`Fill(0)` / `Fill("state")` 语义决策** —— 上面第 4、6 点需要明确决定

### 关联项

- **依赖 #8**（必须）
- **跟在 #7A 之后**（强烈建议）——#7A 是这条路线的可行性验证
- **跟 #16（close_state 的 Auction agg bug）配合**：#16 修完后，close_state 的 agg 规则已经确定为「mode-based」，这套规则需要在 #17 的 `Fill("state")` 路径里复用
- **跟 #4（null aggregation）独立**：但 #17 之后 ratio 的 null 语义会变（num.sum=null → ratio=null 而不是 0），跟 #4 的修复方向天然一致

### 决策清单

- [x] 等 #8 / #7A 落地
- [ ] 在 #7A 落地后做一次小规模 profiling：「同源双 group_by + join」vs「单 group_by 多列」的性能差距
- [ ] 决定 transform 分发方案（A 构造时 vs B 执行时）
- [ ] 决定 `Fill(0)` 在 ratio 上的语义（保留还是改变）
- [ ] 决定 join how（inner / left / outer）
- [ ] 如果 profiling 通过且语义都决策完，从 plan_rethink2.md 升到 plan.md 排期落地
