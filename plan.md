# Draco Model 优化计划

按优先级和依赖关系记录待办优化项。每项独立可落地，不预先一次性实施。

---

## #8 — `Node.id` 改成结构性 hash

### 目标

把 `Node.id` 从全局自增 counter 改为基于 `(kind, op, params, inputs)` 的结构性 hash，让结构相同的子图自动共享 id，从而：

1. **可复现**：同一 model 在任意进程、任意时刻构造，node id 稳定。
2. **自动 CSE**：`Engine._memory` 按 `(universe, node.id, eval_date)` 缓存 lazy plan，结构同构的子图自动命中。
3. **为 #7A 铺路**：`Fill('state')` 内部重建的 close-state 子链自动复用主图节点。
4. **顺手修 Filter 重复 condition**：同一 condition 作用在同一 frame 上不再造两份。

### 现状

```python
# core.py:9, 20
_NODE_COUNTER = itertools.count()

@dataclass(frozen=True)
class Node:
    ...
    id: str = field(default_factory=lambda: f"n{next(_NODE_COUNTER)}")
```

Counter 全局且永不重置 → 同一结构每次构造 id 不同 → 缓存 miss、调试 id 漂移。

### 设计

`Node` 是 `frozen=True` 的 dataclass，`default_factory` 拿不到其他字段。改用 `__post_init__` + `object.__setattr__` 绕开 frozen：

```python
import hashlib
import json

@dataclass(frozen=True)
class Node:
    kind: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, "Node"] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", _structural_id(self.kind, self.op, self.params, self.inputs))


def _structural_id(kind: str, op: str, params: dict, inputs: dict) -> str:
    payload = json.dumps(
        {
            "kind": kind,
            "op": op,
            "params": params,
            "inputs": [(name, child.id) for name, child in sorted(inputs.items())],
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
    return f"n_{digest}"
```

Bottom-up 自然成立：子节点先构造，子的 id 已定；父算 id 时直接读子 id。

### 细节确认

| 项 | 说明 |
|---|---|
| **params 序列化** | 现有 params 全是 `str/int/float/None/list[str]`，可 JSON 序列化。`Threshold.value: Any` 是理论风险点，`default=str` 兜底。 |
| **跨 model 缓存** | `_memory` key 含 `universe`，同 universe 下跨 model 共享子图 → 自动复用 lazy plan，且 universe 保隔离。 |
| **Mermaid 输出** | `Model.explain_mermaid()` 已用局部 alias `f"n{idx}"`，底层 id 改成 hash 不影响渲染。 |
| **Hash 长度** | 8 bytes blake2b → 16 hex chars，碰撞概率对实际规模可忽略。 |
| **`__hash__`** | 现有 `hash(self.id)` 不变，id 仍是字符串。 |
| **`_NODE_COUNTER`** | 删掉，不再需要。 |

### 影响范围

- `draco_model/core.py` — 改 `Node`，加 `_structural_id`，删 `_NODE_COUNTER`
- 其他构造 `Node()` 的地方（`layers/filters.py`、`layers/inputs/input.py`、`Condition.to_node`）—— 不动，`__post_init__` 自动跑
- `tests/` — 跑一遍现有测试看有没有间接断言 node id 格式（预期不会断，但需验证）

### 验证

1. **回归**：现有 `tests/` 全部通过。
2. **新增测试**：
   - 结构相同的两个 model output node id 相等。
   - 结构不同（params 改一个值、inputs 顺序逻辑不同）id 不同。
   - 同一进程多次构造 model，id 稳定。
3. **CSE 验证**：构造两个共享子图的 model，跑 `Engine.collect()`，断言 `_memory` 在第二次求值时命中。

### 关联项

- **#7A**（`Fill('state')` 复用 close 子链）依赖本项落地后才能干净实现。落地顺序：先 #8，再 #7A。
- 不影响 #5（schema 单次调用）、#6（bucket map 缓存），可独立。

---

## #14 — `Node`/`Layer` 增加可选 `name` 字段

### 目标

让 `Engine.trace()` 和 `Model.explain_mermaid()` 使用同一套**人类可读节点名**，便于把 trace step 和流程图互相对应。当前调试时只能看 `node.id`（自增计数或 #8 之后的 hash），可读性差。

### 现状

- `Node.id` 是 `n0`/`n1`/... 或（#8 之后）`n_<hash>`，调试不友好。
- `Model.explain_mermaid()` 已经用局部 alias `f"n{idx}"` 渲染，但仍然没有语义信息。
- `TraceStep` 只暴露 `node`，打印时拿不到稳定的人类可读名字。

### 设计（README 草案已定）

```python
raw = Input(source="trades_tbar", name="raw_trades")
close = Field("close", name="close_1m")(raw)
volume = Field("volume", name="volume_1m")(raw)
features = Concat(name="close_volume")({"close": close, "volume": volume})
filtered = Filter(TopQuantile("volume", q=0.8), name="top_volume_filter")(features)
output = DailyAgg(value_col="close", agg="mean", name="daily_mean_close")(filtered)
```

### 规则

| 规则 | 说明 |
|---|---|
| **可选字段** | 用户不传时，`Engine`/`Model` 自动按拓扑顺序分配稳定可读名字，如 `input_0`、`field_close_1`、`filter_4`。 |
| **仅 debug/visualization** | `name` 不参与 executor dispatch，不改变计算语义，不参与 #8 的结构性 hash。 |
| **Node 字段** | `Node.name: str \| None`，`Layer.__call__()` 创建节点时把 layer 的 `name` 传下去。 |
| **直接构造入口** | `Input(...)`、`Condition.to_node` 等直接返回 `Node` 的入口也接受 `name`。 |
| **TraceStep** | 保留 `node`，额外暴露 `resolved_name`。打印/示例用 resolved name。 |
| **Mermaid** | `Model.explain_mermaid()` 用 resolved name 当主标签，同时保留 `op` 和关键 params，避免图里只有名字看不出节点类型。 |

### 自动命名策略

- 拓扑序遍历，每个 op 维护一个 counter
- 默认格式：`{op}_{counter}` 或 `{op}_{key_param}_{counter}`
  - `Input(source="trades_tbar")` → `input_trades_tbar_0`
  - `Field("close")` → `field_close_0`
  - `Auction("drop")` → `auction_drop_0`
  - `Resample("5m", "last")` → `resample_5m_0`
  - `Fill("state")` → `fill_state_0`
  - `DailyAgg(value_col="close", agg="mean")` → `daily_agg_close_mean_0`
- 同名 counter 自增保证全局唯一

### 影响范围

- `draco_model/core.py` — `Node` 加 `name` 字段，`Layer.__init__` 接收 `name`，`Layer.__call__` 透传到 Node。
- `draco_model/layers/inputs/input.py` — `Input.__new__` 加 `name` 参数。
- `draco_model/layers/filters.py` — `Filter.__init__` 加 `name` 参数；`Condition.to_node` 可考虑接 `name`（condition 节点是否需要单独命名待定，默认从 frame 派生即可）。
- `draco_model/core.py` 或新建 `naming.py` — 实现 `_resolve_names(nodes) -> dict[node_id, str]` 自动命名函数。
- `draco_model/runtime/execution.py` — `TraceStep` 加 `resolved_name` 字段。
- `draco_model/core.py` `Model.explain_mermaid()` — 渲染用 resolved name。
- `draco_model/runtime/engine.py` `Engine.trace()` — 把 resolved name 塞进 `TraceStep`。

### 验证

1. **回归**：现有 `tests/` 通过。
2. **新增测试**：
   - 显式传 `name` → trace 和 mermaid 出现该名字。
   - 不传 `name` → 自动命名稳定（同一 model 跑两次结果相同）。
   - 重名（用户两个节点用同一个 `name`）→ raise 清晰错误，或者后缀自增（设计决定）。
3. **示例更新**：`examples/close_last.py` 加上 `name`，README 同步。

### 关联项

- **依赖 #8**：#8 让 `Node.id` 稳定，自动命名走拓扑序时才能跨进程复现。先 #8 后 #14。
- **不影响计算路径**：完全是 metadata 层。
- 与 #15（CLAUDE.md / AGENTS.md 同步）无关，但落地时记得更新 README 示例。

---

## #12 — 加 logging（等 cloud 部署规范定了再落地）

### 目标

为「上云机按日并行」部署做准备。当前 `draco_model/` 完全没有 logging 调用，正常路径静默，线上跑挂了没现场可看。

### 现状

- 异常抛出是唯一可观测信号
- `Engine.trace()` 和 `Model.explain_mermaid()` 只能交互式用
- Cloud runner 收集 stdout/stderr，但项目不产出任何结构化日志

### 不立即做的理由

Logging 规范应该**先定再写**。现在随手加 `logger.info(...)` 几行，等 cloud 平台日志规范定了（JSON 格式、字段命名、级别约定等）很可能要返工。等具体的日志聚合栈（ELK / Loki / Datadog / 公司自研）和 schema 确定后再做。

### 关键打点位置（按价值排序）

| 位置 | 日志内容 | 级别 |
|---|---|---|
| `Engine.collect` 进入/退出 | model name, universe, dates, 耗时 | INFO |
| `Engine.evaluate` 单 date 结束 | model, date, output rows, 耗时 | INFO |
| `SourceCatalog._scan_date` | source, date, 路径, schema hash | DEBUG |
| `Fill('state')` 触发 close_state 重建 | source, lookback_days, transforms | DEBUG |
| `Engine.trace` 每个 step | step index, node op, name, row count, 耗时 | DEBUG |
| Executor 失败入口 | node id/name, op, params, 上游 inputs | ERROR |

最值钱的两个：`Engine.collect` 的进入/退出（每个 model run 一对）和 `SourceCatalog._scan_date`（数据 IO 现场）。

### 实现风格

**stdlib `logging`**（不引入 `structlog` 等额外依赖）：

```python
# 每个模块顶部
import logging
logger = logging.getLogger(__name__)

# Engine.collect 里
logger.info("collect.start model=%s universe=%s dates=%s", model.name, model.universe, dates)
t0 = time.perf_counter()
...
logger.info("collect.done model=%s rows=%d elapsed=%.3fs", model.name, rows, time.perf_counter() - t0)
```

结构化输出靠 `python-json-logger` 或类似 formatter，**在应用层配置**，库代码不绑死格式。

### 约束 / 坑

| 项 | 规则 |
|---|---|
| **logger 实例化** | 每个模块 `logging.getLogger(__name__)`，**不要**拿 root logger |
| **handler 配置** | 库代码绝不调 `logging.basicConfig()` 或加 handler——应用层职责 |
| **计时** | 用 `time.perf_counter()`，不是 `time.time()` |
| **hot path** | `_aggregate_frame` / `_resample_frame` 这种每 transform 都进的函数不打。100 model × 几十 transform 一秒会刷几万行 |
| **异常处理** | 库代码 `raise` 不打日志；只在 `Engine.collect` 等对外入口 `logger.exception(...)` 后 re-raise |

### 影响范围

- `draco_model/runtime/engine.py` — `collect` / `evaluate` 加日志
- `draco_model/data/source.py` — `_scan_date` 加 DEBUG
- `draco_model/layers/transforms.py` — `_close_state_frame` 加 DEBUG
- `draco_model/runtime/execution.py` — 可选：`get_executor` 失败时打 ERROR
- `pyproject.toml` — 可选：声明 `python-json-logger` 为 optional dep（如果选 JSON formatter）

### 落地时序

1. **等 cloud 部署规范明确**：日志格式（plain / JSON）、级别约定、字段命名、聚合栈
2. **先定 logger naming / 关键字段** 再写代码
3. **加完后跑一次本地 `examples/`** 看输出 noise 是否可接受
4. **写一个 logging 配置示例** 进 README 或 `examples/logging_setup.py`

### 关联项

- 与 #8 / #14 独立，可任何顺序
- 加日志时把 #14 的 `resolved_name` 一起带上，trace step 日志才有可读 id

---

## #10 — `.gitignore` 排除生成物

### 目标

把 Python / pytest 自动生成的目录从仓库里排除掉，避免 `__pycache__/` 和 `.pytest_cache/` 污染 git status 和 diff。

### 现状

`tests/` 和 `draco_model/` 下到处是 `__pycache__/`，根目录有 `.pytest_cache/`。检查 `.gitignore` 是否已经覆盖这些（如果没有 `.gitignore` 或规则不全，下一次 commit 容易把 `.pyc` 带进去）。

### 建议规则

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class

# Pytest
.pytest_cache/

# 虚拟环境
.venv/
venv/
env/

# 包构建产物
build/
dist/
*.egg-info/

# IDE
.idea/
.vscode/

# OS
.DS_Store
Thumbs.db
```

`.idea/` 当前在仓库里——如果是有意 check-in（团队共享 IDE 配置），从规则里删掉这行。

### 落地步骤

1. 检查现有 `.gitignore`（如果没有就新建）
2. 合并上述规则
3. 对已 tracked 的 `__pycache__` 做 `git rm -r --cached __pycache__`（如果 git 已经追踪了）
4. commit

### 影响范围

- 仅 `.gitignore` 一个文件
- 可能伴随一次 `git rm --cached` 清理已追踪的 cache 文件

### 关联项

- 独立项，不依赖其他改动
- 跟 #11（pyproject dev deps）合并做也合理——都是工程基础设施

---

## #4 — 修正 aggregation 的 `null` 语义

### 目标

让 `.sum()` 对**全 null** 的 group 返回 `null`，而不是 `0`。0 和 null 在因子语义里完全不同——0 是「真的没有交易/没有量」，null 是「这个 group 没数据」。把 null 误当 0 会污染下游计算（除法、ranking、winsorize）。

### 现状

```python
# layers/aggregate.py:38, layers/transforms.py:204
def _agg_expr(column, method):
    if method == "sum":
        return expr.sum()
    ...
```

Polars 的 `.sum()` 对全 null group 返回 `0`，不是 `null`。

README TODO 里已经标记：

> 修正 aggregation 的 null 语义：当前 `.agg(pl.col(...).sum())` 会把全 null 聚合成 `0`，后续需要改成全 null 保持 null，避免把缺失值误当成真实 0。

### 改法

把受影响的 aggregation 包一层 `when/then/otherwise`：

```python
def _null_safe_sum(column: str | pl.Expr) -> pl.Expr:
    expr = pl.col(column) if isinstance(column, str) else column
    return pl.when(expr.is_not_null().any()).then(expr.sum()).otherwise(None)
```

`_agg_expr` 里 `"sum"` 分支替换：

```python
if method == "sum":
    return _null_safe_sum(expr)
```

### 需要同时检查的其他方法

| 方法 | 全 null 行为 | 是否要改 |
|---|---|---|
| `sum` | → 0 | **要改** |
| `mean` | → null | 不用改 |
| `max` / `min` | → null | 不用改 |
| `std` / `median` | → null | 不用改 |
| `first` / `last` | 当前 `drop_nulls().first/last()` → null | 不用改 |

只有 `sum` 有这个 bug。但要在两个 `_agg_expr`（`aggregate.py` 和 `transforms.py`）都改，并且 `transforms.py` 里的 ratio 分支（numerator/denominator 分别 sum）也要同步。

### 影响范围

- `draco_model/layers/aggregate.py` — `_agg_expr` 的 `sum` 分支
- `draco_model/layers/transforms.py` — `_agg_expr` 的 `sum` 分支，以及 `_aggregate_frame` 里 ratio numerator/denominator 的 sum
- `draco_model/layers/inputs/field.py` — `_volume_field` / `_no_field` / `_amount_field` 里的 `pl.col(...).sum()`，以及 `_component_expr` 末尾的 `sum()`。这些 field builder 也要用 null-safe sum，否则在 field 这一步就把全 null 变成 0 了，下游 fill 也救不回来。

### 验证

1. **新增测试**：
   - 构造一个 source，某个 `(date, secu_code, minute)` group 的 `volume` 全是 null → `Field("volume")` 输出该 group 的 volume **是 null，不是 0**
   - 同样测 `DailyAgg(value_col="volume", agg="sum")`
   - 测 ratio：numerator 全 null + denominator 有值 → ratio 是 null（而不是 0/x = 0）
2. **回归**：现有 sum 相关测试可能需要更新——如果之前断言全 null group 输出 0，现在改成 null。

### 关联项

- 独立于 #8 / #14 / #12 / #10，可任何顺序
- 跟 #5（schema 单次调用）/ #6（bucket map 缓存）也独立
- README TODO 里的「修正 aggregation 的 null 语义」对应这一项，做完后从 README TODO 删除

---

## #16 — `_close_state_frame` 的 Auction agg 误传

### 目标

修正 `_close_state_frame` 重放 Auction transform 时**错误地透传原始 agg** 的语义 bug。当主链路是 `RatioField + Auction("merge", "sum")` 时，close_state 内部会对 close 做 `sum`，得到无意义的数值（例如 925.close + 930.close）。

### 现状

```python
# layers/transforms.py:280-286
for transform in lineage.transforms:
    if transform.op == "auction":
        close = _auction_frame(close, str(transform.params["mode"]), transform.params.get("agg"))
    elif transform.op == "resample":
        close = _resample_frame(close, str(transform.params["frequency"]), "last", context)
```

**Resample 强制用 `"last"`** —— 正确（「这根 5 分钟 bar 的 close 是哪个值」只能是 last）

**Auction 原样透传 `transform.params.get("agg")`** —— 在 ratio 场景下有问题。

### 触发场景

```python
raw = Input(source="trades_tbar")
vwap = Fill("state")(
    Resample("5m", "sum")(
        Auction("merge", agg="sum")(
            RatioField("amount", "volume", alias="vwap")(raw)
        )
    )
)
```

- 对 vwap 来说，`Auction("merge", "sum")` 是正确的——sum num + sum den + 重算 ratio
- 但 `_close_state_frame` 会把 `("merge", "sum")` 应用到 close 上：
  - 925 → 930 合并后，close[930] = `925.close + 930.close`
  - 1456 → 1500 合并后，close[1500] = `1456.close + 1500.close`
- 这个无意义的「close 和」被 forward_fill 后用来填 vwap 的 null，污染下游

### 修法

close_state 永远用 `"last"`，跟 Resample 保持一致：

```python
for transform in lineage.transforms:
    if transform.op == "auction":
        mode = str(transform.params["mode"])
        # close_state 永远用 last 语义，不跟随原始 agg
        agg = "last" if mode == "merge" else None
        close = _auction_frame(close, mode, agg)
    elif transform.op == "resample":
        close = _resample_frame(close, str(transform.params["frequency"]), "last", context)
```

`mode == "drop"` 时 `_auction_frame` 内部就不需要 agg，传 None 即可。

### 待讨论的细节

1. **`"last"` 是不是放之四海皆准？**
   - 对 `merge, "first"`：close[930] 应该是 925 的 close 还是 930 的 close？合并语义上是「auction 期间的所有信息聚合到一根 bar」，「这根 bar 的 close」应该是 last。
   - 对 `merge, "sum"`（vwap 场景）：同上，last 是对的。
   - 对 `merge, "last"`：跟原始 agg 一致，无歧义。
   - 结论：**任何 merge mode 下，close_state 都用 last**。
2. **是否影响其他依赖 close_state 的字段？**
   - `_preclose_from_state` 用的是 `__close_state.shift(1)`，依赖于 close_state 的正确性。
   - `open/high/low/ratio` 的 Fill('state') 也用 close_state 填 null。
   - 都受益于这个修正。

### 影响范围

- `draco_model/layers/transforms.py:280-286` — 修一行 dispatch 逻辑

### 验证

1. **新增测试**：
   - 构造 trades_tbar，让 925 和 930 有不同的 close 价格
   - 跑 `Fill("state")(Auction("merge", "sum")(RatioField("amount", "volume", alias="vwap")(raw)))`
   - 断言：填充 null 时用的 close_state[930] **是 930 的 close，而不是 925.close + 930.close**
2. **回归**：现有 `tests/test_price_transforms.py` 里如果有覆盖 ratio + auction merge 场景，可能要检查断言值是否需要更新。

### 关联项

- **跟 #7A 高度相关**：#7A 在重写 `_close_state_frame` 的 transform 链构造，**正好顺手把这个 bug 一起修了**。可以合并到 #7A 落地里。
- 也可以**先于 #7A 单独修**——只改一行 dispatch 逻辑，不需要 #8 / #7A 的任何基础设施。
- 独立于 #4（null 语义），但两者都是「数据语义正确性」类的 bug，可以一起做一轮回归。
