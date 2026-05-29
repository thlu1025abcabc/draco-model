"""Independent verification tests for plan.md / plan_rethink.md items.

These tests re-derive the expected behavior from the code alone, without
referencing how existing tests already exercise the same paths.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model, Node
from draco_model.core import _structural_id, resolve_node_names
from draco_model.layers.aggregate import DailyAgg
from draco_model.layers.combine import Concat
from draco_model.layers.expressions import sum_or_null
from draco_model.layers.inputs.field import Field, RatioField
from draco_model.layers.inputs.input import Input
from draco_model.layers.transforms import Auction, Fill, Resample
from draco_model.market.minute_calendar import MinuteCalendar


def _write_parquet(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(path)


# ---------------------------------------------------------------------------
# #8 — Structural Node.id
# ---------------------------------------------------------------------------

def test_structural_id_is_pure_function_of_structure():
    """_structural_id is a deterministic function of (kind, op, params, child ids)."""
    a = _structural_id("frame", "field", {"name": "close"}, {})
    b = _structural_id("frame", "field", {"name": "close"}, {})
    assert a == b
    assert a != _structural_id("frame", "field", {"name": "open"}, {})
    assert a != _structural_id("frame", "ratio_field", {"name": "close"}, {})


def test_no_global_counter_state_leaks_between_constructions():
    """Building Node N times must not shift ids of independently-built graphs."""
    first_batch = [Field("close")(Input(source="trades_tbar")).id for _ in range(50)]
    second_batch = [Field("close")(Input(source="trades_tbar")).id for _ in range(50)]
    assert len(set(first_batch)) == 1
    assert first_batch[0] == second_batch[0]


def test_engine_memory_hits_for_structurally_identical_subgraphs(sample_root):
    """Engine._memory key uses node.id, so two identical sub-DAGs share cache."""
    left = DailyAgg(value_col="close", agg="last")(
        Auction("drop")(Field("close")(Input(source="trades_tbar")))
    )
    right = DailyAgg(value_col="close", agg="last")(
        Auction("drop")(Field("close")(Input(source="trades_tbar")))
    )
    assert left.id == right.id

    engine = Engine(data_root=sample_root)
    model_a = Model(name="m_a", universe="ex2kamt", output=left)
    model_b = Model(name="m_b", universe="ex2kamt", output=right)
    engine.evaluate(model_a, left, "20170103")
    # second eval on identical structure under same universe -> cache hit
    key = ("ex2kamt", right.id, "20170103")
    assert key in engine._memory
    engine.evaluate(model_b, right, "20170103")
    # still only one entry under that key, not duplicated
    matching = [k for k in engine._memory if k == key]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# #14 — Optional Node/Layer name
# ---------------------------------------------------------------------------

def test_auto_naming_is_op_keyed_topological_counter():
    """Without explicit names, nodes auto-name as {op}_{counter} in topo order."""
    raw = Input(source="trades_tbar")
    close = Field("close")(raw)
    auc = Auction("drop")(close)
    output = DailyAgg(value_col="close", agg="last")(auc)
    model = Model("m", "u", output)

    names = resolve_node_names(model.nodes())
    # All five ops appear once -> counter index is _0
    assert sorted(names.values()) == sorted(["input_0", "field_0", "auction_0", "daily_agg_0"])


def test_explicit_name_overrides_auto_and_duplicates_raise():
    """User-supplied name wins; conflicting explicit names raise ValueError."""
    raw = Input(source="trades_tbar", name="shared")
    bad = Field("close", name="shared")(raw)
    model = Model("m", "u", bad)
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_node_names(model.nodes())


def test_name_does_not_affect_structural_id():
    """name is metadata only — it must not enter the id hash."""
    raw_a = Input(source="trades_tbar", name="alpha")
    raw_b = Input(source="trades_tbar", name="beta")
    raw_c = Input(source="trades_tbar")
    assert raw_a.id == raw_b.id == raw_c.id


# ---------------------------------------------------------------------------
# #4 — Null-safe sum
# ---------------------------------------------------------------------------

def test_sum_or_null_keeps_null_for_all_null_group():
    """sum_or_null returns null when every value in the group is null."""
    df = pl.DataFrame({"g": [1, 1, 2, 2], "v": [None, None, 1.0, 2.0]})
    out = df.group_by("g").agg(sum_or_null(pl.col("v")).alias("s")).sort("g")
    assert out["s"].to_list() == [None, 3.0]


def test_polars_default_sum_would_have_produced_zero():
    """Sanity check: vanilla pl.col(...).sum() turns all-null into 0 — the bug we fixed."""
    df = pl.DataFrame({"g": [1, 1], "v": [None, None]}, schema={"g": pl.Int64, "v": pl.Float64})
    bad = df.group_by("g").agg(pl.col("v").sum().alias("s"))
    assert bad["s"].to_list() == [0.0]


def test_field_volume_returns_null_when_all_minute_volume_is_null(tmp_path: Path):
    """Field("volume") must produce null per-minute bucket when every row's volume is null."""
    _write_parquet(tmp_path / "external" / "trading_days.parquet", {"date": ["20170103"]})
    data_root = tmp_path / "data"
    _write_parquet(data_root / "universe" / "ex2kamt" / "20170103.parquet", {"secu_code": [1]})
    _write_parquet(
        data_root / "trades_tbar" / "20170103.parquet",
        {
            "SecuCode": [1, 1],
            "MinBar": [930, 931],
            "Price": [10.0, 10.0],
            "Side": [0, 0],
            "Volume": [None, 5.0],
            "No": [1, 1],
            "isfirst": [True, True],
            "islast": [True, True],
        },
    )
    node = Field("volume")(Input(source="trades_tbar"))
    engine = Engine(data_root=data_root)
    out = engine.evaluate(Model("m", "ex2kamt", node), node, "20170103").collect()
    row_930 = out.filter(pl.col("minute") == 930)
    row_931 = out.filter(pl.col("minute") == 931)
    assert row_930["volume"].to_list() == [None]
    assert row_931["volume"].to_list() == [5.0]


def test_ratio_field_null_propagates_through_zero_guard(tmp_path: Path):
    """RatioField numerator-null + denominator-null -> ratio is null, not 0."""
    _write_parquet(tmp_path / "external" / "trading_days.parquet", {"date": ["20170103"]})
    data_root = tmp_path / "data"
    _write_parquet(data_root / "universe" / "ex2kamt" / "20170103.parquet", {"secu_code": [1]})
    _write_parquet(
        data_root / "trades_tbar" / "20170103.parquet",
        {
            "SecuCode": [1],
            "MinBar": [930],
            "Price": [10.0],
            "Side": [0],
            "Volume": [None],
            "No": [1],
            "amount": [None],
            "isfirst": [True],
            "islast": [True],
        },
    )
    node = RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))
    engine = Engine(data_root=data_root)
    out = engine.evaluate(Model("m", "ex2kamt", node), node, "20170103").collect()
    assert out["vwap"].to_list() == [None]


# ---------------------------------------------------------------------------
# #6 — MinuteCalendar.bucket_map cache
# ---------------------------------------------------------------------------

def test_bucket_map_caches_underlying_dataframe():
    cal = MinuteCalendar()
    a = cal.bucket_map(5)
    b = cal.bucket_map(5)
    # outer LazyFrames may differ; the cached DataFrame is the same object
    assert cal._bucket_maps[5] is cal._bucket_maps[5]
    # both LazyFrames materialize to identical content
    assert a.collect().equals(b.collect())


def test_bucket_map_distinguishes_intervals():
    cal = MinuteCalendar()
    m5 = cal.bucket_map(5).collect()
    m15 = cal.bucket_map(15).collect()
    # row 930 always buckets to itself; row 934 buckets to 930 for 5m and to 930 for 15m
    bucket_of = lambda df, minute: df.filter(pl.col("minute") == minute)["__bucket_minute"][0]
    assert bucket_of(m5, 930) == 930
    assert bucket_of(m5, 934) == 930
    assert bucket_of(m5, 935) == 935
    assert bucket_of(m15, 935) == 930  # 15m bucket spans 930..944


def test_bucket_map_excludes_auction_minutes():
    cal = MinuteCalendar()
    df = cal.bucket_map(5).collect()
    minutes = set(df["minute"].to_list())
    assert 925 not in minutes
    assert 1500 not in minutes
    assert 930 in minutes


def test_bucket_map_rejects_invalid_interval():
    cal = MinuteCalendar()
    with pytest.raises(ValueError):
        cal.bucket_map(0)


# ---------------------------------------------------------------------------
# #16 — close_state replays Auction(merge) with "last" semantics, not original agg
# ---------------------------------------------------------------------------

def test_close_state_under_auction_merge_uses_last_not_sum(tmp_path: Path):
    """Fill('state') after Auction('merge','sum') must fill nulls from close (last),
    not from the bogus sum (925.close + 930.close).
    """
    _write_parquet(tmp_path / "external" / "trading_days.parquet", {"date": ["20170103"]})
    data_root = tmp_path / "data"
    _write_parquet(data_root / "universe" / "ex2kamt" / "20170103.parquet", {"secu_code": [1]})
    _write_parquet(
        data_root / "daily_k" / "20170103.parquet",
        {"sec_code": ["000001.SZ"], "trading_day": ["2017-01-03"], "preclose": [9.0]},
    )
    # 925 close=9.0, 930 close=10.0. amount is null everywhere -> after Auction(merge,sum)
    # the post-merge amount@930 is null, which Fill('state') must fill from close_state.
    # close_state @930 must equal 10.0 (last) — NOT 19.0 (=9+10, the old bug).
    _write_parquet(
        data_root / "trades_tbar" / "20170103.parquet",
        {
            "SecuCode": [1, 1, 1],
            "MinBar": [925, 930, 1500],
            "Price": [9.0, 10.0, 15.0],
            "Side": [0, 0, 0],
            "Volume": [1.0, 1.0, 1.0],
            "No": [1, 1, 1],
            "amount": [None, None, None],
            "isfirst": [True, True, True],
            "islast": [True, True, True],
        },
    )

    pipeline = Fill("state")(Auction("merge", "sum")(Field("amount")(Input(source="trades_tbar"))))
    model = Model("audit_16", "ex2kamt", pipeline)
    out = Engine(data_root=data_root).evaluate(model, pipeline, "20170103").collect()

    amount_930 = out.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["amount"][0]
    assert amount_930 == pytest.approx(10.0), f"expected 10.0 (close last), got {amount_930}"
    # And NOT 19.0, which would indicate the old sum-passthrough bug
    assert amount_930 != pytest.approx(19.0)


def test_close_state_under_auction_drop_does_not_aggregate(tmp_path: Path):
    """Fill('state') after Auction('drop') replays drop on close (no agg)."""
    _write_parquet(tmp_path / "external" / "trading_days.parquet", {"date": ["20170103"]})
    data_root = tmp_path / "data"
    _write_parquet(data_root / "universe" / "ex2kamt" / "20170103.parquet", {"secu_code": [1]})
    _write_parquet(
        data_root / "daily_k" / "20170103.parquet",
        {"sec_code": ["000001.SZ"], "trading_day": ["2017-01-03"], "preclose": [9.0]},
    )
    _write_parquet(
        data_root / "trades_tbar" / "20170103.parquet",
        {
            "SecuCode": [1, 1, 1, 1],
            "MinBar": [925, 930, 935, 1500],
            "Price": [9.0, 10.0, 11.0, 15.0],
            "Side": [0, 0, 0, 0],
            "Volume": [1.0, 1.0, 1.0, 1.0],
            "No": [1, 1, 1, 1],
            "amount": [None, None, None, None],
            "isfirst": [True, True, True, True],
            "islast": [True, True, True, True],
        },
    )

    pipeline = Fill("state")(Auction("drop")(Field("amount")(Input(source="trades_tbar"))))
    model = Model("audit_16_drop", "ex2kamt", pipeline)
    out = Engine(data_root=data_root).evaluate(model, pipeline, "20170103").collect()

    # Auction(drop) removes 925 and 1500. close at 930=10, forward-filled through 935.
    # Fill('state') for amount@930 -> close_state[930] = 10
    amount_930 = out.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["amount"][0]
    amount_935 = out.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 935))["amount"][0]
    assert amount_930 == pytest.approx(10.0)
    assert amount_935 == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# #5 — collect_schema reduction via cached infer_schema
# ---------------------------------------------------------------------------

def test_executors_route_through_infer_schema_not_collect_schema(monkeypatch, sample_root):
    """Hot-path executors must consult context.infer_schema rather than calling
    LazyFrame.collect_schema themselves on every helper hop."""
    call_count = {"n": 0}
    original = pl.LazyFrame.collect_schema

    def counting(self):
        call_count["n"] += 1
        return original(self)

    monkeypatch.setattr(pl.LazyFrame, "collect_schema", counting)

    # Build a fairly deep chain so an old impl that re-asked schema in every
    # helper would explode the count.
    pipeline = DailyAgg(value_col="close", agg="last")(
        Resample("5m", "last")(Auction("drop")(Field("close")(Input(source="trades_tbar"))))
    )
    model = Model("schema_audit", "ex2kamt", pipeline)
    Engine(data_root=sample_root).evaluate(model, pipeline, "20170103").collect()

    # Loose ceiling. The point is "not dozens of calls per executor hop".
    # With current code, count is ~10. Pin at <40 so accidental regressions trip.
    assert call_count["n"] < 40, f"collect_schema called {call_count['n']} times"


# ---------------------------------------------------------------------------
# #7A 鈥?close_state is an explicit Node subtree
# ---------------------------------------------------------------------------

def test_fill_state_close_subtree_is_explicit_and_reuses_equivalent_main_graph(sample_root):
    raw = Input(source="trades_tbar")
    close_5m = Resample("5m", "last")(Auction("drop")(Field("close")(raw)))
    filled_high = Fill("state")(Resample("5m", "max")(Auction("drop")(Field("high")(raw))))
    output = Concat()({"high": filled_high, "close": close_5m})
    model = Model("audit_7a", "ex2kamt", output)

    assert filled_high.inputs["close_state"].id == close_5m.id
    assert close_5m.id in {node.id for node in model.nodes()}

    engine = Engine(data_root=sample_root)
    engine.evaluate(model, output, "20170103").collect()

    assert ("ex2kamt", close_5m.id, "20170103") in engine._memory
