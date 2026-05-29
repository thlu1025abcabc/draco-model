from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Engine, Model
from draco_model.layers import Field, Input, RatioField
from draco_model.layers.inputs.field import FIELD_BUILDERS, get_field_builder, register_field


def test_builtin_field_builders_are_registered() -> None:
    assert {"open", "high", "low", "close", "volume", "no", "amount", "preclose"}.issubset(FIELD_BUILDERS)
    assert "vwap" not in FIELD_BUILDERS
    for name, builder in FIELD_BUILDERS.items():
        assert get_field_builder(name) is builder


def test_unknown_field_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported field 'missing'"):
        get_field_builder("missing")


def test_duplicate_field_registration_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Duplicate field builder 'close'"):

        @register_field("close")
        def duplicate_close(raw: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
            return raw


def test_field_alias_renames_single_value_column(sample_root: Path) -> None:
    trade_volume = Field("volume", alias="trade_volume")(Input(source="trades_tbar"))
    model = Model(name="trade_volume_probe", universe="ex2kamt", output=trade_volume)

    frame = Engine(data_root=sample_root).evaluate(model, trade_volume, "20170103").collect()

    assert "trade_volume" in frame.columns
    assert "volume" not in frame.columns
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["trade_volume"].sum() > 0


def test_field_alias_cannot_use_internal_payload_prefix() -> None:
    with pytest.raises(ValueError, match="must not start with '__'"):
        Field("volume", alias="__volume")
    with pytest.raises(ValueError, match="must not start with '__'"):
        RatioField("amount", "volume", alias="__vwap")


def test_volume_field_keeps_all_null_group_null(sample_root: Path) -> None:
    source = sample_root / "null_volume_tbar" / "20170103.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "SecuCode": [1, 1, 1],
            "MinBar": [930, 930, 931],
            "Volume": [None, None, 5.0],
        }
    ).write_parquet(source)

    volume = Field("volume")(Input(source="null_volume_tbar"))
    model = Model(name="null_volume_probe", universe="ex2kamt", output=volume)

    frame = Engine(data_root=sample_root).evaluate(model, volume, "20170103").collect()

    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))["volume"].to_list() == [None]
    assert frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 931))["volume"].to_list() == [5.0]


def test_ratio_field_builds_public_ratio_and_payload(sample_root: Path) -> None:
    vwap = RatioField("amount", "volume", alias="vwap")(Input(source="trades_tbar"))
    model = Model(name="vwap_probe", universe="ex2kamt", output=vwap)

    frame = Engine(data_root=sample_root).evaluate(model, vwap, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["vwap"].to_list() == pytest.approx([152.0 / 15.0])
    assert {"__ratio_vwap_num", "__ratio_vwap_den"}.issubset(frame.columns)


def test_ratio_field_keeps_all_null_numerator_null(sample_root: Path) -> None:
    source = sample_root / "null_amount_tbar" / "20170103.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "SecuCode": [1, 1],
            "MinBar": [930, 930],
            "Amount": [None, None],
            "Volume": [5.0, 5.0],
        }
    ).write_parquet(source)

    vwap = RatioField("amount", "volume", alias="vwap")(Input(source="null_amount_tbar"))
    model = Model(name="null_ratio_probe", universe="ex2kamt", output=vwap)

    frame = Engine(data_root=sample_root).evaluate(model, vwap, "20170103").collect()

    row = frame.filter((pl.col("secu_code") == 1) & (pl.col("minute") == 930))
    assert row["__ratio_vwap_num"].to_list() == [None]
    assert row["__ratio_vwap_den"].to_list() == [10.0]
    assert row["vwap"].to_list() == [None]
