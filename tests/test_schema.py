from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model
from draco_model.layers import Auction, Fill, Input, Resample, RatioField


def test_engine_infers_transform_output_schema(sample_root: Path) -> None:
    raw = Input(source="trades_tbar")
    output = Fill("state")(
        Resample("5m", "sum")(
            Auction("merge", agg="sum")(
                RatioField("amount", "volume", alias="vwap")(raw)
            )
        )
    )
    model = Model(name="schema_probe", universe="ex2kamt", output=output)
    engine = Engine(data_root=sample_root)
    engine._ensure_calendar()

    schema = engine._infer_schema(model, output, "20170103")

    assert schema.columns == ("date", "secu_code", "minute", "vwap")
