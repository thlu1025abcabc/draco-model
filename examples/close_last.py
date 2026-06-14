from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model
from draco_model.layers import Aggregate, Source
from draco_model.recipes import metric


DATA_ROOT = Path("data")


raw = Source("trades_tbar", name="raw_trades")
close = metric("close")(raw)
output = Aggregate("1d", "last", value_col="close", alias="value", name="daily_close_last")(close)
model = Model(name="close_last", universe="ex2kamt", output=output)


if __name__ == "__main__":
    engine = Engine(data_root=DATA_ROOT)
    print(engine.collect(model, dates=["20170103"]))
    print(model.explain_mermaid())
