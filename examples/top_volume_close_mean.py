from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model
from draco_model.layers import Aggregate, Join, Source, TopQuantile, Where
from draco_model.recipes import metric


DATA_ROOT = Path("data")
DATE = "20170103"


raw = Source("trades_tbar")
features = Join()({
    "close": metric("close")(raw),
    "volume": metric("volume")(raw),
})
filtered = Where(TopQuantile("volume", q=0.8, over=["date", "secu_code"]))(features)
output = Aggregate("1d", "mean", value_col="close", alias="value")(filtered)
model = Model(name="top_volume_close_mean", universe="ex2kamt", output=output)


if __name__ == "__main__":
    engine = Engine(data_root=DATA_ROOT)
    print("\nFinal result")
    print(engine.collect(model, dates=[DATE]))
    print("\nDAG")
    print(model.explain_mermaid())
