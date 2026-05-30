from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model
from draco_model.layers import Aggregate, Join, Metric, Source, TopQuantile, Where


DATA_ROOT = Path("data")
DATE = "20170103"


raw = Source("trades_tbar")
features = Join()({
    "close": Metric("close", raw),
    "volume": Metric("volume", raw),
})
filtered = Where(TopQuantile("volume", q=0.8, over=["date", "secu_code"]))(features)
output = Aggregate("1d", "mean", value_col="close", alias="value")(filtered)
model = Model(name="top_volume_close_mean", universe="ex2kamt", output=output)


def trace_(engine: Engine) -> dict:
    out = {}
    for step in engine.trace(model, DATE):
        key = " ".join([str(step.index), step.node.op, str(step.node.params), str(step.frame.shape)])
        out[key] = step.frame
    return out


if __name__ == "__main__":
    engine = Engine(data_root=DATA_ROOT)
    print("\nFinal result")
    print(engine.collect(model, dates=[DATE]))
    print("\nDAG")
    print(model.explain_mermaid())
