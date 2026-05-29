from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model
from draco_model.layers import Concat, DailyAgg, Field, Filter, Input, TopQuantile


DATA_ROOT = Path("data")
DATE = "20170103"


raw = Input(source="trades_tbar")
close = Field("close")(raw)
volume = Field("volume")(raw)
features = Concat()({"close": close, "volume": volume})
filtered = Filter(TopQuantile("volume", q=0.8, over=["date", "secu_code"]))(features)
output = DailyAgg(value_col="close", agg="mean")(filtered)
model = Model(name="top_volume_close_mean", universe="ex2kamt", output=output)


def trace_(engine: Engine) -> dict:
    mapper = {}
    for step in engine.trace(model, DATE):
        key = " ".join([str(i) for i in [step.index,
                        step.node.op,
                        step.node.params,
                        step.frame.shape]])
        value = step.frame
        mapper[key] = value
    return mapper

if __name__ == "__main__":
    engine = Engine(data_root=DATA_ROOT)
    step_trace = trace_(engine)
    print("\nFinal result")
    print(engine.collect(model, dates=[DATE]))
    print("\nDAG")
    print(model.explain_mermaid())
