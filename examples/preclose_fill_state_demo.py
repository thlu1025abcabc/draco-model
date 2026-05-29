from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model
from draco_model.layers import Field, Fill, Input, Resample


DATA_ROOT = Path("data")
DATE = "20170103"


raw = Input(source="trades_tbar")

direct_preclose = Field("preclose")(raw)
direct_model = Model(name="direct_preclose", universe="ex2kamt", output=direct_preclose)

filled_preclose = Fill("state")((Field("preclose")(raw)))
filled_model = Model(name="filled_preclose_5m", universe="ex2kamt", output=filled_preclose)


if __name__ == "__main__":
    engine = Engine(data_root=DATA_ROOT)

    print("Direct Field(\"preclose\") model")
    try:
        print(engine.evaluate(direct_model, direct_preclose, DATE).collect().head(8))
    except ValueError as error:
        print(f"Expected error: {error}")

    print("\nFill(\"state\")(Resample(\"5m\", \"last\")(Field(\"preclose\"))) model")
    print(engine.evaluate(filled_model, filled_preclose, DATE).collect().head(8))

    print("\nDAG")
    print(filled_model.explain_mermaid())
