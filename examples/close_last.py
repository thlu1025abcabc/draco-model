from __future__ import annotations

from pathlib import Path

from draco_model import Engine, Model
from draco_model.layers import Auction, DailyAgg, Field, Input


DATA_ROOT = Path("data")


close = Auction("drop", name="drop_auction")(
    Field("close", name="close_1m")(
        Input(source="trades_tbar", name="raw_trades")
    )
)
output = DailyAgg(value_col="close", agg="last", name="daily_close_last")(close)
model = Model(name="close_last", universe="ex2kamt", output=output)


if __name__ == "__main__":
    engine = Engine(data_root=DATA_ROOT)
    result = engine.collect(model, dates=["20170103"])
    print(result)
    print(model.explain_mermaid())
