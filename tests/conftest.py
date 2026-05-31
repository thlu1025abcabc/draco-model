from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draco_model import Node
from draco_model.runtime.execution import EvalContext, register_executor


@register_executor("constant_test_frame")
def _constant_test_frame(node: Node, context: EvalContext) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [context.eval_date],
            "secu_code": [1],
            "value": [float(node.params["value"])],
        }
    ).lazy()


@pytest.fixture
def engine_data_root(tmp_path: Path) -> Path:
    """Minimal data root with only the trading calendar populated."""
    _write_parquet(tmp_path / "external" / "trading_days.parquet", {"date": ["20170103", "20170104"]})
    return tmp_path / "data"


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    _write_parquet(tmp_path / "external" / "trading_days.parquet", {"date": ["20170103", "20170104"]})
    _write_parquet(data_root / "universe" / "ex2kamt" / "20170103.parquet", {"secu_code": [1, 2, 2]})
    _write_parquet(
        data_root / "trades_tbar" / "20170103.parquet",
        {
            "SecuCode": [1, 1, 1, 1, 1, 1, 1, 1],
            "MinBar": [925, 930, 930, 931, 932, 933, 935, 1500],
            "Price": [9.85, 10.1, 10.2, None, 10.5, 10.3, None, 10.85],
            "Side": [0, 0, 0, 0, 0, 0, 0, 0],
            "Volume": [10.0, 10.0, 5.0, 0.0, 30.0, 20.0, 0.0, 10.0],
            "vw_wait_time": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "isfirst": [True, True, False, True, True, True, True, True],
            "islast": [True, False, True, True, True, True, True, True],
            "No": [1, 99, 1, 3, 4, 5, 6, 7],
        },
    )
    _write_parquet(
        data_root / "daily_k" / "20170103.parquet",
        {
            "sec_code": ["000001.SZ", "000002.SZ"],
            "trading_day": ["2017-01-03", "2017-01-03"],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "shares": [100.0, 200.0],
            "amount": [1000.0, 2000.0],
            "limit_up": [11.0, 22.0],
            "limit_down": [9.0, 18.0],
            "preclose": [9.5, 20.0],
            "isSuspend": [False, False],
            "isST": [False, False],
            "adjfactor": [1.0, 1.0],
            "total_share": [1000.0, 2000.0],
            "float_share": [900.0, 1800.0],
            "free_share": [800.0, 1600.0],
            "list_date": ["19910403", "19910403"],
        },
    )
    return data_root


def _write_parquet(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(path)
