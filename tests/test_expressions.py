from __future__ import annotations

import polars as pl

from draco_model.layers.expressions import sum_or_null


def test_sum_or_null_preserves_all_null_groups() -> None:
    frame = pl.DataFrame(
        {
            "group": ["all_null", "all_null", "mixed", "mixed"],
            "value": [None, None, None, 2.0],
        }
    )

    result = (
        frame.lazy()
        .group_by("group")
        .agg(sum_or_null(pl.col("value")).alias("total"))
        .sort("group")
        .collect()
    )

    assert result.to_dict(as_series=False) == {
        "group": ["all_null", "mixed"],
        "total": [None, 2.0],
    }
