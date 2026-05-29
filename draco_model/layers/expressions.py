from __future__ import annotations

import polars as pl


def sum_or_null(expr: pl.Expr) -> pl.Expr:
    """Sum non-null values, preserving null when the whole group is null."""
    return pl.when(expr.count() > 0).then(expr.sum()).otherwise(None)
