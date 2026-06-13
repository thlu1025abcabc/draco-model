# Data Sources

`Source(source, lookback_days=1)` creates a raw source frame. Source frames do not automatically add an intraday grid.

```python
from draco_model.layers import Source

raw = Source("trades_tbar")
raw_5d = Source("trades_tbar", lookback_days=5)
```

## Directory Layout

```text
data/
  trades_tbar/
    20170103.parquet
  quotes_tbar/
    20170103.parquet
  cancels_tbar/
    20170103.parquet
  snapshot_tbar/
    20170103.parquet
  daily_k/
    20170103.parquet
  universe/
    ex2kamt/
      20170103.parquet
external/
  trading_days.parquet
```

## Normalized Columns

Source scan normalizes common vendor column names:

| Vendor column | Normalized column |
|---|---|
| `SecuCode` | `secu_code` |
| `MinBar` | `minute` |
| `Price` | `price` |
| `Amount` | `amount` |
| `Volume` | `volume` |
| `No` | `no` |
| `Side` | `side` |
| `isfirst` | `is_first` |
| `islast` | `is_last` |
| `trading_day` | `date` |

If `sec_code` exists and `secu_code` does not, `secu_code` is derived from the first six characters of `sec_code`.

## Fixed Source Contracts

Known sources use fixed schemas for stable planning:

- `trades_tbar` / `cancels_tbar`: `secu_code`, `minute`, `price`, `side`, `volume`, `vw_wait_time`, `is_first`, `is_last`, `no`, `date`.
- `quotes_tbar`: `secu_code`, `minute`, `price`, `side`, `volume`, `is_first`, `is_last`, `no`, `date`.
- `daily_k`: `sec_code`, `date`, `open`, `high`, `low`, `close`, `shares`, `amount`, `limit_up`, `limit_down`, `preclose`, `isSuspend`, `isST`, `adjfactor`, `total_share`, `float_share`, `free_share`, `list_date`, `secu_code`.
- `snapshot_tbar`: `AskPrice1`-`AskPrice10`, `BidPrice1`-`BidPrice10`, `AskVolume1`-`AskVolume10`, `BidVolume1`-`BidVolume10`, `aVOI1`-`aVOI5`, `secu_code`, `minute`, `date`.
- `universe/ex2kamt`: `sec_code`, `preclose`, `close`, `adjfactor`, `secu_code`, `date`.

Extra normalized columns are intentionally dropped by the `Source` executor. Missing fixed-contract columns raise a clear `ValueError` that includes the source, date, missing columns, and actual normalized columns.

Unknown sources fall back to `scan().collect_schema()` and use their actual normalized columns. Every source must expose identity keys: either registered fixed keys, or the standard key columns (`date`, `secu_code`, `minute`) or (`date`, `secu_code`) present in the normalized schema. Sources without identity keys raise a `ValueError`.

The trading calendar file `external/trading_days.parquet` must contain a `date` or `trading_day` column.
