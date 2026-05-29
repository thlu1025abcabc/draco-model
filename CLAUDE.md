---
description: Behavioral guidelines to reduce common LLM coding mistakes. Use when writing, reviewing, or refactoring code to avoid overcomplication, make surgical changes, surface assumptions, and define verifiable success criteria.
alwaysApply: true
---

# Karpathy behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Draco Lite project conventions

- Metric `side` parameters should use exactly three semantic options: `"buy"`, `"sell"`, or `null`/omitted.
- Do not introduce string aliases like `"raw"`, `"all"`, or `"both"` for side. `null`/omitted side means no side filter.

## Draco Model data conventions

- `trades_tbar` is minute-level aggregated tick-trade data. Each row is one stock, one minute, one price, and one side, with `Volume` and `No` aggregated for that bucket.
- `quotes_tbar` is minute-level aggregated tick-order data. Each row is one stock, one minute, one price, and one side, with order volume and record count aggregated for that bucket.
- `cancels_tbar` is minute-level aggregated tick-cancel data. Each row is one stock, one minute, one price, and one side, with cancel volume and record count aggregated for that bucket.
- `snapshot_tbar` is minute-level snapshot data with averaged bid/ask price and volume levels 1 through 10, such as `AskPrice1`-`AskPrice10`, `BidPrice1`-`BidPrice10`, `AskVolume1`-`AskVolume10`, and `BidVolume1`-`BidVolume10`.
- `daily_k` is daily data with fields such as `open`, `high`, `low`, `close`, `preclose`, `volume`, and `amount`.
- `universe/ex2kamt` is the stock universe. It contains stock identifiers and daily reference fields such as `preclose`, `close`, and `adjfactor`.
- `external/trading_days.parquet` is the trading calendar source.
- Source files commonly use vendor names such as `SecuCode`, `MinBar`, `Price`, `Amount`, `Volume`, `No`, `Side`, `isfirst`, `islast`, and `trading_day`; code may normalize these to `secu_code`, `minute`, `price`, `amount`, `volume`, `no`, `side`, `is_first`, `is_last`, and `date`.
- `No`/`no` means number of records in that minute/price/side bucket. It is not price order. Use `is_first` and `is_last` for first/last semantics; assume at most one `is_first=True` and one `is_last=True` per stock-minute when deriving open/close style values.
- Do not automatically add an intraday grid to raw minute sources. Grid alignment should be an explicit layer/operation when needed.
- `close.fill("state")` semantics: first forward-fill `close` over each `(secu_code, date)` series, then fill remaining nulls with `daily_k.preclose`. Do not conflate this rule with the separate minute `preclose` field design.

## Draco Model documentation conventions

- After changing user-visible behavior, public APIs, examples, or data semantics, check whether `README.md` needs to be updated in the same change.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
