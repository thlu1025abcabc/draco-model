from __future__ import annotations

from draco_model.market.schema import KEY_COLUMNS


RESERVED_PUBLIC_NAMES = set(KEY_COLUMNS)


def validate_public_alias(alias: str, *, subject: str = "Alias") -> None:
    """Reject public output names reserved for internal payload columns."""
    if alias.startswith("__"):
        raise ValueError(f"{subject} must not start with '__'; this prefix is reserved for internal payload columns.")
    if alias in RESERVED_PUBLIC_NAMES:
        raise ValueError(f"{subject} must not be a key column: {alias!r}.")
