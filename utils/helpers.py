"""
Route helper utilities shared across blueprints.
"""
from __future__ import annotations


def float_or_none(value: str | None) -> float | None:
    """Parse a query-param string as float, returning None on failure."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def int_or(value, default: int) -> int:
    """Parse a query-param as int, returning default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
