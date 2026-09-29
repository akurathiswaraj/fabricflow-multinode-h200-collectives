"""Human-readable byte-size parsing shared by the CLIs."""

from __future__ import annotations

import re


_UNITS = {
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
}


def parse_size(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("size must be positive")
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\s*", value)
    if not match:
        raise ValueError(f"invalid size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    if unit not in _UNITS:
        raise ValueError(f"unsupported size unit: {unit}")
    result = int(number * _UNITS[unit])
    if result <= 0:
        raise ValueError("size must be positive")
    return result


def format_size(byte_count: int) -> str:
    for label, divisor in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if byte_count >= divisor and byte_count % divisor == 0:
            return f"{byte_count // divisor}{label}"
    return f"{byte_count}B"

