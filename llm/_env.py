"""Environment-variable parsing shared by every provider configuration."""

from __future__ import annotations

import math
import os


def env_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def env_optional_int(name: str, *, minimum: int) -> int | None:
    if os.getenv(name) is None:
        return None
    return env_int(name, minimum, minimum=minimum)


def env_float(name: str, default: float, *, minimum: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value
