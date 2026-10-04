"""Durable case blobs remain synthetic-only until a separate design is accepted."""
from __future__ import annotations

import os
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from .pilot import PilotBlocked

REFUSAL = "Durable case storage is excluded from the controlled pilot. Use the reviewed session-only service."
P = ParamSpec("P")
T = TypeVar("T")


def synthetic_storage_mode() -> bool:
    return os.getenv("VA_LSE_MODE", "synthetic").strip().lower() == "synthetic"


def require_synthetic_storage() -> None:
    if not synthetic_storage_mode():
        raise PilotBlocked(REFUSAL) from None


def synthetic_storage(function: Callable[P, T]) -> Callable[P, T]:
    """Refuse before effects and before returning a delayed response or error."""
    @wraps(function)
    def checked(*args: P.args, **kwargs: P.kwargs) -> T:
        require_synthetic_storage()
        try:
            result = function(*args, **kwargs)
        except Exception:
            require_synthetic_storage()
            raise
        require_synthetic_storage()
        return result
    return checked
