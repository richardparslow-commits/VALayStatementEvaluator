"""Queue execution is synthetic-only until a separate real-data design is accepted."""
from __future__ import annotations

import os
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from .pilot import PilotBlocked

REFUSAL = "Queued processing is excluded from the controlled pilot. Use the reviewed in-process service."
P = ParamSpec("P")
T = TypeVar("T")


def synthetic_mode() -> bool:
    return os.getenv("VA_LSE_MODE", "synthetic").strip().lower() == "synthetic"


def require_synthetic_queue() -> None:
    # Reject unknown modes too. Never infer permission from a cached backend,
    # an injected worker/client, a flag or a queue reference.
    if not synthetic_mode():
        raise PilotBlocked(REFUSAL)


def synthetic_queue(function: Callable[P, T]) -> Callable[P, T]:
    """Check before effects and before a delayed response is returned to a caller."""
    @wraps(function)
    def checked(*args: P.args, **kwargs: P.kwargs) -> T:
        require_synthetic_queue()
        try:
            result = function(*args, **kwargs)
        except Exception:
            require_synthetic_queue()  # Fixed refusal supersedes a stale private error.
            raise
        require_synthetic_queue()
        return result
    return checked
