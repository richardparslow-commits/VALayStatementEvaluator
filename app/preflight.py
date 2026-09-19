"""Decide whether the configured endpoint can serve the configured models.

Why this module exists: a run costs minutes and, on a bundle, a lot of tokens, and
the configuration that makes every one of its calls fail — a model id the account
cannot use, a key belonging to another provider — is detectable in one cheap
request before the run starts. Without that check the user learns about a dead
configuration from a failed run that took seven minutes to get there (and, before
the failure-reporting fix, told them the endpoint was "temporarily unavailable").

The policy is deliberately asymmetric. A preflight **blocks** only on evidence
that is unambiguous and about the models or the key, and it never blocks when the
probe could not tell: an unreachable host, a provider that serves completions
without listing models, a 5xx from someone else's outage. A gate that refuses to
start a working run is a worse failure than the one it prevents, so the ambiguous
outcomes are reported and the run proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable

from .config import Settings
from .llm import ModelProbe, probe_models

# A run may start. The endpoint answered and lists every configured model.
OK = "ok"
# A run may not start without a waiver: a call to this endpoint cannot succeed as
# configured, and no amount of retrying or waiting changes that.
BLOCKED = "blocked"
# Nothing could be established either way, so the run is allowed. Recorded (not
# silent) because the user is running without the check they think they have.
UNVERIFIED = "unverified"

ProbeFn = Callable[[str, str], ModelProbe]


@dataclass(frozen=True)
class Verdict:
    """What the preflight established, and what to do about it."""

    kind: str
    headline: str
    fix: str = ""
    # The HTTP status the probe saw, when it saw one. None means no response
    # arrived at all (DNS, connection refused, timeout).
    status: int | None = None
    # Configured models the endpoint does not list. Non-empty only when the model
    # list was available, so an empty tuple never means "nothing was wrong".
    missing: tuple[str, ...] = ()
    # Model names compared, and how many the endpoint listed. Both are evidence:
    # "checked 2 against 4 listed" reads very differently from "checked 0".
    checked: tuple[str, ...] = field(default_factory=tuple)
    listed: int = 0

    @property
    def blocks(self) -> bool:
        return self.kind == BLOCKED

    @property
    def checked_anything(self) -> bool:
        return bool(self.checked)


def signature(settings: Settings) -> str:
    """A stable id for "which endpoint + key + models is configured".

    The API key is hashed rather than embedded: this value is a session-state key
    and may end up in a log line, and a digest is enough to detect that the key
    changed. The base URL and model names are not secrets and stay readable, which
    makes a stale entry diagnosable rather than opaque.
    """
    key = settings.api_key or ""
    digest = sha256(key.encode("utf-8")).hexdigest()[:12] if key else "no-key"
    return "|".join(
        (
            (settings.base_url or "").strip().rstrip("/"),
            (settings.model_main or "").strip(),
            (settings.model_fast or "").strip(),
            digest,
        )
    )


def _quoted(names: tuple[str, ...]) -> str:
    """``a`` / ``a`` and ``b`` / ``a``, ``b`` and ``c`` — for a sentence."""
    quoted = [f"`{n}`" for n in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def _model_present(model: str, available: set[str]) -> bool:
    """True when the endpoint lists *model*, exactly or as a dated variant.

    Providers routinely serve `qwen3.7-max` as `qwen3.7-max-2025-04-16`, and the
    configured id is the one that works, so a listed id *extending* the configured
    one counts as present. The reverse does not: a configured id longer than
    anything listed is a different model, and blocking on it is the point.
    """
    return any(candidate == model or candidate.startswith(model) for candidate in available)


def check_endpoint(settings: Settings, *, probe: ProbeFn = probe_models) -> Verdict:
    """Probe the configured endpoint and judge whether a run can start.

    ``probe`` is a parameter so the policy can be tested without a socket, and so
    a caller can reuse a result it already has.
    """
    models = tuple(
        m.strip() for m in (settings.model_main, settings.model_fast) if m and m.strip()
    )
    result = probe(settings.base_url, settings.api_key)
    checked = models

    if not models:
        return Verdict(
            UNVERIFIED,
            headline="No model names are configured, so there is nothing to check.",
            fix=(
                "The run will use whatever default the provider applies. Set "
                "`LLM_MODEL_MAIN`/`LLM_MODEL_FAST` (or the sidebar fields) to have the "
                "check compare real ids."
            ),
            status=result.status,
            listed=len(result.models or ()),
        )

    if result.ok:
        available = result.models or set()
        missing = tuple(m for m in models if not _model_present(m, available))
        if missing:
            return Verdict(
                BLOCKED,
                headline=f"The endpoint does not offer {_quoted(missing)}.",
                fix=(
                    "Every call would be rejected, so the run was not started. "
                    "`COMPATIBILITY.md` lists the ids each provider serves — or click "
                    "**Test connection** in the sidebar to see the whole list."
                ),
                status=result.status,
                missing=missing,
                checked=checked,
                listed=len(available),
            )
        return Verdict(
            OK,
            headline=f"The endpoint offers every configured model ({len(available)} listed).",
            status=result.status,
            checked=checked,
            listed=len(available),
        )

    if result.status in (401, 403):
        return Verdict(
            BLOCKED,
            headline=f"The endpoint rejected this API key (HTTP {result.status}).",
            fix=(
                "A rejected key fails every call, so the run was not started. The key and "
                "the base URL must come from the same provider account — and on Perplexity's "
                "Router API the account also needs Router access (private preview). "
                "**Test connection** in the sidebar names the status and the provider's own "
                "message."
            ),
            status=result.status,
            checked=checked,
        )

    if result.status == 404:
        return Verdict(
            UNVERIFIED,
            headline=f"The endpoint has no `/models` route at this base URL (HTTP 404).",
            fix=(
                "Some OpenAI-compatible servers serve completions without listing models, so "
                "the run is allowed — but the model names cannot be checked. If the base URL "
                "is wrong, calls will fail for that reason instead."
            ),
            status=result.status,
            checked=checked,
        )

    if result.status is None:
        return Verdict(
            UNVERIFIED,
            headline="The preflight request got no response from the endpoint.",
            fix=(
                "The host or the network is the problem, not the key, and a single failed "
                "probe is not proof that calls will fail — the run is allowed. "
                f"`{result.error}`"
            ),
            status=None,
            checked=checked,
        )

    if result.status == 200:
        return Verdict(
            UNVERIFIED,
            headline="The endpoint answered but published no model ids.",
            fix="Nothing could be compared, so the run is allowed.",
            status=200,
            checked=checked,
        )

    return Verdict(
        UNVERIFIED,
        headline=f"The endpoint failed the preflight request (HTTP {result.status}).",
        fix=(
            "That is a provider-side failure and the run is allowed; if the provider is "
            "really down, the run's own error will say so."
        ),
        status=result.status,
        checked=checked,
    )


def verdict_from_session(value: Any) -> Verdict | None:
    """Read a Verdict back out of session state, or None when it is anything else.

    Session state is shared with other tabs and with anything that survived a
    reload, so the shape is checked rather than trusted.
    """
    return value if isinstance(value, Verdict) else None
