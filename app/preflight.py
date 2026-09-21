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

Two probes, and the second one is the reason this module has a second probe at all:
``GET /models`` is a *listing*, and a listing is not a promise. Perplexity's Router API
answers it with its ids and then refuses every completion — ``403 The Router API is
currently in limited preview`` (measured on an account without preview access), which
looks exactly like a healthy endpoint until something calls it. So once the listing has
not already blocked the run, the preflight asks for a short answer from each configured
model and judges that: a refusal blocks, a served call confirms (even one whose model
wrote no visible text — see ``ChatProbe.silent``), and anything ambiguous (no response,
a rate limit, a 5xx) leaves the listing's verdict alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

from .config import DEFAULT_BASE_URL, Settings
from .llm import ChatProbe, ModelProbe, probe_chat, probe_models

# A run may start. The endpoint answered and lists every configured model.
OK = "ok"
# A run may not start without a waiver: a call to this endpoint cannot succeed as
# configured, and no amount of retrying or waiting changes that.
BLOCKED = "blocked"
# Nothing could be established either way, so the run is allowed. Recorded (not
# silent) because the user is running without the check they think they have.
UNVERIFIED = "unverified"

ProbeFn = Callable[[str, str], ModelProbe]
#: ``(base_url, api_key, model) -> ChatProbe`` — the short chat call.
ChatProbeFn = Callable[[str, str, str], ChatProbe]

#: Statuses that mean *every* call will be refused as configured. A model listing that
#: came back with ids does not rule them out: that is the whole reason the chat probe
#: exists (see the module docstring), and a 404 on the completions path is just as
#: deterministic as a rejected key.
REFUSAL_STATUSES = (401, 403, 404)

#: How long a verdict may stand in for a fresh probe. **Test connection** in the
#: sidebar probes the same configuration the run gate probes seconds later, so the
#: second pair of requests adds nothing inside this window — but a key revoked or
#: an endpoint that died in between must be re-probed rather than assumed, so the
#: window is short.
VERDICT_REUSE_SECONDS = 300.0

#: Session-state key holding the last verdict (see :func:`remember_verdict`). One
#: entry only: the most recent configuration is the one a run will use.
VERDICT_SESSION_KEY = "endpoint_preflight_last_verdict"

# ----------------------------------------------------------- Vercel credentials
#
# Three unrelated credentials carry Vercel's name, and handing one to the other's
# endpoint is precisely the "a key belonging to another provider" failure this
# module exists to name before a run starts:
#
# * an **AI Gateway API key** (``vck_…``) authenticates the gateway's own
#   OpenAI-compatible API, and nothing else;
# * a Vercel **access token** (or the OIDC token a Function gets) is what the
#   Sandbox product takes — see DEPLOYMENT.md section 6 — and no LLM endpoint
#   wants it;
# * a provider key (``pplx-…``, ``sk-…``) is what the endpoint in ``base_url``
#   wants.
#
# The gateway also serves ids from *its own* catalog — ``owner/model``, e.g.
# ``moonshotai/kimi-k3``, ``alibaba/qwen3.7-flash``, ``perplexity/sonar-pro`` — so
# another provider's model names are not in it, which the missing-model verdict
# below says out loud when that is the endpoint in question.
VERCEL_GATEWAY_HOST = "ai-gateway.vercel.sh"
VERCEL_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
VERCEL_GATEWAY_KEY_PREFIX = "vck_"
GATEWAY_ID_NOTE = (
    "On the AI Gateway the ids come from its own catalog (`moonshotai/kimi-k3`, "
    "`alibaba/qwen3.7-flash`, `perplexity/sonar-pro`, …), not from another provider's "
    "model names."
)


def _is_vercel_gateway_url(base_url: str) -> bool:
    """Whether *base_url* addresses Vercel's AI Gateway."""
    host = (urlparse(base_url or "").hostname or "").lower()
    return host == VERCEL_GATEWAY_HOST or host.endswith(f".{VERCEL_GATEWAY_HOST}")


def _looks_like_vercel_gateway_key(api_key: str) -> bool:
    return (api_key or "").strip().startswith(VERCEL_GATEWAY_KEY_PREFIX)


def _credential_mismatch(settings: Settings) -> str:
    """A sentence naming a Vercel-credential mix-up, or "" when there is none.

    Shape only, never a substitute for the probe: a key and a URL can still be
    paired through a proxy that fronts the gateway, so this is used to make a
    *rejection* specific rather than to decide the verdict itself.
    """
    if _looks_like_vercel_gateway_key(settings.api_key) and not _is_vercel_gateway_url(
        settings.base_url
    ):
        return (
            f"This looks like a Vercel **AI Gateway** key (`{VERCEL_GATEWAY_KEY_PREFIX}…`), which is "
            f"only valid against the gateway (`{VERCEL_GATEWAY_BASE_URL}`) with that catalog's model "
            "ids. Either set `OPENAI_BASE_URL` there, or use the key this endpoint needs."
        )
    if _is_vercel_gateway_url(settings.base_url) and not _looks_like_vercel_gateway_key(
        settings.api_key
    ):
        return (
            "`OPENAI_BASE_URL` is Vercel's AI Gateway, which needs an **AI Gateway API key** "
            f"(`{VERCEL_GATEWAY_KEY_PREFIX}…`, from the project's AI Gateway → API Keys; a Vercel "
            "*access token*, which the Sandbox product takes, is a different credential again)."
        )
    return ""


#: Model ids that identify a **retired** configuration even when the base URL does not.
#: These are the settings this repo shipped between 2026-09-18 and 2026-09-19 (Vercel AI
#: Gateway era) and the Router-era default before that; a deployment still carrying them
#: is one that has not picked up the Agent-API defaults. Matched exactly, case- and
#: whitespace-insensitively — a `provider/` prefix or a suffix the endpoint may add
#: (``-2025-04-16``) is not a retired id, and flagging it would be crying wolf.
RETIRED_MODEL_IDS = frozenset(
    {
        # Vercel AI Gateway catalog ids this repo's defaults/tests referenced.
        "moonshotai/kimi-k3",
        "alibaba/qwen3.7-flash",
        "openai/gpt-4.1-nano",
        # Router-era default ids, before the Agent API pivot.
        "perplexity/glm-5",
        "perplexity/kimi-k",
    }
)


def _is_router_url(base_url: str) -> bool:
    """True when *base_url* addresses Perplexity's Router API — a path on its host.

    The host decides, not the word: ``openrouter.ai`` and any proxy URL containing
    "router" are other endpoints, and an advisory that fires on them is noise the
    user learns to ignore. ``PERPLEXITY_HOST`` match mirrors
    ``app.llm._is_perplexity_base_url``; the *path* is what separates the retired
    Router route from the supported Agent API on the same host.
    """
    from .llm import _is_perplexity_base_url  # local: keeps module import order stable

    return _is_perplexity_base_url(base_url) and "router" in urlparse(base_url or "").path.lower()


def _stripped_lower(value: str) -> str:
    return (value or "").strip().lower()


RETIRED_PROVIDER_KINDS = {
    "vercel_ai_gateway": "Vercel AI Gateway",
    "perplexity_router": "Perplexity Router API",
    "gateway_era_model_ids": "gateway-era model ids",
}


def retired_endpoint_kind(settings: Settings) -> str:
    """Which retired provider *settings* name — ``""`` when they name none.

    The machine-readable counterpart of :func:`retired_endpoint_kind`'s prose twin
    :func:`retired_endpoint_reason`, for audit fields and tests, and the reason text
    is derived from it — one set of signals, two representations, no drift. The kinds
    are exactly the three signals documented there, checked in the same order so the
    strongest signal wins.
    """
    if _is_vercel_gateway_url(settings.base_url):
        return "vercel_ai_gateway"
    if _is_router_url(settings.base_url):
        return "perplexity_router"
    ids = (
        _stripped_lower(settings.model_main),
        _stripped_lower(settings.model_fast),
    )
    if RETIRED_MODEL_IDS & set(ids):
        return "gateway_era_model_ids"
    return ""


def retired_endpoint_reason(settings: Settings) -> str:
    """Why the configuration is a retired provider, or "" when it is not.

    Retired means: this app worked against it once and the project has moved on, and
    every failure it still produces is one already paid for — the gateway's free-tier
    per-model rate limit turned a 329-chunk run into 315 rejections in under a minute
    (measured 2026-09-19), and the Router refuses completions outright without
    private-preview access. The failure is deterministic, so the place for the
    remedy is before the run, not in its error report.

    Three independent signals, any one of which is enough:

    * **Vercel AI Gateway base URL** — the strongest: the endpoint itself is the
      retired one, whatever models are configured.
    * **Perplexity Router path** — ``…/router/v1``; private preview, completions
      refused without entitlement.
    * **Gateway-era model ids on a non-gateway endpoint** — the ids identify the era
      even when the URL was corrected; that combination is at least half stale and
      never something this app's current defaults produce.

    Deliberately *not* here: ordinary Perplexity hosts (the supported default) and
    any other OpenAI-compatible endpoint the user has chosen — an advisory must not
    name a working setup as retired.
    """
    kind = retired_endpoint_kind(settings)
    if not kind:
        return ""
    if kind == "vercel_ai_gateway":
        return (
            "⚠️ The base URL is Vercel's **AI Gateway**, which this app has retired in "
            "favour of Perplexity's Agent API. The gateway's free tier is rate-limited "
            "per model — a large run was measured dying with 315 of 329 chunks rejected "
            "as `429` — so runs here are expected to fail. Set the base URL to "
            f"`{DEFAULT_BASE_URL}` with a `pplx-` key."
        )
    if kind == "perplexity_router":
        return (
            "⚠️ The base URL is Perplexity's **Router API**, a private preview this app "
            "has retired: the models route answers, but every completion is refused with "
            "`403 The Router API is currently in limited preview` unless the account has "
            f"preview entitlement. Set the base URL to `{DEFAULT_BASE_URL}`."
        )
    ids = (
        _stripped_lower(settings.model_main),
        _stripped_lower(settings.model_fast),
    )
    retired = sorted(RETIRED_MODEL_IDS & set(ids))
    return (
        "⚠️ "
        + ", ".join(f"`{m}`" for m in retired)
        + " is a retired model id (Vercel AI Gateway / Router era) — this app's default "
        f"models now live on Perplexity's Agent API. Set the base URL to "
        f"`{DEFAULT_BASE_URL}` and pick ids the endpoint lists, e.g. "
        "`perplexity/kimi-k3` (main) and `perplexity/glm-5.3-flash` (fast)."
    )


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
    # Set when the *configuration* names a provider this app has retired, whatever
    # the probes found — see :func:`retired_endpoint_kind` and
    # :func:`retired_endpoint_reason`. Stamped on every verdict
    # :func:`check_endpoint` produces, so a reused verdict carries it too and the
    # audit trail can record a run that started on a retired configuration anyway.
    retired: str = ""
    retired_kind: str = ""

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


def _router_note(settings: Settings) -> str:
    """The Router-API hint, when that is the endpoint being refused.

    This is the case the models probe cannot see: Perplexity's Router API publishes its
    ids and refuses completions until the account is granted preview access, so the
    provider's message is the actionable part and the remedy belongs beside it.
    """
    if "router" not in (settings.base_url or "").lower():
        return ""
    return (
        "Perplexity's **Router API** is a private preview: the models route answers, but "
        "calls are refused until the account is granted access (request it from "
        "api@perplexity.ai). Their standard API (`https://api.perplexity.ai`, `sonar` "
        "models) is a different endpoint that serves normal plans."
    )


def _verified_by_a_call(
    settings: Settings,
    models: tuple[str, ...],
    chat_probe: ChatProbeFn,
    allowed: Verdict,
) -> Verdict:
    """Strengthen a would-be-allowed verdict with one real call to each model.

    Only ever called when the models probe has *not* blocked the run, because that is
    the case a listing cannot settle: ids are published, and every completion is then
    refused. A refusal blocks (with the provider's own words, which is where the
    remedy is), an answer confirms, and anything ambiguous — no response, a rate limit,
    a 5xx — leaves *allowed* exactly as the listing left it.
    """
    outcomes = [
        (model, chat_probe(settings.base_url, settings.api_key, model))
        for model in dict.fromkeys(models)
    ]
    refused = [(model, probe) for model, probe in outcomes if probe.status in REFUSAL_STATUSES]
    answered = tuple(model for model, probe in outcomes if probe.ok)

    if refused:
        model, probe = refused[0]
        # Which sentence is true depends on whether a listing came back at all: with one,
        # the point is that a listing is not a promise; without one, the call is the only
        # evidence there was and it says the endpoint does not serve this model.
        lede = (
            "A listing is not a promise: the models route answered, and a real call is "
            "refused in the same way every call would be, so the run was not started."
            if allowed.listed
            else "A real call is refused in the same way every call in the run would be, "
            "so the run was not started."
        )
        fix = f"{lede} The provider's own words: {probe.error}"
        note = _router_note(settings) or _credential_mismatch(settings)
        if note:
            fix = f"{note} {fix}"
        return Verdict(
            BLOCKED,
            headline=f"The endpoint refused a real call to `{model}` (HTTP {probe.status}).",
            fix=fix,
            status=probe.status,
            checked=allowed.checked,
            listed=allowed.listed,
        )

    if answered:
        silent = tuple(model for model, probe in outcomes if probe.ok and probe.silent)
        if silent and len(silent) == len(answered):
            headline = (
                f"{allowed.headline} A real call to {_quoted(answered)} was served — the "
                "model produced no visible text, which is what a reasoning model can do "
                "when it spends the probe's budget thinking — and a served call is the "
                "evidence this check needs."
            )
        else:
            headline = (
                f"{allowed.headline} A real call to {_quoted(answered)} answered, so the "
                "configured key, endpoint and ids all work."
            )
        return replace(allowed, kind=OK, headline=headline)

    return allowed


def _stamp_retired(settings: Settings, verdict: Verdict) -> Verdict:
    """Carry the retired-provider finding on *verdict*, whatever the probes found.

    The advisory does not change the verdict — a healthy gateway stays ``ok``, and
    the run gate keeps its own policy — but the finding must travel with the
    verdict object so the gate's audit fields and the run buttons can report it,
    including when the verdict is a *reused* one that skipped today's probes.
    """
    kind = retired_endpoint_kind(settings)
    if not kind or verdict.retired_kind:
        return verdict
    return replace(verdict, retired_kind=kind, retired=retired_endpoint_reason(settings))


def check_endpoint(
    settings: Settings,
    *,
    probe: ProbeFn = probe_models,
    chat_probe: ChatProbeFn = probe_chat,
) -> Verdict:
    """Probe the configured endpoint and judge whether a run can start.

    ``probe`` and ``chat_probe`` are parameters so the policy can be tested without a
    socket, and so a caller can reuse a result it already has. The chat probe is the
    second half of the check because the first half is only a listing; see the module
    docstring for the endpoint that makes that difference matter.

    Every verdict this function produces also states whether the *configuration*
    names a provider this app has retired (:func:`retired_endpoint_kind`) — a
    finding the probes themselves cannot make, stamped so a reused verdict and the
    audit trail carry it without re-deriving it.
    """
    return _stamp_retired(
        settings, _probe_endpoint(settings, probe=probe, chat_probe=chat_probe)
    )


def _probe_endpoint(
    settings: Settings,
    *,
    probe: ProbeFn = probe_models,
    chat_probe: ChatProbeFn = probe_chat,
) -> Verdict:
    """The probe-and-judge half of :func:`check_endpoint`, without the retired stamp."""
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
            fix = (
                "Every call would be rejected, so the run was not started. "
                "`COMPATIBILITY.md` lists the ids each provider serves — or click "
                "**Test connection** in the sidebar to see the whole list."
            )
            if _is_vercel_gateway_url(settings.base_url):
                fix = f"{fix} {GATEWAY_ID_NOTE}"
            return Verdict(
                BLOCKED,
                headline=f"The endpoint does not offer {_quoted(missing)}.",
                fix=fix,
                status=result.status,
                missing=missing,
                checked=checked,
                listed=len(available),
            )
        return _verified_by_a_call(
            settings,
            checked,
            chat_probe,
            Verdict(
                OK,
                headline=f"The endpoint offers every configured model ({len(available)} listed).",
                status=result.status,
                checked=checked,
                listed=len(available),
            ),
        )

    if result.status in (401, 403):
        fix = (
            "A rejected key fails every call, so the run was not started. The key and "
            "the base URL must come from the same provider account — and on Perplexity's "
            "Router API the account also needs Router access (private preview). "
            "**Test connection** in the sidebar names the status and the provider's own "
            "message."
        )
        mismatch = _credential_mismatch(settings)
        if mismatch:
            fix = f"{mismatch} {fix}"
        return Verdict(
            BLOCKED,
            headline=f"The endpoint rejected this API key (HTTP {result.status}).",
            fix=fix,
            status=result.status,
            checked=checked,
        )

    if result.status == 404:
        # No listing route, so the short chat call is the only thing that can tell a
        # working endpoint from a wrong base URL — exactly what it is for.
        return _verified_by_a_call(
            settings,
            checked,
            chat_probe,
            Verdict(
                UNVERIFIED,
                headline=f"The endpoint has no `/models` route at this base URL (HTTP 404).",
                fix=(
                    "Some OpenAI-compatible servers serve completions without listing models, so "
                    "the run is allowed — but the model names cannot be checked. If the base URL "
                    "is wrong, calls will fail for that reason instead."
                ),
                status=result.status,
                checked=checked,
            ),
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
        return _verified_by_a_call(
            settings,
            checked,
            chat_probe,
            Verdict(
                UNVERIFIED,
                headline="The endpoint answered but published no model ids.",
                fix="Nothing could be compared, so the run is allowed.",
                status=200,
                checked=checked,
            ),
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


class VerdictStore(Protocol):
    """The slice of a session-state store these helpers use.

    ``st.session_state`` satisfies it, and so does a plain dict in tests. Reading
    through ``[]`` rather than ``get`` keeps this to plain methods: ``Mapping.get``
    is overloaded in typeshed, and an overloaded implementation does not satisfy a
    single-signature protocol member.
    """

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...


def remember_verdict(
    store: VerdictStore,
    settings: Settings,
    verdict: Verdict,
    *,
    now: float | None = None,
) -> None:
    """Record *verdict* as evidence a later check of the same configuration can reuse.

    Called by both ends that pay for probes — **Test connection** and the run gate —
    so the next check of the same configuration, seconds later, does not pay for them
    again. The verdict is keyed by :func:`signature`, so any change to the endpoint,
    key or model names makes it unreadable rather than misleading.
    """
    store[VERDICT_SESSION_KEY] = {
        "signature": signature(settings),
        "at": time.monotonic() if now is None else now,
        "verdict": verdict,
    }


def _reusable_entry(
    store: VerdictStore,
    settings: Settings,
    now: float | None,
) -> tuple[dict[str, Any], float] | None:
    """The stored entry for *settings* and its age, when it may stand in for a probe.

    ``None`` whenever the entry is missing, describes a different configuration, or
    is older than :data:`VERDICT_REUSE_SECONDS` — the caller then probes as usual.
    Session state is shared with other tabs and survives reruns, so the shape and
    the age are checked rather than trusted.
    """
    try:
        entry = store[VERDICT_SESSION_KEY]
    except KeyError:
        return None
    if not isinstance(entry, dict) or entry.get("signature") != signature(settings):
        return None
    at = entry.get("at")
    if not isinstance(at, (int, float)):
        return None
    age = (time.monotonic() if now is None else now) - at
    if age < 0 or age > VERDICT_REUSE_SECONDS:
        return None
    return entry, age


def reusable_verdict(
    store: VerdictStore,
    settings: Settings,
    *,
    now: float | None = None,
) -> Verdict | None:
    """The stored verdict for *settings*, if recent enough to stand in for a probe."""
    found = _reusable_entry(store, settings, now)
    return verdict_from_session(found[0].get("verdict")) if found else None


def verdict_age(
    store: VerdictStore,
    settings: Settings,
    *,
    now: float | None = None,
) -> float | None:
    """Seconds since the probe behind the verdict that would be reused, or ``None``.

    The companion to :func:`reusable_verdict`, for callers that report *why* no probe
    happened. Measured from the probe, never from a later read — see
    :data:`VERDICT_REUSE_SECONDS`.
    """
    found = _reusable_entry(store, settings, now)
    return found[1] if found else None
