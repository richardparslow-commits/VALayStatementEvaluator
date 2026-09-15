"""Distributed cache for VA reference data (condition topics, rating tables).

Provides a two-tier cache:
  1. **Shared** (Upstash Redis / Vercel KV via HTTP REST) — shared across all
     Streamlit instances so every pod hits the same cached data.  Enabled when
     ``VA_LSE_SHARED_CACHE_URL`` and ``VA_LSE_SHARED_CACHE_TOKEN`` are set.
  2. **Local LRU** (process-local ``OrderedDict``) — always available as a
     fallback; absorbs the common hot-key traffic when a shared cache is
     configured but also handles the single-instance case gracefully.

Hit-rate tracking is built in so the health endpoint and observability
dashboards can surface cache effectiveness without external tooling.

Upstash REST API docs: https://upstash.com/docs/rest/api/get-key
All HTTP calls use ``urllib.request`` (stdlib) — no ``redis`` package needed.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from . import config

logger = logging.getLogger("app.shared_cache")


# ---------------------------------------------------------------------------
# Hit-rate / stats
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Mutable hit/miss counters for a cache instance."""

    hits: int = 0
    misses: int = 0
    errors: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Cache protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CacheBackend(Protocol):
    """Minimal interface both backends implement."""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    @property
    def is_shared(self) -> bool: ...
    @property
    def stats(self) -> CacheStats: ...
    def health(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Local LRU backend
# ---------------------------------------------------------------------------

class LocalLRUCache:
    """Process-local bounded LRU cache.  Always available."""

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, str] = OrderedDict()
        self._expires: dict[str, float] = {}
        self._stats = CacheStats()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            self._evict_expired()
            if key in self._data:
                self._data.move_to_end(key)
                self._stats.hits += 1
                return self._data[key]
            self._stats.misses += 1
            return None

    def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        with self._lock:
            self._evict_expired()
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            self._stats.sets += 1
            if ttl_seconds is not None:
                self._expires[key] = time.monotonic() + ttl_seconds
            # Evict oldest when over capacity.
            while len(self._data) > self._maxsize:
                evicted_key, _ = self._data.popitem(last=False)
                self._expires.pop(evicted_key, None)
                self._stats.evictions += 1

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._expires.pop(key, None)

    def exists(self, key: str) -> bool:
        with self._lock:
            self._evict_expired()
            return key in self._data

    @property
    def is_shared(self) -> bool:
        return False

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def health(self) -> dict[str, Any]:
        return {
            "backend": "local_lru",
            "is_shared": False,
            "size": len(self._data),
            "maxsize": self._maxsize,
            **self._stats.snapshot(),
        }

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, exp in self._expires.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)
            self._expires.pop(k, None)


# ---------------------------------------------------------------------------
# Upstash Redis REST backend
# ---------------------------------------------------------------------------

class UpstashRedisCache:
    """Upstash Redis / Vercel KV via HTTP REST API (no ``redis`` package).

    All calls go through ``urllib.request`` with a short timeout so a slow or
    unreachable backend never blocks the Streamlit thread.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds
        self._stats = CacheStats()
        self._last_ping_ok: bool = False
        self._last_ping_at: float = 0.0
        self._lock = threading.Lock()

    # -- low-level HTTP ------------------------------------------------------

    def _request(self, path: str, body: dict[str, Any]) -> Any:
        """POST to the Upstash REST endpoint and return the parsed response."""
        url = f"{self._url}/{path}"
        payload = json.dumps(body).encode("utf-8")
        auth = base64.b64encode(self._token.encode("utf-8")).decode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                return json.loads(raw)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self._stats.errors += 1
            logger.warning(
                "upstash request failed path=%s error=%s",
                path, f"{type(exc).__name__}: {exc}",
            )
            raise

    # -- public interface ----------------------------------------------------

    def get(self, key: str) -> str | None:
        try:
            result = self._request("get", {"key": key})
            # Upstash returns {"result": <value>} or {"result": null}
            value = result.get("result") if isinstance(result, dict) else None
            if value is None:
                self._stats.misses += 1
                return None
            self._stats.hits += 1
            # Upstash may return bytes or str depending on how it was SET.
            return value if isinstance(value, str) else str(value)
        except Exception:  # noqa: BLE001
            self._stats.misses += 1
            return None

    def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        body: dict[str, Any] = {"key": key, "value": value}
        if ttl_seconds is not None:
            body["ex"] = ttl_seconds
        try:
            self._request("set", body)
            self._stats.sets += 1
        except Exception:  # noqa: BLE001
            pass  # best-effort — local LRU catches the miss on next read

    def delete(self, key: str) -> None:
        try:
            self._request("del", {"key": key})
        except Exception:  # noqa: BLE001
            pass

    def exists(self, key: str) -> bool:
        try:
            result = self._request("exists", {"key": key})
            val = result.get("result") if isinstance(result, dict) else 0
            return bool(val)
        except Exception:  # noqa: BLE001
            return False

    @property
    def is_shared(self) -> bool:
        return True

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def ping(self) -> bool:
        """Lightweight connectivity check (cached for 30 s)."""
        now = time.monotonic()
        with self._lock:
            if (now - self._last_ping_at) < 30.0 and self._last_ping_at != 0.0:
                return self._last_ping_ok
        try:
            self._request("ping", {})
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
        with self._lock:
            self._last_ping_ok = ok
            self._last_ping_at = now
        return ok

    def health(self) -> dict[str, Any]:
        reachable = self.ping()
        return {
            "backend": "upstash_redis",
            "is_shared": True,
            "reachable": reachable,
            "url": self._url,
            **self._stats.snapshot(),
        }


# ---------------------------------------------------------------------------
# Tiered cache (shared + local LRU)
# ---------------------------------------------------------------------------

class TieredCache:
    """Two-tier cache: shared (Upstash) → local LRU fallback.

    Reads check shared first, then local.  Writes populate both tiers.
    If the shared backend is unreachable, the local LRU absorbs traffic
    transparently so the app never hangs on cache I/O.
    """

    def __init__(self, shared: UpstashRedisCache | None, local: LocalLRUCache) -> None:
        self._shared = shared
        self._local = local
        self._shared_stats = CacheStats()
        self._local_stats = local.stats  # reference to same object

    def get(self, key: str) -> str | None:
        # Try shared first.
        if self._shared is not None:
            try:
                value = self._shared.get(key)
                if value is not None:
                    # Back-fill local so hot keys are fast on next rerun.
                    self._local.set(key, value)
                    self._shared_stats.hits += 1
                    return value
            except Exception:  # noqa: BLE001
                self._shared_stats.errors += 1
        # Fallback to local.
        return self._local.get(key)

    def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        self._local.set(key, value, ttl_seconds=ttl_seconds)
        if self._shared is not None:
            try:
                self._shared.set(key, value, ttl_seconds=ttl_seconds)
                self._shared_stats.sets += 1
            except Exception:  # noqa: BLE001
                self._shared_stats.errors += 1

    def delete(self, key: str) -> None:
        self._local.delete(key)
        if self._shared is not None:
            try:
                self._shared.delete(key)
            except Exception:  # noqa: BLE001
                pass

    def exists(self, key: str) -> bool:
        if self._shared is not None:
            try:
                if self._shared.exists(key):
                    return True
            except Exception:  # noqa: BLE001
                pass
        return self._local.exists(key)

    @property
    def is_shared(self) -> bool:
        return self._shared is not None

    @property
    def stats(self) -> CacheStats:
        """Combined stats (shared + local)."""
        combined = CacheStats(
            hits=self._shared_stats.hits + self._local_stats.hits,
            misses=self._shared_stats.misses + self._local_stats.misses,
            errors=self._shared_stats.errors + self._local_stats.errors,
            sets=self._shared_stats.sets + self._local_stats.sets,
            evictions=self._local_stats.evictions,
        )
        return combined

    @property
    def local_stats(self) -> CacheStats:
        return self._local_stats

    @property
    def shared_stats(self) -> CacheStats:
        return self._shared_stats

    def health(self) -> dict[str, Any]:
        local_health = self._local.health()
        if self._shared is not None:
            shared_health = self._shared.health()
            return {
                "tiered": True,
                "shared": shared_health,
                "local": local_health,
                "combined": self._stats_snapshot(),
            }
        return {
            "tiered": False,
            "local": local_health,
            "combined": self._stats_snapshot(),
        }

    def _stats_snapshot(self) -> dict[str, Any]:
        return self.stats.snapshot()


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_cache: TieredCache | None = None
_lock = threading.Lock()


def get_cache() -> TieredCache:
    """Return (and lazily create) the process-global tiered cache."""
    global _cache  # noqa: PLW0603
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        local = LocalLRUCache(maxsize=config.SHARED_CACHE_LOCAL_MAXSIZE)
        shared: UpstashRedisCache | None = None
        url = config.SHARED_CACHE_URL
        token = config.SHARED_CACHE_TOKEN
        if url and token:
            shared = UpstashRedisCache(
                url=url,
                token=token,
                timeout_seconds=config.SHARED_CACHE_TIMEOUT_SECONDS,
            )
            logger.info(
                "shared cache enabled url=%s timeout=%.1fs",
                url,
                config.SHARED_CACHE_TIMEOUT_SECONDS,
            )
        else:
            logger.info("shared cache disabled; using local LRU only")
        _cache = TieredCache(shared=shared, local=local)
        return _cache


def reset_cache_for_tests() -> None:
    """Reset the singleton so tests get a fresh cache."""
    global _cache  # noqa: PLW0603
    with _lock:
        _cache = None
