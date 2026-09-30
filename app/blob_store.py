"""Blob storage for job payloads that are too large for the queue.

Why this exists
---------------
``app/job_queue.py`` moves a run from the web pod to a worker. The job's inputs
include the *extracted record text* (``ExtractedDocument``), which the web pod
produces because it is the only place that parses PDFs/DOCX. For a large bundle
that text is tens of megabytes. Pushing it through the queue works but has real
costs:

* Redis holds every in-flight payload in memory (the reference StatefulSet ships
  with a 256 MB ``maxmemory`` and refuses writes when full).
* Upstash REST bills per request and per byte, so a 15 MB payload is an expensive
  round trip on both ends.
* ``VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES`` has to be generous enough for the worst
  case, which means the cap stops protecting Redis from the pathological case.

So large jobs put their text in a blob store instead, and the queue carries a
small reference. Small jobs stay inline — an extra round trip for a 40 KB job
would be pure overhead.

Backends
--------
* :class:`FilesystemBlobStore` — stdlib only. Needs a volume shared between the
  web tier and the workers (a Kubernetes ``ReadWriteMany`` PVC, or a named volume
  in Docker Compose). **This is the one thing that can silently misconfigure:** a
  per-pod ``emptyDir`` looks fine on the writing pod and fails on the worker, so
  a missing blob raises a message that names the sharing requirement.
* :class:`S3BlobStore` — any S3-compatible endpoint (AWS S3, Cloudflare R2,
  MinIO, DigitalOcean Spaces). Requires ``boto3``, which is deliberately *not* in
  ``requirements.txt``: install ``requirements-s3.txt`` when you use it.

Keys are content-addressed (a SHA-256 of the stored bytes), so re-uploading the
same record bundle reuses one blob and two users running the same file do not
double the storage.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger("app.blob_store")

# Content-addressed keys are hex digests we generate ourselves; validating the
# shape on every read keeps a hostile or corrupt reference from escaping the
# blob root (``../`` in a key would otherwise read arbitrary files).
_KEY_RE = re.compile(r"^blobs/[0-9a-f]{2}/[0-9a-f]{64}\.json$")

# Retention starts again on each put because several jobs can share a blob.
# Cleanup also needs an independent schedule; writes alone do not bound retention.
_DEFAULT_SWEEP_AGE_SECONDS = 24 * 3600
_SWEEP_MIN_INTERVAL_SECONDS = 300.0
_STORAGE_LOCK_TIMEOUT_SECONDS = 5.0
_SWEEP_LOCK_BUDGET_SECONDS = 5.0


class BlobStoreError(RuntimeError):
    """Raised when a blob cannot be written, read, or is missing when required."""


class BlobNotFound(BlobStoreError):
    """Raised when a referenced blob is absent from a shared store.

    Carries the sharing hint because the overwhelmingly common cause is a blob
    store that is not actually shared between the writing pod and the worker.
    """


def content_key(data: bytes) -> str:
    """Return the content-addressed key for ``data`` (``blobs/ab/<sha>.json``)."""
    digest = hashlib.sha256(data).hexdigest()
    return f"blobs/{digest[:2]}/{digest}.json"


@dataclass
class BlobRef:
    """A pointer to a job's documents in the blob store.

    ``sha256`` travels with the reference so a reader can verify it received
    exactly what the writer stored — a truncated write on a network filesystem
    should be a loud failure, not a silently short record set.
    """

    key: str
    size: int
    sha256: str
    backend: str

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "size": self.size, "sha256": self.sha256, "backend": self.backend}

    @classmethod
    def from_json(cls, raw: Any) -> BlobRef | None:
        if not isinstance(raw, dict):
            return None
        key = raw.get("key")
        if not isinstance(key, str) or not key:
            return None
        return cls(
            key=key,
            size=int(raw.get("size") or 0),
            sha256=str(raw.get("sha256") or ""),
            backend=str(raw.get("backend") or ""),
        )


class BlobStore:
    """Storage contract for job documents.

    Not a ``typing.Protocol`` so the backends can share the reference-validation
    and verification logic that must behave identically everywhere.
    """

    name = "base"
    is_shared = False

    def put(self, data: bytes) -> BlobRef:
        raise NotImplementedError

    def get(self, ref: BlobRef) -> bytes:
        """Return the stored bytes, raising :class:`BlobNotFound` if absent."""
        raise NotImplementedError

    def delete(self, ref: BlobRef) -> None:
        raise NotImplementedError

    @contextmanager
    def submission_guard(self, ref: BlobRef) -> Iterator[None]:
        """Refuse submissions referencing absent/corrupt documents."""
        self.get(ref)
        yield

    def sweep(self, *, max_age_seconds: int | None = None) -> int:
        """Delete blobs older than the age limit. Returns the count removed."""
        return 0

    def ping(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {"backend": self.name, "is_shared": self.is_shared, "reachable": self.ping()}

    # -- shared helpers -----------------------------------------------------
    def _check_size(self, data: bytes) -> None:
        """Refuse a blob larger than the job-size ceiling.

        Reuses ``VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES`` on purpose: moving documents
        into a blob store must not become a way around the cap that protects the
        queue, Redis, and the blob volume from one pathological record bundle.
        """
        if len(data) > config.JOB_QUEUE_MAX_PAYLOAD_BYTES:
            raise BlobStoreError(
                f"job documents are {len(data):,} bytes, over the "
                f"{config.JOB_QUEUE_MAX_PAYLOAD_BYTES:,}-byte "
                "VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES limit. Split the record set."
            )

    def _verify(self, ref: BlobRef, data: bytes) -> bytes:
        """Check a read against the reference that asked for it."""
        if ref.size and len(data) != ref.size:
            raise BlobStoreError(
                f"blob {ref.key} is {len(data):,} bytes but the job reference says "
                f"{ref.size:,} — the store is probably not shared between the web pod "
                "and the worker (see DEPLOYMENT.md → Pattern C)"
            )
        if ref.sha256:
            actual = hashlib.sha256(data).hexdigest()
            if actual != ref.sha256:
                raise BlobStoreError(
                    f"blob {ref.key} failed its integrity check (expected sha256 "
                    f"{ref.sha256[:12]}…, got {actual[:12]}…)"
                )
        return data


class FilesystemBlobStore(BlobStore):
    """Blob store on a shared filesystem (stdlib only).

    Requires the directory to be visible to the web tier *and* every worker —
    a ``ReadWriteMany`` PVC in Kubernetes, a named volume in Docker Compose.
    """

    name = "filesystem"
    # A filesystem store is only shared if the volume is; the operator declares
    # that by configuring it, so report it as shared and let a missing blob
    # surface the mistake with a specific message.
    is_shared = True

    def __init__(self, root: str | Path, *, sweep_age_seconds: int = _DEFAULT_SWEEP_AGE_SECONDS,
                 retained_keys: Callable[[], set[str]] | None = None) -> None:
        if sweep_age_seconds <= 0:
            raise BlobStoreError("blob retention age must be positive")
        self._root = Path(root).expanduser()
        self._sweep_age = sweep_age_seconds
        self._retained_keys = retained_keys or (lambda: set())
        self._lock = threading.Lock()
        # ``-inf`` rather than 0.0: this is compared against ``time.monotonic()``,
        # which counts from an arbitrary origin — on a freshly booted host (a CI
        # runner, a just-started container) 0.0 is *not* "long ago", so the first
        # put would skip its sweep and the next one would land inside the interval.
        self._last_sweep = float("-inf")

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, key: str) -> Path:
        if not _KEY_RE.match(key):
            raise BlobStoreError(f"refusing to use an unrecognized blob key: {key!r}")
        path = self._root / key
        if (self._root / "blobs").is_symlink() or path.parent.is_symlink() or path.is_symlink():
            raise BlobStoreError("refusing a symlink in the filesystem blob namespace")
        return path

    @contextmanager
    def _storage_lock(self) -> Iterator[None]:
        """Serialize writers and cleanup across processes sharing a POSIX volume.

        Keep the lock inode permanently: unlinking it could let a second process
        lock a different inode. Closing the descriptor releases crashed writers'
        locks. The deployed volume must support cross-pod advisory flock.
        """
        try:
            import fcntl
        except ImportError as exc:
            raise BlobStoreError("filesystem blobs require POSIX advisory file locking") from exc
        self._root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._root / ".blob-store.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+b") as handle:
            deadline = time.monotonic() + _STORAGE_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise BlobStoreError("filesystem blob lock timed out") from exc
                    time.sleep(0.01)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def put(self, data: bytes) -> BlobRef:
        self._check_size(data)
        key = content_key(data)
        path = self._path_for(key)
        digest = hashlib.sha256(data).hexdigest()
        tmp_path: Path | None = None
        try:
            with self._storage_lock():
                path.parent.mkdir(parents=True, exist_ok=True)
                # Rewrite even reused keys: renew retention and repair an existing
                # corrupt blob. Cleanup cannot unlink a refreshed file using an
                # old stat result while this shared lock is held.
                with tempfile.NamedTemporaryFile(
                    dir=str(path.parent), prefix=".tmp-", delete=False
                ) as handle:
                    tmp_path = Path(handle.name)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, path)
        except OSError as exc:
            raise BlobStoreError(f"could not write blob {key}: {type(exc).__name__}: {exc}") from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("could not remove an interrupted blob write; scheduled cleanup is required")
        self._maybe_sweep()
        return BlobRef(key=key, size=len(data), sha256=digest, backend=self.name)

    def get(self, ref: BlobRef) -> bytes:
        path = self._path_for(ref.key)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise BlobNotFound(
                f"blob {ref.key} is not in the blob store at {self._root}. The store must "
                "be shared between the web pod that wrote it and the worker reading it "
                "(ReadWriteMany PVC — see DEPLOYMENT.md → Pattern C)."
            ) from exc
        except OSError as exc:
            raise BlobStoreError(f"could not read blob {ref.key}: {type(exc).__name__}: {exc}") from exc
        return self._verify(ref, data)

    @contextmanager
    def submission_guard(self, ref: BlobRef) -> Iterator[None]:
        with self._storage_lock():
            self.get(ref)  # Verify before a retained wire payload is admitted.
            os.utime(self._path_for(ref.key), None)
            yield

    def delete(self, ref: BlobRef) -> None:
        try:
            path = self._path_for(ref.key)
            with self._storage_lock():
                path.unlink(missing_ok=True)
        except (OSError, BlobStoreError) as exc:
            logger.warning("could not delete blob %s: %s", ref.key, exc)

    def _maybe_sweep(self) -> None:
        """Best-effort TTL sweep, rate-limited to one per few minutes per process."""
        now = time.monotonic()
        with self._lock:
            if (now - self._last_sweep) < _SWEEP_MIN_INTERVAL_SECONDS:
                return
            self._last_sweep = now
        try:
            self.sweep()
        except Exception as exc:  # noqa: BLE001 - housekeeping is never fatal
            logger.warning("blob sweep failed: %s", exc)

    def sweep(self, *, max_age_seconds: int | None = None, dry_run: bool = False,
              retained_keys: Callable[[], set[str]] | None = None) -> int:
        limit = self._sweep_age if max_age_seconds is None else max_age_seconds
        if limit <= 0:
            raise BlobStoreError("blob retention age must be positive")
        cutoff = time.time() - limit
        removed = 0
        if not self._root.exists():
            return 0
        # Capture the cutoff BEFORE inventory, without holding the writer lock.
        # Every concurrently admitted blob is renewed under that lock, so its
        # mtime is newer than this cutoff even if SCAN misses the new input key.
        protected = (retained_keys or self._retained_keys)()
        try:
            with self._storage_lock():
                deadline = time.monotonic() + _SWEEP_LOCK_BUDGET_SECONDS
                def check_budget() -> None:
                    if time.monotonic() >= deadline:
                        raise BlobStoreError("filesystem blob cleanup exceeded its lock budget")
                # Crash-interrupted writes contain record text too. Expire their
                # temporary files on the same schedule, without touching fresh ones.
                namespace = self._root / "blobs"
                if namespace.is_symlink():
                    raise BlobStoreError("refusing a symlink in the filesystem blob namespace")
                # Path.glob suppresses directory scan errors on some Python
                # versions. Explicit scans must report an unreadable mount.
                paths: list[Path] = []
                shards: list[Path] = []
                try:
                    with os.scandir(namespace) as entries:
                        for entry in entries:
                            check_budget()
                            if not re.fullmatch(r"[0-9a-f]{2}", entry.name):
                                continue
                            if entry.is_symlink():
                                raise BlobStoreError("refusing a symlink in the filesystem blob namespace")
                            if not entry.is_dir(follow_symlinks=False):
                                continue
                            shard = Path(entry.path)
                            shards.append(shard)
                            with os.scandir(shard) as files:
                                for item in files:
                                    check_budget()
                                    if item.is_file(follow_symlinks=False) and (
                                        _KEY_RE.fullmatch(f"blobs/{entry.name}/{item.name}")
                                        or item.name.startswith(".tmp-")
                                    ):
                                        paths.append(Path(item.path))
                except FileNotFoundError:
                    if namespace.exists():
                        raise  # A disappearing shard is not a successful pass.
                for path in paths:
                    check_budget()
                    if path.relative_to(self._root).as_posix() in protected:
                        continue
                    if path.stat().st_mtime < cutoff:
                        if not dry_run:
                            path.unlink()
                        removed += 1
                if not dry_run:
                    for shard in shards:
                        check_budget()
                        try:
                            shard.rmdir()
                        except OSError:
                            pass  # Nonempty directories do not contain expired data.
        except OSError as exc:
            raise BlobStoreError("filesystem blob cleanup failed") from exc
        if removed:
            logger.info("blob sweep %s %d expired file(s)", "found" if dry_run else "removed", removed)
        return removed

    def ping(self) -> bool:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            return os.access(self._root, os.W_OK | os.R_OK)
        except OSError:
            return False

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "is_shared": self.is_shared,
            "reachable": self.ping(),
            "root": str(self._root),
        }


class S3BlobStore(BlobStore):
    """Blob store on any S3-compatible endpoint (requires ``boto3``).

    Install ``requirements-s3.txt`` to use this. Works with AWS S3, Cloudflare R2,
    MinIO, and DigitalOcean Spaces — anything that speaks the S3 API.
    """

    name = "s3"
    is_shared = True

    def __init__(self, bucket: str, *, prefix: str = "", endpoint_url: str = "") -> None:
        try:
            import boto3  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:  # pragma: no cover - depends on env
            raise BlobStoreError(
                "boto3 is not installed; pip install -r requirements-s3.txt (or use the "
                "filesystem blob store via VA_LSE_BLOB_DIR)"
            ) from exc
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client: Any = boto3.client("s3", endpoint_url=endpoint_url or None)

    def _object_key(self, key: str) -> str:
        if not _KEY_RE.match(key):
            raise BlobStoreError(f"refusing to use an unrecognized blob key: {key!r}")
        return f"{self._prefix}/{key}" if self._prefix else key

    def put(self, data: bytes) -> BlobRef:
        self._check_size(data)
        key = content_key(data)
        digest = hashlib.sha256(data).hexdigest()
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._object_key(key),
                Body=data,
                ContentType="application/json",
            )
        except Exception as exc:  # noqa: BLE001 - botocore raises many types
            raise BlobStoreError(f"could not write blob {key}: {type(exc).__name__}: {exc}") from exc
        return BlobRef(key=key, size=len(data), sha256=digest, backend=self.name)

    def get(self, ref: BlobRef) -> bytes:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._object_key(ref.key))
            body: bytes = response["Body"].read()
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in ("NoSuchKey", "404"):
                raise BlobNotFound(
                    f"blob {ref.key} is not in bucket {self._bucket!r}. Check the bucket, "
                    "prefix, and credentials on both the web tier and the workers."
                ) from exc
            raise BlobStoreError(f"could not read blob {ref.key}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise BlobStoreError(f"could not read blob {ref.key}: {type(exc).__name__}: {exc}") from exc
        return self._verify(ref, body)

    def delete(self, ref: BlobRef) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=self._object_key(ref.key))
        except Exception as exc:  # noqa: BLE001 - a failed delete falls back to the bucket's lifecycle rule
            logger.warning("could not delete blob %s: %s", ref.key, exc)

    def ping(self) -> bool:
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return True
        except Exception:  # noqa: BLE001
            return False

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "is_shared": self.is_shared,
            "reachable": self.ping(),
            "bucket": self._bucket,
            "prefix": self._prefix,
        }


class NullBlobStore(BlobStore):
    """No blob store: every job payload is inline.

    Selected when the queue is disabled and no store is configured. Keeps the
    small-job path working with zero configuration, and makes ``put`` fail loudly
    if something tries to externalize a payload anyway.
    """

    name = "none"
    is_shared = False

    def put(self, data: bytes) -> BlobRef:
        raise BlobStoreError(
            "no blob store is configured, so a job too large to inline cannot be queued. "
            "Set VA_LSE_BLOB_DIR (shared volume) or VA_LSE_BLOB_S3_BUCKET."
        )

    def get(self, ref: BlobRef) -> bytes:
        raise BlobStoreError("no blob store is configured (VA_LSE_BLOB_DIR / VA_LSE_BLOB_S3_BUCKET)")

    def delete(self, ref: BlobRef) -> None:
        return


_store: BlobStore | None = None
_store_lock = threading.Lock()


def build_blob_store() -> BlobStore:
    """Select a backend from configuration (see the module docstring)."""
    mode = config.BLOB_STORE_MODE
    if mode == "none":
        return NullBlobStore()
    if mode in ("auto", "s3") and config.BLOB_S3_BUCKET:
        try:
            store = S3BlobStore(
                config.BLOB_S3_BUCKET,
                prefix=config.BLOB_S3_PREFIX,
                endpoint_url=config.BLOB_S3_ENDPOINT_URL,
            )
        except BlobStoreError as exc:
            logger.error("S3 blob store unavailable: %s", exc)
        else:
            logger.info(
                "blob store backend=s3 bucket=%s prefix=%s",
                config.BLOB_S3_BUCKET,
                config.BLOB_S3_PREFIX,
            )
            return store
    if mode in ("auto", "s3") and mode == "s3":
        logger.warning("VA_LSE_BLOB_STORE=s3 but VA_LSE_BLOB_S3_BUCKET is unset")
    if mode == "s3":
        return NullBlobStore()
    if config.JOB_QUEUE_ENABLED or mode == "filesystem":
        logger.info(
            "blob store backend=filesystem root=%s (must be shared between the web tier "
            "and every worker)",
            config.BLOB_DIR,
        )
        from .job_queue import build_job_backend, get_job_backend
        def retained_keys() -> set[str]:
            # Queue-off web tiers may still share storage with workers draining
            # older jobs. Inspect the configured remote tier even in that case.
            if config.JOB_QUEUE_REDIS_URL or (config.SHARED_CACHE_URL and config.SHARED_CACHE_TOKEN):
                return build_job_backend(require_distributed=True).retained_blob_keys()
            return get_job_backend().retained_blob_keys()
        return FilesystemBlobStore(config.BLOB_DIR, sweep_age_seconds=config.JOB_QUEUE_TTL_SECONDS,
                                   retained_keys=retained_keys)
    return NullBlobStore()


def get_blob_store() -> BlobStore:
    """Return (and lazily create) the process-global blob store."""
    global _store  # noqa: PLW0603
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:
            _store = build_blob_store()
        return _store


def reset_blob_store_for_tests() -> None:
    """Drop the cached store so tests get a fresh one."""
    global _store  # noqa: PLW0603
    with _store_lock:
        _store = None


def dumps_documents(payload: dict[str, Any]) -> bytes:
    """Canonical JSON bytes for a documents bundle (stable for content addressing)."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def loads_documents(data: bytes) -> dict[str, Any]:
    """Parse a stored documents bundle, raising :class:`BlobStoreError` if unusable."""
    try:
        parsed = json.loads(data)
    except ValueError as exc:
        raise BlobStoreError("stored job documents are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise BlobStoreError("stored job documents are not a JSON object")
    return parsed
