"""Audit-log retention and off-pod backup.

Why this exists
---------------
``app/audit.py`` writes a compliance stream to ``{VA_LSE_AUDIT_LOG_DIR}/audit.log``
with size-based rotation. Two gaps made that insufficient for forensics:

* **Rotation bounds size, not time.** ``AUDIT_LOG_MAX_BYTES`` ×
  (``AUDIT_LOG_BACKUPS`` + 1) is a hard ~110 MiB ceiling, and
  ``RotatingFileHandler`` deletes the oldest file to make room — during a busy
  period a file younger than the retention window is destroyed. There was no
  age-based rule at all, and nothing shipped the files anywhere.
* **The logs lived only on the pod.** In the reference Kubernetes manifests
  ``/app/logs`` was an ``emptyDir``, so a pod restart reclaimed the whole
  stream. That is the failure this module exists to prevent, and it is why the
  volume must be a PVC before any of this is useful (``deploy/k8s/k8s-logs.yaml``).

What gets uploaded
------------------
*Rotated* files (``audit.log.1`` …) in full, oldest first.
*Plus* a **byte-watermark window of the live ``audit.log``** — the bytes written
since the last successful pass, cut back to the last complete line.

That second part is the one a naive implementation misses, and it is the whole
ballgame: at a few hundred runs a day a 10 MiB file takes *weeks* to rotate, so a
backup that only ships rotated files leaves every recent event, including the
events from the run that just failed, on the pod. The window closes that gap
without ever reading a partial line.

Idempotency
-----------
Object keys are derived from content and byte range::

    audit/2026/09/16/live-audit.log-g3a1b2c4d5e6f-0-48213-9f31c2ab7d10.jsonl

so a pass that uploads and then dies before checkpointing re-uploads byte range
``0-48213`` to *the same key* on the next run. At-least-once delivery therefore
degrades into a harmless overwrite rather than a duplicate, which is why there is
no dedupe table to corrupt. (A retry after a crash is still possible for a range
whose *end* shifts because more lines arrived; those are separate ranges with
separate keys, not duplicates.)

The ``g<tag>`` segment identifies *which* ``audit.log`` the range came from
(:func:`generation_tag`). It is what makes a restore able to stitch windows back
together: every generation starts at offset 0, so without it a window from the
file written after a rotation is indistinguishable from a retry of a window from
the file before it.

Old key shape (no ``g`` segment)::

    audit/2026/09/16/live-audit.log-0-48213-9f31c2ab7d10.jsonl

Restore still reads that shape — it is treated as an unknown generation, which is
reported rather than silently merged. Keys are otherwise unchanged.

Failure behaviour
-----------------
Failures are recorded in the state file and surfaced through ``/health``; they
are never raised at the caller. The state file — not process memory — is the
source of truth, because the writer (a CronJob or sidecar) and the reader (the
web pod's ``/health``) are different processes on a shared volume.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from . import config

logger = logging.getLogger("app.audit_backup")

STATE_VERSION = 1

# A window bigger than this is refused rather than read into memory. The audit
# stream is capped at ~110 MiB in total, so hitting this means the watermark was
# lost and the destination is far behind — upload it in chunks by rerunning.
_MAX_WINDOW_BYTES = 64 * 1024 * 1024


class BackupError(RuntimeError):
    """Raised for an unusable destination configuration or a failed transfer."""


# --------------------------------------------------------------------- helpers


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime | None = None) -> str:
    return (moment or _utc_now()).isoformat()


def _parse_iso(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _sha12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _file_sha12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def audit_log_dir() -> Path:
    """Resolve the audit log directory (mirrors ``app/audit.py``)."""
    return Path(config.AUDIT_LOG_DIR).expanduser()


def audit_log_path() -> Path:
    return audit_log_dir() / config.AUDIT_LOG_FILE


def state_path() -> Path:
    return audit_log_dir() / config.AUDIT_BACKUP_STATE_FILE


def rotated_audit_files(directory: Path | None = None) -> list[Path]:
    """Return rotated audit files, oldest first.

    ``RotatingFileHandler`` names them ``<file>.1`` … ``<file>.<n>`` where ``.1``
    is the newest. Sorting by size descending is a decent proxy for age; sorting
    by the numeric suffix is exact, so do that.
    """
    base = directory if directory is not None else audit_log_dir()
    name = config.AUDIT_LOG_FILE
    found: list[tuple[int, Path]] = []
    if not base.exists():
        return []
    for path in base.glob(f"{name}.*"):
        suffix = path.name[len(name) + 1 :]
        if suffix.isdigit():
            found.append((int(suffix), path))
    # Highest suffix = oldest file, so upload that first.
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in found]


# ----------------------------------------------------------------- destinations


@dataclass
class RemoteObject:
    """One object in the destination, as reported by ``list_objects``."""

    key: str
    size: int
    last_modified: datetime | None


class BackupDestination:
    """Contract every destination backend implements."""

    name = "none"
    off_pod = False

    def put_object(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        raise NotImplementedError

    def get_object(self, key: str) -> bytes:
        """Download one object. Only the restore path needs this."""
        raise NotImplementedError

    def delete_object(self, key: str) -> None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "off_pod": self.off_pod}


class FilesystemDestination(BackupDestination):
    """Write to another directory — an NFS mount, a synced share, or a test tmpdir.

    Usable for real off-pod backup when the directory is a *different* mount from
    the audit log (an NFS/Azure Files/EFS mount, a hostPath on a different disk).
    Pointing it at a directory on the same volume is permitted but reported as
    ``off_pod: false`` in ``/health``, because it cannot survive the pod failure
    the backup exists to survive.
    """

    name = "filesystem"

    def __init__(self, root: str | Path, *, same_volume: bool = False) -> None:
        self._root = Path(root).expanduser()
        self.off_pod = not same_volume

    @property
    def root(self) -> Path:
        return self._root

    def put_object(self, key: str, data: bytes) -> None:
        target = self._root / key
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError as exc:
            raise BackupError(f"could not write {key}: {type(exc).__name__}: {exc}") from exc

    def get_object(self, key: str) -> bytes:
        try:
            return (self._root / key).read_bytes()
        except OSError as exc:
            raise BackupError(f"could not read {key}: {type(exc).__name__}: {exc}") from exc

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        if not self._root.exists():
            return []
        out: list[RemoteObject] = []
        for path in sorted(self._root.rglob("*.jsonl")):
            rel = path.relative_to(self._root).as_posix()
            if not rel.startswith(prefix):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            out.append(
                RemoteObject(
                    key=rel,
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
        return out

    def delete_object(self, key: str) -> None:
        try:
            (self._root / key).unlink(missing_ok=True)
        except OSError:
            logger.debug("could not delete backup object %s", key)

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({"root": str(self._root), "same_volume": not self.off_pod})
        return base


class S3Destination(BackupDestination):
    """Any S3-compatible endpoint via ``boto3`` (AWS S3, R2, MinIO, Spaces, GCS interop)."""

    name = "s3"
    off_pod = True

    def __init__(self, bucket: str, *, prefix: str = "", endpoint_url: str = "") -> None:
        try:
            import boto3  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise BackupError(
                "boto3 is not installed; pip install -r requirements-backup.txt "
                "(or use VA_LSE_AUDIT_BACKUP_DESTINATION=filesystem)"
            ) from exc
        if not bucket:
            raise BackupError("VA_LSE_AUDIT_BACKUP_S3_BUCKET is empty")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client: Any = boto3.client("s3", endpoint_url=endpoint_url or None)

    def put_object(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._object_key(key),
                Body=data,
                ContentType="application/x-ndjson",
            )
        except Exception as exc:  # noqa: BLE001 - botocore raises many types
            raise BackupError(f"could not upload {key}: {type(exc).__name__}: {exc}") from exc

    def get_object(self, key: str) -> bytes:
        try:
            body = self._client.get_object(Bucket=self._bucket, Key=self._object_key(key))["Body"]
            return bytes(body.read())
        except Exception as exc:  # noqa: BLE001 - botocore raises many types
            raise BackupError(f"could not download {key}: {type(exc).__name__}: {exc}") from exc

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        import botocore.exceptions  # noqa: PLC0415

        out: list[RemoteObject] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=self._object_key(prefix)):
                for item in page.get("Contents", []) or []:
                    out.append(
                        RemoteObject(
                            key=self._strip(item.get("Key", "")),
                            size=int(item.get("Size") or 0),
                            last_modified=item.get("LastModified"),
                        )
                    )
        except botocore.exceptions.BotoCoreError as exc:
            raise BackupError(f"could not list {self._bucket}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise BackupError(f"could not list {self._bucket}: {type(exc).__name__}: {exc}") from exc
        return out

    def delete_object(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=self._object_key(key))
        except Exception as exc:  # noqa: BLE001 - a failed prune is not fatal
            logger.warning("could not prune %s: %s", key, exc)

    def _object_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _strip(self, key: str) -> str:
        return key[len(self._prefix) + 1 :] if self._prefix and key.startswith(self._prefix) else key

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({"bucket": self._bucket, "prefix": self._prefix})
        return base


class GCSDestination(BackupDestination):
    """Google Cloud Storage via ``google-cloud-storage``."""

    name = "gcs"
    off_pod = True

    def __init__(self, bucket: str, *, prefix: str = "") -> None:
        try:
            from google.cloud import storage  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise BackupError(
                "google-cloud-storage is not installed; pip install -r requirements-backup.txt "
                "(or use GCS interoperability mode with the s3 destination)"
            ) from exc
        if not bucket:
            raise BackupError("VA_LSE_AUDIT_BACKUP_GCS_BUCKET is empty")
        self._bucket_name = bucket
        self._prefix = prefix.strip("/")
        self._client: Any = storage.Client()

    def put_object(self, key: str, data: bytes) -> None:
        try:
            blob = self._client.bucket(self._bucket_name).blob(self._object_key(key))
            blob.upload_from_string(data, content_type="application/x-ndjson")
        except Exception as exc:  # noqa: BLE001 - google.api_core raises many types
            raise BackupError(f"could not upload {key}: {type(exc).__name__}: {exc}") from exc

    def get_object(self, key: str) -> bytes:
        try:
            blob = self._client.bucket(self._bucket_name).blob(self._object_key(key))
            return bytes(blob.download_as_bytes())
        except Exception as exc:  # noqa: BLE001 - google.api_core raises many types
            raise BackupError(f"could not download {key}: {type(exc).__name__}: {exc}") from exc

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        out: list[RemoteObject] = []
        try:
            for blob in self._client.list_blobs(self._bucket_name, prefix=self._object_key(prefix)):
                out.append(
                    RemoteObject(
                        key=self._strip(blob.name),
                        size=int(getattr(blob, "size", 0) or 0),
                        last_modified=getattr(blob, "updated", None),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise BackupError(f"could not list {self._bucket_name}: {type(exc).__name__}: {exc}") from exc
        return out

    def delete_object(self, key: str) -> None:
        try:
            self._client.bucket(self._bucket_name).blob(self._object_key(key)).delete()
        except Exception as exc:  # noqa: BLE001 - a failed prune is not fatal
            logger.warning("could not prune %s: %s", key, exc)

    def _object_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _strip(self, key: str) -> str:
        return key[len(self._prefix) + 1 :] if self._prefix and key.startswith(self._prefix) else key

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({"bucket": self._bucket_name, "prefix": self._prefix})
        return base


class AzureBlobDestination(BackupDestination):
    """Azure Blob Storage via ``azure-storage-blob``.

    Azure has no S3-compatible API, so it needs its own backend. Auth comes from
    ``AZURE_STORAGE_CONNECTION_STRING`` when set, otherwise ``account_url`` plus
    ``DefaultAzureCredential`` (managed identity / workload identity in AKS).
    """

    name = "azure"
    off_pod = True

    def __init__(self, container: str, *, prefix: str = "", account_url: str = "") -> None:
        try:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise BackupError(
                "azure-storage-blob is not installed; pip install -r requirements-backup.txt"
            ) from exc
        if not container:
            raise BackupError("VA_LSE_AUDIT_BACKUP_AZURE_CONTAINER is empty")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
        try:
            if connection_string:
                self._service: Any = BlobServiceClient.from_connection_string(connection_string)
            elif account_url:
                from azure.identity import DefaultAzureCredential  # noqa: PLC0415

                self._service = BlobServiceClient(
                    account_url=account_url, credential=DefaultAzureCredential()
                )
            else:
                raise BackupError(
                    "set AZURE_STORAGE_CONNECTION_STRING or VA_LSE_AUDIT_BACKUP_AZURE_ACCOUNT_URL"
                )
        except BackupError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BackupError(f"could not create the Azure client: {type(exc).__name__}: {exc}") from exc
        self._container = container
        self._prefix = prefix.strip("/")

    def put_object(self, key: str, data: bytes) -> None:
        try:
            client = self._service.get_blob_client(container=self._container, blob=self._object_key(key))
            client.upload_blob(data, overwrite=True)
        except Exception as exc:  # noqa: BLE001 - azure-core raises many types
            raise BackupError(f"could not upload {key}: {type(exc).__name__}: {exc}") from exc

    def get_object(self, key: str) -> bytes:
        try:
            client = self._service.get_blob_client(container=self._container, blob=self._object_key(key))
            return bytes(client.download_blob().readall())
        except Exception as exc:  # noqa: BLE001 - azure-core raises many types
            raise BackupError(f"could not download {key}: {type(exc).__name__}: {exc}") from exc

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        out: list[RemoteObject] = []
        try:
            container = self._service.get_container_client(self._container)
            for blob in container.list_blobs(name_starts_with=self._object_key(prefix)):
                out.append(
                    RemoteObject(
                        key=self._strip(blob.name),
                        size=int(getattr(blob, "size", 0) or 0),
                        last_modified=getattr(blob, "last_modified", None),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise BackupError(f"could not list {self._container}: {type(exc).__name__}: {exc}") from exc
        return out

    def delete_object(self, key: str) -> None:
        try:
            client = self._service.get_blob_client(container=self._container, blob=self._object_key(key))
            client.delete_blob()
        except Exception as exc:  # noqa: BLE001 - a failed prune is not fatal
            logger.warning("could not prune %s: %s", key, exc)

    def _object_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _strip(self, key: str) -> str:
        return key[len(self._prefix) + 1 :] if self._prefix and key.startswith(self._prefix) else key

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({"container": self._container, "prefix": self._prefix})
        return base


class NullDestination(BackupDestination):
    """No destination configured: every operation is a recorded no-op."""

    name = "none"

    def get_object(self, key: str) -> bytes:
        raise BackupError("no backup destination is configured, so there is nothing to restore")


def build_destination(*, overrides: Mapping[str, str] | None = None) -> BackupDestination:
    """Select a destination from config. Raises :class:`BackupError` when unusable.

    ``overrides`` maps a ``config`` attribute name (``AUDIT_BACKUP_S3_BUCKET`` and
    friends) to a replacement value. The restore CLI uses it so an operator can
    read from a *different* bucket than the app writes to — the common case when
    the deployment's credentials are scoped to the writer and recovery is done
    with a read-only key — without mutating process-global config.
    """

    def value(name: str) -> str:
        if overrides and name in overrides:
            return str(overrides[name]).strip()
        return str(getattr(config, name, "") or "").strip()

    choice = (overrides.get("AUDIT_BACKUP_DESTINATION", "") if overrides else "") or (
        config.AUDIT_BACKUP_DESTINATION
    )
    choice = str(choice).strip().lower()
    if not choice or choice == "none":
        return NullDestination()
    if choice == "filesystem":
        root_raw = value("AUDIT_BACKUP_DIR")
        if not root_raw:
            raise BackupError(
                "VA_LSE_AUDIT_BACKUP_DESTINATION=filesystem requires VA_LSE_AUDIT_BACKUP_DIR. "
                "Point it at a mount that is NOT the audit log's own volume — a destination on "
                "the same volume does not survive the pod failure a backup exists to survive."
            )
        root = Path(root_raw).expanduser()
        return FilesystemDestination(root, same_volume=_same_volume(root, audit_log_dir()))
    if choice == "s3":
        return S3Destination(
            value("AUDIT_BACKUP_S3_BUCKET"),
            prefix=value("AUDIT_BACKUP_S3_PREFIX"),
            endpoint_url=value("AUDIT_BACKUP_S3_ENDPOINT_URL"),
        )
    if choice == "gcs":
        return GCSDestination(
            value("AUDIT_BACKUP_GCS_BUCKET"), prefix=value("AUDIT_BACKUP_GCS_PREFIX")
        )
    if choice == "azure":
        return AzureBlobDestination(
            value("AUDIT_BACKUP_AZURE_CONTAINER"),
            prefix=value("AUDIT_BACKUP_AZURE_PREFIX"),
            account_url=value("AUDIT_BACKUP_AZURE_ACCOUNT_URL"),
        )
    raise BackupError(
        f"unknown VA_LSE_AUDIT_BACKUP_DESTINATION {choice!r} "
        "(expected filesystem, s3, gcs, or azure)"
    )


def _same_volume(left: Path, right: Path) -> bool:
    """True when two paths resolve to the same device, or one contains the other.

    ``st_dev`` catches "two directories on one filesystem"; the containment check
    catches "the backup dir is inside the log dir" even before either exists.
    """
    try:
        left_resolved = left.resolve()
        right_resolved = right.resolve()
    except OSError:
        return False
    if left_resolved == right_resolved or right_resolved in left_resolved.parents:
        return True
    try:
        return left_resolved.stat().st_dev == right_resolved.stat().st_dev
    except OSError:
        # Neither path exists yet: fall back to the containment check above.
        return False


# ----------------------------------------------------------------------- state


@dataclass
class BackupState:
    """Persisted backup progress. Written by the backup process, read by health."""

    destination: str = ""
    last_attempt_utc: str = ""
    last_success_utc: str = ""
    last_error: str = ""
    uploaded_objects: int = 0
    uploaded_bytes: int = 0
    runs: int = 0
    last_duration_ms: int = 0
    # Live-file watermark: how far into audit.log we have successfully shipped.
    live_offset: int = 0
    live_size: int = 0
    live_name: str = ""
    live_identity: str = ""
    # Which generation of ``live_name`` the watermark belongs to — see
    # :func:`generation_tag`. Recorded so a restore can tell one file's windows
    # from the next file's without guessing.
    live_generation: str = ""
    # Rotated files already shipped, keyed by file name.
    rotated: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "destination": self.destination,
            "last_attempt_utc": self.last_attempt_utc,
            "last_success_utc": self.last_success_utc,
            "last_error": self.last_error,
            "uploaded_objects": self.uploaded_objects,
            "uploaded_bytes": self.uploaded_bytes,
            "runs": self.runs,
            "last_duration_ms": self.last_duration_ms,
            "live_offset": self.live_offset,
            "live_size": self.live_size,
            "live_name": self.live_name,
            "live_identity": self.live_identity,
            "live_generation": self.live_generation,
            "rotated": self.rotated,
        }

    @classmethod
    def from_json(cls, raw: Any) -> "BackupState":
        if not isinstance(raw, dict):
            return cls()
        state = cls()
        state.destination = str(raw.get("destination") or "")
        state.last_attempt_utc = str(raw.get("last_attempt_utc") or "")
        state.last_success_utc = str(raw.get("last_success_utc") or "")
        state.last_error = str(raw.get("last_error") or "")
        state.uploaded_objects = int(raw.get("uploaded_objects") or 0)
        state.uploaded_bytes = int(raw.get("uploaded_bytes") or 0)
        state.runs = int(raw.get("runs") or 0)
        state.last_duration_ms = int(raw.get("last_duration_ms") or 0)
        state.live_offset = int(raw.get("live_offset") or 0)
        state.live_size = int(raw.get("live_size") or 0)
        state.live_name = str(raw.get("live_name") or "")
        state.live_identity = str(raw.get("live_identity") or "")
        state.live_generation = str(raw.get("live_generation") or "")
        rotated = raw.get("rotated")
        if isinstance(rotated, dict):
            for key, value in rotated.items():
                if isinstance(value, dict):
                    state.rotated[str(key)] = value
        return state


def load_state(path: Path | None = None) -> BackupState:
    """Read the state file. A missing or corrupt file reads as "never run"."""
    target = path if path is not None else state_path()
    try:
        return BackupState.from_json(json.loads(target.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return BackupState()
    except (OSError, ValueError) as exc:
        logger.warning("could not read backup state at %s: %s", target, exc)
        return BackupState()


def save_state(state: BackupState, path: Path | None = None) -> bool:
    """Persist the state file atomically. Returns False when unwritable."""
    target = path if path is not None else state_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(state.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
        return True
    except OSError as exc:
        logger.warning("could not write backup state at %s: %s", target, exc)
        return False


# ---------------------------------------------------------------- object naming


def generation_tag(stat: os.stat_result) -> str:
    """A short, stable tag identifying **which file** a live window came from.

    Without this the key records a byte range but not *which* ``audit.log`` that
    range belonged to, and that is genuinely ambiguous after a rotation: every
    generation's first window starts at offset 0, so a window from the new file
    is indistinguishable from a retry of a window from the old one. The
    consequence was real — a restore would concatenate the old file's head with
    the new file's head and call it contiguous.

    The tag is derived from the file identity ``_live_window`` already uses to
    detect rotation, hashed so the key does not leak a raw inode. It is stable
    for the life of one generation, which keeps the retry-overwrite contract
    intact, and it changes exactly when the file rotates. ``"0"`` means the
    platform exposed no identity — the same condition under which rotation
    cannot be detected at all.
    """
    identity = _identity(stat)
    return _sha12(identity.encode("utf-8")) if identity else "0"


def object_key(
    live: bool,
    filename: str,
    *,
    start: int = 0,
    end: int = 0,
    sha12: str,
    generation: str = "0",
) -> str:
    """Build a content/range-addressed object key.

    Deterministic for a given (file generation, byte range) so a retry overwrites
    the same object instead of adding a duplicate — see the module docstring.
    """
    stamp = _utc_now()
    kind = "live" if live else "rotated"
    if live:
        return (
            f"audit/{stamp:%Y/%m/%d}/{kind}-{filename}-g{generation}-{start}-{end}-{sha12}.jsonl"
        )
    size = end - start
    return f"audit/{stamp:%Y/%m/%d}/{kind}-{filename}-{size}-{sha12}.jsonl"


# ------------------------------------------------------------------------- pass


@dataclass
class BackupResult:
    """Outcome of one backup pass."""

    configured: bool
    destination: str
    uploaded: int = 0
    uploaded_bytes: int = 0
    skipped_rotated: int = 0
    pruned_local: int = 0
    pruned_remote: int = 0
    error: str = ""
    detail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error


def _live_window(path: Path, state: BackupState) -> tuple[int, int] | None:
    """Return ``(start, end)`` byte offsets of the shippable live-log window.

    ``end`` is trimmed back to the last complete newline so a half-written line is
    never uploaded. Returns ``None`` when there is nothing new.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    size = stat.st_size
    offset = state.live_offset if state.live_name == path.name else 0
    if offset:
        identity = _identity(stat)
        # Two independent rotation detectors, because neither is sufficient:
        # ``RotatingFileHandler`` renames the old file and creates a *new* one, so
        # the inode changes — and the new file can be the same size as the
        # watermark (identical line lengths), which is why a size check alone
        # silently skips the new file's head. On platforms without a usable
        # inode, the size regression still catches truncation and growth-rotation.
        inode_changed = bool(state.live_identity and identity and state.live_identity != identity)
        shrank = offset > size
        if inode_changed or shrank:
            logger.info(
                "audit.log is a different file than the watermark (%s); resetting watermark",
                "rotation" if inode_changed else f"size {size} < {offset}",
            )
            offset = 0
    if size <= offset:
        return None
    window = size - offset
    if window > _MAX_WINDOW_BYTES:
        raise BackupError(
            f"the un-uploaded window of {path.name} is {window:,} bytes "
            f"(limit {_MAX_WINDOW_BYTES:,}); the destination is too far behind. "
            "Shipping it in one pass would need a contiguous read of the whole file."
        )
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(window)
    except OSError as exc:
        raise BackupError(f"could not read {path.name}: {type(exc).__name__}: {exc}") from exc
    newline = chunk.rfind(b"\n")
    if newline < 0:
        # Only a partial line so far — wait for the writer to finish it.
        return None
    return offset, offset + newline + 1


def _identity(stat: os.stat_result) -> str:
    """A rotation fingerprint for a stat result (inode where the platform has one)."""
    inode = getattr(stat, "st_ino", 0)
    device = getattr(stat, "st_dev", 0)
    if inode:
        return f"{device}:{inode}"
    return ""


def _load_bytes(path: Path, start: int, end: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(end - start)


def run_backup(
    *,
    destination: BackupDestination | None = None,
    state: BackupState | None = None,
    dry_run: bool = False,
    prune_remote: bool = False,
) -> BackupResult:
    """Run one backup pass: rotated files, then the live window, then retention.

    Never raises for an operational failure — the error is recorded in the state
    file and in the returned result so ``/health`` and the CronJob's exit code can
    both report it.
    """
    dest: BackupDestination
    try:
        dest = destination if destination is not None else build_destination()
    except BackupError as exc:
        return BackupResult(configured=True, destination=config.AUDIT_BACKUP_DESTINATION, error=str(exc))

    if isinstance(dest, NullDestination):
        # Local retention is a disk-hygiene concern that exists independently of
        # whether anything is being shipped, so it runs even here. Returning early
        # without sweeping meant the retention policy silently did nothing in the
        # default (unconfigured) deployment — the one place it is load-bearing for
        # "no protection against disk-space exhaustion".
        retention = BackupResult(configured=False, destination="none")
        try:
            retention.pruned_local = sweep_local_retention(dry_run=dry_run)
        except OSError as exc:
            logger.warning("local audit retention sweep failed: %s", exc)
        return retention

    current = state if state is not None else load_state()
    current.destination = dest.name
    started = time.monotonic()
    current.last_attempt_utc = _iso()
    current.runs += 1
    result = BackupResult(configured=True, destination=dest.name)

    try:
        _ship_rotated(dest, current, result, dry_run=dry_run)
        _ship_live_window(dest, current, result, dry_run=dry_run)
    except BackupError as exc:
        result.error = str(exc)
        current.last_error = str(exc)
        logger.warning("audit backup pass failed: %s", exc)
    else:
        current.last_success_utc = _iso()
        current.last_error = ""
        current.uploaded_objects += result.uploaded
        current.uploaded_bytes += result.uploaded_bytes

    try:
        result.pruned_local = sweep_local_retention(dry_run=dry_run)
    except OSError as exc:
        logger.warning("local audit retention sweep failed: %s", exc)

    if prune_remote and not dry_run and not result.error:
        try:
            result.pruned_remote = prune_cloud_retention(dest)
        except BackupError as exc:
            result.detail.append(f"cloud prune failed: {exc}")
            logger.warning("cloud retention prune failed: %s", exc)

    current.last_duration_ms = int((time.monotonic() - started) * 1000)
    if not dry_run:
        save_state(current)
    return result


def _ship_rotated(
    dest: BackupDestination, state: BackupState, result: BackupResult, *, dry_run: bool
) -> None:
    """Upload every rotated file we have not already shipped, oldest first."""
    for path in rotated_audit_files():
        try:
            stat = path.stat()
        except OSError:
            continue
        digest = _file_sha12(path)
        fingerprint = f"{stat.st_size}:{digest}"
        if state.rotated.get(path.name, {}).get("fingerprint") == fingerprint:
            result.skipped_rotated += 1
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.warning("could not read rotated audit file %s: %s", path.name, exc)
            continue
        key = object_key(False, path.name, start=0, end=len(data), sha12=digest)
        if not dry_run:
            dest.put_object(key, data)
            state.rotated[path.name] = {
                "fingerprint": fingerprint,
                "size": stat.st_size,
                "sha256_12": digest,
                "object": key,
                "uploaded_utc": _iso(),
            }
        result.uploaded += 1
        result.uploaded_bytes += len(data)
        result.detail.append(f"rotated {path.name} ({len(data):,} bytes) -> {key}")


def _ship_live_window(
    dest: BackupDestination, state: BackupState, result: BackupResult, *, dry_run: bool
) -> None:
    """Upload the un-shipped tail of the live audit.log, cut to the last newline."""
    path = audit_log_path()
    if not path.exists():
        return
    window = _live_window(path, state)
    if window is None:
        return
    start, end = window
    try:
        stat = path.stat()
    except OSError as exc:
        raise BackupError(f"could not stat {path.name}: {type(exc).__name__}: {exc}") from exc
    data = _load_bytes(path, start, end)
    digest = _sha12(data)
    key = object_key(
        True, path.name, start=start, end=end, sha12=digest, generation=generation_tag(stat)
    )
    if not dry_run:
        dest.put_object(key, data)
        state.live_offset = end
        state.live_size = stat.st_size
        state.live_name = path.name
        state.live_identity = _identity(stat)
        state.live_generation = generation_tag(stat)
    result.uploaded += 1
    result.uploaded_bytes += len(data)
    result.detail.append(f"live window {start:,}-{end:,} ({len(data):,} bytes) -> {key}")


# -------------------------------------------------------------------- retention


def sweep_local_retention(*, retention_days: int | None = None, dry_run: bool = False) -> int:
    """Delete rotated audit files older than the local retention window.

    Rotation already deletes by *count*, which can be far more aggressive than the
    age policy during a busy period. This sweep is the age half of the rule; the
    effective local retention is whichever limit is reached first, and the live
    ``audit.log`` is never removed.
    """
    days = config.AUDIT_RETENTION_DAYS if retention_days is None else retention_days
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for path in rotated_audit_files():
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            if not dry_run:
                path.unlink()
            removed += 1
            logger.info("audit retention removed %s (older than %d day(s))", path.name, days)
        except OSError as exc:
            logger.debug("could not remove %s: %s", path.name, exc)
    return removed


def prune_cloud_retention(
    destination: BackupDestination, *, retention_days: int | None = None
) -> int:
    """Delete destination objects older than the cloud retention window.

    A bucket lifecycle rule is the better instrument (it survives a broken backup
    job), so this is a fallback for destinations that cannot set one. Objects with
    no ``last_modified`` are left alone rather than guessed at.
    """
    days = config.AUDIT_BACKUP_CLOUD_RETENTION_DAYS if retention_days is None else retention_days
    if days <= 0:
        return 0
    cutoff = _utc_now() - timedelta(days=days)
    removed = 0
    for obj in destination.list_objects(""):
        modified = obj.last_modified
        if modified is None:
            continue
        if modified.tzinfo is None:
            modified = modified.replace(tzinfo=timezone.utc)
        if modified >= cutoff:
            continue
        destination.delete_object(obj.key)
        removed += 1
    if removed:
        logger.info("cloud retention pruned %d object(s) older than %d day(s)", removed, days)
    return removed


# ----------------------------------------------------------------------- health


def disk_status(path: Path | None = None) -> dict[str, Any]:
    """Free/total space for the volume holding the logs (best-effort)."""
    target = path if path is not None else audit_log_dir()
    try:
        probe = target if target.exists() else target.parent
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {"checked": False, "error": f"{type(exc).__name__}: {exc}"}
    floor = config.DISK_MIN_FREE_BYTES
    return {
        "checked": True,
        "path": str(probe),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "used_percent": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
        "min_free_bytes": floor,
        "below_floor": usage.free < floor,
    }


def pending_bytes(state: BackupState | None = None) -> int:
    """Bytes present locally but not yet shipped — what a pod death would lose.

    Compares recorded *size* only, never a hash: this runs on the ``/health``
    path, and hashing ~110 MiB of rotated files on every probe (kubelet polls
    every 10 s) would blow the endpoint's 2 s budget for a number that only needs
    to be indicative.
    """
    current = state if state is not None else load_state()
    total = 0
    for path in rotated_audit_files():
        try:
            stat = path.stat()
        except OSError:
            continue
        entry = current.rotated.get(path.name) or {}
        if int(entry.get("size", -1)) != stat.st_size:
            total += stat.st_size
    live = audit_log_path()
    try:
        live_size = live.stat().st_size
    except OSError:
        live_size = 0
    if live_size > current.live_offset:
        total += live_size - current.live_offset
    return total


def destination_summary() -> dict[str, Any]:
    """Config-only description of the destination, for ``/health``.

    Deliberately does *not* call :func:`build_destination`: that would construct a
    cloud client on every probe (kubelet polls every 10 s) and can raise. The only
    thing worth reporting before a pass has run is whether the destination is
    genuinely off the pod, and that much is answerable from a ``stat``.
    """
    choice = config.AUDIT_BACKUP_DESTINATION or "none"
    summary: dict[str, Any] = {"backend": choice, "off_pod": choice not in ("", "none", "filesystem")}
    if choice == "filesystem":
        if config.AUDIT_BACKUP_DIR:
            same_volume = _same_volume(Path(config.AUDIT_BACKUP_DIR).expanduser(), audit_log_dir())
            summary["off_pod"] = not same_volume
            summary["same_volume"] = same_volume
            if same_volume:
                summary["warning"] = (
                    "the destination is on the audit log's own volume, so it does not "
                    "survive the pod failure a backup exists to survive"
                )
        else:
            summary["off_pod"] = False
            summary["error"] = "VA_LSE_AUDIT_BACKUP_DIR is unset"
    return summary


def audit_backup_health(state: BackupState | None = None) -> dict[str, Any]:
    """Health payload for ``GET /health``.

    Deliberately reads the *state file* and never the network: the backup runs in
    a different process, and `/health` must stay under its 2 s SLO. Reachability
    of the destination is the backup job's problem, reported here through
    ``last_error`` and ``stale``.
    """
    destination = config.AUDIT_BACKUP_DESTINATION or "none"
    configured = destination not in ("", "none")
    current = state if state is not None else load_state()
    payload: dict[str, Any] = {
        "configured": configured,
        "destination": destination,
        "off_pod": destination_summary().get("off_pod"),
        "interval_hours": config.AUDIT_BACKUP_INTERVAL_HOURS,
        "local_retention_days": config.AUDIT_RETENTION_DAYS,
        "cloud_retention_days": config.AUDIT_BACKUP_CLOUD_RETENTION_DAYS,
        "uploaded_objects": current.uploaded_objects,
        "uploaded_bytes": current.uploaded_bytes,
        "runs": current.runs,
        "last_success_utc": current.last_success_utc or None,
        "last_error": current.last_error or None,
        "state_file": str(state_path()),
    }
    try:
        payload["pending_bytes"] = pending_bytes(current)
    except Exception as exc:  # noqa: BLE001 - health must not fail on housekeeping
        payload["pending_bytes"] = None
        payload["pending_error"] = f"{type(exc).__name__}: {exc}"

    last_success = _parse_iso(current.last_success_utc)
    if not configured:
        payload["status"] = "disabled"
        payload["reason"] = "set VA_LSE_AUDIT_BACKUP_DESTINATION to ship audit logs off-pod"
        return payload
    if last_success is None:
        # A recorded error outranks "never ran": a job that has been failing since
        # deployment looked identical to one that was never scheduled, which hides
        # the incident an operator actually needs to see.
        payload["status"] = "error" if current.last_error else "never_ran"
        payload["reason"] = current.last_error or (
            "no successful backup recorded; the backup job has not run against this volume yet"
        )
        payload["age_seconds"] = None
        payload["stale"] = True
        return payload

    age = (_utc_now() - last_success).total_seconds()
    # Audit logs are forensics, not real-time alerting, so "stale" is one missed
    # interval, not a hard failure: /health stays 200 and the operator decides.
    stale_after = max(config.AUDIT_BACKUP_INTERVAL_HOURS * 3600 * 2, 3600)
    payload["age_seconds"] = int(age)
    payload["stale"] = age > stale_after
    payload["stale_after_seconds"] = int(stale_after)
    if current.last_error:
        payload["status"] = "error"
        payload["reason"] = current.last_error
    elif payload["stale"]:
        payload["status"] = "stale"
        payload["reason"] = (
            f"last successful backup was {int(age / 3600)}h ago, "
            f"over {config.AUDIT_BACKUP_INTERVAL_HOURS:g}h interval"
        )
    else:
        payload["status"] = "ok"
    return payload


# -------------------------------------------------------------------------- lock


class BackupLock:
    """Cross-process lock so two backup jobs cannot fight over one watermark.

    A sidecar and a CronJob can be deployed against the same volume by accident;
    without this, both would read the same offset and upload the same window (the
    content-addressed key makes that harmless, but the state file write is a
    last-writer-wins race that could *lose* a watermark and cause re-uploads).

    Stale locks are taken over after ``stale_after_seconds``. That matters because
    the shipped CronJob sets ``activeDeadlineSeconds``: a pass that overruns is
    SIGKILLed, and SIGKILL cannot run a cleanup handler, so the lock file survives.
    Without takeover, one hung upload would disable **every** subsequent backup —
    the backup would be silently dead until someone noticed a lock file. An
    abandoned lock is therefore treated as evidence of a crashed run, not as
    authority over the next one.
    """

    def __init__(self, path: Path | None = None, *, stale_after_seconds: float | None = None) -> None:
        self._path = path if path is not None else state_path().with_suffix(".lock")
        self._stale_after = (
            float(config.AUDIT_BACKUP_LOCK_STALE_SECONDS)
            if stale_after_seconds is None
            else float(stale_after_seconds)
        )
        self._held = False

    def acquire(self) -> bool:
        for attempt in range(2):
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if attempt == 0 and self._take_over_if_stale():
                    continue
                return False
            except OSError as exc:
                logger.warning("could not create the backup lock: %s", exc)
                # A volume that cannot hold a lock file is not a reason to skip the
                # backup; proceed and rely on the idempotent object keys.
                self._held = True
                return True
            try:
                os.write(fd, f"{os.getpid()} {_iso()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            self._held = True
            return True
        return False

    def _take_over_if_stale(self) -> bool:
        """Remove an abandoned lock so one crashed pass cannot stop all future ones."""
        try:
            raw = self._path.read_text(encoding="utf-8").strip()
            age = time.time() - self._path.stat().st_mtime
        except OSError:
            return False
        if age < self._stale_after:
            return False
        logger.warning(
            "taking over a stale backup lock held for %.0fs (over %.0fs) by %s",
            age,
            self._stale_after,
            raw or "unknown",
        )
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    def release(self) -> None:
        if not self._held:
            return
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass
        self._held = False

    def __enter__(self) -> "BackupLock":
        if not self.acquire():
            raise BackupError(
                f"another backup process holds {self._path}; remove it only if that process is gone"
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
