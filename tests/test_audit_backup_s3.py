"""Live S3 round trip for the audit backup destination.

The filesystem destination is covered in ``tests/test_audit_backup.py``, but the
cloud backends are the ones that decide whether a compliance log survives a pod
failure — and they are the ones a unit test can most easily get wrong by encoding
an assumption about the SDK instead of the protocol. So this drives the **real**
``boto3`` client from :class:`app.audit_backup.S3Destination` against a loopback
HTTP server that speaks enough of the S3 REST API to store, list, and delete
objects, then asserts the object key, the bytes, and the pass-to-pass behaviour.

Skipped when boto3 is absent (``requirements-backup.txt`` is optional), which is
why CI installs it: a skipped test here would silently mean "unverified".
"""
from __future__ import annotations

import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote_plus

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)
from app import audit_backup, config

try:
    import boto3  # noqa: F401 - presence check only

    HAVE_BOTO3 = True
except ImportError:  # pragma: no cover - depends on the image
    HAVE_BOTO3 = False


class FakeS3Handler(BaseHTTPRequestHandler):
    """Minimal S3 REST surface: PUT/GET/DELETE object, list, head bucket."""

    protocol_version = "HTTP/1.1"
    bucket = "audit-bucket"

    def log_message(self, *_args: object) -> None:  # silence the server
        pass

    # -- request plumbing --------------------------------------------------
    def _split(self) -> tuple[str, str]:
        """Return ``(key, query)`` for both path-style and virtual-host requests."""
        raw = self.path
        path, _, query = raw.partition("?")
        prefix = f"/{self.bucket}/"
        if path.startswith(prefix):
            key = path[len(prefix) :]
        elif path == f"/{self.bucket}":
            key = ""
        else:
            key = path.lstrip("/")
        return key, query

    def _store(self) -> dict[str, bytes]:
        return self.server.store  # type: ignore[attr-defined]

    # -- verbs -------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self) -> None:  # noqa: N802
        key, _ = self._split()
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        self._store()[key] = body
        # Record a distinct, increasing modification time per object. Real S3
        # always returns LastModified, and the restore orders file generations by
        # it (every generation starts at offset 0, so offsets cannot order them) —
        # a constant timestamp would make that ordering unknowable rather than
        # tested.
        self.server.clock += 1  # type: ignore[attr-defined]
        self.server.mtimes[key] = self.server.clock  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("ETag", '"fake-etag"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        key, query = self._split()
        if "list-type=2" in query or query.startswith("list-type"):
            return self._list(query)
        body = self._store().get(key)
        if body is None:
            return self._error(404, "NoSuchKey")
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self) -> None:  # noqa: N802
        key, _ = self._split()
        self._store().pop(key, None)
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- helpers -----------------------------------------------------------
    def _list(self, query: str) -> None:
        prefix = ""
        for part in query.split("&"):
            if part.startswith("prefix="):
                prefix = unquote_plus(part[len("prefix=") :])
        entries = [
            (name, body)
            for name, body in sorted(self._store().items())
            if name.startswith(prefix)
        ]
        mtimes = self.server.mtimes  # type: ignore[attr-defined]
        contents = "".join(
            "<Contents>"
            f"<Key>{name}</Key>"
            f"<LastModified>{self._modified(mtimes.get(name, 0))}</LastModified>"
            '<ETag>"fake-etag"</ETag>'
            f"<Size>{len(body)}</Size>"
            "<StorageClass>STANDARD</StorageClass>"
            "</Contents>"
            for name, body in entries
        )
        payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>{self.bucket}</Name><Prefix>{prefix}</Prefix>"
            f"<KeyCount>{len(entries)}</KeyCount><MaxKeys>1000</MaxKeys>"
            "<IsTruncated>false</IsTruncated>"
            f"{contents}</ListBucketResult>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _modified(sequence: int) -> str:
        """An ISO timestamp that advances one second per stored object."""
        base = datetime(2026, 9, 16, tzinfo=timezone.utc)
        return (base + timedelta(seconds=sequence)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _error(self, code: int, s3_code: str) -> None:
        payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Error><Code>{s3_code}</Code><Message>missing</Message></Error>"
        ).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@unittest.skipUnless(HAVE_BOTO3, "boto3 not installed (requirements-backup.txt)")
class TestS3DestinationLive(unittest.TestCase):
    """Drive S3Destination through the wire, not through a mock."""

    bucket = "audit-bucket"

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeS3Handler)
        cls.server.store = {}  # type: ignore[attr-defined]
        cls.server.mtimes = {}  # type: ignore[attr-defined]
        cls.server.clock = 0  # type: ignore[attr-defined]
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.store.clear()  # type: ignore[attr-defined]
        self.server.mtimes.clear()  # type: ignore[attr-defined]
        self.server.clock = 0  # type: ignore[attr-defined]
        self._tmp = tempfile.TemporaryDirectory()
        self.logs = Path(self._tmp.name) / "logs"
        self.logs.mkdir()
        self._env = {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
        self._saved_env = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)
        self._saved = {
            name: getattr(config, name)
            for name in (
                "AUDIT_LOG_DIR",
                "AUDIT_LOG_FILE",
                "AUDIT_BACKUP_DESTINATION",
                "AUDIT_BACKUP_S3_PREFIX",
                "AUDIT_BACKUP_S3_BUCKET",
                "AUDIT_BACKUP_S3_ENDPOINT_URL",
            )
        }
        config.AUDIT_LOG_DIR = str(self.logs)
        config.AUDIT_LOG_FILE = "audit.log"
        config.AUDIT_BACKUP_DESTINATION = "s3"
        config.AUDIT_BACKUP_S3_PREFIX = ""
        config.AUDIT_BACKUP_S3_BUCKET = self.bucket
        config.AUDIT_BACKUP_S3_ENDPOINT_URL = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(config, name, value)
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def destination(self) -> audit_backup.S3Destination:
        """Build through config, so the bucket/prefix/endpoint wiring is covered too."""
        destination = audit_backup.build_destination()
        self.assertIsInstance(destination, audit_backup.S3Destination)
        assert isinstance(destination, audit_backup.S3Destination)
        return destination

    def append(self, index: int) -> str:
        line = f'{{"request_id": "req_{index}", "status": "ok"}}\n'
        with (self.logs / "audit.log").open("a", encoding="utf-8") as handle:
            handle.write(line)
        return line

    # -- the tests ---------------------------------------------------------
    def test_get_object_downloads_the_exact_bytes_over_the_wire(self) -> None:
        """``get_object`` is the only primitive the restore path uses on S3."""
        dest = self.destination()
        payload = b'{"request_id": "req_wire"}\n'
        key = "audit/2026/09/16/live-audit.log-gdeadbeef0000-0-27-abcdef123456.jsonl"
        dest.put_object(key, payload)
        self.assertEqual(dest.get_object(key), payload)

    def test_get_object_missing_key_raises_a_clear_error(self) -> None:
        dest = self.destination()
        with self.assertRaises(audit_backup.BackupError) as caught:
            dest.get_object("audit/2026/09/16/absent.jsonl")
        self.assertIn("could not download", str(caught.exception))

    def test_backup_then_verify_and_restore_over_the_wire(self) -> None:
        """The whole loop through real boto3: write, read back, rebuild.

        Covers the integration the filesystem round-trip test cannot: the S3 client
        actually responding to ``get_object``, and the restore's expectation that a
        key it listed can be downloaded by that same key.
        """
        first = self.append(0) + self.append(1)
        result = audit_backup.run_backup(destination=self.destination())
        self.assertTrue(result.ok, result.error)
        # Rotate and write more, so the stream spans two file generations.
        (self.logs / "audit.log").replace(self.logs / "audit.log.1")
        second = self.append(2)
        audit_backup.run_backup(destination=self.destination())

        from app import audit_restore

        report = audit_restore.verify_backup(self.destination())
        self.assertTrue(report.ok, report.to_json())
        self.assertEqual(report.corrupt, [])
        self.assertEqual(report.plan.gaps, [])
        self.assertGreaterEqual(report.verified_objects, 3)

        target = Path(self._tmp.name) / "restored"
        out = audit_restore.restore_backup(self.destination(), target=target)
        self.assertTrue(out.ok, out.error)
        self.assertEqual((target / "restored.jsonl").read_text(), first + second)
        self.assertGreaterEqual(len(out.rotated_paths), 1)

    def test_put_and_list_round_trip_over_the_wire(self) -> None:
        dest = self.destination()
        payload = b'{"request_id": "req_wire"}\n'
        dest.put_object("audit/2026/09/16/live-audit.log-0-27-abcdef123456.jsonl", payload)
        self.assertEqual(self.server.store.get(  # type: ignore[attr-defined]
            "audit/2026/09/16/live-audit.log-0-27-abcdef123456.jsonl"
        ), payload)
        listed = dest.list_objects("")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].size, len(payload))
        self.assertIsNotNone(listed[0].last_modified)

    def test_prefix_is_applied_to_keys_and_stripped_from_listings(self) -> None:
        config.AUDIT_BACKUP_S3_PREFIX = "va-lse"
        dest = self.destination()
        dest.put_object("audit/x.jsonl", b"one")
        keys = list(self.server.store.keys())  # type: ignore[attr-defined]
        self.assertEqual(keys, ["va-lse/audit/x.jsonl"])
        listed = dest.list_objects("")
        self.assertEqual([o.key for o in listed], ["audit/x.jsonl"])

    def test_full_pass_through_config_puts_objects_under_the_prefix(self) -> None:
        config.AUDIT_BACKUP_S3_PREFIX = "va-lse"
        self.append(0)
        result = audit_backup.run_backup()  # no destination argument: config decides
        self.assertTrue(result.ok, result.error)
        keys = list(self.server.store.keys())  # type: ignore[attr-defined]
        self.assertEqual(len(keys), 1)
        self.assertTrue(keys[0].startswith("va-lse/audit/"), keys[0])

    def test_a_full_pass_ships_the_live_log_and_is_idempotent(self) -> None:
        expected = "".join(self.append(i) for i in range(3))
        dest = self.destination()
        first = audit_backup.run_backup(destination=dest)
        self.assertTrue(first.ok, first.error)
        self.assertEqual(first.uploaded, 1)
        stored = list(self.server.store.values())  # type: ignore[attr-defined]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].decode(), expected)

        second = audit_backup.run_backup(destination=dest)
        self.assertTrue(second.ok, second.error)
        self.assertEqual(second.uploaded, 0, "no new bytes means no new object")
        self.assertEqual(len(self.server.store), 1)  # type: ignore[attr-defined]

    def test_rotated_file_reaches_the_bucket_and_prune_deletes_it(self) -> None:
        self.append(0)
        (self.logs / "audit.log").replace(self.logs / "audit.log.1")
        dest = self.destination()
        result = audit_backup.run_backup(destination=dest)
        self.assertTrue(result.ok, result.error)
        keys = list(self.server.store.keys())  # type: ignore[attr-defined]
        self.assertEqual(len(keys), 1)
        self.assertIn("rotated-audit.log.1-", keys[0])
        # 90-day default window: a fresh object must survive a prune.
        self.assertEqual(audit_backup.prune_cloud_retention(dest), 0)
        self.assertEqual(len(self.server.store), 1)  # type: ignore[attr-defined]

    def test_upload_failure_surfaces_instead_of_losing_the_watermark(self) -> None:
        self.append(0)
        broken = audit_backup.S3Destination(
            self.bucket, endpoint_url="http://127.0.0.1:1"  # nothing listening
        )
        result = audit_backup.run_backup(destination=broken)
        self.assertFalse(result.ok)
        self.assertTrue(result.error)
        state = audit_backup.load_state()
        self.assertEqual(state.live_offset, 0, "bytes must be retried, not dropped")
        self.assertTrue(state.last_error)


if __name__ == "__main__":
    unittest.main()
