"""Tests for the blob store and for externalized job payloads.

Large jobs put their extracted record text in a blob store instead of in Redis.
Two failure modes matter most here and both are covered directly:

* a blob store that is **not actually shared** between the web pod and the worker
  (the writing side looks fine; the reading side must fail loudly with a message
  that names the sharing requirement), and
* a **truncated/corrupt** blob being silently digested into a report, which the
  sha256 + size verification on every read is there to prevent.
"""
import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import blob_store as blob_mod  # noqa: E402
from app import config  # noqa: E402
from app.blob_store import (  # noqa: E402
    BlobNotFound,
    BlobRef,
    BlobStoreError,
    FilesystemBlobStore,
    NullBlobStore,
    build_blob_store,
    content_key,
    dumps_documents,
    loads_documents,
)
from app.documents import document_from_text  # noqa: E402
from app.job_payload import (  # noqa: E402
    KIND_EVALUATE,
    EvaluateJob,
    PayloadError,
    decode_job,
    documents_bundle,
    encode_job,
    encode_job_with_blob,
    payload_needs_blob,
)


def _job(text: str = "EVT knee pain noted.", name: str = "a.txt") -> EvaluateJob:
    return EvaluateJob(
        statement_text="I injured my knee.",
        records=[document_from_text(name, text)],
        request_id="req_blob",
        record_sources=["Upload"],
    )


class TestContentAddressing(unittest.TestCase):
    def test_key_is_derived_from_content(self):
        key = content_key(b"hello")
        self.assertEqual(key, content_key(b"hello"))
        self.assertNotEqual(key, content_key(b"hello "))
        self.assertRegex(key, r"^blobs/[0-9a-f]{2}/[0-9a-f]{64}\.json$")

    def test_identical_content_yields_one_blob(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp)
            first = store.put(b"same bytes")
            second = store.put(b"same bytes")
            self.assertEqual(first.key, second.key)
            stored = list((Path(tmp) / "blobs").glob("*/*.json"))
            self.assertEqual(len(stored), 1)


class TestFilesystemBlobStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = FilesystemBlobStore(self._tmp.name)

    def test_put_get_roundtrip(self):
        data = b'{"documents": []}'
        ref = self.store.put(data)
        self.assertEqual(ref.size, len(data))
        self.assertEqual(self.store.get(ref), data)

    def test_missing_blob_names_the_sharing_requirement(self):
        ref = BlobRef(key=content_key(b"never written"), size=10, sha256="a" * 64, backend="filesystem")
        with self.assertRaises(BlobNotFound) as ctx:
            self.store.get(ref)
        message = str(ctx.exception)
        self.assertIn("shared", message)
        self.assertIn("ReadWriteMany", message)

    def test_truncated_blob_is_refused(self):
        ref = self.store.put(b"x" * 100)
        # Simulate a partial write landing in the store.
        (Path(self._tmp.name) / ref.key).write_bytes(b"x" * 40)
        with self.assertRaises(BlobStoreError) as ctx:
            self.store.get(ref)
        self.assertIn("not shared", str(ctx.exception))

    def test_corrupt_blob_fails_its_integrity_check(self):
        ref = self.store.put(b"x" * 100)
        (Path(self._tmp.name) / ref.key).write_bytes(b"y" * 100)
        with self.assertRaises(BlobStoreError) as ctx:
            self.store.get(ref)
        self.assertIn("integrity check", str(ctx.exception))

    def test_path_traversal_in_a_key_is_refused(self):
        hostile = BlobRef(key="blobs/../etc/passwd", size=0, sha256="", backend="filesystem")
        with self.assertRaises(BlobStoreError):
            self.store.get(hostile)
        # delete() is deliberately best-effort (it only logs), so the guarantee is
        # that it does not touch the filesystem — not that it raises.
        self.store.delete(hostile)
        self.assertFalse((Path(self._tmp.name).parent / "etc" / "passwd").exists())

    def test_keys_outside_the_blob_namespace_are_refused(self):
        for bad in ("other/ab/cd.json", "blobs/ab/not-a-digest.json", "blobs/AABB/" + "a" * 64 + ".json"):
            with self.assertRaises(BlobStoreError):
                self.store.get(BlobRef(key=bad, size=0, sha256="", backend="filesystem"))

    def test_delete_removes_the_blob(self):
        ref = self.store.put(b"gone soon")
        self.store.delete(ref)
        with self.assertRaises(BlobNotFound):
            self.store.get(ref)

    def test_delete_of_a_missing_blob_is_silent(self):
        ref = BlobRef(key=content_key(b"absent"), size=1, sha256="b" * 64, backend="filesystem")
        self.store.delete(ref)  # must not raise

    def test_oversize_blob_is_refused(self):
        with patch.object(config, "JOB_QUEUE_MAX_PAYLOAD_BYTES", 64):
            with self.assertRaises(BlobStoreError) as ctx:
                self.store.put(b"x" * 65)
        self.assertIn("VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES", str(ctx.exception))

    def test_sweep_removes_expired_blobs_and_prunes_shards(self):
        ref = self.store.put(b"old")
        self.assertEqual(self.store.sweep(max_age_seconds=-1), 1)
        with self.assertRaises(BlobNotFound):
            self.store.get(ref)
        # The empty shard directory is cleaned up too.
        self.assertEqual(list((Path(self._tmp.name) / "blobs").glob("*")), [])

    def test_sweep_keeps_fresh_blobs(self):
        ref = self.store.put(b"fresh")
        self.assertEqual(self.store.sweep(max_age_seconds=3600), 0)
        self.assertEqual(self.store.get(ref), b"fresh")

    def test_ping_reports_writability(self):
        self.assertTrue(self.store.ping())
        self.assertTrue(self.store.health()["reachable"])

    def test_writes_are_atomic_no_temp_files_left_behind(self):
        self.store.put(b"atomic")
        leftovers = [p for p in Path(self._tmp.name).rglob(".tmp-*")]
        self.assertEqual(leftovers, [])


class TestStoreSelection(unittest.TestCase):
    def test_null_when_nothing_configured(self):
        with patch.object(config, "BLOB_STORE_MODE", "auto"), patch.object(
            config, "JOB_QUEUE_ENABLED", False
        ), patch.object(config, "BLOB_S3_BUCKET", ""):
            self.assertIsInstance(build_blob_store(), NullBlobStore)

    def test_filesystem_selected_when_the_queue_is_on(self):
        with patch.object(config, "BLOB_STORE_MODE", "auto"), patch.object(
            config, "JOB_QUEUE_ENABLED", True
        ), patch.object(config, "BLOB_S3_BUCKET", ""), patch.object(
            config, "BLOB_DIR", "/tmp/va-lse-blob-selection-test"
        ):
            self.assertIsInstance(build_blob_store(), FilesystemBlobStore)

    def test_explicit_none_mode_wins(self):
        with patch.object(config, "BLOB_STORE_MODE", "none"), patch.object(
            config, "JOB_QUEUE_ENABLED", True
        ):
            self.assertIsInstance(build_blob_store(), NullBlobStore)

    def test_s3_mode_without_a_bucket_degrades_to_null(self):
        with patch.object(config, "BLOB_STORE_MODE", "s3"), patch.object(
            config, "BLOB_S3_BUCKET", ""
        ):
            self.assertIsInstance(build_blob_store(), NullBlobStore)

    def test_s3_mode_without_boto3_degrades_with_a_clear_log(self):
        with patch.object(config, "BLOB_STORE_MODE", "s3"), patch.object(
            config, "BLOB_S3_BUCKET", "bucket"
        ), patch.dict(sys.modules, {"boto3": None}), self.assertLogs(
            "app.blob_store", level="ERROR"
        ) as captured:
            self.assertIsInstance(build_blob_store(), NullBlobStore)
        self.assertTrue(any("boto3 is not installed" in line for line in captured.output))

    def test_singleton_is_cached_until_reset(self):
        blob_mod.reset_blob_store_for_tests()
        try:
            with patch.object(config, "BLOB_STORE_MODE", "none"):
                self.assertIs(blob_mod.get_blob_store(), blob_mod.get_blob_store())
        finally:
            blob_mod.reset_blob_store_for_tests()

    def test_null_store_refuses_a_put(self):
        with self.assertRaises(BlobStoreError) as ctx:
            NullBlobStore().put(b"data")
        self.assertIn("VA_LSE_BLOB_DIR", str(ctx.exception))


class TestDocumentBundles(unittest.TestCase):
    def test_bundle_roundtrip(self):
        bundle = dumps_documents({"version": 1, "documents": [{"filename": "a.txt", "pages": []}]})
        self.assertEqual(loads_documents(bundle)["version"], 1)

    def test_bundle_is_stable_for_content_addressing(self):
        self.assertEqual(documents_bundle(_job()), documents_bundle(_job()))

    def test_bundle_excludes_the_statement_so_shared_records_dedupe(self):
        """Two users with the same records must share one blob."""
        first = _job()
        second = EvaluateJob(
            statement_text="a completely different statement",
            records=first.records,
            request_id="req_other",
            record_sources=["VA.gov"],
        )
        self.assertEqual(documents_bundle(first), documents_bundle(second))

    def test_garbage_bundle_raises(self):
        with self.assertRaises(BlobStoreError):
            loads_documents(b"{not json")
        with self.assertRaises(BlobStoreError):
            loads_documents(b"[1, 2]")


class TestBlobReferencedPayloads(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = FilesystemBlobStore(self._tmp.name)

    def _externalized(self, job: EvaluateJob) -> str:
        ref = self.store.put(documents_bundle(job))
        return encode_job_with_blob(KIND_EVALUATE, job, ref)

    def test_referenced_payload_decodes_via_the_blob_store(self):
        job = _job()
        payload = self._externalized(job)
        decoded = decode_job(KIND_EVALUATE, payload, blob_store=self.store)
        self.assertEqual(decoded.statement_text, job.statement_text)
        self.assertEqual(len(decoded.records), 1)
        self.assertIn("knee pain", decoded.records[0].full_text)

    def test_referenced_payload_is_far_smaller_than_inline(self):
        job = EvaluateJob(
            statement_text="s",
            records=[document_from_text("a.txt", "EVT " + ("pain noted. " * 5000))],
            request_id="req_big",
        )
        inline = encode_job(KIND_EVALUATE, job)
        referenced = self._externalized(job)
        self.assertLess(len(referenced), len(inline) / 10)

    def test_reference_without_a_store_is_a_hard_error(self):
        """Never fall back to running with zero records."""
        payload = self._externalized(_job())
        with self.assertRaises(PayloadError) as ctx:
            decode_job(KIND_EVALUATE, payload)
        self.assertIn("no blob store is configured", str(ctx.exception))
        self.assertIn("VA_LSE_BLOB_DIR", str(ctx.exception))

    def test_missing_blob_surfaces_the_sharing_hint(self):
        payload = self._externalized(_job())
        empty = TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        with self.assertRaises(PayloadError) as ctx:
            decode_job(KIND_EVALUATE, payload, blob_store=FilesystemBlobStore(empty.name))
        self.assertIn("shared", str(ctx.exception))

    def test_unreadable_reference_is_rejected(self):
        payload = json.dumps({"kind": KIND_EVALUATE, "documents_ref": "not a ref", "statement_text": "s"})
        with self.assertRaises(PayloadError) as ctx:
            decode_job(KIND_EVALUATE, payload, blob_store=self.store)
        self.assertIn("unreadable documents reference", str(ctx.exception))

    def test_kind_mismatch_still_rejected_for_referenced_payloads(self):
        payload = self._externalized(_job())
        with self.assertRaises(PayloadError):
            decode_job("draft", payload, blob_store=self.store)

    def test_inline_payloads_still_work_without_a_store(self):
        decoded = decode_job(KIND_EVALUATE, encode_job(KIND_EVALUATE, _job()))
        self.assertEqual(len(decoded.records), 1)


class TestInlineThreshold(unittest.TestCase):
    def test_small_job_stays_inline(self):
        with patch.object(config, "JOB_QUEUE_INLINE_MAX_BYTES", 1024 * 1024):
            self.assertFalse(payload_needs_blob(KIND_EVALUATE, _job()))

    def test_large_job_is_externalized(self):
        big = EvaluateJob(
            statement_text="s",
            records=[document_from_text("a.txt", "EVT " + ("pain noted. " * 5000))],
        )
        with patch.object(config, "JOB_QUEUE_INLINE_MAX_BYTES", 1024):
            self.assertTrue(payload_needs_blob(KIND_EVALUATE, big))

    def test_job_over_the_hard_cap_is_externalized_rather_than_refused(self):
        """The blob store is what makes an over-cap job queueable at all."""
        big = EvaluateJob(
            statement_text="s",
            records=[document_from_text("a.txt", "EVT " + ("pain noted. " * 5000))],
        )
        with patch.object(config, "JOB_QUEUE_INLINE_MAX_BYTES", 1024), patch.object(
            config, "JOB_QUEUE_MAX_PAYLOAD_BYTES", 2048
        ):
            self.assertTrue(payload_needs_blob(KIND_EVALUATE, big))


class TestSweepTiming(unittest.TestCase):
    def test_opportunistic_sweep_is_rate_limited(self):
        """Many puts in a burst must not walk the whole tree each time."""
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp)
            calls = []
            with patch.object(store, "sweep", side_effect=lambda **kw: calls.append(1) or 0):
                for i in range(10):
                    store.put(f"blob-{i}".encode())
            self.assertLessEqual(len(calls), 1)

    def test_sweep_interval_permits_a_later_sweep(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp)
            calls = []
            with patch.object(store, "sweep", side_effect=lambda **kw: calls.append(1) or 0):
                store.put(b"first")
                # Pretend the interval elapsed.
                store._last_sweep = time.monotonic() - 10_000
                store.put(b"second")
            self.assertEqual(len(calls), 2)

    def test_sweep_age_defaults_to_the_job_ttl(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp, sweep_age_seconds=60)
            old = store.put(b"old")
            fresh = store.put(b"fresh")
            past = time.time() - 120
            os.utime(Path(tmp) / old.key, (past, past))
            self.assertEqual(store.sweep(), 1)
            self.assertFalse((Path(tmp) / old.key).exists())
            self.assertTrue((Path(tmp) / fresh.key).exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
