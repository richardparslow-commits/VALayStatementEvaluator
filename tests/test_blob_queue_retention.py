"""Retained queue inputs protect shared records across all queue transports."""
import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from tests import test_job_queue as queue_fixture
from app import blob_store, config, job_queue
from app.blob_store import BlobNotFound, FilesystemBlobStore


class _BlobRetentionContract:
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.backend = self.make_backend()
        self.store = FilesystemBlobStore(self.tmp.name, sweep_age_seconds=60,
                                         retained_keys=self.backend.retained_blob_keys)
        patcher = patch.object(blob_store, "get_blob_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def age(self, ref):
        old = time.time() - 120
        os.utime(Path(self.tmp.name) / ref.key, (old, old))

    def expire_input(self, job_id):
        if isinstance(self.backend, job_queue.InProcessJobBackend):
            self.backend._payloads.pop(job_id)
            self.backend._records.pop(job_id)
        else:
            self.backend._command("DEL", job_queue._payload_key(self.backend._prefix, job_id),
                                  job_queue._meta_key(self.backend._prefix, job_id))

    def test_aged_blob_survives_claim_heartbeat_requeue_and_terminal_retention(self):
        ref = self.store.put(b"shared synthetic records")
        payload = json.dumps({"documents_ref": ref.to_json()})
        record = self.backend.enqueue("evaluate", payload)
        self.age(ref)
        self.assertEqual(self.store.sweep(), 0)
        claimed, _ = self.backend.claim(["evaluate"], worker_id="dead-worker")
        self.backend.set_progress(record.job_id, 0.5, "working", claim_token=claimed.claim_token)
        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            self.assertEqual(self.backend.requeue_stale(), 1)
        self.assertEqual(self.store.sweep(), 0)
        retry, _ = self.backend.claim(["evaluate"], worker_id="healthy-worker")
        self.assertEqual(self.store.get(ref), b"shared synthetic records")
        self.backend.complete(record.job_id, claim_token=retry.claim_token)
        self.assertEqual(self.store.sweep(), 0)
        self.expire_input(record.job_id)
        self.assertEqual(self.store.sweep(), 1)

    def test_old_wire_submission_renews_but_deleted_input_cannot_create_a_job(self):
        ref = self.store.put(b"synthetic records")
        payload = json.dumps({"documents_ref": ref.to_json()})
        self.age(ref)
        record = self.backend.enqueue("evaluate", payload, request_id="original")
        self.assertGreater((Path(self.tmp.name) / ref.key).stat().st_mtime, time.time() - 60)
        self.store.delete(ref)
        with self.assertRaises(BlobNotFound):
            self.backend.enqueue("evaluate", payload, request_id="new")
        candidate = job_queue._submission("evaluate", payload, "new", "")
        self.assertIsNone(self.backend.get(candidate.job_id))
        self.assertEqual(self.backend.depth(), 1)
        self.assertIsNotNone(self.backend.get(record.job_id))

    def test_identical_content_is_kept_until_both_users_inputs_expire(self):
        ref = self.store.put(b"same synthetic bundle")
        payload = json.dumps({"documents_ref": ref.to_json()})
        first = self.backend.enqueue("evaluate", payload, request_id="one", owner_id="owner-one")
        second = self.backend.enqueue("draft", payload, request_id="two", owner_id="owner-two")
        self.age(ref)
        self.expire_input(first.job_id)
        self.assertEqual(self.store.sweep(), 0)
        self.assertEqual(self.store.get(ref), b"same synthetic bundle")
        self.expire_input(second.job_id)
        self.assertEqual(self.store.sweep(), 1)

    def test_uninspectable_retained_input_refuses_cleanup(self):
        ref = self.store.put(b"synthetic record to preserve")
        self.backend.enqueue("evaluate", "legacy opaque input")
        self.age(ref)
        with self.assertRaises(job_queue.JobQueueError):
            self.store.sweep()
        self.assertEqual(self.store.get(ref), b"synthetic record to preserve")


class TestInProcessBlobRetention(_BlobRetentionContract, unittest.TestCase):
    make_backend = queue_fixture.TestInProcessBackend.make_backend


class TestRedisBlobRetention(_BlobRetentionContract, unittest.TestCase):
    make_backend = queue_fixture.TestRedisBackend.make_backend

    def test_scan_uses_all_pages_and_ignores_other_queue_prefixes(self):
        ref = self.store.put(b"synthetic retained record")
        protected = self.backend.enqueue("evaluate", json.dumps({"documents_ref": ref.to_json()}))
        for index in range(220):
            self.client.set(f"t:job:inline-{index}:payload", "{}")
        self.client.set("foreign:job:other:payload", "not our data")
        self.assertEqual(self.backend.retained_blob_keys(), {ref.key})
        self.assertIsNotNone(self.backend.get(protected.job_id))

    def test_invalid_scan_replies_and_transport_failure_refuse_cleanup(self):
        for reply in (None, ["not-a-cursor", []], ["0", "not-a-list"], ["0", ["foreign"]]):
            with patch.object(self.backend, "_command", return_value=reply):
                with self.assertRaises(job_queue.JobQueueError):
                    self.backend.retained_blob_keys()
        with patch.object(self.backend, "_command", side_effect=job_queue.JobQueueError("offline")):
            with self.assertRaises(job_queue.JobQueueError):
                self.backend.retained_blob_keys()

    def test_prefix_with_glob_characters_is_matched_literally(self):
        self.backend._prefix = "t[1]*?"
        self.client.set("t[1]*?:job:own:payload", json.dumps({"documents_ref": {"key": "own-key"}}))
        self.client.set("t1foreign:job:other:payload", "malformed foreign input")
        self.assertEqual(self.backend.retained_blob_keys(), {"own-key"})


class TestUpstashBlobRetention(_BlobRetentionContract, unittest.TestCase):
    make_backend = queue_fixture.TestUpstashBackend.make_backend


class TestCleanupBackendSelection(unittest.TestCase):
    def test_unconfigured_cleanup_never_uses_an_empty_local_queue(self):
        with patch.object(config, "JOB_QUEUE_ENABLED", False), \
             patch.object(config, "JOB_QUEUE_REDIS_URL", ""), \
             patch.object(config, "SHARED_CACHE_URL", ""), \
             patch.object(config, "SHARED_CACHE_TOKEN", ""):
            with self.assertRaises(job_queue.JobQueueUnavailable):
                job_queue.build_job_backend(require_distributed=True)

    def test_queue_off_tier_still_inventories_its_configured_remote_queue(self):
        with TemporaryDirectory() as tmp:
            with patch.object(config, "BLOB_STORE_MODE", "filesystem"), \
                 patch.object(config, "BLOB_DIR", tmp), \
                 patch.object(config, "JOB_QUEUE_ENABLED", False), \
                 patch.object(config, "JOB_QUEUE_REDIS_URL", "redis://synthetic"), \
                 patch.object(job_queue, "build_job_backend") as build:
                build.return_value.retained_blob_keys.return_value = {"protected"}
                store = blob_store.build_blob_store()
                self.assertEqual(store._retained_keys(), {"protected"})
                build.assert_called_once_with(require_distributed=True)
