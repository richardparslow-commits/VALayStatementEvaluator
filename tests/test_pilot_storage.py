"""R16 synthetic regressions: durable exclusion, visible deletion and S3 uncertainty."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import traceback
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from botocore.exceptions import ClientError
from app import blob_cleanup, blob_store as storage, config, pilot
from app.storage_policy import REFUSAL, synthetic_storage

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = ("controlled-pilot", "controlled-pilto", "", "production")


def missing():
    return ClientError({"Error": {"Code": "404", "Message": "invented absent object"},
                        "ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject")


class PilotStorageTests(unittest.TestCase):
    def setUp(self):
        mode = patch.dict(os.environ, VA_LSE_MODE="synthetic")
        mode.start()
        self.addCleanup(mode.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = storage.FilesystemBlobStore(self.root)
        self.ref = self.store.put(b"INVENTED_DURABLE_CASE_CANARY")

    def refuse(self, function):
        with self.assertRaises(pilot.PilotBlocked) as caught:
            function()
        self.assertEqual(str(caught.exception), REFUSAL)

    def s3(self):
        client = MagicMock()
        client.get_bucket_versioning.return_value = {}
        client.delete_object.return_value = {}
        client.head_object.side_effect = missing()
        module = SimpleNamespace(client=lambda *a, **k: client)
        with patch.dict(sys.modules, boto3=module):
            store = storage.S3BlobStore("invented-private-bucket", prefix="invented-prefix")
        return store, client

    def test_cached_factory_and_constructors_refuse_before_configuration_or_clients(self):
        with patch.object(storage, "_store", self.store), patch("boto3.client") as client:
            for mode in EXCLUDED:
                with self.subTest(mode=mode), patch.dict(os.environ, VA_LSE_MODE=mode):
                    calls = (storage.get_blob_store, storage.build_blob_store,
                             lambda: storage.FilesystemBlobStore(self.root / "must-not-exist"),
                             lambda: storage.S3BlobStore("invented-private-bucket"))
                    for call in calls:
                        self.refuse(call)
            client.assert_not_called()
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_cached_filesystem_operations_refuse_without_reads_writes_or_delete(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        for mode in EXCLUDED:
            with patch.dict(os.environ, VA_LSE_MODE=mode, VA_LSE_BLOB_STORE="filesystem", VA_LSE_STORAGE_APPROVED="1"):
                calls = (lambda: self.store.put(b"private"), lambda: self.store.get(self.ref),
                         lambda: self.store.delete(self.ref), self.store.ping, self.store.sweep,
                         self.store._maybe_sweep, lambda: self.store._path_for(self.ref.key),
                         lambda: self.store._storage_lock().__enter__(),
                         lambda: self.store.submission_guard(self.ref).__enter__())
                for call in calls:
                    self.refuse(call)
                self.assertEqual(self.store.health(), {"backend": "excluded", "is_shared": False, "reachable": False})
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_cached_s3_operations_and_health_never_contact_backend_in_excluded_modes(self):
        store, client = self.s3()
        for mode in EXCLUDED:
            with patch.dict(os.environ, VA_LSE_MODE=mode):
                for call in (lambda: store.put(b"private"), lambda: store.get(self.ref),
                             lambda: store.delete(self.ref), store.ping, store.sweep, store._require_unversioned,
                             lambda: store._object_key(self.ref.key),
                             lambda: store.submission_guard(self.ref).__enter__()):
                    self.refuse(call)
                self.assertEqual(store.health()["backend"], "excluded")
        self.assertEqual(client.mock_calls, [])

    def test_null_backend_is_not_an_exclusion_bypass(self):
        store = storage.NullBlobStore()
        with patch.dict(os.environ, VA_LSE_MODE="controlled-pilot"):
            for call in (lambda: store.put(b"private"), lambda: store.get(self.ref),
                         lambda: store.delete(self.ref), store.sweep, store.ping):
                self.refuse(call)
        self.assertEqual(store.health()["backend"], "none")

    def test_cleanup_cli_refuses_even_dry_run_before_backend_mount_or_inventory(self):
        with patch.object(blob_cleanup, "build_blob_store") as blobs, \
                patch.object(blob_cleanup, "build_job_backend") as queue:
            for mode in EXCLUDED:
                with patch.dict(os.environ, VA_LSE_MODE=mode):
                    for args in ([], ["--dry-run"]):
                        output = io.StringIO()
                        with redirect_stderr(output):
                            self.assertEqual(blob_cleanup.main(args), 2)
                        self.assertEqual(output.getvalue().strip(), REFUSAL)
            blobs.assert_not_called()
            queue.assert_not_called()

    def test_actual_cleanup_cli_returns_fixed_refusal_without_private_configuration(self):
        env = dict(os.environ, VA_LSE_MODE="controlled-pilot", VA_LSE_BLOB_STORE="filesystem",
                   VA_LSE_BLOB_DIR="INVENTED_PRIVATE_PATH_CANARY")
        result = subprocess.run([sys.executable, "-m", "app.blob_cleanup", "--dry-run"], env=env,
                                cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr.strip(), REFUSAL)
        self.assertEqual(result.stdout, "")

    def test_late_cleanup_factory_cannot_inspect_mount_or_construct_queue(self):
        def late():
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return self.store
        output = io.StringIO()
        with patch.object(blob_cleanup, "build_blob_store", side_effect=late), \
                patch.object(Path, "is_dir") as probe, patch.object(blob_cleanup, "build_job_backend") as queue, \
                redirect_stderr(output):
            self.assertEqual(blob_cleanup.main([]), 2)
        self.assertEqual(output.getvalue().strip(), REFUSAL)
        probe.assert_not_called()
        queue.assert_not_called()

    def test_late_filesystem_read_cannot_return_case_bytes(self):
        def late(*a, **k):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return b"INVENTED_PRIVATE_CANARY"
        with patch.object(Path, "read_bytes", side_effect=late):
            self.refuse(lambda: self.store.get(self.ref))

    def test_late_s3_reply_closes_body_without_reading_it(self):
        store, client = self.s3()
        body = MagicMock()
        def late(**k):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return {"Body": body}
        client.get_object.side_effect = late
        self.refuse(lambda: store.get(self.ref))
        body.read.assert_not_called()
        body.close.assert_called_once()

    def test_late_inventory_does_not_scan_or_delete_files(self):
        def late():
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return set()
        with patch.object(os, "scandir") as scan:
            self.refuse(lambda: self.store.sweep(retained_keys=late))
        scan.assert_not_called()
        self.assertTrue((self.root / self.ref.key).exists())

    def test_late_private_error_has_fixed_refusal_without_traceback_canary(self):
        @synthetic_storage
        def late():
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            raise RuntimeError("INVENTED_PRIVATE_CANARY")
        try:
            late()
        except pilot.PilotBlocked as exc:
            self.assertEqual(str(exc), REFUSAL)
            self.assertNotIn("INVENTED_PRIVATE_CANARY", "".join(traceback.format_exception(exc)))
        else:
            self.fail("excluded delayed error was returned")

    def test_late_cleanup_failure_reports_exclusion_not_success_or_raw_error(self):
        def late(**k):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            raise storage.BlobStoreError("INVENTED_PRIVATE_CANARY")
        errors = io.StringIO()
        with patch.object(blob_cleanup, "build_blob_store", return_value=self.store), \
                patch.object(blob_cleanup, "build_job_backend", return_value=MagicMock()), \
                patch.object(self.store, "sweep", side_effect=late), redirect_stderr(errors):
            self.assertEqual(blob_cleanup.main([]), 2)
        self.assertEqual(errors.getvalue().strip(), REFUSAL)

    def test_filesystem_delete_permission_failure_is_visible_and_preserves_object(self):
        with patch.object(Path, "unlink", side_effect=PermissionError("INVENTED_PRIVATE_CANARY")):
            with self.assertRaises(storage.BlobStoreError) as caught:
                self.store.delete(self.ref)
        self.assertEqual(str(caught.exception), "Filesystem blob deletion is unconfirmed.")
        self.assertEqual(self.store.get(self.ref), b"INVENTED_DURABLE_CASE_CANARY")

    def test_filesystem_delete_retry_and_missing_object_are_idempotent(self):
        self.store.delete(self.ref)
        self.store.delete(self.ref)
        with self.assertRaises(storage.BlobNotFound):
            self.store.get(self.ref)

    def test_namespace_traversal_and_trailing_newline_keys_never_delete_other_files(self):
        for key in ("blobs/../foreign.json", self.ref.key + "\n"):
            ref = storage.BlobRef(key, 0, "", "filesystem")
            with self.assertRaises(storage.BlobStoreError):
                self.store.delete(ref)
        self.assertTrue((self.root / self.ref.key).exists())

    def test_s3_versioned_suspended_unknown_and_invalid_states_refuse_before_delete(self):
        store, client = self.s3()
        for state in ({"Status": "Enabled"}, {"Status": "Suspended"}, {"Status": "Disabled"},
                      {"Status": None}, {"MFADelete": "Disabled"}, {"Error": "private"}, [], None,
                      {"ResponseMetadata": None}, {"ResponseMetadata": {"HTTPStatusCode": 403}}):
            with self.subTest(state=state):
                client.get_bucket_versioning.return_value = state
                with self.assertRaises(storage.BlobStoreError):
                    store.delete(self.ref)
        client.delete_object.assert_not_called()
        client.head_object.assert_not_called()

    def test_s3_unsupported_versioning_query_preserves_object_and_reports_failure(self):
        store, client = self.s3()
        client.get_bucket_versioning.side_effect = RuntimeError("INVENTED_PRIVATE_CANARY")
        with self.assertRaises(storage.BlobStoreError) as caught:
            store.delete(self.ref)
        self.assertEqual(str(caught.exception), "S3 blob deletion is unconfirmed.")
        client.delete_object.assert_not_called()

    def test_s3_never_versioned_delete_verifies_absence_and_rechecks_versioning(self):
        store, client = self.s3()
        store.delete(self.ref)
        client.delete_object.assert_called_once_with(Bucket="invented-private-bucket",
                                                    Key="invented-prefix/" + self.ref.key)
        self.assertEqual(client.get_bucket_versioning.call_count, 2)
        client.head_object.assert_called_once()

    def test_s3_delete_failure_is_visible_and_does_not_claim_lifecycle_success(self):
        store, client = self.s3()
        client.delete_object.side_effect = RuntimeError("INVENTED_PRIVATE_CANARY")
        with self.assertRaises(storage.BlobStoreError) as caught:
            store.delete(self.ref)
        self.assertEqual(str(caught.exception), "S3 blob deletion is unconfirmed.")
        client.head_object.assert_not_called()

    def test_s3_marker_version_and_invalid_delete_replies_are_unconfirmed(self):
        store, client = self.s3()
        for response in ({"DeleteMarker": True}, {"DeleteMarker": False}, {"VersionId": "invented"}, None,
                         {"Error": "private"}, {"ResponseMetadata": None},
                         {"ResponseMetadata": {"HTTPStatusCode": 503}}):
            client.delete_object.return_value = response
            with self.assertRaises(storage.BlobStoreError):
                store.delete(self.ref)
        client.head_object.assert_not_called()

    def test_s3_contradictory_missing_object_response_is_unconfirmed(self):
        store, client = self.s3()
        client.head_object.side_effect = ClientError({"Error": {"Code": "404"},
            "ResponseMetadata": {"HTTPStatusCode": 503}}, "HeadObject")
        with self.assertRaises(storage.BlobStoreError):
            store.delete(self.ref)

    def test_s3_missing_status_and_delete_marker_head_responses_are_unconfirmed(self):
        store, client = self.s3()
        for metadata in ({}, {"HTTPStatusCode": 404, "HTTPHeaders": {"x-amz-delete-marker": "true"}},
                         {"HTTPStatusCode": 404, "HTTPHeaders": {"x-amz-version-id": "invented"}}):
            client.head_object.side_effect = ClientError({"Error": {"Code": "404"},
                "ResponseMetadata": metadata}, "HeadObject")
            with self.assertRaises(storage.BlobStoreError):
                store.delete(self.ref)

    def test_s3_recreated_object_and_denied_verification_are_unconfirmed(self):
        store, client = self.s3()
        for response in ({"ContentLength": 1}, ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")):
            client.head_object.side_effect = response if isinstance(response, Exception) else None
            client.head_object.return_value = response
            with self.assertRaises(storage.BlobStoreError):
                store.delete(self.ref)

    def test_s3_changed_bucket_versioning_after_delete_is_not_reported_complete(self):
        store, client = self.s3()
        client.get_bucket_versioning.side_effect = [{}, {"Status": "Enabled"}]
        with self.assertRaises(storage.BlobStoreError):
            store.delete(self.ref)
        client.delete_object.assert_called_once()

    def test_s3_mode_change_during_delete_preflight_never_sends_delete(self):
        store, client = self.s3()
        def late(**k):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return {}
        client.get_bucket_versioning.side_effect = late
        self.refuse(lambda: store.delete(self.ref))
        client.delete_object.assert_not_called()

    def test_policy_import_has_no_clients_ui_or_settings_side_effects(self):
        result = subprocess.run([sys.executable, "-c",
            'import sys; import app.storage_policy; assert not {"app.config", "app.llm", "streamlit", "boto3", "app.telemetry"}.intersection(sys.modules)'],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
