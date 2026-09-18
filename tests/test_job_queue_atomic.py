"""Fault injection runs the production Lua, including partial script failures."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, job_queue
import test_job_queue as queue_tests


def interrupted_script(index, after):
    # Redis does not roll back writes made before a Lua runtime error. Wrap each
    # real Redis command to test that every partial state remains discoverable.
    return f"""
local count = 0
local function call(...)
    count = count + 1
    if count == {index} and {str(not after).lower()} then error('injected') end
    local result = redis.call(...)
    if count == {index} and {str(after).lower()} then error('injected') end
    return result
end
""" + job_queue._TRANSITION_SCRIPT.replace("redis.call(", "call(")


class _AtomicFailures:
    def test_claim_transport_failure_before_and_after_each_request(self):
        for command in ("LINDEX", "EVAL"):
            for after in (False, True):
                with self.subTest(command=command, after=after):
                    backend = self.make_backend()
                    job = backend.enqueue("evaluate", "evidence")
                    original = backend._command

                    def broken(*args):
                        if args[0] == command:
                            if after:
                                original(*args)
                            raise job_queue.JobQueueError("lost connection")
                        return original(*args)

                    with patch.object(backend, "_command", side_effect=broken):
                        with self.assertRaises(job_queue.JobQueueError):
                            backend.claim(["evaluate"], worker_id="lost")
                    with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
                        backend.requeue_stale()
                    self.assertEqual(backend.depth(), 1)
                    recovered, payload = backend.claim(["evaluate"], worker_id="replacement")
                    self.assertEqual(recovered.job_id, job.job_id)
                    self.assertEqual(payload, "evidence")

    def test_claim_script_failure_at_every_command_boundary(self):
        # TYPE x5, GET metadata, LINDEX, GET payload, ZADD, SET, EXPIRE,
        # DEL result, LREM queue. Neither pre-lease nor post-lease failures lose it.
        self._interrupt_transition("claim", 13)

    def test_recovery_script_failure_at_every_command_boundary(self):
        # TYPE x5, GET, ZSCORE, LREM, LPUSH, DEL, SET, ZREM.
        self._interrupt_transition("recover", 12)

    def test_draining_requeue_failure_at_every_command_boundary(self):
        # TYPE x5, GET, LREM, LPUSH, DEL, SET, ZREM.
        self._interrupt_transition("requeue", 11)

    def _interrupt_transition(self, operation, commands):
        for index in range(1, commands + 1):
            for after in (False, True):
                with self.subTest(operation=operation, index=index, after=after):
                    backend = self.make_backend()
                    job = backend.enqueue("evaluate", "evidence")
                    if operation != "claim":
                        owned, _ = backend.claim(["evaluate"], worker_id="lost")
                    script = interrupted_script(index, after)
                    with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
                        with patch.object(job_queue, "_TRANSITION_SCRIPT", script):
                            if operation == "requeue":
                                with self.assertLogs("app.job_queue", level="ERROR"):
                                    self.assertFalse(backend.requeue(
                                        job.job_id, claim_token=owned.claim_token
                                    ))
                            else:
                                with self.assertRaises(job_queue.JobQueueError):
                                    if operation == "claim":
                                        backend.claim(["evaluate"], worker_id="lost")
                                    else:
                                        backend.requeue_stale()
                        backend.requeue_stale()
                    self.assertEqual(backend.depth(), 1)
                    current, payload = backend.claim(["evaluate"], worker_id="replacement")
                    self.assertEqual(current.job_id, job.job_id)
                    self.assertEqual(payload, "evidence")

    def test_two_sweepers_do_not_duplicate_recovery(self):
        backend = self.make_backend()
        backend.enqueue("evaluate", "evidence")
        backend.claim(["evaluate"], worker_id="lost")
        command = backend._command

        def sweep_after_scan(*args):
            result = command(*args)
            if args[0] == "ZRANGEBYSCORE" and result:
                with patch.object(backend, "_command", side_effect=command):
                    self.assertEqual(backend.requeue_stale(), 1)
            return result

        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0), patch.object(
            backend, "_command", side_effect=sweep_after_scan
        ):
            self.assertEqual(backend.requeue_stale(), 0)
        self.assertEqual(backend.depth(), 1)

    def test_lost_recovery_reply_is_safe_to_retry(self):
        backend = self.make_backend()
        backend.enqueue("evaluate", "evidence")
        backend.claim(["evaluate"], worker_id="lost")
        command = backend._command

        def lose_reply(*args):
            result = command(*args)
            if args[0] == "EVAL":
                raise job_queue.JobQueueError("response lost after commit")
            return result

        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            with patch.object(backend, "_command", side_effect=lose_reply):
                with self.assertRaises(job_queue.JobQueueError):
                    backend.requeue_stale()
            self.assertEqual(backend.requeue_stale(), 0)
        self.assertEqual(backend.depth(), 1)

    def test_heartbeat_between_stale_scan_and_recovery_is_not_overwritten(self):
        backend = self.make_backend()
        backend.enqueue("evaluate", "evidence")
        with patch.object(job_queue, "_now", return_value=100):
            owned, _ = backend.claim(["evaluate"], worker_id="live")
        command = backend._command

        def renew_after_scan(*args):
            result = command(*args)
            if args[0] == "ZRANGEBYSCORE" and result:
                backend.set_progress(owned.job_id, 0.5, "alive", claim_token=owned.claim_token)
            return result

        with patch.object(job_queue, "_now", return_value=1000), patch.object(
            config, "JOB_QUEUE_LEASE_SECONDS", 10
        ), patch.object(backend, "_command", side_effect=renew_after_scan):
            self.assertEqual(backend.requeue_stale(), 0)
        self.assertEqual(backend.get(owned.job_id).status, "running")
        self.assertEqual(backend.depth(), 0)

    def test_reclaim_between_ownership_read_and_write_is_fenced(self):
        backend = self.make_backend()
        backend.enqueue("evaluate", "evidence")
        old, _ = backend.claim(["evaluate"], worker_id="old")
        read = backend._read_record

        def reclaim_after_read(job_id):
            result = read(job_id)
            with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
                backend.requeue_stale()
            backend.claim(["evaluate"], worker_id="new")
            return result

        with patch.object(backend, "_read_record", side_effect=reclaim_after_read):
            with self.assertRaises(job_queue.JobLeaseLost):
                backend.set_progress(old.job_id, 0.9, "late", claim_token=old.claim_token)
        self.assertEqual(backend.get(old.job_id).worker_id, "new")
        self.assertEqual(backend.get(old.job_id).progress, 0)

    def test_missing_payload_is_explicitly_failed(self):
        backend = self.make_backend()
        job = backend.enqueue("evaluate", "evidence")
        backend._command("DEL", job_queue._payload_key(backend._prefix, job.job_id))
        self.assertIsNone(backend.claim(["evaluate"], worker_id="w"))
        self.assertEqual(backend.get(job.job_id).error_class, "JobPayloadMissing")
        self.assertEqual(backend.depth(), 0)

    def test_wrong_key_type_does_not_pop_the_job(self):
        backend = self.make_backend()
        job = backend.enqueue("evaluate", "evidence")
        lease = job_queue._lease_key(backend._prefix, "evaluate")
        backend._command("SET", lease, "wrong type")
        with self.assertRaises(job_queue.JobQueueError):
            backend.claim(["evaluate"], worker_id="w")
        self.assertEqual(backend.depth(), 1)
        self.assertEqual(backend.get(job.job_id).status, "queued")
        backend._command("DEL", lease)
        self.assertIsNotNone(backend.claim(["evaluate"], worker_id="w"))


class TestRedisAtomic(_AtomicFailures, unittest.TestCase):
    make_backend = queue_tests.TestRedisBackend.make_backend


class TestUpstashAtomic(_AtomicFailures, unittest.TestCase):
    make_backend = queue_tests.TestUpstashBackend.make_backend
