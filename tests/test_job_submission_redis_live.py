"""Crash/pressure tests own a temporary Redis; never alter a shared server."""
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from tests import hermetic  # noqa: F401
from tests.test_job_submission import SubmissionContract, SubmissionFailures, SubmissionExpiry
from app import job_queue
from app.job_queue import JobQueueError, RedisJobBackend

REDIS_SERVER = os.getenv('VA_LSE_TEST_REDIS_SERVER', '') or shutil.which('redis-server')


@unittest.skipUnless(REDIS_SERVER, 'temporary Redis tests require redis-server')
class TestIsolatedRedisSubmission(SubmissionContract, SubmissionFailures, SubmissionExpiry, unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='lse-redis-', dir='/tmp')
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)
        self.socket = self.directory / 'redis.sock'
        self.url = 'unix://' + str(self.socket)
        self.process = None
        self.addCleanup(self._stop)
        self._start()

    def _start(self):
        self.process = subprocess.Popen([
            REDIS_SERVER, '--port', '0', '--unixsocket', str(self.socket),
            '--unixsocketperm', '700', '--save', '', '--appendonly', 'yes',
            '--appendfsync', 'always', '--maxmemory-policy', 'noeviction',
            '--dir', str(self.directory), '--logfile', str(self.directory / 'server.log'),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import redis
        self.admin = redis.Redis(unix_socket_path=str(self.socket), decode_responses=True,
                                 socket_timeout=1, socket_connect_timeout=1)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and self.process.poll() is None:
            try:
                if self.admin.ping():
                    return
            except redis.RedisError:
                time.sleep(0.02)
        self.fail('temporary Redis did not start')

    def _stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if hasattr(self, 'admin'):
            self.admin.close()

    def _crash_and_restart(self):
        self.process.kill()
        self.process.wait(timeout=5)
        self.admin.close()
        self._start()

    def make_backend(self):
        return RedisJobBackend(self.url, prefix='test_' + uuid.uuid4().hex,
                               ttl_seconds=120, timeout_seconds=2)

    def test_memory_pressure_refuses_new_work_without_losing_accepted_jobs(self):
        backend = self.make_backend()
        accepted = backend.enqueue('evaluate', 'accepted synthetic input',
                                   request_id='accepted-reference', owner_id='owner')
        self.admin.config_set('maxmemory', 1)
        with self.assertRaises(JobQueueError):
            backend.enqueue('evaluate', 'new input', request_id='new-reference', owner_id='owner')
        self.assertEqual(backend.get(accepted.job_id).status, 'queued')
        self.assertEqual(backend.depth(), 1)
        self.assertEqual(backend.lookup_by_request_id('accepted-reference'), accepted.job_id)
        self.assertIsNone(backend.lookup_by_request_id('new-reference'))
        candidate = job_queue._submission('evaluate', 'new input', 'new-reference', 'owner')
        self.assertIsNone(backend.get(candidate.job_id))
        self.assertIsNone(self.admin.get(job_queue._payload_key(backend._prefix, candidate.job_id)))
        self.admin.config_set('maxmemory', 0)
        claimed, payload = backend.claim(['evaluate'], worker_id='worker')
        self.assertEqual(payload, 'accepted synthetic input')
        self.assertEqual(claimed.job_id, accepted.job_id)

    def test_crash_restart_preserves_queued_and_completed_retry_state(self):
        backend = self.make_backend()
        original = backend.enqueue('evaluate', 'synthetic input', request_id='reference', owner_id='owner')
        self.assertEqual(self.admin.config_get('appendfsync')['appendfsync'], 'always')
        self._crash_and_restart()
        retried = backend.enqueue('evaluate', 'synthetic input', request_id='reference', owner_id='owner')
        self.assertEqual(retried.job_id, original.job_id)
        self.assertEqual(backend.lookup_by_request_id('reference'), original.job_id)
        self.assertEqual(backend.depth(), 1)
        claimed, payload = backend.claim(['evaluate'], worker_id='worker')
        self.assertEqual(payload, 'synthetic input')
        backend.store_result(claimed.job_id, 'synthetic result', claim_token=claimed.claim_token)
        backend.complete(claimed.job_id, claim_token=claimed.claim_token)
        self._crash_and_restart()
        retried = backend.enqueue('evaluate', 'synthetic input', request_id='reference', owner_id='owner')
        self.assertEqual(retried.status, 'done')
        self.assertEqual(backend.get_result(original.job_id), 'synthetic result')
        self.assertEqual(backend.depth(), 0)

    def test_memory_pressure_can_confirm_retained_work_without_reference_repair(self):
        for status in ('queued', 'running', 'done'):
            with self.subTest(status=status):
                backend = self.make_backend()
                original = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
                if status != 'queued':
                    claimed, _ = backend.claim(['evaluate'], worker_id='worker')
                    if status == 'done':
                        backend.store_result(claimed.job_id, 'result', claim_token=claimed.claim_token)
                        backend.complete(claimed.job_id, claim_token=claimed.claim_token)
                reference_key = job_queue._recovery_key(backend._prefix, 'reference')
                self.admin.pexpire(reference_key, 1)
                time.sleep(0.02)
                self.assertIsNone(backend.lookup_by_request_id('reference'))
                self.admin.config_set('maxmemory', 1)
                try:
                    retried = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
                    self.assertEqual((retried.job_id, retried.status), (original.job_id, status))
                    self.assertFalse(retried.recovery_available)
                    self.assertIsNone(backend.lookup_by_request_id('reference'))
                    self.assertEqual(backend.depth(), 1 if status == 'queued' else 0)
                finally:
                    self.admin.config_set('maxmemory', 0)
                restored = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
                self.assertTrue(restored.recovery_available)
                self.assertEqual(backend.lookup_by_request_id('reference'), original.job_id)
                self.assertEqual(backend.depth(), 1 if status == 'queued' else 0)

    def test_acl_write_rejection_happens_before_partial_submission(self):
        import redis
        backend = self.make_backend()
        self.admin.execute_command('ACL', 'SETUSER', 'restricted', 'on', '>synthetic-only',
                                   '~' + backend._prefix + '*', '+@all', '-lpush')
        restricted = RedisJobBackend(self.url, prefix=backend._prefix, ttl_seconds=120)
        restricted._client = redis.Redis(unix_socket_path=str(self.socket),
                                         username='restricted', password='synthetic-only',
                                         decode_responses=True)
        self.addCleanup(restricted._client.close)
        with self.assertRaises(JobQueueError):
            restricted.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        self.assertEqual(backend.depth(), 0)
        self.assertIsNone(backend.lookup_by_request_id('reference'))
        candidate = job_queue._submission('evaluate', 'input', 'reference', 'owner')
        self.assertIsNone(backend.get(candidate.job_id))
        self.assertIsNone(self.admin.get(job_queue._payload_key(backend._prefix, candidate.job_id)))
