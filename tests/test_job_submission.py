"""Producer retries, concurrency, admission and partial-write regressions."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from tests import hermetic  # noqa: F401
from app import config, job_queue
from app.job_queue import JobQueueError


class SubmissionContract:
    def test_submission_also_commits_recovery_reference_and_owner(self):
        backend = self.make_backend()
        job = backend.enqueue('evaluate', 'synthetic input', request_id='reference', owner_id='owner')
        self.assertEqual(backend.lookup_by_request_id('reference'), job.job_id)
        self.assertEqual(job.owner_id, 'owner')

    def test_concurrent_retries_create_one_claimable_job(self):
        backend = self.make_backend()
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(lambda _: backend.enqueue('evaluate', 'synthetic input',
                                    request_id='reference', owner_id='owner'), range(8)))
        self.assertEqual(len({job.job_id for job in jobs}), 1)
        self.assertEqual(backend.depth(), 1)
        claimed, payload = backend.claim(['evaluate'], worker_id='worker')
        self.assertEqual(payload, 'synthetic input')
        self.assertEqual(claimed.job_id, jobs[0].job_id)

    def test_changed_inputs_or_identity_cannot_retarget_a_reference(self):
        backend = self.make_backend()
        original = backend.enqueue('evaluate', 'original input', request_id='reference', owner_id='owner')
        for kind, payload, owner in [('evaluate', 'changed input', 'owner'),
                                     ('evaluate', 'original input', 'other'),
                                     ('draft', 'original input', 'owner')]:
            with self.subTest(kind=kind, owner=owner), self.assertRaises(JobQueueError):
                backend.enqueue(kind, payload, request_id='reference', owner_id=owner)
        self.assertEqual(backend.depth(), 1)
        self.assertEqual(backend.lookup_by_request_id('reference'), original.job_id)

    def test_running_and_completed_work_is_returned_without_resubmission(self):
        backend = self.make_backend()
        original = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        claimed, _ = backend.claim(['evaluate'], worker_id='worker')
        duplicate = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        self.assertEqual(duplicate.status, 'running')
        self.assertEqual(duplicate.claim_token, claimed.claim_token)
        backend.store_result(claimed.job_id, 'synthetic result', claim_token=claimed.claim_token)
        backend.complete(claimed.job_id, claim_token=claimed.claim_token)
        duplicate = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        self.assertEqual(duplicate.status, 'done')
        self.assertEqual(duplicate.job_id, original.job_id)
        # Redis cjson rounds timestamps when workers update the metadata.
        self.assertAlmostEqual(duplicate.created_at, original.created_at, places=3)
        self.assertEqual(backend.get_result(original.job_id), 'synthetic result')
        self.assertEqual(backend.depth(), 0)

    def test_concurrent_admission_is_bounded_but_existing_retries_are_allowed(self):
        backend = self.make_backend()
        def submit(i):
            try:
                return backend.enqueue('evaluate', 'input', request_id=f'reference-{i}', owner_id='owner')
            except JobQueueError:
                return None
        with patch.object(config, 'JOB_QUEUE_MAX_PENDING', 2), ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [job for job in pool.map(submit, range(8)) if job]
            self.assertEqual(len(jobs), 2)
            repeated = backend.enqueue('evaluate', 'input', request_id=jobs[0].request_id, owner_id='owner')
        self.assertEqual(repeated.job_id, jobs[0].job_id)
        self.assertEqual(backend.depth(), 2)

    def test_oversized_input_is_rejected_before_creating_state(self):
        backend = self.make_backend()
        with patch.object(config, 'JOB_QUEUE_MAX_PAYLOAD_BYTES', 8), self.assertRaises(JobQueueError):
            backend.enqueue('evaluate', '\u00e9' * 5, request_id='reference', owner_id='owner')
        self.assertEqual(backend.depth(), 0)
        self.assertIsNone(backend.lookup_by_request_id('reference'))

    def test_legacy_reference_helper_cannot_overwrite_a_submission(self):
        backend = self.make_backend()
        job = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        backend.set_recovery_index('reference', job.job_id)
        with self.assertRaises(JobQueueError):
            backend.set_recovery_index('reference', 'another-job')
        self.assertEqual(backend.lookup_by_request_id('reference'), job.job_id)


class TestInProcessSubmission(SubmissionContract, unittest.TestCase):
    def make_backend(self):
        from tests.test_job_queue import TestInProcessBackend
        return TestInProcessBackend.make_backend(self)


class TestRedisSubmission(SubmissionContract, unittest.TestCase):
    def make_backend(self):
        from tests.test_job_queue import TestRedisBackend
        return TestRedisBackend.make_backend(self)


class TestUpstashSubmission(SubmissionContract, unittest.TestCase):
    def make_backend(self):
        from tests.test_job_queue import TestUpstashBackend
        return TestUpstashBackend.make_backend(self)


def interrupt_submission(index, after):
    return f'''
local count = 0
local function call(...)
    count = count + 1
    if count == {index} and {str(not after).lower()} then error('injected') end
    local result = redis.call(...)
    if count == {index} and {str(after).lower()} then error('injected') end
    return result
end
''' + job_queue._ENQUEUE_SCRIPT.replace('redis.call(', 'call(')


class SubmissionFailures:
    def test_failure_before_and_after_every_command_leaves_no_partial_job(self):
        for index in range(1, 13):
            for after in (False, True):
                with self.subTest(index=index, after=after):
                    backend = self.make_backend()
                    candidate = job_queue._submission('evaluate', 'input', 'reference', 'owner')
                    with patch.object(job_queue, '_ENQUEUE_SCRIPT', interrupt_submission(index, after)):
                        with self.assertRaises(JobQueueError):
                            backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
                    self.assertEqual(backend.depth(), 0)
                    self.assertIsNone(backend.get(candidate.job_id))
                    self.assertIsNone(backend._command('GET', job_queue._payload_key(backend._prefix, candidate.job_id)))
                    self.assertIsNone(backend.lookup_by_request_id('reference'))
                    retried = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
                    self.assertEqual(retried.job_id, candidate.job_id)
                    self.assertEqual(backend.depth(), 1)

    def test_lost_reply_after_commit_returns_the_original_job_on_retry(self):
        backend = self.make_backend()
        command = backend._command
        def lose_reply(*args):
            command(*args)
            raise JobQueueError('synthetic lost response')
        with patch.object(backend, '_command', side_effect=lose_reply), self.assertRaises(JobQueueError):
            backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        original = backend.lookup_by_request_id('reference')
        job = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        self.assertEqual(job.job_id, original)
        self.assertEqual(backend.depth(), 1)

    def test_wrong_key_type_is_rejected_before_creating_any_job(self):
        backend = self.make_backend()
        backend._command('SET', job_queue._queue_key(backend._prefix, 'evaluate'), 'wrong type')
        with self.assertRaises(JobQueueError):
            backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        candidate = job_queue._submission('evaluate', 'input', 'reference', 'owner')
        self.assertIsNone(backend.get(candidate.job_id))
        self.assertIsNone(backend.lookup_by_request_id('reference'))


class TestRedisSubmissionFailures(SubmissionFailures, unittest.TestCase):
    def make_backend(self):
        from tests.test_job_queue import TestRedisBackend
        return TestRedisBackend.make_backend(self)


class TestUpstashSubmissionFailures(SubmissionFailures, unittest.TestCase):
    def make_backend(self):
        from tests.test_job_queue import TestUpstashBackend
        return TestUpstashBackend.make_backend(self)


class TestBrowserSubmissionRetry(unittest.TestCase):
    def test_lost_reply_retains_original_inputs_and_reference(self):
        from app.views import job_runner
        from app.job_payload import EvaluateJob
        backend = job_queue.InProcessJobBackend(prefix='retry-test', ttl_seconds=60)
        enqueue = backend.enqueue
        first = True
        def lose_first_reply(*args, **kwargs):
            nonlocal first
            result = enqueue(*args, **kwargs)
            if first:
                first = False
                raise JobQueueError('synthetic lost reply')
            return result
        state = {}
        ui = MagicMock()
        ui.session_state = state
        arguments = dict(slot='eval', job=EvaluateJob(statement_text='Original input', records=[]),
                         request_id='original-reference', condition='original condition', sources=['Upload'],
                         files=1, pages=1, action_label='Evaluation')
        with patch.object(job_runner, 'st', ui), patch.object(job_runner, 'get_job_backend', return_value=backend), \
                patch.object(job_runner.pilot, 'current_owner', return_value='owner'), \
                patch.object(job_runner, '_encode_payload', return_value='original payload') as encode, \
                patch.object(backend, 'enqueue', side_effect=lose_first_reply), \
                patch.object(job_runner, '_poll', return_value=MagicMock(ok=True, still_running=False)), \
                patch.object(job_runner, 'run_log_event'), \
                patch.object(job_runner, 'report_failure', return_value='synthetic failure'):
            self.assertIsNone(job_runner.submit_job(**arguments))
            self.assertIn('eval_queue_submission', state)
            arguments['request_id'] = 'different-reference'
            arguments['job'] = EvaluateJob(statement_text='Changed input', records=[])
            job_runner.submit_job(**arguments)
        encode.assert_called_once()
        self.assertNotIn('eval_queue_submission', state)
        self.assertEqual(backend.depth(), 1)
        claimed, payload = backend.claim(['evaluate'], worker_id='worker')
        self.assertEqual(claimed.request_id, 'original-reference')
        self.assertEqual(payload, 'original payload')
        self.assertIsNone(backend.lookup_by_request_id('different-reference'))
