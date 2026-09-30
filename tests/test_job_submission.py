"""Producer retries, concurrency, admission and partial-write regressions."""
import sys
import time
from dataclasses import replace
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


class SubmissionExpiry:
    def test_expired_entries_do_not_block_admission_without_a_worker(self):
        backend = self.make_backend()
        with patch.object(config, 'JOB_QUEUE_MAX_PENDING', 2):
            expired = backend.enqueue('evaluate', 'expired', request_id='expired', owner_id='owner')
            live = backend.enqueue('evaluate', 'live', request_id='live', owner_id='owner')
            for key in (job_queue._meta_key(backend._prefix, expired.job_id),
                        job_queue._payload_key(backend._prefix, expired.job_id),
                        job_queue._recovery_key(backend._prefix, expired.request_id)):
                backend._command('PEXPIRE', key, 1)
            time.sleep(0.02)
            self.assertIsNone(backend.get(expired.job_id))
            fresh = backend.enqueue('evaluate', 'fresh', request_id='fresh', owner_id='owner')
        self.assertEqual(backend.depth(), 2)
        claimed, payload = backend.claim(['evaluate'], worker_id='worker')
        self.assertEqual((claimed.job_id, payload), (live.job_id, 'live'))
        claimed, payload = backend.claim(['evaluate'], worker_id='worker')
        self.assertEqual((claimed.job_id, payload), (fresh.job_id, 'fresh'))

    def test_retry_restores_expired_reference_for_retained_job(self):
        backend = self.make_backend()
        original = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        claimed, _ = backend.claim(['evaluate'], worker_id='worker')
        reference_key = job_queue._recovery_key(backend._prefix, 'reference')
        backend._command('PEXPIRE', reference_key, 1)
        time.sleep(0.02)
        self.assertIsNone(backend.lookup_by_request_id('reference'))
        retried = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        self.assertEqual(retried.job_id, original.job_id)
        self.assertEqual(retried.claim_token, claimed.claim_token)
        self.assertEqual(backend.lookup_by_request_id('reference'), original.job_id)
        remaining = backend._command('PTTL', job_queue._meta_key(backend._prefix, original.job_id))
        self.assertAlmostEqual(backend._command('PTTL', reference_key), remaining, delta=100)
        self.assertEqual(backend.depth(), 0)

    def test_reference_repair_failure_still_confirms_retained_work(self):
        backend = self.make_backend()
        original = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        reference_key = job_queue._recovery_key(backend._prefix, 'reference')
        backend._command('DEL', reference_key)
        script = job_queue._ENQUEUE_SCRIPT.replace(
            "return redis.call('SET', KEYS[4], id, 'PX', remaining)",
            "error('synthetic reference repair refusal')")
        with patch.object(job_queue, '_ENQUEUE_SCRIPT', script):
            retried = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        self.assertEqual(retried.job_id, original.job_id)
        self.assertFalse(retried.recovery_available)
        self.assertNotIn('recovery_available', retried.to_json())
        self.assertIsNone(backend.lookup_by_request_id('reference'))
        self.assertEqual(backend.depth(), 1)
        restored = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        self.assertTrue(restored.recovery_available)
        self.assertEqual(backend.lookup_by_request_id('reference'), original.job_id)
        self.assertEqual(backend.depth(), 1)


class TestInProcessSubmission(SubmissionContract, unittest.TestCase):
    def make_backend(self):
        from tests.test_job_queue import TestInProcessBackend
        return TestInProcessBackend.make_backend(self)


class TestRedisSubmission(SubmissionContract, SubmissionExpiry, unittest.TestCase):
    def make_backend(self):
        from tests.test_job_queue import TestRedisBackend
        return TestRedisBackend.make_backend(self)


class TestUpstashSubmission(SubmissionContract, SubmissionExpiry, unittest.TestCase):
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
    def _pending(self):
        return dict(payload='original payload', request_id='original-reference', owner_id='owner',
                    kind='evaluate', files=1, pages=1, condition=None, sources=['Upload'],
                    action_label='Evaluation')

    def test_check_earlier_submission_needs_no_current_form_inputs(self):
        from app.views import job_runner
        backend = job_queue.InProcessJobBackend(prefix='check-test', ttl_seconds=60)
        ui = MagicMock()
        ui.session_state = {'eval_queue_submission': self._pending()}
        ui.button.side_effect = lambda label, **kwargs: label == 'Check earlier submission'
        outcome = job_runner.QueueOutcome(ok=True, request_id='original-reference')
        with patch.object(job_runner, 'st', ui), \
                patch.object(job_runner, 'get_job_backend', return_value=backend), \
                patch.object(job_runner.pilot, 'current_owner', return_value='owner'), \
                patch.object(job_runner, '_encode_payload') as encode, \
                patch.object(job_runner, '_poll', return_value=outcome), \
                patch.object(job_runner, 'run_log_event'), \
                patch.object(job_runner.pilot, 'display') as display:
            job_runner._render_uncertain_submission('eval')
        encode.assert_not_called()
        claimed, payload = backend.claim(['evaluate'], worker_id='worker')
        self.assertEqual((claimed.request_id, payload), ('original-reference', 'original payload'))
        self.assertNotIn('eval_queue_submission', ui.session_state)
        self.assertIn('original-reference', display.call_args.args[0])

    def test_unavailable_reference_is_explained_without_hiding_the_job(self):
        from app.views import job_runner
        from app.job_payload import EvaluateJob
        backend = job_queue.InProcessJobBackend(prefix='warning-test', ttl_seconds=60)
        record = backend.enqueue('evaluate', 'input', request_id='reference', owner_id='owner')
        ui = MagicMock()
        ui.session_state = {}
        with patch.object(job_runner, 'st', ui), \
                patch.object(job_runner, 'get_job_backend', return_value=backend), \
                patch.object(backend, 'enqueue', side_effect=[replace(record, recovery_available=False), record]), \
                patch.object(job_runner.pilot, 'current_owner', return_value='owner'), \
                patch.object(job_runner, '_encode_payload', return_value='input'), \
                patch.object(job_runner, '_poll', side_effect=[
                    job_runner.QueueOutcome(ok=False, still_running=True),
                    job_runner.QueueOutcome(ok=True, request_id='reference')]) as poll, \
                patch.object(job_runner, 'run_log_event'), \
                patch.object(job_runner.pilot, 'display') as display:
            result = job_runner.submit_job(slot='eval', job=EvaluateJob(statement_text='input', records=[]),
                                           request_id='reference', condition=None, sources=[], files=0,
                                           pages=0, action_label='Evaluation')
            self.assertTrue(result.still_running)
            self.assertEqual(ui.session_state['eval_queue_submission']['confirmed_job_id'], record.job_id)
            ui.button.side_effect = lambda label, **kwargs: label == 'Check earlier submission'
            job_runner._render_uncertain_submission('eval')
        self.assertEqual(poll.call_count, 2)
        self.assertNotIn('eval_queue_submission', ui.session_state)
        warnings = [call.args[0] for call in display.call_args_list if call.kwargs.get('method') == 'warning']
        self.assertTrue(any('Keep this tab open' in warning for warning in warnings))
        self.assertEqual(backend.depth(), 1)

    def test_discard_requires_explicit_acknowledgement_and_allows_current_inputs(self):
        from app.views import job_runner
        from app.job_payload import EvaluateJob
        for acknowledgement in (False, True):
            with self.subTest(acknowledgement=acknowledgement):
                backend = job_queue.InProcessJobBackend(prefix='discard-test', ttl_seconds=60)
                original = backend.enqueue('evaluate', 'original payload',
                                           request_id='original-reference', owner_id='owner')
                ui = MagicMock()
                ui.session_state = {'eval_queue_submission': self._pending()}
                ui.button.side_effect = lambda label, **kwargs: label == 'Discard earlier submission'
                ui.checkbox.return_value = acknowledgement
                with patch.object(job_runner, 'st', ui), \
                        patch.object(job_runner, 'get_job_backend', return_value=backend), \
                        patch.object(job_runner.pilot, 'current_owner', return_value='owner'), \
                        patch.object(job_runner, '_encode_payload', return_value='changed payload'), \
                        patch.object(job_runner, '_poll', return_value=job_runner.QueueOutcome(ok=True)), \
                        patch.object(job_runner, 'run_log_event'), \
                        patch.object(job_runner.pilot, 'display'):
                    job_runner._render_uncertain_submission('eval')
                    self.assertEqual('eval_queue_submission' in ui.session_state, not acknowledgement)
                    if acknowledgement:
                        job_runner.submit_job(slot='eval', job=EvaluateJob(statement_text='Changed', records=[]),
                                              request_id='new-reference', condition=None, sources=[], files=0,
                                              pages=0, action_label='Evaluation')
                self.assertIsNotNone(backend.get(original.job_id))
                self.assertEqual(backend.depth(), 2 if acknowledgement else 1)
                if acknowledgement:
                    backend.claim(['evaluate'], worker_id='worker')
                    fresh, payload = backend.claim(['evaluate'], worker_id='worker')
                    self.assertEqual((fresh.request_id, payload), ('new-reference', 'changed payload'))

    def test_completed_callers_show_the_confirmed_reference(self):
        from app.views import job_runner, evaluate_view, draft_view
        for module in (evaluate_view, draft_view):
            with self.subTest(view=module.__name__), \
                    patch.object(module, 'check_shutdown_gate', return_value=True), \
                    patch.object(module, 'audit_record_meta', return_value=([], 0, 0)), \
                    patch.object(module.pilot, 'display') as display, \
                    patch.object(job_runner, 'worker_config_error', return_value=''), \
                    patch.object(job_runner, 'submit_job',
                                 return_value=job_runner.QueueOutcome(ok=True, request_id='confirmed-reference')):
                if module is evaluate_view:
                    with patch.object(module, 'new_run_request_id', return_value='new-click-reference'), \
                            patch.object(module, 'audit_condition_for_slot', return_value=None):
                        module._run_evaluation_queued('input', [], {})
                else:
                    module._run_draft_queued(rid='new-click-reference', records=[], condition='',
                                            claim_type='', witness={}, observations='')
                self.assertIn('confirmed-reference', display.call_args.args[0])
                self.assertNotIn('new-click-reference', display.call_args.args[0])

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
