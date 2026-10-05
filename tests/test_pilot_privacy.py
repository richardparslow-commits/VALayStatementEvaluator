"""Synthetic R08 lifecycle/retention regressions; no provider or account calls."""
from __future__ import annotations

import contextvars
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.test_controlled_pilot import approval
from app import pilot, log_retention, audit, audit_backup, audit_restore, logging_config, run_log
from app.documents import document_from_text

CANARY = 'SYNTHETIC_R08_PRIVATE_RECORD_CANARY'


def line(moment, **fields):
    return json.dumps({'timestamp': datetime.fromtimestamp(moment, timezone.utc).isoformat(), **fields}) + '\n'


class TestLogAge(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'app.log'
        self.now = time.time()

    def handler(self, **kwargs):
        h = log_retention.PilotLogHandler(str(self.path), maxBytes=kwargs.get('maxBytes', 2000),
                                          backupCount=kwargs.get('backupCount', 3), days=1)
        h.setFormatter(logging.Formatter('%(message)s'))
        self.addCleanup(h.close)
        return h

    def emit(self, h, text='Application event'):
        h.handle(logging.LogRecord('test', logging.INFO, '', 0, line(time.time(), message=text).rstrip(), (), None))

    def test_startup_expires_live_file_even_with_recent_mtime(self):
        self.path.write_text(line(self.now - 86401, request_id='req_aaaaaaaaaaaa'))
        os.utime(self.path, (self.now, self.now))
        self.handler()
        self.assertFalse(self.path.exists())

    def test_exact_expiry_removes_open_live_file_and_writer_reopens(self):
        h = self.handler()
        self.emit(h)
        start = h._oldest
        h.prune(now=start + 86400)
        self.assertFalse(self.path.exists())
        self.emit(h, 'New event')
        self.assertIn('New event', self.path.read_text())
        self.assertEqual(len(self.path.read_text().splitlines()), 1)

    def test_rotations_beyond_current_count_expire_without_touching_other_files(self):
        old = self.path.with_name('app.log.99')
        recent = self.path.with_name('app.log.1')
        other = self.path.with_name('private-case.txt')
        old.write_text(line(self.now - 86401))
        recent.write_text(line(self.now - 60))
        other.write_text(CANARY)
        self.handler()
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())
        self.assertEqual(other.read_text(), CANARY)

    def test_unknown_legacy_and_future_age_are_not_preserved(self):
        for data in ('unknown age\n', '{"timestamp":"bad"}\n', line(self.now + 86400),
                     line(self.now).replace('+00:00', ''), 'x' * 4097):
            with self.subTest(data=data[:30]):
                self.path.write_text(data)
                h = self.handler()
                self.assertFalse(self.path.exists())
                h.close()

    def test_continuing_writes_do_not_renew_live_file_age(self):
        self.path.write_text(line(self.now - 86399))
        h = self.handler()
        self.emit(h)
        h.prune(now=self.now + 2)
        self.assertFalse(self.path.exists())

    def test_size_rotation_still_bounds_storage(self):
        h = self.handler(maxBytes=180, backupCount=2)
        for _ in range(20):
            self.emit(h, 'event ' * 10)
        self.assertLessEqual(len(list(self.path.parent.glob('app.log*'))), 3)
        self.assertTrue(self.path.with_name('app.log.1').exists())

    def test_idle_sweep_executes_without_another_event(self):
        h = self.handler()
        self.emit(h)
        # Time-travel only during a sweep: exercise the registered idle path.
        with patch.object(log_retention.time, 'time', return_value=self.now + 86402):
            log_retention.sweep_registered()
        self.assertFalse(self.path.exists())
        self.assertTrue(log_retention.retention_health()['active'])

    def test_background_scheduler_really_cleans_without_another_write(self):
        # Fresh process gives the scheduler a short test interval before its
        # first wait, without disturbing the other tests' live singleton.
        script = '''
import json, logging, time
from datetime import datetime, timezone
from pathlib import Path
from app import log_retention
log_retention.SWEEP_SECONDS = 0.02
path = Path(__import__('sys').argv[1])
h = log_retention.PilotLogHandler(str(path), maxBytes=2000, backupCount=2, days=1)
payload = json.dumps({'timestamp': datetime.fromtimestamp(time.time()-86401, timezone.utc).isoformat()})
h.handle(logging.LogRecord('test', logging.INFO, '', 0, payload, (), None))
assert path.exists()
deadline = time.monotonic() + 2
while path.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
assert not path.exists()
h.close()
print('IDLE_SWEEP_OK')
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.path)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('IDLE_SWEEP_OK', result.stdout)

    def test_restart_does_not_renew_retention(self):
        h = self.handler()
        self.emit(h)
        h.close()
        with patch.object(log_retention.time, 'time', return_value=self.now + 86402):
            self.handler()
        self.assertFalse(self.path.exists())

    def test_cleanup_failure_is_visible_without_exception_content(self):
        h = self.handler()
        self.emit(h)
        with patch.object(Path, 'unlink', side_effect=PermissionError(CANARY)):
            with self.assertRaises(PermissionError):
                h.prune(now=self.now + 86402)
        self.assertTrue(h.failed)
        self.assertTrue(log_retention.retention_health()['failed'])
        self.assertNotIn(CANARY, repr(log_retention.retention_health()))

    def test_file_write_failure_drops_caller_text_from_stderr(self):
        h = self.handler()
        with patch.object(h, '_open', side_effect=OSError(CANARY)), patch('sys.stderr', new_callable=io.StringIO) as stderr:
            self.emit(h, CANARY)
        self.assertEqual(stderr.getvalue(), '')
        self.assertTrue(h.write_failed)

    def test_writer_and_cleanup_use_same_lock(self):
        h = self.handler(maxBytes=100000)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.emit, h) for _ in range(100)]
            futures += [executor.submit(h.prune) for _ in range(30)]
            for future in futures:
                future.result()
        self.assertEqual(len(self.path.read_text().splitlines()), 100)
        for item in self.path.read_text().splitlines():
            self.assertEqual(json.loads(item)['message'], 'Application event')

    def test_files_are_private_and_symlink_writes_are_refused(self):
        h = self.handler()
        self.emit(h)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        h.close()
        other = self.path.parent / 'other.txt'
        other.write_text(line(self.now))
        self.path.unlink()
        self.path.symlink_to(other)
        with self.assertRaises(OSError):
            self.handler()
        self.assertNotIn(CANARY, other.read_text())

    def test_retained_rotations_get_private_modes_before_admission(self):
        for name in ('app.log', 'app.log.1', 'app.log.99'):
            path = self.path.parent / name
            path.write_text(line(self.now))
            path.chmod(0o644)
        self.handler()
        for path in self.path.parent.glob('app.log*'):
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_non_regular_or_foreign_owned_rotations_are_refused(self):
        os.mkfifo(self.path.with_name('app.log.1'))
        with self.assertRaises(OSError):
            self.handler()
        self.path.with_name('app.log.1').unlink()
        self.path.write_text(line(self.now))
        with patch.object(log_retention.os, 'geteuid', return_value=os.geteuid() + 1), self.assertRaises(OSError):
            self.handler()

    def test_policy_requires_explicit_bounded_days(self):
        for value in ('', '0', '-1', '31', '7.0', 'true'):
            with patch.dict(os.environ, {'VA_LSE_PILOT_LOG_RETENTION_DAYS': value}), self.assertRaises(ValueError):
                log_retention.retention_days()
        with patch.dict(os.environ, {'VA_LSE_PILOT_LOG_RETENTION_DAYS': '7'}):
            self.assertEqual(log_retention.retention_days(), 7)

    def test_all_three_sinks_are_content_free_and_age_limited(self):
        env = {'VA_LSE_MODE': 'controlled-pilot', 'VA_LSE_PILOT_LOG_RETENTION_DAYS': '1',
               'VA_LSE_LOG_DIR': str(self.path.parent), 'VA_LSE_AUDIT_LOG_DIR': str(self.path.parent),
               'VA_LSE_RUN_LOG_DIR': str(self.path.parent)}
        # Restore logger configuration after the pilot env is restored.
        self.addCleanup(lambda: audit.configure_audit_logging(log_dir=tempfile.gettempdir(), force=True))
        self.addCleanup(lambda: logging_config.configure_logging(log_dir='', force=True))
        with patch.dict(os.environ, env):
            app_logger = logging_config.configure_logging(force=True)
            audit.configure_audit_logging(force=True)
            app_logger.error(CANARY, extra={'record_pages': 3, 'private': CANARY})
            audit.audit_event('evaluate', 'error', request_id='req_aaaaaaaaaaaa', error=ValueError(CANARY), condition=CANARY)
            run_log.run_log_event('draft', 'error', request_id='req_aaaaaaaaaaaa', error=CANARY, pages=3)
            paths = [self.path.parent / name for name in ('app.log', 'audit.log', 'runs.jsonl')]
            for path in paths:
                self.assertTrue(path.exists())
                self.assertNotIn(CANARY, path.read_text())
            with patch.object(log_retention.time, 'time', return_value=self.now + 86402):
                log_retention.sweep_registered()
            self.assertTrue(all(not path.exists() for path in paths))


class TestPolicyAdmission(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(lambda: audit.configure_audit_logging(log_dir=tempfile.gettempdir(), force=True))
        self.addCleanup(lambda: logging_config.configure_logging(log_dir='', force=True))
        context = patch.dict(os.environ, {'VA_LSE_MODE': 'controlled-pilot',
            'VA_LSE_PILOT_LOG_RETENTION_DAYS': '7', 'VA_LSE_LOG_DIR': directory.name,
            'VA_LSE_AUDIT_LOG_DIR': directory.name, 'VA_LSE_RUN_LOG_DIR': directory.name})
        context.start()
        self.addCleanup(context.stop)
        logging_config.configure_logging(force=True)
        audit.configure_audit_logging(force=True)

    def test_server_initializes_cleanup_before_cli_and_without_browser(self):
        # Run the process entrypoint, with CLI replaced only after initialize.
        from app import pilot_server
        from tests.pilot_budget_fixtures import install_budget
        data = approval()
        install_budget(self, data)
        with patch.dict(os.environ, {'VA_LSE_HEALTH_PORT': '0'}), \
                patch('app.upload_temp.validate'), \
                patch('streamlit.web.cli.main') as cli, patch.object(sys, 'argv', ['fixture']), \
                patch.object(pilot, 'load_approval', return_value=data):
            pilot_server.main(['--server.port=8501'])
            self.assertEqual(sys.argv, ['streamlit', 'run', 'app/pilot_asgi.py', '--server.port=8501'])
        cli.assert_called_once()
        self.assertTrue(log_retention.retention_health()['active'])

    def test_fresh_process_cleans_existing_volume_without_browser(self):
        script = r'''from tests import hermetic
import os, sys, json
from pathlib import Path
from datetime import datetime, timezone
from app import pilot_server
from app import pilot, pilot_budget
from app import upload_temp
# Retention test uses an ordinary temporary directory; actual tmpfs is tested
# separately by the required non-root runtime image probe.
upload_temp.validate=lambda: None
from tests.test_controlled_pilot import approval
root=Path(sys.argv[1])
os.environ.update({'VA_LSE_MODE':'controlled-pilot','VA_LSE_PILOT_LOG_RETENTION_DAYS':'1','VA_LSE_LOG_DIR':str(root),'VA_LSE_AUDIT_LOG_DIR':str(root),'VA_LSE_RUN_LOG_DIR':str(root),'VA_LSE_HEALTH_PORT':'0'})
data=approval()
pilot.load_approval=lambda: data
pilot_budget.provision(root/'pilot-budget.sqlite3', data)
os.environ['VA_LSE_PILOT_BUDGET_FILE']=str(root/'pilot-budget.sqlite3')
for name in ('app.log','audit.log','runs.jsonl'):
    (root/name).write_text(json.dumps({'timestamp':datetime.fromtimestamp(1,timezone.utc).isoformat(),'pages':84923})+'\n')
assert all(json.loads((root/name).read_text())['pages']==84923 for name in ('app.log','audit.log','runs.jsonl'))
pilot_server.initialize()
assert all('84923' not in (root/name).read_text() for name in ('app.log','audit.log','runs.jsonl'))
print('SERVER_STARTUP_CLEANUP_OK')'''
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-c',script,directory],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('SERVER_STARTUP_CLEANUP_OK',result.stdout)

    def test_failed_audit_write_keeps_counter_and_retention_signal(self):
        logger = audit.configure_audit_logging(force=True)
        file_handler = next(h for h in logger.handlers if isinstance(h, log_retention.PilotLogHandler))
        before = audit.audit_health()['write_failures']
        with patch.object(file_handler, '_open', side_effect=OSError(CANARY)):
            audit.audit_event('evaluate', 'error', request_id='req_aaaaaaaaaaaa', error=ValueError(CANARY))
        self.assertEqual(audit.audit_health()['write_failures'],before+1)
        self.assertTrue(file_handler.write_failed)
        self.assertNotIn(CANARY,repr(audit.audit_health()))

    def test_matching_healthy_policy_is_admitted(self):
        pilot.validate_log_policy(approval())

    def test_absent_or_mismatched_policy_is_refused(self):
        for value in ('', '1'):
            with patch.dict(os.environ, {'VA_LSE_PILOT_LOG_RETENTION_DAYS': value}), self.assertRaises(pilot.PilotBlocked):
                pilot.validate_log_policy(approval())

    def test_disabled_run_sink_is_refused(self):
        with patch.dict(os.environ, {'VA_LSE_RUN_LOG_DISABLED': '1'}), self.assertRaises(pilot.PilotBlocked):
            pilot.validate_log_policy(approval())

    def test_unwritable_file_and_overlapping_writers_are_refused(self):
        with patch.object(log_retention.PilotLogHandler, '_open', side_effect=PermissionError(CANARY)), self.assertRaises(pilot.PilotBlocked):
            pilot.validate_log_policy(approval())
        with patch.dict(os.environ, {'VA_LSE_LOG_FILE': 'runs.jsonl'}):
            logging_config.configure_logging(force=True)
            with self.assertRaises(pilot.PilotBlocked):
                pilot.validate_log_policy(approval())

    def test_failed_stale_or_dead_cleanup_is_refused(self):
        for change in ({'failed': True}, {'stale': True}, {'active': False}):
            health = {'failed': False, 'stale': False, 'active': True, **change}
            with patch.object(log_retention, 'retention_health', return_value=health), self.assertRaises(pilot.PilotBlocked):
                pilot.validate_log_policy(approval())


class TestNoticeLifecycle(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        from app import llm
        llm._sdk_name('NOT_GIVEN')  # Normal client construction binds this sentinel.
        self.data = approval()
        self.now = time.time()
        from tests.pilot_budget_fixtures import install_budget
        install_budget(self, self.data)
        self.claims = {'is_logged_in': True, 'iss': self.data['issuer'], 'sub': 'participant',
                       'iat': self.now - 1, 'exp': self.now + 300}
        self.owner = pilot.authorized_identity(self.claims, self.data)
        self.state = {'_pilot_owner': self.owner,
                      '_pilot_consent_grant': pilot.ConsentGrant(pilot.notice_binding(self.owner, self.data)),
                      '_pilot_notice_consent': pilot.notice_binding(self.owner, self.data)}
        for context in (patch.dict(os.environ, {'VA_LSE_MODE': 'controlled-pilot'}),
                        patch.object(pilot, 'load_approval', side_effect=lambda: self.data),
                        patch.object(st, 'user', self.claims), patch.object(st, 'session_state', self.state)):
            context.start()
            self.addCleanup(context.stop)

    def test_current_notice_allows_owner_and_changed_notice_refuses_cached_case(self):
        pilot.require_session_access()
        self.data['participant_notice'] += ' Changed provider retention.'
        with self.assertRaises(pilot.PilotBlocked):
            pilot.require_session_access()

    def test_policy_provider_model_and_review_changes_require_fresh_consent(self):
        for field, value in (('local_log_retention_days', 1), ('provider_terms', 'new terms'),
                             ('privacy_review', 'new review'), ('retention_policy', 'new policy'),
                             ('ingestion_security', 'new ingestion policy review'),
                             ('provider_base_url', 'https://new.example.test'), ('models', ['new-model'])):
            original = self.data[field]
            with self.subTest(field=field):
                self.data[field] = value
                with self.assertRaises(pilot.PilotBlocked):
                    pilot.require_consent(self.owner)
            self.data[field] = original

    def test_missing_consent_cannot_start_provider_work(self):
        from app.llm import LLMClient
        client = LLMClient.__new__(LLMClient)
        client._client = MagicMock()
        self.state.pop('_pilot_notice_consent')
        self.state.pop('_pilot_consent_grant')
        with self.assertRaises(pilot.PilotBlocked):
            client._call_openai('primary', 'test-model', 'system', CANARY, 0, 100, None)
        client._client.chat.completions.create.assert_not_called()

    def test_run_copies_consent_to_workers_and_detects_mid_run_notice_change(self):
        docs = [document_from_text('synthetic.txt', 'Synthetic knee observation.')]
        with self.assertRaises(pilot.PilotBlocked), pilot.action_budget(docs):
            context = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.assertEqual(executor.submit(context.run, pilot.require_consent, self.owner).result().binding,
                                 self.state['_pilot_notice_consent'])
            self.data['participant_notice'] += ' Changed notice.'
            with self.assertRaises(pilot.PilotBlocked):
                context.run(pilot.recheck_owner, self.owner)
        self.assertIsNone(pilot._run_notice.get())
        self.assertIsNone(pilot._run_claims.get())
        self.assertEqual(pilot._active, 0)

    def test_clearing_case_releases_notice_case_and_registered_uploads(self):
        self.state['record'] = CANARY
        context = MagicMock(session_id='synthetic-session')
        with patch('streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx', return_value=context):
            pilot.clear_case()
        self.assertEqual(self.state, {})
        context.uploaded_file_mgr.remove_session_files.assert_called_once_with('synthetic-session')
        with self.assertRaises(pilot.PilotBlocked):
            pilot.require_session_access()

    def test_other_identity_cannot_reuse_consent(self):
        self.claims['sub'] = 'operator'
        with self.assertRaises(pilot.PilotBlocked):
            pilot.require_consent(pilot.current_owner())

    def test_clearing_revokes_copied_workers_before_their_next_provider_request(self):
        from app.llm import LLMClient
        docs=[document_from_text('synthetic.txt','Synthetic knee observation.')]
        client=LLMClient.__new__(LLMClient)
        client._client=MagicMock()
        with self.assertRaises(pilot.PilotBlocked), pilot.action_budget(docs):
            context=contextvars.copy_context()
            pilot.clear_case()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future=executor.submit(context.run, client._call_openai,'primary','test-model','system',CANARY,0,100,None)
                with self.assertRaises(pilot.PilotBlocked):
                    future.result()
        client._client.chat.completions.create.assert_not_called()
        self.assertIsNone(pilot._run_notice.get())

    def test_withdrawal_during_preparation_stops_both_request_formats(self):
        from app import llm
        for responses in (False,True):
            self.state['_pilot_consent_grant']=pilot.ConsentGrant(pilot.notice_binding(self.owner,self.data))
            client=llm.LLMClient.__new__(llm.LLMClient)
            client._client=MagicMock()
            client._settings=MagicMock(base_url=self.data['provider_base_url'])
            client._pilot_calls=0
            client._pilot_prompt_chars=0
            client._pilot_call_lock=threading.Lock()
            with self.subTest(responses=responses), \
                    patch.object(pilot,'require_destination',side_effect=lambda *a: self.state['_pilot_consent_grant'].revoked.set()), \
                    patch.object(llm,'_uses_responses_schema',return_value=responses), self.assertRaises(pilot.PilotBlocked):
                client._call_openai('primary','test-model','system',CANARY,0,100,None)
            client._client.chat.completions.create.assert_not_called()
            client._client.responses.create.assert_not_called()

    def test_in_flight_response_is_refused_after_session_clear(self):
        from app import llm
        docs=[document_from_text('synthetic.txt','Synthetic knee observation.')]
        client=llm.LLMClient.__new__(llm.LLMClient)
        client._client=MagicMock()
        client._settings=MagicMock(base_url=self.data['provider_base_url'])
        client._pilot_calls=0
        client._pilot_prompt_chars=0
        client._pilot_call_lock=threading.Lock()
        def clear(**kwargs):
            pilot.clear_case()
            return MagicMock()
        client._client.chat.completions.create.side_effect=clear
        with self.assertRaises(pilot.PilotBlocked), pilot.action_budget(docs), \
                patch.object(pilot,'require_destination'), patch.object(llm,'_uses_responses_schema',return_value=False):
            client._call_openai('primary','test-model','system',CANARY,0,100,None)
        client._client.chat.completions.create.assert_called_once()
        self.assertEqual(self.state,{})

    def test_backup_and_restore_cannot_bypass_exclusion_with_injected_destination(self):
        destination = MagicMock()
        with self.assertRaises(audit_backup.BackupError):
            audit_backup.build_destination(overrides={'AUDIT_BACKUP_DESTINATION': 's3'})
        self.assertTrue(audit_backup.run_backup(destination=destination).error)
        self.assertTrue(audit_restore.verify_backup(destination=destination).list_error)
        self.assertTrue(audit_restore.restore_backup(destination=destination, target='/unused').error)
        self.assertEqual(destination.mock_calls, [])
