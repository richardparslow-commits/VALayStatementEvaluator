"""R09 synthetic durable reservation, restart and wire-call regressions."""
from __future__ import annotations

import contextvars
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.test_controlled_pilot import approval
from tests.pilot_budget_fixtures import install_budget
from app import pilot, pilot_budget, llm
from app.documents import document_from_text

CANARY = 'SYNTHETIC_R09_PRIVATE_CONTENT_CANARY'


class TestDurableLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'pilot-budget.sqlite3'
        self.data = approval()
        self.data['quota_policy'].update(run_attempts=3, pilot_total_attempts=10,
                                         participant_daily_starts=3, pilot_total_microusd=10_000_000)
        self.now = pilot_budget._now()
        clock = patch.object(pilot_budget, '_now', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        pilot_budget.provision(self.path, self.data)

    def ledger(self):
        result = pilot_budget.Ledger(self.path, self.data)
        self.addCleanup(result.close)
        return result

    def counters(self):
        with sqlite3.connect(self.path) as db:
            return db.execute('SELECT charged, reserved_attempts FROM control').fetchone()

    def debit(self, ledger, run, chars):
        number=ledger.attempt(run,chars)
        ledger.complete_attempt(run,number)  # Simulated local call has returned/raised.
        return number

    def test_entire_run_envelope_reserved_before_attempts(self):
        ledger = self.ledger()
        run = ledger.start('opaque-test-owner')
        self.assertEqual(self.counters(), (3_000_000, 3))
        ledger.finish(run)
        self.assertEqual(self.counters(), (0, 0))

    def test_unstarted_slots_released_but_failed_attempts_keep_full_charge(self):
        ledger = self.ledger()
        run = ledger.start('owner')
        self.debit(ledger, run, 10)  # No response or reconciliation.
        ledger.finish(run)
        self.assertEqual(self.counters(), (1_000_000, 1))

    def test_reported_usage_is_recorded_once_and_never_refunds_attempt(self):
        ledger = self.ledger()
        run = ledger.start('owner')
        number = self.debit(ledger, run, 10)
        ledger.reconcile(run, number, 4, 2)
        ledger.reconcile(run, number, 1, 1)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT reconciled,input_tokens,output_tokens FROM attempts').fetchone(), (1,4,2))
        ledger.finish(run)
        self.assertEqual(self.counters(), (1_000_000, 1))

    def test_missing_invalid_or_boolean_usage_is_unknown(self):
        ledger = self.ledger()
        run = ledger.start('owner')
        for tokens in ((None, None), (True, -1), (1.2, 'private')):
            ledger.reconcile(run, self.debit(ledger, run, 1), *tokens)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT input_tokens,output_tokens FROM attempts').fetchall(), [(None,None)]*3)
        ledger.finish(run)
        self.assertEqual(self.counters(), (3_000_000, 3))

    def test_attempt_and_prompt_limits_are_shared_and_exact(self):
        self.data['quota_policy']['run_prompt_chars'] = 6
        self.path.unlink()
        Path(str(self.path)+'.lock').unlink()
        pilot_budget.provision(self.path, self.data)
        ledger = self.ledger()
        run = ledger.start('owner')
        self.debit(ledger, run, 6)
        with self.assertRaises(pilot.PilotBlocked):
            self.debit(ledger, run, 1)
        self.debit(ledger, run, 0)
        self.debit(ledger, run, 0)
        with self.assertRaises(pilot.PilotBlocked):
            self.debit(ledger, run, 0)
        ledger.finish(run)
        self.assertEqual(self.counters(), (3_000_000, 3))

    def test_concurrent_workers_cannot_overspend_reservation(self):
        ledger = self.ledger()
        run = ledger.start('owner')
        def attempt(_):
            try:
                self.debit(ledger, run, 1)
                return True
            except pilot.PilotBlocked:
                return False
        with ThreadPoolExecutor(max_workers=12) as pool:
            self.assertEqual(sum(pool.map(attempt, range(24))), 3)
        ledger.finish(run)
        self.assertEqual(self.counters(), (3_000_000, 3))

    def test_two_tabs_cannot_start_concurrent_runs(self):
        ledger = self.ledger()
        barrier = threading.Barrier(2)
        def start(owner):
            barrier.wait(timeout=2)
            try:
                return ledger.start(owner)
            except pilot.PilotBlocked:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(start, ['owner-one','owner-two']))
        self.assertEqual(len([r for r in runs if r]), 1)

    def test_two_hourly_starts_survive_close_and_reopen(self):
        ledger = self.ledger()
        for _ in range(2):
            ledger.finish(ledger.start('owner'))
        ledger.close()
        ledger = self.ledger()
        with self.assertRaises(pilot.PilotBlocked):
            ledger.start('owner')
        self.now += 3600
        ledger.finish(ledger.start('owner'))
        with self.assertRaises(pilot.PilotBlocked):
            ledger.start('owner')  # Daily three-start rolling limit.
        self.now += 86400
        ledger.finish(ledger.start('owner'))

    def test_global_money_and_attempt_ceiling_refuse_before_new_run(self):
        for key in ('pilot_total_microusd', 'pilot_total_attempts'):
            with self.subTest(key=key):
                data = approval()
                data['quota_policy'].update(run_attempts=3, pilot_total_microusd=100_000_000,
                                            pilot_total_attempts=100)
                data['quota_policy'][key] = 3_000_000 if key.endswith('microusd') else 3
                path = Path(self.tmp.name) / (key + '.sqlite3')
                pilot_budget.provision(path, data)
                ledger = pilot_budget.Ledger(path, data)
                self.addCleanup(ledger.close)
                run = ledger.start('one')
                for _ in range(3):
                    self.debit(ledger, run, 1)
                ledger.finish(run)
                with self.assertRaises(pilot.PilotBlocked):
                    ledger.start('another')

    def test_crashed_run_keeps_full_reservation_on_restart(self):
        ledger = self.ledger()
        run = ledger.start('owner')
        self.debit(ledger, run, 1)
        ledger.close()
        ledger = self.ledger()
        self.assertEqual(self.counters(), (3_000_000, 3))
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT state FROM runs WHERE id=?',(run,)).fetchone()[0], 'abandoned')
        with self.assertRaises(pilot.PilotBlocked):
            ledger.finish(run)
        ledger.finish(ledger.start('another'))
        self.assertEqual(self.counters(), (3_000_000, 3))

    def child(self, code):
        return subprocess.run([sys.executable, '-c', code, str(self.path), json.dumps(self.data)],
                              capture_output=True, text=True, timeout=12)

    def test_actual_process_restart_preserves_owner_quota(self):
        first = self.child('''from tests import hermetic
import sys,json
from pathlib import Path
from app.pilot_budget import Ledger
l=Ledger(Path(sys.argv[1]),json.loads(sys.argv[2]))
for _ in range(2): l.finish(l.start('synthetic-owner'))
print('TWO_STARTS')''')
        self.assertEqual(first.returncode,0,first.stderr)
        second = self.child('''from tests import hermetic
import sys,json
from pathlib import Path
from app.pilot_budget import Ledger
from app.pilot import PilotBlocked
l=Ledger(Path(sys.argv[1]),json.loads(sys.argv[2]))
try: l.start('synthetic-owner')
except PilotBlocked: print('RESTART_BLOCKED')
else: raise AssertionError('quota reset')''')
        self.assertEqual(second.returncode,0,second.stderr)
        self.assertIn('RESTART_BLOCKED',second.stdout)

    def test_actual_abrupt_process_exit_keeps_all_reserved_spend(self):
        child = self.child('''from tests import hermetic
import sys,json,os
from pathlib import Path
from app.pilot_budget import Ledger
l=Ledger(Path(sys.argv[1]),json.loads(sys.argv[2]))
l.attempt(l.start('synthetic-owner'),1)
os._exit(0)''')
        self.assertEqual(child.returncode,0,child.stderr)
        self.now=datetime.now(timezone.utc).timestamp()
        self.ledger()
        self.assertEqual(self.counters(),(3_000_000,3))

    def test_second_process_cannot_take_single_instance_lock(self):
        self.ledger()
        result = self.child('''from tests import hermetic
import sys,json
from pathlib import Path
from app.pilot_budget import Ledger
from app.pilot import PilotBlocked
try: Ledger(Path(sys.argv[1]),json.loads(sys.argv[2]))
except PilotBlocked: print('SECOND_PROCESS_BLOCKED')
else: raise AssertionError('second instance admitted')''')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('SECOND_PROCESS_BLOCKED',result.stdout)

    def test_inherited_process_context_cannot_use_parent_lease(self):
        result=self.child('''from tests import hermetic
import sys,json,os
from pathlib import Path
from app.pilot_budget import Ledger
from app.pilot import PilotBlocked
l=Ledger(Path(sys.argv[1]),json.loads(sys.argv[2]))
child=os.fork()
if child==0:
    try: l.start('synthetic-owner')
    except PilotBlocked: os._exit(0)
    else: os._exit(9)
_,status=os.waitpid(child,0)
assert os.waitstatus_to_exitcode(status)==0
l.finish(l.start('synthetic-owner'))
print('INHERITED_LEASE_REFUSED')''')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('INHERITED_LEASE_REFUSED',result.stdout)

    def test_missing_corrupt_or_replaced_file_never_autocreates(self):
        ledger = self.ledger()
        self.path.unlink()
        with self.assertRaises(pilot.PilotBlocked):
            ledger.start('owner')
        self.assertFalse(self.path.exists())
        ledger.close()
        self.path.write_text(CANARY)
        self.path.chmod(0o600)
        with self.assertRaises(pilot.PilotBlocked) as error:
            self.ledger()
        self.assertNotIn(CANARY,str(error.exception))

    def test_replaced_database_inode_is_refused(self):
        ledger = self.ledger()
        replacement = self.path.with_name('replacement.sqlite3')
        pilot_budget.provision(replacement,self.data)
        replacement.replace(self.path)
        with self.assertRaises(pilot.PilotBlocked):
            ledger.start('owner')

    def test_symlink_fifo_hardlink_and_public_permissions_refused(self):
        self.path.unlink()
        for kind in ('symlink','fifo','hardlink','public-file','public-directory'):
            with self.subTest(kind=kind):
                self.path.parent.chmod(0o700)
                target = self.path.with_name('target')
                target.write_text(CANARY)
                if kind == 'symlink': self.path.symlink_to(target)
                elif kind == 'fifo': os.mkfifo(self.path)
                elif kind == 'hardlink': os.link(target,self.path)
                else:
                    pilot_budget.provision(self.path,self.data)
                    if kind == 'public-file': self.path.chmod(0o644)
                    else: self.path.parent.chmod(0o755)
                with self.assertRaises(pilot.PilotBlocked): self.ledger()
                self.assertEqual(target.read_text(),CANARY)
                self.path.unlink()
                target.unlink()
                Path(str(self.path)+'.lock').unlink(missing_ok=True)

    def test_existing_ledger_cannot_be_reprovisioned(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            pilot_budget.provision(self.path,self.data)
        self.assertEqual(self.path.read_bytes(),before)

    def test_policy_changes_expiry_and_clock_rollback_close_admission(self):
        ledger = self.ledger()
        changed = {**self.data,'quota_policy':{**self.data['quota_policy'],'run_attempts':2}}
        with self.assertRaises(pilot.PilotBlocked): ledger.verify(changed)
        self.now -= 1
        with self.assertRaises(pilot.PilotBlocked): ledger.start('owner')
        self.now += 2
        with self.assertRaises(pilot.PilotBlocked): ledger.start('owner')  # Failure latched.

    def test_metadata_has_no_owner_or_record_content_and_old_rows_prune(self):
        ledger = self.ledger()
        run = ledger.start(CANARY)
        ledger.reconcile(run,self.debit(ledger, run,10),2,1)
        ledger.finish(run)
        self.assertNotIn(CANARY.encode(),self.path.read_bytes())
        self.now += 86400
        ledger.verify(self.data)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM runs').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],0)
        self.assertEqual(self.counters(),(1_000_000,1))  # Global spend is NEVER reset by pruning.

    def test_strict_policy_requires_explicit_integer_reviewed_values(self):
        for key,value in (('run_attempts',True),('participant_daily_starts',0),('pilot_id','new-pilot'),
                          ('attempt_charge_microusd',1.0),('expires_at','2000-01-01T00:00:00Z'),
                          ('pilot_total_attempts',2),('pilot_total_microusd',2)):
            data={**self.data,'quota_policy':{**self.data['quota_policy'],key:value}}
            with self.subTest(key=key), self.assertRaises(pilot.PilotBlocked): pilot_budget.policy(data)
        with self.assertRaises(pilot.PilotBlocked): pilot_budget.policy({})


class TestProviderBudgetIntegration(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        self.data=approval()
        self.data['quota_policy'].update(run_attempts=2,attempt_output_tokens=100,run_prompt_chars=50)
        self.path=install_budget(self,self.data)
        self.claims={'is_logged_in':True,'iss':self.data['issuer'],'sub':'participant',
                     'iat':pilot_budget._now()-1,'exp':pilot_budget._now()+300}
        self.owner=pilot.authorized_identity(self.claims,self.data)
        self.state={'_pilot_consent_grant':pilot.ConsentGrant(pilot.notice_binding(self.owner,self.data))}
        for context in (patch.dict(os.environ,{'VA_LSE_MODE':'controlled-pilot'}),
                        patch.object(pilot,'load_approval',side_effect=lambda:self.data),
                        patch.object(st,'user',self.claims),patch.object(st,'session_state',self.state),
                        patch.object(pilot,'require_destination')):
            context.start()
            self.addCleanup(context.stop)
        llm._sdk_name('NOT_GIVEN')
        self.docs=[document_from_text('synthetic.txt','Synthetic knee observation.')]

    def client(self):
        client=llm.LLMClient.__new__(llm.LLMClient)
        client._client=MagicMock()
        client._settings=MagicMock(base_url=self.data['provider_base_url'],model_main='test-model')
        client.usage=MagicMock()
        client._new_attempt_client=MagicMock(return_value=client._client)
        return client

    def wire(self,client,*,responses=False,tokens=200):
        with patch.object(llm,'_uses_responses_schema',return_value=responses):
            return client._call_openai('primary','test-model','system','synthetic',0.2,tokens,None)

    def test_pilot_attempts_use_private_pools_and_deadline_clipped_watchdogs(self):
        client = self.client()
        del client._new_attempt_client  # Exercise the real constructor path.
        owned = [MagicMock(), MagicMock()]
        for sdk in owned:
            sdk.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Invented result", refusal=None), finish_reason="stop")],
                usage=MagicMock(prompt_tokens=2, completion_tokens=2))
        factory = MagicMock(side_effect=owned)
        transports = [MagicMock(), MagicMock()]
        timers = [MagicMock(), MagicMock()]
        rate = MagicMock(enabled=False)
        with patch.object(llm, '_sdk_name', return_value=factory), \
                patch.object(llm, '_pilot_transport', side_effect=[{'http_client': t} for t in transports]), \
                patch.object(llm, '_uses_responses_schema', return_value=False), \
                patch.object(llm, 'get_llm_breaker', return_value=MagicMock()), \
                patch.object(llm, 'get_llm_limiter', return_value=MagicMock(queue_timeout=10)), \
                patch.object(llm, 'get_llm_rate_gate', return_value=rate), \
                patch.object(llm, '_stall_watchdog_seconds', return_value=0), \
                patch.object(llm, 'pipeline_remaining_seconds', return_value=.5), \
                patch.object(llm.threading, 'Timer', side_effect=timers) as timer, pilot.action_budget(self.docs):
            self.assertEqual(client._chat_on_endpoint('primary', 'system', 'synthetic'), 'Invented result')
            self.assertEqual(client._chat_on_endpoint('primary', 'system', 'synthetic'), 'Invented result')
        for index, sdk in enumerate(owned):
            sdk.chat.completions.create.assert_called_once()
            sdk.close.assert_called_once()
            self.assertEqual(timer.call_args_list[index].args[0], .5)
            self.assertIs(timer.call_args_list[index].kwargs['args'][-1], sdk)
            self.assertIs(factory.call_args_list[index].kwargs['http_client'], transports[index])
            self.assertEqual(factory.call_args_list[index].kwargs['max_retries'], 0)
            timers[index].cancel.assert_called_once()
        client._client.close.assert_not_called()
        self.assertIsNone(llm._attempt_client.get())

    def test_attempt_watchdog_does_not_close_or_rebuild_other_provider_pools(self):
        client = self.client()
        owned = MagicMock()
        with patch.object(llm, '_shutdown_pool_sockets', return_value=1) as shutdown, \
                patch.object(client, '_rebuild_client') as rebuild:
            client._stall_watchdog_fired('primary', 'invented', 'invented', 'test-model', 1, .5, owned)
        shutdown.assert_called_once_with(owned)
        owned.close.assert_called_once()
        client._client.close.assert_not_called()
        rebuild.assert_not_called()

    def test_failed_attempt_client_construction_closes_the_new_transport(self):
        client = self.client()
        del client._new_attempt_client
        transport = MagicMock()
        with patch.object(llm, '_sdk_name', return_value=MagicMock(side_effect=RuntimeError('Invented setup failure'))), \
                patch.object(llm, '_pilot_transport', return_value={'http_client': transport}):
            with self.assertRaises(RuntimeError):
                client._new_attempt_client('primary')
        transport.close.assert_called_once()

    def test_direct_provider_call_without_reserved_run_never_reaches_sdk(self):
        client=self.client()
        with self.assertRaises(pilot.PilotBlocked): self.wire(client)
        client._client.chat.completions.create.assert_not_called()

    def test_paid_credential_probe_never_contacts_provider(self):
        with patch.object(llm,'_probe_open') as transport:
            result=llm.probe_chat(self.data['provider_base_url'],CANARY,'test-model')
        self.assertIsNone(result.status)
        self.assertNotIn(CANARY,result.error)
        transport.assert_not_called()

    def test_raw_post_probe_cannot_bypass_reserved_sdk_path(self):
        import urllib.request
        request=urllib.request.Request(self.data['provider_base_url']+'/responses',data=b'{}',method='POST')
        with patch('urllib.request.build_opener') as opener,self.assertRaises(pilot.PilotBlocked):
            llm._probe_open(request,timeout=1)
        opener.assert_not_called()

    def test_both_formats_cap_output_and_distinct_clients_share_attempt_budget(self):
        clients=[self.client(),self.client(),self.client()]
        with pilot.action_budget(self.docs):
            self.wire(clients[0])
            self.wire(clients[1],responses=True)
            with self.assertRaises(pilot.PilotBlocked): self.wire(clients[2])
        self.assertEqual(clients[0]._client.chat.completions.create.call_args.kwargs['max_tokens'],100)
        self.assertEqual(clients[1]._client.responses.create.call_args.kwargs['max_output_tokens'],100)
        clients[2]._client.chat.completions.create.assert_not_called()

    def test_workers_share_run_reservation_and_finished_context_cannot_spend(self):
        client=self.client()
        with pilot.action_budget(self.docs):
            context=contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(context.run,self.wire,client).result()
        with self.assertRaises(pilot.PilotBlocked): context.run(self.wire,client)
        self.assertEqual(client._client.chat.completions.create.call_count,1)

    def test_aborted_run_keeps_active_slot_until_outstanding_worker_returns(self):
        entered,released=threading.Event(),threading.Event()
        client=self.client()
        def wait_for_release(**kwargs):
            entered.set()
            if not released.wait(4): raise RuntimeError('Synthetic worker deadline')
            return MagicMock()
        client._client.chat.completions.create.side_effect=wait_for_release
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                with self.assertRaises(ValueError),pilot.action_budget(self.docs):
                    context=contextvars.copy_context()
                    future=pool.submit(context.run,self.wire,client)
                    self.assertTrue(entered.wait(2))
                    raise ValueError('Synthetic pipeline refusal')
                with sqlite3.connect(self.path) as db:
                    self.assertEqual(db.execute('SELECT state FROM runs').fetchone()[0],'closing')
                    self.assertEqual(db.execute('SELECT charged,reserved_attempts FROM control').fetchone(),(2_000_000,2))
                with self.assertRaises(pilot.PilotBlocked),pilot.action_budget(self.docs):
                    self.fail('A second run overlapped an outstanding worker')
            finally:
                released.set()
            future.result(timeout=3)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT state FROM runs').fetchone()[0],'finished')
            self.assertEqual(db.execute('SELECT charged,reserved_attempts FROM control').fetchone(),(1_000_000,1))
        with pilot.action_budget(self.docs): pass

    def test_withdrawal_during_durable_debit_blocks_both_wire_formats(self):
        ledger=pilot_budget.get_ledger(self.data)
        original=ledger.attempt
        def withdraw(run,chars):
            number=original(run,chars)
            self.state['_pilot_consent_grant'].revoked.set()
            return number
        for responses in (False,True):
            self.state['_pilot_consent_grant']=pilot.ConsentGrant(pilot.notice_binding(self.owner,self.data))
            client=self.client()
            with self.subTest(responses=responses),self.assertRaises(pilot.PilotBlocked), \
                    pilot.action_budget(self.docs),patch.object(ledger,'attempt',side_effect=withdraw):
                self.wire(client,responses=responses)
            client._client.chat.completions.create.assert_not_called()
            client._client.responses.create.assert_not_called()
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT DISTINCT state FROM runs').fetchall(),[('finished',)])

    def test_error_attempt_is_charged_and_retry_cannot_bypass_run_cap(self):
        client=self.client()
        client._client.chat.completions.create.side_effect=llm.LLMTimeoutError('Synthetic timeout',retriable=True)
        breaker,limiter,rate=MagicMock(),MagicMock(),MagicMock()
        rate.enabled=False
        with patch.object(llm,'get_llm_breaker',return_value=breaker), \
                patch.object(llm,'get_llm_limiter',return_value=limiter), \
                patch.object(llm,'get_llm_rate_gate',return_value=rate), \
                patch.object(llm,'_uses_responses_schema',return_value=False), \
                patch.object(llm,'_stall_watchdog_seconds',return_value=0), \
                patch.object(llm,'wait_with_cancellation'), self.assertRaises(pilot.PilotBlocked), \
                pilot.action_budget(self.docs):
            client._chat_on_endpoint('primary','system','synthetic')
        self.assertEqual(client._client.chat.completions.create.call_count,2)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT charged,reserved_attempts FROM control').fetchone(),(2_000_000,2))

    def test_new_quota_policy_or_spending_review_requires_fresh_consent(self):
        for key in ('quota_policy','spending_controls'):
            saved=self.data[key]
            self.data[key]={**saved,'run_attempts':1} if key=='quota_policy' else 'changed review'
            with self.subTest(key=key),self.assertRaises(pilot.PilotBlocked): pilot.require_consent(self.owner)
            self.data[key]=saved

    def test_invalid_output_or_failed_ledger_blocks_before_paid_work(self):
        client=self.client()
        with self.assertRaises(pilot.PilotBlocked), pilot.action_budget(self.docs):
            with self.assertRaises(pilot.PilotBlocked): self.wire(client,tokens=True)
            self.path.unlink()
            with self.assertRaises(pilot.PilotBlocked): self.wire(client)
            # Unwinding preserves failure; no unsafe return of a completed case.
            with self.assertRaises(pilot.PilotBlocked):
                pilot_budget.get_ledger(self.data)
        client._client.chat.completions.create.assert_not_called()
