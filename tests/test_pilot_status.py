"""Safe service diagnostics without host Docker access or private log output."""
from __future__ import annotations
import json
import subprocess
import unittest
from unittest.mock import MagicMock, patch
from tests import hermetic  # noqa: F401
from scripts import pilot_status

class TestPilotStatus(unittest.TestCase):
    def test_reads_only_fixed_service_state_fields(self):
        safe={'status':'running','exit_code':0,'oom_killed':False,'restarts':2,'health':'healthy'}
        with patch.object(pilot_status.subprocess,'run',side_effect=[
                MagicMock(stdout='a'*64),MagicMock(stdout=json.dumps(safe))]*3) as calls:
            self.assertEqual(pilot_status.status(),{s:safe for s in pilot_status.SERVICES})
        self.assertEqual(calls.call_count,6)
        for call in calls.call_args_list:
            args=call.args[0]
            self.assertNotIn('logs',args)
            self.assertNotIn('exec',args)
            self.assertNotIn('.Config',repr(args))
            self.assertNotIn('.State.Health.Log',repr(args))
            self.assertTrue(call.kwargs['check'])
            self.assertEqual(call.kwargs['timeout'],10)

    def test_errors_oversized_output_and_extra_fields_do_not_echo_content(self):
        canary='SYNTHETIC_PRIVATE_DIAGNOSTIC_CANARY'
        bad=[subprocess.CalledProcessError(1,['docker'],stderr=canary),MagicMock(stdout='x'*4097),
             MagicMock(stdout='a'*64+'\n'+'b'*64)]
        for value in bad:
            with self.subTest(value=type(value).__name__),patch.object(pilot_status.subprocess,'run',side_effect=[value]*3):
                self.assertEqual(pilot_status.status(),{s:{'status':'unavailable'} for s in pilot_status.SERVICES})
        with patch.object(pilot_status,'_read',side_effect=['a'*64,json.dumps({'status':'running','private':canary})]*3):
            self.assertNotIn(canary,json.dumps(pilot_status.status()))

    def test_absent_services_and_non_running_state_are_visible(self):
        with patch.object(pilot_status,'_read',return_value=''):
            self.assertEqual(pilot_status.status(),{s:{'status':'absent'} for s in pilot_status.SERVICES})
        with patch.object(pilot_status,'status',return_value={s:{'status':'exited'} for s in pilot_status.SERVICES}),patch('builtins.print'):
            self.assertEqual(pilot_status.main(),2)

    def test_success_requires_healthy_web_and_parser_but_not_nginx_healthcheck(self):
        base={s:{'status':'running','health':'healthy'} for s in pilot_status.SERVICES}
        base['nginx']['health']='unavailable'
        with patch.object(pilot_status,'status',return_value=base),patch('builtins.print'):
            self.assertEqual(pilot_status.main(),0)
            for service in ('streamlit-web','parser-launcher'):
                for health in ('starting','unavailable','unhealthy'):
                    base[service]['health']=health
                    with self.subTest(service=service,health=health):
                        self.assertEqual(pilot_status.main(),2)
                base[service]['health']='healthy'
