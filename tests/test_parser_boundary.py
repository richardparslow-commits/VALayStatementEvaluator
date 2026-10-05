"""Synthetic parser protocol and launcher regressions; no application providers."""
from __future__ import annotations

import hashlib
import io
import os
import socket
import subprocess
import time
import struct
import threading
import unittest
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from app import parser_protocol as wire
from app.isolated_extract import IsolatedExtractor, _documents, parser_health
from app.documents import ExtractionError
from app.parser_service import DockerParser, handle

IMAGE = 'sha256:' + 'a' * 64
DATA = b'Synthetic observation of knee pain.'


def request():
    return {'version': 1, 'label': 'record.txt', 'size': len(DATA),
            'sha256': hashlib.sha256(DATA).hexdigest(), 'nonce': 'b' * 32, 'page_limit': 500}


def response():
    return {**{k: v for k, v in request().items() if k not in ('size', 'page_limit')}, 'image': IMAGE,
            'documents': [{'filename': 'record.txt', 'schema_version': 3, 'total_pages': 1,
                           'unreadable_pages': [], 'pagination': 'block', 'coverage_known': True, 'source_sha256': hashlib.sha256(DATA).hexdigest(), 'extraction_method': 'strict-unicode', 'text_encoding': 'utf-8',
                           'pages': [{'page': 1, 'kind': 'block', 'text': DATA.decode(), 'source_part': ''}]}], 'skipped': []}


class ProtocolTests(unittest.TestCase):
    def test_worker_uses_lower_request_and_deployment_page_limit(self):
        from app.parser_worker import extraction_page_limit
        for requested, deployed, expected in ((5000, 500, 500), (500, 5000, 500), (5000, 5000, 5000)):
            with patch.dict(os.environ, {'VA_LSE_PARSER_MAX_PAGES': str(deployed)}):
                self.assertEqual(extraction_page_limit({'page_limit': requested}), expected)
        with patch.dict(os.environ, {'VA_LSE_PARSER_MAX_PAGES': '0'}), self.assertRaises(ValueError):
            extraction_page_limit({'page_limit': 500})

    def test_json_graph_depth_and_alternative_encoding_are_refused_before_decode(self):
        values = [b'[' + b'{},' * 70000 + b'{}]', b'[' * 17 + b'0' + b']' * 17,
                  '{"x":1}'.encode('utf-16-le'), b'{"x":"\xc3\xa9"}']
        for value in values:
            with self.subTest(size=len(value)), patch('app.parser_protocol.json.loads') as parse, self.assertRaises(wire.ParserRefused):
                wire.decode(value)
            parse.assert_not_called()

    def test_json_string_punctuation_and_escaped_unicode_remain_valid(self):
        value = {'text': 'Observation \" quoted \\ path [ {},: ] \U0001f600' * 20000}
        self.assertEqual(wire.decode(wire.encode(value)), value)

    def test_synthetic_page_limit_is_preserved_while_pilot_limit_is_enforced(self):
        from app import config
        value, req = response(), request()
        value['label'] = req['label'] = 'records.zip'
        req['page_limit'] = 5000
        original = value['documents'][0]
        value['documents'] = [{**original, 'filename': f'records/{n}.txt', 'total_pages': 3,
                               'pages': [{'page': p, 'kind': 'block', 'text': 'Synthetic observation', 'source_part': ''} for p in (1,2,3)]}
                              for n in range(200)]
        with patch.object(config, 'MAX_RECORD_PAGES', 5000):
            self.assertEqual(len(_documents(value, req, IMAGE)[0]), 200)
        with patch.object(config, 'MAX_RECORD_PAGES', 500), self.assertRaises(wire.ParserRefused):
            _documents(value, req, IMAGE)

    def test_refused_empty_upload_is_skipped_and_other_files_continue(self):
        from app import documents
        good = documents.document_from_text('good.txt', DATA.decode())
        selected = MagicMock()
        protected = IsolatedExtractor()
        selected.extract.side_effect = lambda label, data: protected.extract(label, data) if not data else ([good], [])
        empty, valid = MagicMock(name='empty'), MagicMock(name='valid')
        empty.name, valid.name = 'empty.txt', 'good.txt'
        empty.getvalue.return_value, valid.getvalue.return_value = b'', DATA
        with patch.dict(os.environ, {'VA_LSE_PARSER_IMAGE': IMAGE}), patch.object(documents, '_ACTIVE_EXTRACTOR', selected), \
                patch('app.documents.InProcessExtractor.extract') as fallback:
            docs, skipped = documents.extract_uploaded_documents([empty, valid])
        self.assertEqual(docs, [good])
        self.assertEqual(len(skipped), 1)
        self.assertIn('empty.txt', skipped[0])
        self.assertEqual(selected.extract.call_count, 2)
        fallback.assert_not_called()

    def test_busy_client_refusal_occurs_before_file_bytes_are_sent(self):
        with patch.dict(os.environ, {'VA_LSE_PARSER_IMAGE': IMAGE}), patch('app.isolated_extract.socket.socket') as constructor, \
                patch('app.isolated_extract.recv_frame', return_value=wire.encode({'busy': True})), self.assertRaisesRegex(ExtractionError, 'busy'):
            IsolatedExtractor().extract('record.txt', DATA)
        self.assertEqual(constructor.return_value.__enter__.return_value.sendall.call_count, 1)

    def test_trickled_socket_bytes_cannot_reset_absolute_deadline(self):
        connection = MagicMock()
        connection.recv.return_value = b'x'
        with patch('app.parser_protocol.time.monotonic', side_effect=[0, 1, 2]), self.assertRaises(wire.ParserRefused):
            wire.recv_exact(connection, 4, deadline=2)
        self.assertEqual(connection.recv.call_count, 2)

    def test_bounded_input_round_trip(self):
        self.assertEqual(wire.read_request(io.BytesIO(wire.frame(wire.encode(request())) + DATA)), (request(), DATA))

    def test_truncated_extra_and_wrong_hash_inputs_are_rejected(self):
        for data in (DATA[:-1], DATA + b'x', b'x' * len(DATA)):
            with self.subTest(data=data), self.assertRaises(wire.ParserRefused):
                wire.read_request(io.BytesIO(wire.frame(wire.encode(request())) + data))

    def test_request_schema_limits_and_duplicate_keys(self):
        for key, value in (('size', True), ('page_limit', True), ('page_limit', 0), ('page_limit', 5001), ('size', wire.MAX_INPUT + 1), ('label', 'a\n.txt'),
                           ('label', ''), ('nonce', '../path'), ('version', True), ('sha256', 'x')):
            with self.subTest(key=key), self.assertRaises(wire.ParserRefused):
                wire.validate_request({**request(), key: value})
        with self.assertRaises(wire.ParserRefused):
            wire.decode(b'{"label":"a", "label":"b"}')
        with self.assertRaises(wire.ParserRefused):
            wire.validate_request({**request(), 'command': 'arbitrary'})

    def test_valid_text_and_unreadable_pages_preserve_coverage(self):
        value = response()
        value['documents'][0]['total_pages'] = 2
        value['documents'][0]['unreadable_pages'] = [2]
        docs, skipped = _documents(value, request(), IMAGE)
        self.assertEqual((docs[0].total_pages, docs[0].unreadable_pages, skipped), (2, [2], []))

    def test_entire_scan_is_preserved_instead_of_silently_dropped(self):
        value = response()
        value['documents'][0].update(pages=[], unreadable_pages=[1])
        self.assertEqual(_documents(value, request(), IMAGE)[0][0].unreadable_pages, [1])

    def test_other_request_image_label_and_unknown_response_fields_rejected(self):
        for key, value in (('nonce', 'c' * 32), ('image', 'sha256:' + 'c' * 64), ('label', 'other.txt'),
                           ('sha256', 'c' * 64), ('version', True), ('extra', 1)):
            with self.subTest(key=key), self.assertRaises(wire.ParserRefused):
                _documents({**response(), key: value}, request(), IMAGE)

    def test_page_and_coverage_schema_refuses_partial_coercion(self):
        changes = [('page', True), ('page', 0), ('page', 2), ('text', 123), ('text', '  '), ('kind', 'page')]
        for key, change in changes:
            value = response()
            value['documents'][0]['pages'][0][key] = change
            with self.subTest(key=key, change=change), self.assertRaises(wire.ParserRefused):
                _documents(value, request(), IMAGE)
        for change in ({'total_pages': 2}, {'total_pages': True}, {'coverage_known': False},
                       {'unreadable_pages': [1]}, {'unreadable_pages': [2]}, {'schema_version': True},
                       {'pages': []}, {'filename': 'other.txt'}):
            value = response()
            value['documents'][0].update(change)
            with self.subTest(change=change), self.assertRaises(wire.ParserRefused):
                _documents(value, request(), IMAGE)

    def test_duplicate_documents_pages_and_text_limit_rejected(self):
        value = response()
        value['documents'] *= 2
        with self.assertRaises(wire.ParserRefused):
            _documents(value, request(), IMAGE)
        value = response()
        value['documents'][0]['pages'] *= 2
        with self.assertRaises(wire.ParserRefused):
            _documents(value, request(), IMAGE)
        with patch('app.isolated_extract.MAX_TEXT', 10), self.assertRaises(wire.ParserRefused):
            _documents(response(), request(), IMAGE)

    def test_archive_members_remain_bound_to_archive_stem(self):
        value, req = response(), request()
        value['label'] = req['label'] = 'records.zip'
        value['documents'][0]['filename'] = 'records/folder/one.txt'
        self.assertEqual(_documents(value, req, IMAGE)[0][0].filename, 'records/folder/one.txt')
        for name in ('other/one.txt', 'records//absolute.txt', 'records/sub/../record.txt'):
            value['documents'][0]['filename'] = name
            with self.subTest(name=name), self.assertRaises(wire.ParserRefused):
                _documents(value, req, IMAGE)

    def test_absent_launcher_or_invalid_image_never_falls_back(self):
        with patch.dict(os.environ, {'VA_LSE_PARSER_IMAGE': IMAGE}), \
                patch('app.isolated_extract.socket.socket') as constructor, \
                patch('app.documents.InProcessExtractor.extract') as fallback:
            constructor.return_value.__enter__.return_value.connect.side_effect = OSError('synthetic')
            with self.assertRaises(ExtractionError):
                IsolatedExtractor().extract('record.txt', DATA)
            fallback.assert_not_called()
        with patch.dict(os.environ, {'VA_LSE_PARSER_IMAGE': 'mutable:latest'}), self.assertRaises(ExtractionError):
            IsolatedExtractor().extract('record.txt', DATA)

    def test_response_frame_size_is_checked_before_allocation(self):
        connection = MagicMock()
        connection.recv.return_value = struct.pack('!I', wire.MAX_OUTPUT + 1)
        with self.assertRaises(wire.ParserRefused):
            wire.recv_frame(connection, wire.MAX_OUTPUT)
        self.assertEqual(connection.recv.call_count, 1)

    def test_health_requires_exact_image_and_revision(self):
        with patch.dict(os.environ, {'VA_LSE_PARSER_IMAGE': IMAGE, 'VA_LSE_BUILD_SHA': 'reviewed'}), \
                patch('app.isolated_extract.socket.socket'), patch('app.isolated_extract.recv_frame', return_value=wire.encode(
                    {'ready': True, 'image': IMAGE, 'revision': 'old'})), self.assertRaises(wire.ParserRefused):
            parser_health()
        from app import config
        with patch.dict(os.environ, {'VA_LSE_PARSER_IMAGE': IMAGE, 'VA_LSE_BUILD_SHA': 'reviewed'}), \
                patch.object(config, 'MAX_RECORD_PAGES', 500), patch('app.isolated_extract.socket.socket'), \
                patch('app.isolated_extract.recv_frame', return_value=wire.encode(
                    {'ready': True, 'image': IMAGE, 'revision': 'reviewed', 'max_pages': 500})):
            parser_health()


class LauncherTests(unittest.TestCase):
    def test_busy_admission_is_prompt_and_health_remains_responsive(self):
        runner = MagicMock()
        runner.parse_lock = threading.Lock()
        runner.ready.return_value = {'ready': True}
        started, release = threading.Event(), threading.Event()
        def run(*args):
            started.set()
            self.assertTrue(release.wait(3))
            return wire.encode(response())
        runner.run.side_effect = run
        pairs = [socket.socketpair() for _ in range(3)]
        threads = []
        try:
            for left, _ in pairs:
                thread = threading.Thread(target=handle, args=(left, runner))
                threads.append(thread)
                thread.start()
            first, second, health = [right for _, right in pairs]
            for connection in (first, second, health): connection.settimeout(2)
            first.sendall(wire.frame(wire.encode(request())))
            self.assertEqual(wire.decode(wire.recv_frame(first, wire.MAX_HEADER)), {'accepted': True})
            first.sendall(DATA)
            self.assertTrue(started.wait(1))
            second.sendall(wire.frame(wire.encode(request())))
            self.assertEqual(wire.decode(wire.recv_frame(second, wire.MAX_HEADER)), {'busy': True})
            health.sendall(wire.frame(wire.encode({'operation': 'health'})))
            self.assertEqual(wire.decode(wire.recv_frame(health, wire.MAX_HEADER)), {'ready': True})
            self.assertEqual(runner.run.call_count, 1)
            release.set()
            self.assertEqual(wire.decode(wire.recv_frame(first, wire.MAX_OUTPUT)), response())
        finally:
            release.set()
            for left, right in pairs: left.close(); right.close()
            for thread in threads: thread.join(3)
        self.assertFalse(any(t.is_alive() for t in threads))

    def test_reap_failure_still_attempts_container_removal(self):
        runner = DockerParser(IMAGE, 'reviewed')
        process = subprocess.Popen([os.sys.executable, '-c', "import sys; sys.stdin.buffer.read(); print('{}')"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        try:
            with patch('app.parser_service.subprocess.Popen', return_value=process), \
                    patch.object(process, 'wait', side_effect=[0, subprocess.TimeoutExpired('synthetic', 5)]), \
                    patch('app.parser_service.subprocess.run') as remove, self.assertRaises(subprocess.TimeoutExpired):
                runner.run(request(), DATA)
            self.assertEqual(remove.call_args.args[0][1:3], ['rm', '--force'])
        finally:
            process.wait(timeout=5)

    def test_narrow_parser_dependency_hashes_match_application_lock(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        full = (root / 'requirements.lock').read_text()
        narrow = (root / 'requirements-parser.lock').read_text()
        lines = [line for line in narrow.splitlines() if line and not line.startswith('#')]
        self.assertEqual([line.split('==')[0] for line in lines if '==' in line], ['pypdf', 'python-dotenv'])
        self.assertTrue(all(line in full for line in lines))

    def test_pilot_mounts_keep_credentials_and_daemon_in_different_services(self):
        from pathlib import Path
        import yaml
        root = Path(__file__).resolve().parents[1]
        services = yaml.safe_load((root / 'docker-compose.pilot.yml').read_text())['services']
        web, launcher = services['streamlit-web'], services['parser-launcher']
        self.assertFalse(any('docker.sock' in x for x in web['volumes']))
        self.assertNotIn('env_file', launcher)
        self.assertEqual(launcher['network_mode'], 'none')
        self.assertEqual(set(launcher['environment']), {'VA_LSE_PARSER_IMAGE', 'VA_LSE_BUILD_SHA', 'VA_LSE_PARSER_MAX_PAGES'})
        self.assertEqual(launcher['volumes'], ['/var/run/docker.sock:/var/run/docker.sock', 'parser-channel:/run/parser'])
        self.assertIn('parser-channel:/run/parser:ro', web['volumes'])
        parser = (root / 'deploy/parser.Dockerfile').read_text()
        self.assertNotIn('COPY app/ ./app/', parser)
        self.assertNotIn('run_app.py', parser)
        self.assertIn('app/documents.py', parser)

    def test_launcher_argv_is_fixed_and_excludes_mounts_shell_credentials_and_network(self):
        runner = DockerParser(IMAGE, 'reviewed')
        argv = runner.command('va-parser-test')
        for item in ('--network=none', '--cap-drop=ALL', '--read-only', '--ipc=none',
                     '--user=65534:65534', '--security-opt=apparmor=va-lse-parser',
                     '--security-opt=no-new-privileges:true', '--pids-limit=32', '--memory-swap=1g',
                     '--log-driver=none', '--pull=never'):
            self.assertIn(item, argv)
        self.assertFalse(any(x.startswith(('--mount', '--volume', '--privileged', '--pid', '--env-file'))
                             for x in argv if x != '--pids-limit=32'))
        self.assertNotIn('seccomp=unconfined', ' '.join(argv))
        self.assertEqual(set(runner.env), {'PATH', 'HOME', 'DOCKER_HOST'})
        self.assertEqual(argv[-3:], [IMAGE, '-m', 'app.parser_worker'])

    def test_mutable_image_and_absent_revision_are_rejected(self):
        for image, revision in (('parser:latest', 'reviewed'), (IMAGE, '')):
            with self.assertRaises(wire.ParserRefused):
                DockerParser(image, revision)

    def test_health_refuses_missing_linux_security_or_mismatched_build(self):
        runner = DockerParser(IMAGE, 'reviewed')
        info = {'OSType': 'linux', 'SecurityOptions': ['name=apparmor', 'name=seccomp,profile=builtin']}
        image = [{'Id': IMAGE, 'Config': {'Env': ['VA_LSE_BUILD_SHA=reviewed']}}]
        with patch.object(runner, '_command', side_effect=[info, image]):
            self.assertTrue(runner.ready()['ready'])
        for change in ({'OSType': 'windows'}, {'SecurityOptions': ['name=seccomp,profile=builtin']},
                       {'SecurityOptions': ['name=apparmor', 'name=seccomp,profile=unconfined']}):
            with patch.object(runner, '_command', return_value={**info, **change}), self.assertRaises(wire.ParserRefused):
                runner.ready()
        with patch.object(runner, '_command', side_effect=[info, [{'Id': IMAGE, 'Config': {'Env': []}}]]), self.assertRaises(wire.ParserRefused):
            runner.ready()

    def test_private_service_transfers_only_verified_bytes(self):
        runner = MagicMock()
        runner.run.return_value = wire.encode(response())
        left, right = socket.socketpair()
        try:
            thread = threading.Thread(target=lambda: handle(left, runner))
            thread.start()
            right.settimeout(3)
            right.sendall(wire.frame(wire.encode(request())))
            self.assertEqual(wire.decode(wire.recv_frame(right, wire.MAX_HEADER)), {'accepted': True})
            right.sendall(DATA)
            self.assertEqual(wire.decode(wire.recv_frame(right, wire.MAX_OUTPUT)), response())
            thread.join(3)
            self.assertFalse(thread.is_alive())
            runner.run.assert_called_once_with(request(), DATA)
        finally:
            left.close()
            right.close()

    def test_invalid_input_is_refused_before_container_launch(self):
        runner, connection = MagicMock(), MagicMock()
        with patch('app.parser_service.recv_frame', return_value=wire.encode({**request(), 'command': 'unsafe'})):
            handle(connection, runner)
        runner.run.assert_not_called()
        self.assertNotIn('unsafe', str(connection.sendall.call_args))

    def test_runtime_exception_content_does_not_reach_reply(self):
        runner, connection = MagicMock(), MagicMock()
        runner.ready.side_effect = OSError('SYNTHETIC_SECRET_CANARY')
        with patch('app.parser_service.recv_frame', return_value=wire.encode({'operation': 'health'})):
            handle(connection, runner)
        self.assertNotIn('SYNTHETIC_SECRET_CANARY', str(connection.sendall.call_args))

    def test_cleanup_is_attempted_when_runtime_exits_without_output(self):
        runner = DockerParser(IMAGE, 'reviewed')
        # A real local synthetic child exercises pipe lifecycle without Docker.
        with patch.object(runner, 'command', return_value=[os.sys.executable, '-c', 'import sys; sys.stdin.buffer.read()']), \
                patch('app.parser_service.subprocess.run') as remove, self.assertRaises(ValueError):
            runner.run(request(), DATA)
        self.assertEqual(remove.call_args.args[0][1:3], ['rm', '--force'])


if __name__ == '__main__':
    unittest.main()
