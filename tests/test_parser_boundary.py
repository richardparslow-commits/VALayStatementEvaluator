"""Synthetic parser protocol and launcher regressions; no application providers."""
from __future__ import annotations

import hashlib
import io
import os
import socket
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
            'sha256': hashlib.sha256(DATA).hexdigest(), 'nonce': 'b' * 32}


def response():
    return {**{k: v for k, v in request().items() if k != 'size'}, 'image': IMAGE,
            'documents': [{'filename': 'record.txt', 'schema_version': 2, 'total_pages': 1,
                           'unreadable_pages': [], 'pagination': 'block', 'coverage_known': True,
                           'pages': [{'page': 1, 'kind': 'block', 'text': DATA.decode()}]}], 'skipped': []}


class ProtocolTests(unittest.TestCase):
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
        for key, value in (('size', True), ('size', wire.MAX_INPUT + 1), ('label', 'a\n.txt'),
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
        for name in ('other/one.txt', 'records/../case.txt', 'records//absolute.txt'):
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


class LauncherTests(unittest.TestCase):
    def test_pilot_mounts_keep_credentials_and_daemon_in_different_services(self):
        from pathlib import Path
        import yaml
        root = Path(__file__).resolve().parents[1]
        services = yaml.safe_load((root / 'docker-compose.pilot.yml').read_text())['services']
        web, launcher = services['streamlit-web'], services['parser-launcher']
        self.assertFalse(any('docker.sock' in x for x in web['volumes']))
        self.assertNotIn('env_file', launcher)
        self.assertEqual(launcher['network_mode'], 'none')
        self.assertEqual(set(launcher['environment']), {'VA_LSE_PARSER_IMAGE', 'VA_LSE_BUILD_SHA'})
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
            right.sendall(wire.frame(wire.encode(request())) + DATA)
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
