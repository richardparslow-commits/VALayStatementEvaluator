"""Synthetic dedicated-engine configuration and trust-boundary regressions."""
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from app import parser_protocol as wire
from app.parser_engine import ParserEngine, parser_engine_id
from app.parser_service import DockerParser
from app.isolated_extract import IsolatedExtractor, _unpack_reply, parser_health
from app.documents import ExtractionError

IMAGE = 'sha256:' + 'a' * 64
ENGINE = ParserEngine('tcp://10.73.0.2:2376', 'synthetic-parser', 'synthetic-application')


class EngineTests(unittest.TestCase):
    def test_plaintext_context_local_socket_public_and_ambiguous_endpoints_refused(self):
        endpoints = ('', 'unix:///var/run/docker.sock', 'ssh://operator@parser', 'http://10.73.0.2:2376',
                     'tcp://10.73.0.2:2375', 'tcp://parser.example:2376', 'tcp://127.0.0.1:2376',
                     'tcp://169.254.1.2:2376', 'tcp://8.8.8.8:2376', 'tcp://10.73.0.2:2376/path',
                     'tcp://user@10.73.0.2:2376', 'tcp://010.73.0.2:2376', ' tcp://10.73.0.2:2376')
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint), self.assertRaises(wire.ParserRefused):
                replace(ENGINE, endpoint=endpoint)
        for endpoint in ('tcp://172.16.0.2:2376', 'tcp://192.168.1.2:2376'):
            self.assertEqual(replace(ENGINE, endpoint=endpoint).environment()['DOCKER_TLS_VERIFY'], '1')

    def test_absent_identity_and_shared_application_engine_refused(self):
        for identity in ('', 'x\n', '../engine', 'x' * 129, ENGINE.application_identity):
            with self.subTest(identity=identity), self.assertRaises(wire.ParserRefused):
                replace(ENGINE, identity=identity)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(wire.ParserRefused):
            ParserEngine.from_environment()
        with patch.dict(os.environ, {'VA_LSE_PARSER_ENGINE_ID': ''}), self.assertRaises(wire.ParserRefused):
            parser_engine_id()

    def test_credentials_missing_unsafe_symlink_and_over_limit_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = replace(ENGINE, tls_directory=Path(directory).resolve())
            with self.assertRaises(wire.ParserRefused):
                engine.verify_credentials()
            for name in ('ca.pem', 'cert.pem', 'key.pem'):
                path = engine.tls_directory / name
                path.write_bytes(b'SYNTHETIC_NOT_A_REAL_CERTIFICATE')
                path.chmod(0o400)
            engine.verify_credentials()  # Docker, not this metadata check, validates X.509.
            for mode in (0o770, 0o777, 0o755):
                engine.tls_directory.chmod(mode)
                with self.subTest(directory_mode=mode), self.assertRaises(wire.ParserRefused):
                    engine.verify_credentials()
            engine.tls_directory.chmod(0o700)
            key = engine.tls_directory / 'key.pem'
            for mode in (0o440, 0o444, 0o420, 0o402):
                key.chmod(mode)
                with self.subTest(mode=mode), self.assertRaises(wire.ParserRefused):
                    engine.verify_credentials()
            key.chmod(0o400)
            target = engine.tls_directory / 'other-key'
            key.rename(target)
            key.symlink_to(target)
            with self.assertRaises(wire.ParserRefused):
                engine.verify_credentials()
            key.unlink()
            target.rename(key)
            key.chmod(0o600)
            key.write_bytes(b'x' * (128 * 1024 + 1))
            with self.assertRaises(wire.ParserRefused):
                engine.verify_credentials()

    def test_ambient_context_or_tls_disable_cannot_change_launcher_environment(self):
        with patch.dict(os.environ, {'DOCKER_HOST': 'unix:///var/run/docker.sock', 'DOCKER_TLS_VERIFY': '',
                                    'DOCKER_CONTEXT': 'default', 'PROVIDER_SECRET': 'SYNTHETIC_CANARY'}):
            env = DockerParser(IMAGE, 'reviewed', engine=ENGINE).env
        self.assertEqual(env['DOCKER_HOST'], ENGINE.endpoint)
        self.assertEqual(env['DOCKER_TLS_VERIFY'], '1')
        self.assertNotIn('DOCKER_CONTEXT', env)
        self.assertNotIn('PROVIDER_SECRET', env)
        with patch.dict(os.environ, {'VA_LSE_PARSER_ENGINE_ENDPOINT': ENGINE.endpoint,
                                    'VA_LSE_PARSER_ENGINE_ID': ENGINE.identity,
                                    'VA_LSE_APPLICATION_ENGINE_ID': ENGINE.application_identity}):
            self.assertEqual(ParserEngine.from_environment(), ENGINE)

    def test_wrong_identity_or_missing_purpose_refused_before_launch(self):
        runner = DockerParser(IMAGE, 'reviewed', engine=ENGINE)
        good = {'ID': ENGINE.identity, 'Labels': ['va-lse-purpose=parser-only'], 'OSType': 'linux',
                'SecurityOptions': ['name=apparmor', 'name=seccomp,profile=builtin']}
        for change in ({'ID': ENGINE.application_identity}, {'ID': 'unknown'}, {'Labels': []},
                       {'ID': None, 'ServerErrors': ['Synthetic TLS refusal']}, {'ID': ''}):
            with self.subTest(change=change), patch.object(runner, '_command', return_value={**good, **change}), \
                    patch('app.parser_service.subprocess.Popen') as launch, self.assertRaises(wire.ParserRefused):
                import hashlib
                data = b'Synthetic record'
                runner.run({'version': 1, 'label': 'record.txt', 'size': len(data),
                            'sha256': hashlib.sha256(data).hexdigest(), 'nonce': 'b' * 32, 'page_limit': 500}, data)
            launch.assert_not_called()

    def test_client_health_and_envelope_require_the_reviewed_engine(self):
        with patch.dict(os.environ, {'VA_LSE_PARSER_ENGINE_ID': ENGINE.identity, 'VA_LSE_PARSER_IMAGE': IMAGE,
                                    'VA_LSE_BUILD_SHA': 'reviewed'}):
            for identity in ('unknown', ENGINE.application_identity):
                with self.subTest(identity=identity), self.assertRaises(wire.ParserRefused):
                    _unpack_reply({'image': IMAGE, 'engine_id': identity, 'response': {}}, IMAGE)
                with patch('app.isolated_extract.socket.socket'), patch('app.isolated_extract.recv_frame', return_value=wire.encode(
                    {'ready': True, 'image': IMAGE, 'revision': 'reviewed', 'max_pages': 500, 'engine_id': identity})), \
                        self.assertRaises(wire.ParserRefused):
                    parser_health()
            self.assertEqual(_unpack_reply({'image': IMAGE, 'engine_id': ENGINE.identity, 'response': {}}, IMAGE), {'image': IMAGE})
            with self.assertRaises(wire.ParserRefused):
                _unpack_reply({'image': IMAGE, 'response': {}}, IMAGE)

    def test_missing_engine_configuration_prevents_any_record_transfer(self):
        with patch.dict(os.environ, {'VA_LSE_PARSER_IMAGE': IMAGE, 'VA_LSE_PARSER_ENGINE_ID': ''}), \
                patch('app.isolated_extract.socket.socket') as connection, self.assertRaises(ExtractionError):
            IsolatedExtractor().extract('record.txt', b'Synthetic record')
        connection.assert_not_called()


if __name__ == '__main__':
    unittest.main()
