"""Executed Linux container probes, opt-in with a reviewed *synthetic* test image.

No host files are mounted into the parser. A temporary test canary stays on the
host and a loopback listener stays in the host namespace. No external packet is
sent: AppArmor rejects socket creation before any connect call.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from app import parser_protocol as wire
from app.parser_service import DockerParser
from app.isolated_extract import _documents, _unpack_reply

IMAGE = os.environ.get('VA_LSE_TEST_PARSER_IMAGE', '')
REVISION = os.environ.get('VA_LSE_TEST_PARSER_REVISION', '')


@unittest.skipUnless(IMAGE and REVISION, 'Opt-in Linux Docker parser image is unset.')
class ParserContainerTests(unittest.TestCase):
    def setUp(self):
        self.runner = DockerParser(IMAGE, REVISION, docker=shutil.which('docker') or '/usr/bin/docker')
        self.runner.ready()
        self.data = b'Synthetic observation of knee pain.'
        self.request = {'version': 1, 'label': 'record.txt', 'size': len(self.data),
                        'sha256': hashlib.sha256(self.data).hexdigest(), 'nonce': uuid.uuid4().hex}

    def probe(self, code):
        command = self.runner.command
        def argv(name):
            return command(name)[:-2] + ['-c', code]
        with patch.object(self.runner, 'command', side_effect=argv):
            return wire.decode(self.runner.run(self.request, self.data))['response']

    def test_normal_file_and_archive_parse_without_local_fallback(self):
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(self.request, self.data)), IMAGE), self.request, IMAGE)
        self.assertEqual(docs[0].pages[0].text, self.data.decode())
        self.assertFalse(skipped)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('one.txt', self.data)
            output.writestr('folder/two.txt', b'Second synthetic observation.')
        data = archive.getvalue()
        req = {**self.request, 'label': 'records.zip', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(req, data)), IMAGE), req, IMAGE)
        self.assertEqual([d.filename for d in docs], ['records/one.txt', 'records/folder/two.txt'])
        # Input staging must allow >32 MB even though returned output is capped at 32 MB.
        # The large unsupported member is synthetic padding and is never extracted.
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as output:
            output.writestr('one.txt', self.data)
            output.writestr('padding.bin', b'0' * (36 * 1024 ** 2))
        data = archive.getvalue()
        req = {**req, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(req, data)), IMAGE), req, IMAGE)
        self.assertEqual(docs[0].pages[0].text, self.data.decode())
        self.assertEqual(len(skipped), 1)

    def test_linux_uid_capabilities_seccomp_apparmor_and_limits_are_active(self):
        result = self.probe('''import sys, os, json, resource
sys.stdin.buffer.read()
status = dict(line.strip().split(':', 1) for line in open('/proc/self/status') if ':' in line)
print(json.dumps({'uid': os.getuid(), 'caps': status['CapEff'].strip(),
 'nnp': status['NoNewPrivs'].strip(), 'seccomp': status['Seccomp'].strip(),
 'profile': open('/proc/self/attr/current').read().strip(),
 'cpu': resource.getrlimit(resource.RLIMIT_CPU), 'fsize': resource.getrlimit(resource.RLIMIT_FSIZE)}))''')
        self.assertEqual(result['uid'], 65534)
        self.assertEqual(int(result['caps'], 16), 0)
        self.assertEqual((result['nnp'], result['seccomp']), ('1', '2'))
        self.assertIn('va-lse-parser', result['profile'])
        self.assertEqual(result['cpu'], [30, 30])
        self.assertEqual(result['fsize'], [wire.MAX_INPUT, wire.MAX_INPUT])

    def test_application_secrets_other_case_and_host_canary_are_absent(self):
        with tempfile.TemporaryDirectory(prefix='va-host-canary-') as directory:
            host = Path(directory) / 'secret.txt'
            host.write_text('SYNTHETIC_HOST_SECRET_CANARY_493')
            paths = [str(host), '/run/pilot/approval.json', '/app/.streamlit/secrets.toml',
                     '/app/logs/other-case.txt', '/var/run/docker.sock', '/app/app/pilot.py']
            result = self.probe(f'''import sys, os, json
sys.stdin.buffer.read()
paths={paths!r}
result={{}}
for path in paths:
 try:
  with open(path, 'rb') as stream: stream.read(1)
  result[path]=True
 except OSError: result[path]=False
print(json.dumps(result))''')
        self.assertFalse(any(result[path] for path in paths))

    def test_loopback_private_public_and_unix_socket_creation_are_denied(self):
        with socket.socket() as host:
            host.bind(('127.0.0.1', 0))
            host.listen(1)
            port = host.getsockname()[1]
            result = self.probe(f'''import sys, socket, json
sys.stdin.buffer.read()
results=[]
for family, address in [(socket.AF_INET, ('127.0.0.1', {port})),
 (socket.AF_INET, ('10.0.0.1', 443)), (socket.AF_INET, ('1.1.1.1', 443)),
 (socket.AF_UNIX, '/tmp/escape.sock')]:
 try:
  s=socket.socket(family); s.settimeout(.5); s.connect(address); results.append(False)
 except PermissionError: results.append(True)
print(json.dumps({{'denied':results}}))''')
        self.assertEqual(result['denied'], [True] * 4)

    def test_filesystem_is_read_only_and_tmp_output_is_bounded(self):
        result = self.probe('''import sys, json, os
sys.stdin.buffer.read()
result={}
for path in ['/app/escape', '/etc/escape']:
 try:
  open(path, 'w').write('x'); result[path]=False
 except OSError: result[path]=True
try:
 with open('/tmp/overflow', 'wb') as stream:
  for _ in range(52): stream.write(b'x' * 1048576)
 result['bounded']=False
except OSError: result['bounded']=True
print(json.dumps(result))''')
        self.assertTrue(all(result[k] for k in ('/app/escape', '/etc/escape', 'bounded')))

    def test_memory_exhaustion_refuses_document_and_launcher_survives(self):
        with self.assertRaises(wire.ParserRefused):
            self.probe("import sys; sys.stdin.buffer.read(); bytearray(2 * 1024**3)")
        _documents(_unpack_reply(wire.decode(self.runner.run(self.request, self.data)), IMAGE), self.request, IMAGE)

    def test_pid_exhaustion_is_bounded_and_descendants_are_removed(self):
        result = self.probe('''import sys, os, time, json
sys.stdin.buffer.read()
children=[]
for _ in range(64):
 try:
  child=os.fork()
  if child==0: time.sleep(120); os._exit(0)
  children.append(child)
 except BlockingIOError: break
print(json.dumps({'count':len(children), 'bounded':len(children)<32}), flush=True)''')
        self.assertTrue(result['bounded'])
        self.assertLess(result['count'], 32)
        running = subprocess.run([self.runner.docker, 'ps', '--filter', 'name=va-parser-', '--format', '{{.Names}}'],
                                 capture_output=True, check=True, timeout=10)
        self.assertEqual(running.stdout.strip(), b'')

    def test_output_flood_and_timeout_remove_container(self):
        with self.assertRaises(wire.ParserRefused):
            self.probe("import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(b'x' * (34*1024**2)); sys.stdout.flush()")
        with patch('app.parser_service.DEADLINE', -8), self.assertRaises(wire.ParserRefused):
            self.probe("import sys,time; sys.stdin.buffer.read(); time.sleep(120)")
        _documents(_unpack_reply(wire.decode(self.runner.run(self.request, self.data)), IMAGE), self.request, IMAGE)

    def test_private_launcher_image_and_channel_work_end_to_end(self):
        name = 'va-parser-launcher-ci-' + uuid.uuid4().hex
        volume = 'va-parser-channel-ci-' + uuid.uuid4().hex
        docker = self.runner.docker
        group = str(os.stat('/var/run/docker.sock').st_gid)
        def call(args, **kwargs):
            return subprocess.run([docker, *args], capture_output=True, check=True, timeout=100, **kwargs)
        call(['volume', 'create', volume])
        try:
            call(['run', '--detach', '--name', name, '--network=none', '--read-only',
                  '--cap-drop=ALL', '--security-opt=no-new-privileges:true', '--group-add', group,
                  '--memory=512m', '--cpus=1', '--pids-limit=64',
                  '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777',
                  '--volume=/var/run/docker.sock:/var/run/docker.sock',
                  '--volume=' + volume + ':/run/parser',
                  '--env=VA_LSE_PARSER_IMAGE=' + IMAGE, '--env=VA_LSE_BUILD_SHA=' + REVISION,
                  'va-lse-parser-launcher:ci'])
            code = ("import time,socket; from app.parser_protocol import *; "
                    "s=socket.socket(socket.AF_UNIX); s.settimeout(90); "
                    "s.connect(SOCKET_PATH); s.sendall(frame(encode(" + repr(self.request) + "))); "
                    "assert decode(recv_frame(s, MAX_HEADER))=={'accepted':True}; "
                    "s.sendall(" + repr(self.data) + "); "
                    "print(encode(decode(recv_frame(s, MAX_OUTPUT))).decode())")
            # Wait for readiness using the launcher user's real channel, not host access.
            import time
            for attempt in range(30):
                ready = subprocess.run([docker, 'exec', name, 'python', '-c',
                                        "from pathlib import Path; from app.parser_protocol import SOCKET_PATH; assert Path(SOCKET_PATH).is_socket()"],
                                       capture_output=True, timeout=10)
                if ready.returncode == 0:
                    break
                if attempt == 29:
                    self.fail('The trusted launcher did not become ready.')
                time.sleep(.5)
            result = call(['run', '--rm', '--network=none', '--read-only', '--user=65534:65534',
                           '--volume=' + volume + ':/run/parser:ro', '--entrypoint=python', IMAGE, '-c', code])
            docs, skipped = _documents(_unpack_reply(wire.decode(result.stdout), IMAGE), self.request, IMAGE)
            self.assertEqual(docs[0].pages[0].text, self.data.decode())
            self.assertFalse(skipped)
        finally:
            subprocess.run([docker, 'rm', '--force', name], capture_output=True, timeout=10)
            call(['volume', 'rm', volume])

    def test_pid_one_supervisor_deadline_survives_launcher_loss(self):
        # Same production supervisor, shortened timer, synthetic stalled read.
        code = "import app.parser_worker as w, time; w.DEADLINE=-8; w.read_request=lambda _: time.sleep(120); w.supervise()"
        with self.assertRaises(wire.ParserRefused):
            self.probe(code)


if __name__ == '__main__':
    unittest.main()
