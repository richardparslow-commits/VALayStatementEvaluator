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
    def test_packaged_parser_keeps_word_stories_and_block_provenance(self):
        from tests.ingestion_fixtures import docx_parts, package
        namespace = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        parts = docx_parts('No PTSD.')
        parts['word/header1.xml'] = f'<w:hdr xmlns:w="{namespace}"><w:p><w:r><w:t>HEADER_DENIAL_CANARY</w:t></w:r></w:p></w:hdr>'.encode()
        parts['word/footnotes.xml'] = f'<w:footnotes xmlns:w="{namespace}"><w:footnote w:id="1"><w:p><w:r><w:t>FOOTNOTE_FINAL_CANARY</w:t></w:r></w:p></w:footnote></w:footnotes>'.encode()
        data = package(parts)
        request = {**self.request, 'label': 'statement.docx', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(request, data)), IMAGE), request, IMAGE)
        self.assertFalse(skipped)
        self.assertIn('HEADER_DENIAL_CANARY', docs[0].full_text)
        self.assertIn('FOOTNOTE_FINAL_CANARY', docs[0].full_text)
        self.assertEqual(docs[0].pagination, 'block')
        self.assertEqual(docs[0].source_sha256, request['sha256'])
        self.assertTrue(all(page.source_part for page in docs[0].pages))

    def test_packaged_parser_unicode_decoding_carries_original_byte_identity(self):
        data = 'Élodie denies β pain; −1; 10 mg.'.encode('utf-16')
        request = {**self.request, 'label': 'unicode.txt', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(request, data)), IMAGE), request, IMAGE)
        self.assertFalse(skipped)
        self.assertEqual(docs[0].text_encoding, 'utf-16-bom')
        self.assertEqual(docs[0].source_sha256, request['sha256'])
        self.assertEqual(docs[0].full_text, data.decode('utf-16'))

    def setUp(self):
        self.runner = DockerParser(IMAGE, REVISION, docker=shutil.which('docker') or '/usr/bin/docker')
        self.runner.ready()
        self.data = b'Synthetic observation of knee pain.'
        self.request = {'version': 1, 'label': 'record.txt', 'size': len(self.data),
                        'sha256': hashlib.sha256(self.data).hexdigest(), 'nonce': uuid.uuid4().hex, 'page_limit': 500}

    def test_packaged_parser_preserves_hidden_archive_corrections(self):
        from tests.ingestion_fixtures import package
        correction = b'Synthetic correction: symptoms were denied.'
        members = {'record.txt': self.data, '.correction.txt': correction,
                   '__MACOSX/notes/.correction.md': correction}
        data = package(members)
        request = {**self.request, 'label': 'records.zip', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(request, data)), IMAGE), request, IMAGE)
        self.assertEqual(skipped, [])
        self.assertEqual([doc.filename for doc in docs], ['records/' + name for name in members])
        for doc, body in zip(docs, members.values()):
            self.assertEqual(doc.full_text, body.decode())
            self.assertEqual(doc.source_sha256, hashlib.sha256(body).hexdigest())

    def test_packaged_parser_names_unreadable_hidden_archive_members(self):
        from tests.ingestion_fixtures import package
        data = package({'record.txt': self.data, '.damaged.pdf': b'Synthetic invalid PDF',
                        '__MACOSX/.DS_Store': b'Synthetic unsupported metadata'})
        request = {**self.request, 'label': 'records.zip', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(request, data)), IMAGE), request, IMAGE)
        self.assertEqual([doc.filename for doc in docs], ['records/record.txt'])
        self.assertEqual(len(skipped), 2)
        for name in ('.damaged.pdf', '__MACOSX/.DS_Store'):
            self.assertTrue(any(name in message for message in skipped), skipped)

    def test_packaged_parser_preserves_split_source_text_and_offsets(self):
        source = 'pad ' * 998 + '2020-01-13: no diagnosis; dose -0.5 mg.\r\n\tSynthetic final denial.  '
        data = source.encode()
        request = {**self.request, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(request, data)), IMAGE), request, IMAGE)
        self.assertEqual(skipped, [])
        self.assertGreater(len(docs[0].pages), 1)
        self.assertEqual(docs[0].full_text, source)
        for page in docs[0].pages:
            self.assertEqual(page.text, source[page.source_start:page.source_end])
            self.assertLessEqual(len(page.text), 4000)
        self.assertTrue(any('2020-01-13' in p.text for p in docs[0].pages))
        self.assertEqual(docs[0].source_sha256, request['sha256'])

    def test_packaged_parser_preserves_docx_story_source_spans(self):
        from tests.ingestion_fixtures import docx_parts, package
        source = 'pad ' * 998 + '2020-01-13: no diagnosis.'
        data = package(docx_parts(source))
        request = {**self.request, 'label': 'record.docx', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(request, data)), IMAGE), request, IMAGE)
        self.assertEqual(skipped, [])
        self.assertEqual(docs[0].full_text, source)
        self.assertGreater(len(docs[0].pages), 1)
        self.assertTrue(all(p.source_part == 'word/document.xml' for p in docs[0].pages))
        self.assertTrue(any('2020-01-13' in p.text for p in docs[0].pages))

    def test_packaged_parser_explicitly_refuses_an_over_limit_source_token(self):
        data = b'X' * 4001
        request = {**self.request, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(request, data)), IMAGE), request, IMAGE)
        self.assertEqual(docs, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn('source token', skipped[0])

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

    def test_pdf_page_limit_is_applied_before_any_text_extraction(self):
        from pypdf import PdfWriter
        output = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_blank_page(width=100, height=100)
        writer.write(output)
        data = output.getvalue()
        req = {**self.request, 'label': 'two-pages.pdf', 'size': len(data),
               'sha256': hashlib.sha256(data).hexdigest(), 'page_limit': 1}
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(req, data)), IMAGE), req, IMAGE)
        self.assertFalse(docs)
        self.assertEqual(len(skipped), 1)
        self.assertIn('physical page limit before text extraction', skipped[0])

    def test_packaged_parser_enforces_passive_formats_with_no_mode_environment(self):
        from tests.ingestion_fixtures import docx_parts, package
        from pypdf import PdfWriter

        def parse(label, data):
            req = {**self.request, 'label': label, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            return _documents(_unpack_reply(wire.decode(self.runner.run(req, data)), IMAGE), req, IMAGE)

        for label, data in (('unicode.txt', 'Synthetic NO dose: 10 mg; β = −1.'.encode('utf-16')),
                            ('passive.docx', package(docx_parts()))):
            with self.subTest(label=label):
                docs, skipped = parse(label, data)
                self.assertEqual(len(docs), 1)
                self.assertFalse(skipped)
        output = io.BytesIO()
        writer = PdfWriter(); writer.add_blank_page(width=100, height=100)
        writer.add_js('app.alert("synthetic");'); writer.write(output)
        parts = docx_parts()
        parts['customXml/item1.xml'] = b'<!DOCTYPE x [<!ENTITY e "synthetic">]><x>&e;</x>'
        for label, data in (('renamed.txt', b'\x89PNG\r\n\x1a\nSynthetic binary'),
                            ('active.pdf', output.getvalue()), ('entities.docx', package(parts))):
            with self.subTest(label=label):
                docs, skipped = parse(label, data)
                self.assertFalse(docs)
                self.assertEqual(len(skipped), 1)

    def test_packaged_parser_refuses_unsafe_archive_before_partial_results(self):
        from tests.ingestion_fixtures import package
        data = package({'valid.txt': self.data, '../unsafe.txt': b'Synthetic unsafe name.'})
        req = {**self.request, 'label': 'records.zip', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        with self.assertRaises(wire.ParserRefused):
            self.runner.run(req, data)
        # A refusal must not poison the next container/job.
        docs, skipped = _documents(_unpack_reply(wire.decode(self.runner.run(self.request, self.data)), IMAGE), self.request, IMAGE)
        self.assertEqual(len(docs), 1)
        self.assertFalse(skipped)

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
