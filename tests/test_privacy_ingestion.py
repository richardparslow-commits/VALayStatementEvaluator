"""Synthetic privacy boundary regressions; no account or veteran data."""
from __future__ import annotations
import asyncio
import contextlib
import concurrent.futures
import copy
import hashlib
import io
import json
import os
import stat
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests import hermetic  # noqa: F401
from tests import test_resource_guards as resources
from tests.test_controlled_pilot import approval
from app import pilot, private_uploads, privacy_acceptance, upload_temp
from app.documents import DocumentPage, ExtractedDocument
from scripts import extract_pdfs as references
import starlette.formparsers as formparsers
from starlette.datastructures import FormData, UploadFile
from starlette.requests import Request

CANARY = 'SYNTHETIC_PRIVACY_CANARY_ONLY'


class MultipartPrivacyTests(unittest.TestCase):
    def setUp(self):
        cleanup = patch.object(private_uploads, '_CLEANUP_FAILED', threading.Event())
        cleanup.start(); self.addCleanup(cleanup.stop)
        self.boundary = resources.UploadTests()
        self.boundary.setUp()
        self.addCleanup(self.boundary.doCleanups)
        self.spools = []
        original = formparsers.SpooledTemporaryFile
        def make(*args, **kwargs):
            spool = original(*args, **kwargs)
            self.spools.append(spool)
            return spool
        context = patch.object(formparsers, 'SpooledTemporaryFile', side_effect=make)
        context.start(); self.addCleanup(context.stop)
        self.addCleanup(lambda: [f.close() for f in self.spools])

    def put(self, files=None, **kwargs):
        return self.boundary.client.put('/_stcore/upload_file/invented-session/new',
            headers=kwargs.pop('headers', self.boundary.headers()), files=files, **kwargs)

    def assert_closed(self):
        self.assertTrue(self.spools)
        self.assertTrue(all(f.closed for f in self.spools))
        self.assertFalse(self.boundary.registry._pending)

    def test_success_closes_rolled_spool_before_storage(self):
        data = b'x' * 1100000
        response = self.put([('file', ('synthetic.txt', data, 'text/plain'))])
        self.assertEqual(response.status_code, 204)
        self.assertTrue(all(f.closed for f in self.spools))
        self.assertFalse(self.boundary.registry._pending)
        self.assertTrue(self.spools[0]._rolled)
        self.assertEqual(self.boundary.manager.get_files('invented-session', ['new'])[0].data, data)
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_distinct_and_duplicate_fields_cannot_insert_two_files(self):
        for field in ('other', 'file'):
            with self.subTest(field=field):
                response = self.put([('file', ('a.txt', b'x'*1100000)), (field, ('b.txt', b'y'*1100000))])
                self.assertEqual(response.status_code, 400)
                self.assert_closed()
                self.assertEqual(self.boundary.manager._total_bytes, 0)

    def test_zero_files_fields_limit_and_field_size_are_refused(self):
        for content in (b'--boundary--\r\n', b'--boundary\r\nContent-Disposition: form-data; name="field"\r\n\r\n'+b'x'*4097+b'\r\n--boundary--\r\n'):
            response = self.put(content=content, headers={**self.boundary.headers(), 'Content-Type':'multipart/form-data; boundary=boundary'})
            self.assertEqual(response.status_code, 400)
        response=self.put([('file',('a.txt',b'x'))], data={str(n):'x' for n in range(5)})
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.boundary.manager._total_bytes,0)
        # The field limit may fire before the first file is allocated.
        self.assertTrue(all(f.closed for f in self.spools))
        self.assertFalse(self.boundary.registry._pending)

    def test_read_failure_and_cancellation_close_allocations(self):
        for error in (OSError(CANARY), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__), patch.object(UploadFile,'read',side_effect=error):
                try:
                    response=self.put([('file',('a.txt',b'x'*1100000))])
                    self.assertNotEqual(response.status_code,204)
                    self.assertNotIn(CANARY,response.text)
                except BaseException as exc:
                    self.assertIsInstance(exc, (asyncio.CancelledError, concurrent.futures.CancelledError, RuntimeError))
                self.assert_closed()
                self.assertEqual(self.boundary.manager._total_bytes,0)

    def test_unfinished_multipart_part_closes_spool_absent_from_form(self):
        body=b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\n'+b'x'*1100000
        response=self.put(content=body,headers={**self.boundary.headers(),'Content-Type':'multipart/form-data; boundary=boundary'})
        self.assertEqual(response.status_code,400)
        self.assert_closed()
        self.assertEqual(self.boundary.manager._total_bytes,0)

    def test_parser_failure_closes_allocations(self):
        real=formparsers.MultiPartParser.on_part_data
        def fail(parser,*args):
            if parser._current_part.file is not None:
                raise ValueError(CANARY)
            return real(parser,*args)
        with patch.object(formparsers.MultiPartParser,'on_part_data',fail):
            response=self.put([('file',('a.txt',b'x'*1100000))])
        self.assertEqual(response.status_code,403)
        self.assertNotIn(CANARY,response.text)
        self.assert_closed()

    def test_finished_part_without_terminal_boundary_is_refused_and_closed(self):
        body=b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\n'+b'x'*1100000+b'\r\n--boundary\r\n'
        response=self.put(content=body,headers={**self.boundary.headers(),'Content-Type':'multipart/form-data; boundary=boundary'})
        self.assertEqual(response.status_code,400)
        self.assert_closed()
        self.assertEqual(self.boundary.manager._total_bytes,0)

    def test_disconnect_stream_error_and_timeout_close_active_spool(self):
        routes=private_uploads.create_upload_routes(self.boundary.current,self.boundary.manager,None)
        endpoint=next(route.endpoint for route in routes if 'PUT' in (route.methods or set()))
        body=b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\n'+b'x'*1100000
        for failure in ('disconnect','error','timeout'):
            sent=False
            async def receive():
                nonlocal sent
                if not sent:
                    sent=True
                    return {'type':'http.request','body':body,'more_body':True}
                if failure=='disconnect': return {'type':'http.disconnect'}
                if failure=='error': raise OSError(CANARY)
                await asyncio.sleep(1)
            scope={'type':'http','method':'PUT','path':'/_stcore/upload_file/invented-session/new','path_params':{'session_id':'invented-session','file_id':'new'},'headers':[(b'content-type',b'multipart/form-data; boundary=boundary')], 'scheme':'https','query_string':b''}
            async def run():
                await asyncio.wait_for(endpoint(Request(scope,receive)),.03)
            with self.subTest(failure=failure),patch('streamlit.web.server.starlette.starlette_routes.is_xsrf_enabled',return_value=False):
                with self.assertRaises(Exception): asyncio.run(run())
                self.assert_closed()
                self.assertEqual(self.boundary.manager._total_bytes,0)

    def test_cleanup_continues_after_one_close_failure(self):
        first=SimpleNamespace(close=lambda: (_ for _ in ()).throw(OSError(CANARY)))
        closed=[]
        second=SimpleNamespace(close=lambda: closed.append(True))
        with self.assertRaises(pilot.PilotBlocked) as error:
            private_uploads.close_spools([first,second])
        self.assertEqual(closed,[True]); self.assertNotIn(CANARY,str(error.exception))
        with self.assertRaises(pilot.PilotBlocked): private_uploads.check_cleanup()
        self.assertEqual(self.put([('file',('a.txt',b'x'))]).status_code,503)
        self.assertEqual(self.boundary.manager._total_bytes,0)

    def test_dependency_mismatch_refuses_adapter(self):
        with patch.object(private_uploads,'version',return_value='unreviewed'):
            with self.assertRaises(pilot.PilotBlocked): private_uploads.install()


class ReferencePrivacyTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root=Path(folder.name).resolve(); self.root.chmod(0o700)
        self.source=self.root/'source.pdf'; self.source.write_bytes(b'%PDF-synthetic')
        self.output=self.root/'result.txt'
        document=ExtractedDocument('reference.pdf',[DocumentPage('reference.pdf',1,CANARY)],1,[],coverage_known=True)
        context=patch.object(references,'IsolatedExtractor')
        self.parser=context.start()
        self.addCleanup(context.stop)
        self.parser.return_value.extract.return_value=([document],[])

    def test_missing_class_sensitive_and_pilot_refuse_before_file_access(self):
        with patch.object(Path,'exists') as exists, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): references.main(['source.pdf'])
            for data_class, enabled in [('sensitive',False),('synthetic',True),('approved-public',True)]:
                with patch.object(pilot,'enabled',return_value=enabled):
                    self.assertEqual(references.main(['--data-class',data_class,'--out',str(self.output),str(self.source)]),3)
                    with self.assertRaises(ValueError): references.extract(self.source,data_class=data_class)
                    with self.assertRaises(ValueError): references.publish(self.output,b'x',data_class=data_class)
            exists.assert_not_called()
        self.parser.assert_not_called()

    def test_synthetic_derivative_is_private_explicit_and_never_overwritten(self):
        args=['--data-class','synthetic','--out',str(self.output),str(self.source)]
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(references.main(args),0)
            self.assertEqual(references.main(args),3)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode),0o600)
        self.assertIn(CANARY,self.output.read_text())
        self.parser.return_value.extract.assert_called_once_with('reference.pdf',b'%PDF-synthetic')

    def test_public_requires_exact_hash_in_protected_manifest(self):
        manifest=self.root/'public.json'
        manifest.write_text(json.dumps({'schema_version':1,'approved_sha256':[hashlib.sha256(self.source.read_bytes()).hexdigest()]}))
        manifest.chmod(0o600)
        self.assertIn(CANARY,references.extract(self.source,data_class='approved-public',public_manifest=manifest))
        self.source.write_bytes(b'%PDF-changed')
        with self.assertRaises(ValueError): references.extract(self.source,data_class='approved-public',public_manifest=manifest)
        manifest.chmod(0o666)
        with self.assertRaises(ValueError): references.extract(self.source,data_class='approved-public',public_manifest=manifest)
        with self.assertRaises(ValueError): references.extract(self.source,data_class='approved-public')
        self.assertEqual(self.parser.return_value.extract.call_count,1)

    def test_repository_public_directory_symlink_and_existing_output_refused(self):
        targets=[references.REPOSITORY/'reference_docs/extracted/new.txt']
        other=self.root/'other'; other.mkdir(mode=0o700); (other/'.git').mkdir(); targets.append(other/'result.txt')
        public=self.root/'public'; public.mkdir(mode=0o755); targets.append(public/'result.txt')
        link=self.root/'link'; link.symlink_to(self.source); targets.append(link)
        for target in targets:
            with self.subTest(target=target.name),self.assertRaises((ValueError,OSError)):
                references.publish(target,b'x',data_class='synthetic')
        self.assertEqual(self.source.read_bytes(),b'%PDF-synthetic')

    def test_bounded_regular_input_refuses_fifo_links_and_oversize(self):
        link=self.root/'alias.pdf'; link.symlink_to(self.source)
        fifo=self.root/'pipe.pdf'; os.mkfifo(fifo)
        for path in (link,fifo):
            with self.assertRaises((ValueError,OSError)): references.read_regular_bounded(path,20)
        with self.assertRaises(ValueError): references.read_regular_bounded(self.source,3)

    def test_parser_error_incomplete_result_and_skip_publish_no_text_or_raw_error(self):
        args=['--data-class','synthetic','--out',str(self.output),str(self.source)]
        for result in [([],['bad']),([],[])]:
            self.parser.return_value.extract.return_value=result
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(references.main(args),3)
            self.assertEqual(stderr.getvalue().strip(),references.REFUSAL)
        self.parser.return_value.extract.side_effect=ValueError(CANARY)
        with contextlib.redirect_stderr(io.StringIO()) as stderr: self.assertEqual(references.main(args),3)
        self.assertNotIn(CANARY,stderr.getvalue()); self.assertFalse(self.output.exists())


class AcceptancePrivacyTests(unittest.TestCase):
    def test_all_observations_are_required_and_bound_to_exact_configuration(self):
        data=approval(); privacy_acceptance.validate(data)
        changes=[None,{}, {**data['privacy_acceptance'],'status':'NOT_RUN'}, {**data['privacy_acceptance'],'schema_version':True}]
        for value in changes:
            with self.assertRaises(ValueError): privacy_acceptance.validate({**data,'privacy_acceptance':value})
        for field in privacy_acceptance.BINDING_FIELDS:
            changed=copy.deepcopy(data); changed[field]='changed'
            with self.subTest(field=field),self.assertRaises((ValueError,TypeError)): privacy_acceptance.validate(changed)
        for key in privacy_acceptance.CHECKS:
            changed=copy.deepcopy(data); changed['privacy_acceptance']['checks'][key]['status']='NOT_RUN'
            with self.subTest(check=key),self.assertRaises(ValueError): privacy_acceptance.validate(changed)

    def test_independent_reviewer_freshness_images_and_evidence_are_required(self):
        base=approval()
        for key,value in [('independent_reviewer',base['privacy_acceptance']['operator']),('operator',' '),('reviewed_at',(datetime.now(timezone.utc)-timedelta(days=31)).isoformat()),('reviewed_at',(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()),('images',{'parser':'latest'}),('checks',{})]:
            data=copy.deepcopy(base); data['privacy_acceptance'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): privacy_acceptance.validate(data)
        for key,value in [('evidence_ref',''),('evidence_sha256','missing')]:
            data=copy.deepcopy(base); data['privacy_acceptance']['checks']['P01'][key]=value
            with self.assertRaises(ValueError): privacy_acceptance.validate(data)
        with patch.dict(os.environ,{'VA_LSE_PARSER_IMAGE':'sha256:'+'d'*64}):
            with self.assertRaises(ValueError): privacy_acceptance.validate(base)
        changed=copy.deepcopy(base); changed['privacy_acceptance']['checks']['P01']['evidence_sha256']='d'*64
        self.assertNotEqual(pilot.notice_binding('owner',base),pilot.notice_binding('owner',changed))

    def test_manifest_references_alone_cannot_open_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'approval.json'; data=approval(); data.pop('privacy_acceptance')
            path.write_text(json.dumps(data))
            with patch.dict(os.environ,{'VA_LSE_PILOT_APPROVAL_FILE':str(path),'VA_LSE_BUILD_SHA':'a'*40}):
                with self.assertRaises(pilot.PilotBlocked): pilot.load_approval()


class UploadTempPrivacyTests(unittest.TestCase):
    def test_wrong_platform_or_effective_destination_refuses(self):
        with patch.object(upload_temp.sys,'platform','linux'),patch('os.geteuid',return_value=65534),patch.dict(os.environ,{'TMPDIR':'/tmp','TEMP':'/tmp','TMP':'/tmp'}):
            with self.assertRaises(pilot.PilotBlocked): upload_temp.validate()

    def test_private_effective_mount_and_capacity_are_checked(self):
        path=upload_temp.UPLOAD_TEMP
        info=SimpleNamespace(st_mode=stat.S_IFDIR|0o700,st_uid=65534)
        line=b'8 7 0:2 / /run/upload-tmp rw,nosuid,nodev,noexec - tmpfs tmpfs rw,size=262144k\n'
        context=contextlib.ExitStack(); self.addCleanup(context.close)
        for p in [patch.object(upload_temp.sys,'platform','linux'),patch('os.geteuid',return_value=65534),patch.dict(os.environ,{key:str(path) for key in ('TMPDIR','TEMP','TMP')}),patch.object(Path,'resolve',return_value=path),patch.object(Path,'lstat',return_value=info),patch.object(tempfile,'gettempdir',return_value=str(path)),patch.object(Path,'open',return_value=io.BytesIO(line)),patch('os.statvfs',return_value=SimpleNamespace(f_blocks=256,f_frsize=1024*1024)),patch.object(tempfile,'TemporaryFile')]: context.enter_context(p)
        with patch('os.fstat',return_value=SimpleNamespace(st_mode=stat.S_IFREG|0o600)): upload_temp.validate()
        for mount in [line.replace(b'tmpfs tmpfs',b'ext4 disk'),line.replace(b',noexec',b''),line.replace(b'/run/upload-tmp',b'/tmp')]:
            with patch.object(Path,'open',return_value=io.BytesIO(mount)),self.assertRaises(pilot.PilotBlocked): upload_temp.validate()
        with patch('os.statvfs',return_value=SimpleNamespace(f_blocks=257,f_frsize=1024*1024)),patch.object(Path,'open',return_value=io.BytesIO(line)),self.assertRaises(pilot.PilotBlocked): upload_temp.validate()


if __name__=='__main__': unittest.main()
