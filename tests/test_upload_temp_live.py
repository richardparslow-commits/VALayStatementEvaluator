"""Required runtime-image probe of effective upload tmpfs; synthetic bytes only."""
import os
import subprocess
import unittest

IMAGE=os.environ.get('VA_LSE_TEST_UPLOAD_RUNTIME_IMAGE','')
PROBE='''
import os, tempfile
from app import upload_temp, pilot
try:
    upload_temp.validate()
except pilot.PilotBlocked:
    print('REFUSED')
    raise SystemExit(3)
with tempfile.SpooledTemporaryFile(max_size=1024*1024) as spool:
    spool.write(b'SYNTHETIC_STARTUP_CANARY' * 60000)
    assert spool._rolled
    fd=spool.fileno()
    assert os.readlink('/proc/self/fd/'+str(fd)).startswith('/run/upload-tmp/')
try:
    os.fstat(fd)
    raise AssertionError('spool descriptor survived closure')
except OSError:
    pass
print('PRIVATE_TMPFS_AND_CLOSED_ROLLED_SPOOL')
'''


@unittest.skipUnless(IMAGE,'Opt-in non-root Linux runtime image is unset.')
class UploadTempLiveTests(unittest.TestCase):
    def run_probe(self,mount='rw,noexec,nosuid,nodev,size=256M,mode=0700,uid=65534,gid=65534',*,destination='/run/upload-tmp',root=False):
        args=['docker','run','--rm','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges:true','--memory=512m','--memory-swap=512m']
        if mount: args += ['--tmpfs','/run/upload-tmp:'+mount]
        if root: args += ['--user','0']
        for key in ('TMPDIR','TEMP','TMP'): args += ['-e',key+'='+destination]
        return subprocess.run([*args,IMAGE,'python','-c',PROBE],capture_output=True,text=True,timeout=40)

    def test_private_tmpfs_and_real_rolled_spool_close(self):
        result=self.run_probe()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PRIVATE_TMPFS_AND_CLOSED_ROLLED_SPOOL',result.stdout)

    def test_missing_mount_wrong_permissions_capacity_flags_and_root_refuse(self):
        cases=[('',False),('rw,noexec,nosuid,nodev,size=256M,mode=0777,uid=65534,gid=65534',False),('rw,noexec,nosuid,nodev,size=300M,mode=0700,uid=65534,gid=65534',False),('rw,size=256M,mode=0700,uid=65534,gid=65534',False),('rw,noexec,nosuid,nodev,size=256M,mode=0700,uid=65534,gid=65534',True)]
        for mount,root in cases:
            with self.subTest(mount=mount,root=root):
                result=self.run_probe(mount,root=root)
                self.assertEqual(result.returncode,3,result.stderr)
                self.assertEqual(result.stdout.strip(),'REFUSED')
        result=self.run_probe(destination='/tmp')
        self.assertEqual(result.returncode,3,result.stderr)


if __name__=='__main__': unittest.main()
