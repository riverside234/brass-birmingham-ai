"""The distributed recorder must survive a normal Git transfer to the lab."""
import subprocess
import unittest

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT


class Package(unittest.TestCase):
    def test_manifest_has_matching_distributable_executable(self):
        directory=SOURCE_ROOT/'training/runtime/local-scoring/ordinary'
        manifest=read(directory/'runtime.json')
        executable=directory/'BrassHost.exe'
        expected=next(f['sha256'] for f in manifest['files'] if f['name']=='BrassHost.exe')
        self.assertTrue(executable.is_file(),'The lab package must include BrassHost.exe, not just runtime.json')
        self.assertEqual(sha(executable),expected)
        # The private licensed dependencies remain in the installed baseline;
        # only the authored executable and its receipt may be distributed.
        self.assertEqual({p.name for p in directory.iterdir()},{'BrassHost.exe','runtime.json'})

    def test_git_keeps_executable_and_preserves_its_bytes(self):
        path='training/runtime/local-scoring/ordinary/BrassHost.exe'
        result=subprocess.run(['git','check-ignore','--no-index','--quiet',path],cwd=SOURCE_ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,1,'Git would omit the recorder: '+result.stderr)
        attrs=subprocess.run(['git','check-attr','text','--',path],cwd=SOURCE_ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(attrs.stdout.strip(),path+': text: unset')


if __name__=='__main__':unittest.main()
