"""Exercise the real launcher and spawned children without running training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/step5.py'

# A lightweight CLI stand-in records the inherited directory, removes the
# now-unused launch folder, and starts a real spawn child as evaluation does.
PROBE = '''
import json
import os
from pathlib import Path
import sys

def child(path):
    Path(path).write_text(json.dumps(dict(cwd=os.getcwd())), encoding='utf-8')

def main():
    import multiprocessing
    from multiprocessing import spawn
    launch=Path(os.environ['STEP5_TEST_LAUNCH'])
    if launch.exists(): launch.rmdir()
    output=Path(os.environ['STEP5_TEST_OUTPUT'])
    worker=multiprocessing.get_context('spawn').Process(target=child,args=(str(output.with_suffix('.child.json')),))
    worker.start();worker.join(20)
    if worker.is_alive(): worker.terminate();worker.join();raise RuntimeError('Spawn child did not finish')
    if worker.exitcode:raise RuntimeError('Spawn child failed')
    output.write_text(json.dumps(dict(cwd=os.getcwd(),original_dir=spawn.process.ORIGINAL_DIR,
        argv=sys.argv[1:],launch_removed=not launch.exists())),encoding='utf-8')
'''


class Step5WorkingDirectoryTests(unittest.TestCase):
    def invoke(self, mode):
        with tempfile.TemporaryDirectory(prefix='step5-cwd-',dir=ROOT/'artifacts') as temp:
            base=Path(temp).resolve();launch=base/'retired-iteration';launch.mkdir()
            (base/'cwd_probe.py').write_text(textwrap.dedent(PROBE),encoding='utf-8')
            output=base/'result.json'
            env=dict(os.environ,STEP5_TEST_LAUNCH=str(launch),STEP5_TEST_OUTPUT=str(output))
            shim=base/'invoke.py'
            shim.write_text(textwrap.dedent('''
                import os, runpy, sys, types
                from pathlib import Path
                import cwd_probe
                script,mode=sys.argv[1:3]
                if mode=='missing':
                    # Simulate Linux getcwd failure until the launcher chdirs;
                    # Windows refuses to unlink a live current directory.
                    real_getcwd,real_chdir=os.getcwd,os.chdir
                    missing=[True]
                    def getcwd():
                        if missing[0]:raise FileNotFoundError(2,'No such file or directory')
                        return real_getcwd()
                    def chdir(path):
                        real_chdir(path);missing[0]=False
                    os.getcwd,os.chdir=getcwd,chdir
                elif mode=='unlinked':
                    os.rmdir(os.environ['STEP5_TEST_LAUNCH'])
                    try:os.getcwd()
                    except FileNotFoundError:pass
                    else:raise AssertionError('Expected an actually unlinked cwd')
                module=types.ModuleType('brass_ai.step5.cli');module.main=cwd_probe.main
                sys.modules[module.__name__]=module
                sys.argv=[script,'--device','cuda','train','--run','artifacts/step5/selfplay-v2','--iterations','1']
                runpy.run_path(script,run_name='__main__')
            '''),encoding='utf-8')
            # A Windows venv redirector keeps its own launch-directory handle
            # after the Python child chdirs. This stdlib-only probe can use the
            # base interpreter directly, so retiring cwd tests the launcher.
            executable=sys._base_executable if os.name=='nt' else sys.executable
            result=subprocess.run([executable,str(shim),str(SCRIPT),mode],cwd=launch,env=env,
                                  capture_output=True,text=True,timeout=40)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads(output.read_text());child=json.loads(output.with_suffix('.child.json').read_text())
            for location in (report['cwd'],report['original_dir'],child['cwd']):
                self.assertEqual(Path(location).resolve(),ROOT)
            self.assertTrue(report['launch_removed'])
            self.assertEqual(report['argv'],['--device','cuda','train','--run','artifacts/step5/selfplay-v2','--iterations','1'])

    def test_launch_folder_can_be_retired_before_evaluation_spawn(self):
        self.invoke('valid')

    def test_missing_current_directory_is_repaired_before_imports_and_spawn(self):
        self.invoke('missing')

    @unittest.skipUnless(os.name=='posix','Unlinking an active cwd requires POSIX')
    def test_real_unlinked_current_directory(self):
        self.invoke('unlinked')


if __name__=='__main__':unittest.main()
