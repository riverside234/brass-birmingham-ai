"""Bounded regression checks for lab prerequisites, receipt reuse and cleanup."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


toolchain = load('rust_lab_toolchain.py')
b01 = load('qualify-rust-lab.py')
b02 = load('qualify-rust-lab-b02.py')


class Prerequisites(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_sealed_compiler_restores_without_developer_cache(self):
        compiler = toolchain.restore(self.root)
        self.assertTrue(compiler.is_file())
        with zipfile.ZipFile(ROOT/toolchain.PACKAGE) as archive:
            for item in archive.infolist():
                if item.filename.startswith(toolchain.PREFIX) and not item.is_dir():
                    self.assertEqual((compiler.parent/item.filename[len(toolchain.PREFIX):]).read_bytes(), archive.read(item))
        with self.assertRaisesRegex(ValueError, 'fresh qualification'):
            toolchain.restore(self.root)

    def test_missing_or_corrupt_compiler_never_creates_output(self):
        with self.assertRaisesRegex(ValueError, 'pinned Roslyn'):
            toolchain.restore(self.root/'out', root=self.root)
        package = self.root/toolchain.PACKAGE
        package.parent.mkdir(parents=True); package.write_bytes(b'wrong compiler')
        with self.assertRaisesRegex(ValueError, 'pinned Roslyn'):
            toolchain.restore(self.root/'out', root=self.root)
        self.assertFalse((self.root/'out').exists())

    def test_restore_rejects_traversal_duplicate_and_symlink_members(self):
        package = self.root/toolchain.PACKAGE; package.parent.mkdir(parents=True)
        for bad in ('tasks/net472/../../outside', 'tasks/net472/sub/../../outside',
                    'tasks/net472/C:/outside', 'tasks/net472/sub\\outside', 'link', 'duplicate'):
            with self.subTest(bad=bad):
                with zipfile.ZipFile(package, 'w') as archive:
                    archive.writestr(toolchain.PREFIX+'csc.exe', b'fixture')
                    if bad == 'link':
                        item = zipfile.ZipInfo(toolchain.PREFIX+'link')
                        item.create_system = 3; item.external_attr = 0o120777 << 16
                        archive.writestr(item, b'../outside')
                    elif bad == 'duplicate':
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', UserWarning)
                            archive.writestr(toolchain.PREFIX+'csc.exe', b'duplicate')
                    else:
                        item = zipfile.ZipInfo(bad)
                        # ZipInfo normalizes backslashes on Windows at creation.
                        item.filename = bad
                        archive.writestr(item, b'escape')
                with patch.object(toolchain, 'SHA256', hashlib.sha256(package.read_bytes()).hexdigest()):
                    with self.assertRaises(ValueError): toolchain.restore(self.root/'out', root=self.root)
                self.assertFalse((self.root/'out').exists())

    def test_b01_reuse_allows_harness_repair_but_rejects_changed_execution_inputs(self):
        def row(path, sha='a'): return dict(path=path, sha256=sha*64)
        production = [row('src/brass_ai/recovery/inference.py'), row('configs/features.json'),
                      row('training/runtime/throughput/ordinary/BrassHost.exe'),
                      row('scripts/build-local-scoring-host.py')]
        recorded = production + [row('scripts/qualify-rust-lab-b02.py')]
        inventory = production + [row('scripts/qualify-rust-lab-b02.py', 'b')]
        with patch.object(b02, 'b01_tools') as base:
            base.return_value.source_inventory.return_value = inventory
            b02.verify_b01_sources(recorded)
            for changed in ([], production[:-1], production+[row('src/brass_ai/new.py')],
                            [row(production[0]['path'], 'b')]+production[1:]):
                with self.subTest(changed=changed):
                    base.return_value.source_inventory.return_value = changed
                    with self.assertRaisesRegex(ValueError, 'rerun B01'): b02.verify_b01_sources(recorded)
            base.return_value.source_inventory.return_value = production
            with self.assertRaisesRegex(ValueError, 'rerun B01'): b02.verify_b01_sources([])

    def test_bounded_command_timeout_reaps_child_and_retains_output(self):
        pidfile = self.root/'pid.txt'
        code = ('import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); '
                'print("started",flush=True); time.sleep(30)')
        with self.assertRaises(subprocess.TimeoutExpired):
            b01.command([sys.executable, '-c', code, pidfile], cwd=ROOT,
                        env=b01.python_environment(), log=self.root/'timeout.log', timeout=1)
        self.assertEqual((self.root/'timeout.log').read_text().strip(), 'started')
        pid = int(pidfile.read_text())
        if os.name == 'nt':
            # A terminated process has no live Windows process handle.
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                status = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(status))
                ctypes.windll.kernel32.CloseHandle(handle)
                self.assertNotEqual(status.value, 259)
        else:
            with self.assertRaises(ProcessLookupError): os.kill(pid, 0)

    def test_child_exit_failure_preserves_log_and_error(self):
        with self.assertRaisesRegex(ValueError, 'Qualification command failed'):
            b01.command([sys.executable, '-c', 'print("reason"); raise SystemExit(7)'], cwd=ROOT,
                        env=b01.python_environment(), log=self.root/'failed.log', timeout=10)
        self.assertIn('reason', (self.root/'failed.log').read_text())


if __name__ == '__main__': unittest.main()
