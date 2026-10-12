"""Fault injection at retention's durable restore boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import socket

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import retention
from brass_ai.step5.league import write_json


class RetentionRestoreAuditTests(unittest.TestCase):
    def test_contender_cannot_remove_newly_acquired_run_marker(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            run = Path(temp)
            checked, proceed = threading.Event(), threading.Event()
            real_active = retention.active
            first_thread = []
            def paused_active(path, allow_owner=False):
                result = real_active(path, allow_owner)
                if threading.get_ident() in first_thread and not checked.is_set():
                    checked.set()
                    if not proceed.wait(5):
                        raise TimeoutError('lock fixture')
                return result
            def acquire_first():
                first_thread.append(threading.get_ident())
                with retention.run_lock(run):
                    return True
            with patch.object(retention, 'active', side_effect=paused_active), ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(acquire_first)
                try:
                    self.assertTrue(checked.wait(5))
                    # The first contender has already acquired the OS guard,
                    # although it has not yet published its active-run marker.
                    try:
                        with retention.run_lock(run):
                            # On the old implementation the second caller can
                            # acquire here. Resume the first while this marker
                            # is still held: it must not steal the live marker.
                            marker = read(run / '.active-run.json')
                            proceed.set()
                            with self.assertRaisesRegex(ValueError, 'active'):
                                first.result(timeout=5)
                            self.assertEqual(read(run / '.active-run.json'), marker)
                    except ValueError as error:
                        self.assertIn('active', str(error))
                finally:
                    proceed.set()
                self.assertTrue(first.result(timeout=5))

    def test_stale_marker_recovery_and_exception_release(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            run = Path(temp)
            write_json(run / '.active-run.json', dict(pid=2147483647,
                host=socket.gethostname(), token='stale'))
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with retention.run_lock(run):
                    self.assertEqual(read(run / '.active-run.json')['pid'], os.getpid())
                    raise RuntimeError('fixture failure')
            self.assertFalse((run / '.active-run.json').exists())
            with retention.run_lock(run):
                self.assertTrue(retention.active(run))

    def fixture(self):
        temporary = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name)
        quarantine = run / '.retention-trash/fixture/old.pt'
        quarantine.parent.mkdir(parents=True)
        quarantine.write_bytes(b'recoverable checkpoint fixture')
        manifest = run / '.retention/cleanup-fixture.json'
        write_json(manifest, dict(run=str(run), created=0, quarantine_hours=0,
            generation=0, status='quarantined', entries=[dict(path='old.pt',
            quarantine=quarantine.relative_to(run).as_posix(),
            sha256=sha(quarantine), bytes=quarantine.stat().st_size, moved=True)]))
        return run, quarantine, manifest

    def test_restore_retries_after_move_before_receipt_commit(self):
        run, quarantine, manifest = self.fixture()
        def interrupted_write(path, data):
            if not quarantine.exists():
                raise OSError('simulated interruption after rename')
            write_json(path, data)
        with patch.object(retention, 'write_json', side_effect=interrupted_write):
            with self.assertRaisesRegex(OSError, 'interruption'):
                retention.restore(SOURCE_ROOT, manifest)
        self.assertEqual((run / 'old.pt').read_bytes(), b'recoverable checkpoint fixture')
        self.assertEqual(retention.restore(SOURCE_ROOT, manifest)['restored_files'], 1)
        self.assertTrue(read(manifest)['entries'][0]['restored'])
        self.assertEqual(retention.restore(SOURCE_ROOT, manifest)['restored_files'], 0)

    def test_interrupted_restore_rejects_changed_destination(self):
        run, quarantine, manifest = self.fixture()
        data = read(manifest)
        data['entries'][0]['restoring'] = True
        write_json(manifest, data)
        quarantine.replace(run / 'old.pt')
        (run / 'old.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            retention.restore(SOURCE_ROOT, manifest)
        self.assertFalse(read(manifest)['entries'][0].get('restored', False))

    def test_restore_never_overwrites_existing_destination(self):
        run, quarantine, manifest = self.fixture()
        (run / 'old.pt').write_bytes(quarantine.read_bytes())
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            retention.restore(SOURCE_ROOT, manifest)
        self.assertTrue(quarantine.exists())

    def test_dependency_scan_retries_retired_external_control_files(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            root = Path(temp)
            run = root / 'artifacts/run'
            run.mkdir(parents=True)
            checkpoint = run / 'old.pt'
            checkpoint.write_bytes(b'pinned')
            pins = root / 'artifacts/consumer/pins.json'
            write_json(pins, dict(paths=['artifacts/run/old.pt']))
            vanished = root / 'artifacts/retired/state.json'
            first = {'state.json': [vanished], 'spec.json': [], 'pins.json': [pins]}
            second = dict(first, **{'state.json': []})
            with patch.object(retention, 'controls', side_effect=[first, second]) as inventory:
                result = retention.references(root, run)
            self.assertIn(checkpoint, result)
            self.assertEqual(inventory.call_count, 2)
            self.assertEqual(checkpoint.read_bytes(), b'pinned')


if __name__ == '__main__':
    unittest.main()
