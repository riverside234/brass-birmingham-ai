"""g3658: the collector over the iteration, size-only scheduled cleanup, the commit path and the value-phase gate."""
import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import collection_gc, league, retention
from brass_ai.recovery.config import RETENTION
from brass_ai.step5.league import write_json


class IterationGc(unittest.TestCase):
    def setUp(self):
        self.thresholds = gc.get_threshold()
        # CPython 3.12 keeps a few hundred interpreter objects in the permanent generation after a full pass.
        gc.unfreeze(); gc.collect(); self.settled = gc.get_freeze_count()

    def tearDown(self):
        collection_gc.settle(); gc.unfreeze(); gc.enable(); gc.set_threshold(*self.thresholds)

    @patch.dict(os.environ, {'BRASS_GC_FULL_EVERY': '1'})  # g5466: the full pass every iteration, as at g3658
    def test_the_heap_stays_frozen_through_the_iteration_and_one_full_pass_ends_it(self):
        window = collection_gc.Window(); window.start()
        self.assertFalse(gc.isenabled()); self.assertGreater(gc.get_freeze_count(), 0)
        for _ in range(100):
            cycle = []; cycle.append(cycle)
        del cycle
        report = window.stop()
        self.assertEqual((report['policy'], report['collections']), ('iteration', {}))
        # After collection: young passes only, the window's survivors frozen with the rest.
        self.assertTrue(gc.isenabled()); self.assertGreater(gc.get_freeze_count(), 0)
        self.assertEqual(gc.get_threshold()[:2], self.thresholds[:2])
        self.assertEqual(gc.get_threshold()[2], collection_gc.NO_FULL)
        gc.collect(0)
        result = window.finish()
        self.assertEqual(result['after_collection']['collections'], {'0': 1})
        self.assertGreaterEqual(result['full_collection']['unreachable'], 100)
        self.assertEqual(gc.get_freeze_count(), self.settled); self.assertEqual(gc.get_threshold(), self.thresholds)
        self.assertNotIn(window._callback, gc.callbacks); self.assertIsNone(window.finish())

    @patch.dict(os.environ, {'BRASS_GC_FULL_EVERY': '1'})
    def test_an_unfinished_iteration_is_settled_by_the_next_window(self):
        first = collection_gc.Window(); first.start(); first.stop()
        second = collection_gc.Window(); second.start()
        self.assertNotIn(first._callback, gc.callbacks); self.assertFalse(first.tail)
        self.assertFalse(gc.isenabled())
        second.stop(); second.finish()
        self.assertTrue(gc.isenabled()); self.assertEqual(gc.get_freeze_count(), self.settled)
        self.assertEqual(gc.get_threshold(), self.thresholds)

    def test_the_switches_keep_the_g3270_window_or_no_freeze(self):
        for switches, freeze in (({'BRASS_ITERATION_GC': '0'}, True), ({'BRASS_COLLECTION_GC_FREEZE': '0'}, False)):
            with patch.dict(os.environ, switches):
                window = collection_gc.Window(); window.start()
                self.assertTrue(gc.isenabled())
                report = window.stop()
            self.assertEqual((report['policy'], report['freeze']), ('window', freeze))
            self.assertEqual(gc.get_freeze_count(), 0); self.assertIsNone(window.finish())
        with patch.dict(os.environ, {'BRASS_ITERATION_GC': '2'}), self.assertRaises(ValueError):
            collection_gc.Window().start()


def fixture(case, count):
    t = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts'); case.addCleanup(t.cleanup)
    root = Path(t.name); run = root / 'artifacts/run'; run.mkdir(parents=True)
    (run / 'keep.pt').write_bytes(b'live')
    write_json(run / 'state.json', dict(best='artifacts/run/keep.pt', last='artifacts/run/keep.pt', epoch=10, iteration=10))
    for i in range(count): (run / f'epoch-{i:04}.pt').write_bytes((str(i) * 4096).encode())
    return root, run, dict(RETENTION, quarantine_hours=0)


class ScheduledCleanup(unittest.TestCase):
    def test_sizes_only_and_one_reference_scan(self):
        root, run, cfg = fixture(self, 8)
        scans = []; references = retention.references
        def counted(*args):
            scans.append(args); return references(*args)
        with patch.object(retention, 'references', counted), patch.object(retention, 'sha', side_effect=AssertionError('hashed')):
            with retention.BackgroundPurge() as background:
                result = retention.scheduled_prune(root, run, cfg, background=background)
                purged = background.join()
        self.assertEqual(len(scans), 1)
        self.assertEqual((result['verification'], result['hash_reuse']['final_delete_rehash']), ('size', False))
        self.assertIsNotNone(result['timings']['references_seconds'])
        self.assertEqual(purged['permanently_deleted_files'], 8)
        manifest = read(next((run / '.retention').glob('cleanup-*.json')))
        self.assertTrue(retention.settled(manifest)); self.assertFalse(any('sha256' in e for e in manifest['entries']))
        self.assertEqual((run / 'keep.pt').read_bytes(), b'live'); self.assertFalse(list(run.glob('epoch-*.pt')))

    def test_a_changed_size_stops_the_purge_and_a_size_only_entry_restores(self):
        root, run, cfg = fixture(self, 4)
        result = retention.prune(root, run, cfg, apply=True, defer_purge=True, automatic=True)
        moved = sorted((run / '.retention-trash').rglob('epoch-*.pt')); original = moved[0].read_bytes()
        moved[0].write_bytes(original + b'!')
        with self.assertRaisesRegex(ValueError, 'Quarantine changed'):
            retention.purge(root, run, cfg, automatic=True, known=result['known_pins'])
        moved[0].write_bytes(original)
        manifest = Path(result['manifest'])
        self.assertEqual(retention._restore(run, manifest, read(manifest))['restored_files'], 4)
        self.assertEqual(len(list(run.glob('epoch-*.pt'))), 4)

    def test_manual_cleanup_keeps_its_hashes(self):
        root, run, cfg = fixture(self, 3)
        result = retention.prune(root, run, cfg, apply=True)
        self.assertTrue(result['hash_reuse']['final_delete_rehash']); self.assertIsNone(result['known_pins'])
        self.assertTrue(all(e.get('sha256') for e in read(Path(result['manifest']))['entries']))


class Commit(unittest.TestCase):
    def test_a_cached_population_member_keeps_its_receipt_unhashed(self):
        t = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts'); self.addCleanup(t.cleanup)
        root = Path(t.name).resolve(); run = root / 'artifacts/run'; (run / 'population').mkdir(parents=True)
        weights = run / 'population/weights-x.pt'; weights.write_bytes(b'weights')
        entry = dict(path=weights.relative_to(root).as_posix(), sha256=sha(weights), model_id='x', generation=3, weights_only=True)
        with patch.object(retention, 'sha', side_effect=AssertionError('rehashed')):
            self.assertEqual(retention.slim_checkpoint(root, run, dict(entry), dict(entry)), entry)
            with self.assertRaisesRegex(ValueError, 'receipt changed'):
                retention.slim_checkpoint(root, run, dict(entry, sha256='0' * 64), dict(entry))
        weights.unlink()
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            retention.slim_checkpoint(root, run, dict(entry), dict(entry))

    def test_commit_cleanups_hands_over_the_written_state_and_writes_compact_json(self):
        class Owner:
            committed_state = True
            def committed(self, generation, *, state=None): self.seen = (generation, state)
        class Legacy:
            def committed(self, generation): self.seen = generation
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'state.json'; state = dict(iteration=4, nested=dict(a=[1, 2]))
            owner, legacy = Owner(), Legacy()
            league.commit_cleanups(path, state, 4, owners=(owner, None, legacy))
            self.assertIs(owner.seen[1], state); self.assertEqual((owner.seen[0], legacy.seen), (4, 4))
            self.assertEqual(read(path), state); self.assertEqual(path.read_text(), '{"iteration":4,"nested":{"a":[1,2]}}\n')

    def test_the_committed_state_satisfies_both_cache_owners(self):
        from brass_ai.recovery.teacher_phase import TeacherPhase
        from brass_ai.recovery.actor_value import TrainingPhases
        self.assertTrue(TeacherPhase.committed_state and TrainingPhases.committed_state)
        from brass_ai.recovery import teacher_buffer
        buffer = teacher_buffer.TeacherBuffer.__new__(teacher_buffer.TeacherBuffer)
        buffer.run = Path('unused'); buffer.retired = []; buffer.pending = [dict(generation=4)]
        buffer.payload = lambda: dict(receipts=[dict(generation=4)])
        with patch.object(teacher_buffer, 'read', side_effect=AssertionError('read state.json')):
            buffer.committed(4, state=dict(iteration=4, teacher_phase=dict(buffer=dict(receipts=[dict(generation=4)]))))
        self.assertEqual(buffer.receipts, [dict(generation=4)])


class ValueGate(unittest.TestCase):
    def test_the_gate_reuses_the_acceptance_scans_with_the_same_numbers(self):
        from brass_ai.recovery import value_phase
        from test_search_rows_g3471 import Phase
        fit = value_phase.fit
        def descriptors(*args, **kwargs):
            return fit(*args, **dict(kwargs, descriptor_diagnostics=True))
        def separate(*args, **kwargs):
            return fit(*args, **dict(kwargs, gate_from_acceptance=False, descriptor_diagnostics=True))
        def no_controller(*args, **kwargs):
            return fit(*args, **dict(kwargs, descriptor_diagnostics=False))
        phase = Phase()
        with patch.object(value_phase, 'fit', descriptors):
            reused = phase.run_phase()
        self.assertEqual(reused['gate_measurement'], 'acceptance')
        self.assertEqual(reused['heldout_before'], reused['acceptance_heldout_before'])
        self.assertEqual(reused['heldout_after'], reused['acceptance_heldout_after'])
        self.assertNotIn('current_validation_seconds', reused['timings'])
        self.assertIn('descriptor_diagnostics_seconds', reused['timings']); self.assertIsNotNone(reused['heldout_unknown'])
        with patch.object(value_phase, 'fit', separate):
            measured = phase.run_phase()
        with patch.object(value_phase, 'fit', no_controller):
            skipped = phase.run_phase()
        self.assertEqual(measured['gate_measurement'], 'separate'); self.assertIn('current_validation_seconds', measured['timings'])
        for key in ('heldout_before', 'heldout_after', 'acceptance_heldout_after', 'optimizer_steps', 'attempted_value_presentations',
                    'heldout_unknown', 'descriptor_gain'):
            self.assertEqual(reused[key], measured[key], key)
        self.assertFalse(skipped['descriptor_gain']['measured']); self.assertIsNone(skipped['heldout_unknown'])
        self.assertNotIn('descriptor_diagnostics_seconds', skipped['timings'])
        for key in ('heldout_before', 'heldout_after', 'optimizer_steps'):
            self.assertEqual(reused[key], skipped[key], key)

    def test_only_the_automatic_rollout_before_snapshots_reads_the_descriptor_gain(self):
        from brass_ai.recovery import critic_data as C
        waiting = dict(snapshots_active=False)
        self.assertTrue(C.reads_descriptor_gain(waiting, C.DEFAULT))
        self.assertFalse(C.reads_descriptor_gain(dict(snapshots_active=True), C.DEFAULT))
        self.assertFalse(C.reads_descriptor_gain(waiting, C.FIXED_DEFAULT))  # the lab recipe (critic-rows/3.0.0)
        self.assertFalse(C.reads_descriptor_gain(waiting, dict(C.DEFAULT, snapshot_rows=False)))
        self.assertFalse(C.reads_descriptor_gain(None, None))


if __name__ == '__main__':
    unittest.main()
