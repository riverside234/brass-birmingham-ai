"""g5466 iteration speed: the cleanup's pass resolver, the GC cycles broken at their source and the full pass every
fifth iteration, the commit-time teacher check limited to retiring ages, and the rating slot without bootstrap
refits or full replay checks."""
import collections
import gc
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import collection_gc


class Resolver(unittest.TestCase):
    def outcome(self, call):
        try: return 'path', call()
        except ValueError as error: return 'refused', str(error)

    def test_the_pass_resolver_returns_and_refuses_exactly_as_project_path(self):
        from brass_ai.learning.paths import PassResolver, resolve_project_path
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            root = Path(tmp).resolve()
            (root / 'artifacts/run/iteration-1').mkdir(parents=True)
            (root / 'artifacts/run/state.json').write_text('{}')
            (root / 'artifacts/run/iteration-1/report.json').write_text('{}')
            (root / 'file.txt').write_text('x')
            linked = root / 'artifacts/linked'
            try: os.symlink(root / 'artifacts/run', linked, target_is_directory=True)
            except OSError: linked = None  # no symlink privilege on this host
            references = [('artifacts/run/state.json', None), ('report.json', 'artifacts/run/iteration-1'),
                          ('teacher-labels/x.jsonl.gz', 'artifacts/run'), ('artifacts/missing/deep/x.pt', None),
                          ('teacher-labels/x.jsonl.gz', root / 'artifacts/run'),
                          (str(root / 'artifacts/run/state.json'), None), (str(root / 'artifacts/run/state.json'), 'artifacts/run'),
                          ('file.txt/child.json', None), ('../escape.json', None), ('artifacts/../x.json', None),
                          ('/elsewhere/x.json', None), ('C:/elsewhere/y.json', None), ('a/b. /c.json', None), ('', None)]
            if linked is not None: references += [('artifacts/linked/state.json', None), ('state.json', 'artifacts/linked')]
            resolver = PassResolver(root)
            for _ in range(2):  # a second pass over the same references hits the directory cache
                for reference, base in references:
                    self.assertEqual(self.outcome(lambda: resolver.resolve(reference, base=base)),
                                     self.outcome(lambda: resolve_project_path(root, reference, base=base)), reference)

    def test_the_pin_scan_finds_the_same_pins(self):
        """A run's own state and another run's state pin files in the run; the scan sees both."""
        from brass_ai.recovery import retention
        from brass_ai.step5.league import write_json
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            root = Path(tmp).resolve(); run = root / 'artifacts/step5/run'; other = root / 'artifacts/step5/other'
            for path in (run / 'population', run / 'teacher-labels', other): path.mkdir(parents=True)
            for name in ('generation-0002.pt', 'population/weights-a.pt', 'teacher-labels/generation-000002-train.jsonl.gz'):
                (run / name).write_bytes(b'x')
            write_json(run / 'manifest.json', dict(weights='artifacts/step5/run/population/weights-a.pt'))
            write_json(run / 'state.json', dict(format='step5-recovery-league/2.2.0', resume=[
                dict(path='artifacts/step5/run/generation-0002.pt')], teacher_phase=dict(buffer=dict(receipts=[
                    dict(path='teacher-labels/generation-000002-train.jsonl.gz')]))))
            write_json(other / 'state.json', dict(format='step5-recovery-league/2.2.0',
                                                  origin=dict(manifest='artifacts/step5/run/manifest.json')))
            pins = retention._references(root, run)
            expected = {run / 'generation-0002.pt', run / 'population/weights-a.pt', run / 'manifest.json',
                        run / 'teacher-labels/generation-000002-train.jsonl.gz'}
            self.assertTrue(expected <= set(pins), sorted(map(str, pins)))
            self.assertIn('active state', pins[run / 'generation-0002.pt'])
            self.assertIn('referenced manifest', pins[run / 'population/weights-a.pt'])


class IterationGc(unittest.TestCase):
    def setUp(self):
        self.thresholds = gc.get_threshold(); gc.unfreeze(); gc.collect(); self.settled = gc.get_freeze_count()

    def tearDown(self):
        collection_gc.settle(); gc.unfreeze(); gc.enable(); gc.set_threshold(*self.thresholds)

    def iteration(self, window, cycles):
        window.start(); cycle = None
        for _ in range(cycles):
            cycle = []; cycle.append(cycle)
        del cycle
        window.stop()
        return window.finish()['full_collection']

    @patch.dict(os.environ, {'BRASS_GC_FULL_EVERY': '3'})
    def test_the_full_pass_runs_every_third_iteration_and_the_heap_stays_frozen_between(self):
        window = collection_gc.Window()
        first = self.iteration(window, 100)
        self.assertEqual((first['ran'], first['seconds'], first['iterations_since_full'], first['next_full_in']), (False, 0., 1, 2))
        # The policy stays in force until the next collection start: frozen heap, young passes only.
        self.assertGreater(gc.get_freeze_count(), self.settled); self.assertEqual(gc.get_threshold()[2], collection_gc.NO_FULL)
        self.assertNotIn(window._callback, gc.callbacks)
        second = self.iteration(window, 100)
        self.assertEqual((second['ran'], second['next_full_in']), (False, 1))
        third = self.iteration(window, 100)
        self.assertTrue(third['ran']); self.assertEqual((third['iterations'], third['next_full_in']), (3, 3))
        self.assertGreaterEqual(third['unreachable'], 300)  # all three iterations' cycles, freed together
        self.assertEqual(gc.get_freeze_count(), self.settled); self.assertEqual(gc.get_threshold(), self.thresholds)

    @patch.dict(os.environ, {'BRASS_GC_FULL_EVERY': '2'})
    def test_high_cyclic_garbage_brings_the_full_pass_back_every_iteration(self):
        window = collection_gc.Window()
        with patch.object(collection_gc, 'GARBAGE_PER_ITERATION', 50):
            self.iteration(window, 200); full = self.iteration(window, 200)
            self.assertTrue(full['ran'] and full['unreachable_per_iteration'] > 50); self.assertEqual(full['next_full_in'], 1)
            again = self.iteration(window, 200); self.assertTrue(again['ran'])
            quiet = self.iteration(window, 0); self.assertTrue(quiet['ran']); self.assertEqual(quiet['next_full_in'], 2)
            self.assertFalse(self.iteration(window, 0)['ran'])
        with patch.dict(os.environ, {'BRASS_GC_FULL_EVERY': '0'}), self.assertRaisesRegex(ValueError, 'integer from 1 to 20'):
            collection_gc.full_every()


class Cycles(unittest.TestCase):
    def test_an_input_audit_and_its_proofs_free_without_a_full_pass(self):
        from brass_ai.recovery.batch_cache import InputAudit
        class Row: pass
        gc.collect(); gc.set_debug(gc.DEBUG_SAVEALL); gc.garbage.clear()
        try:
            audit = InputAudit(); rows = [Row() for _ in range(50)]
            for row in rows:
                key = id(row); reference = __import__('weakref').ref(row, __import__('functools').partial(
                    __import__('brass_ai.recovery.batch_cache', fromlist=['x'])._forget, audit._self, key))
                audit.proofs[key] = (reference, b'seal')
            del rows[:25]  # dead rows drop their proofs through the callback
            self.assertEqual(len(audit.proofs), 25)
            del audit, rows, reference
            gc.collect(); self.assertEqual([o for o in gc.garbage if type(o).__name__ in ('InputAudit', 'partial')], [])
        finally:
            gc.set_debug(0); gc.garbage.clear()

    def test_the_ppg_phase_leaves_no_search_rows_or_audits_for_the_full_pass(self):
        import test_era_head_revision_g4770 as g4770_tests
        g4770_tests.Phase.run_phase(None)  # imports and one-time caches
        gc.collect(); gc.set_debug(gc.DEBUG_SAVEALL); gc.garbage.clear()
        try:
            g4770_tests.Phase.run_phase(None)
            gc.collect(); kinds = collections.Counter(type(o).__name__ for o in gc.garbage)
        finally:
            gc.set_debug(0); gc.garbage.clear()
        self.assertFalse({'SearchRows', 'Corrections', 'InputAudit', 'Tensor'} & set(kinds), kinds)
        self.assertLess(sum(kinds.values()), 500, kinds)  # what remains: the standard library's JSON encoder closures


class TeacherCommit(unittest.TestCase):
    def test_only_retiring_ages_are_checked_at_the_commit(self):
        from brass_ai.recovery import teacher_buffer
        buffer = teacher_buffer.TeacherBuffer.__new__(teacher_buffer.TeacherBuffer)
        retired = [dict(sha256=f'{i:064x}', generation=4970 + i, split='train', path='teacher-labels/x') for i in range(975)]
        buffer.retired, buffer.pending, buffer._rows = retired, [], {retired[-1]['sha256']: ('row',)}
        buffer.payload = lambda: 'committed'
        checked = []
        buffer.paths = lambda receipt: (checked.append(receipt['generation']) or Path('missing-a'), Path('missing-b'))
        buffer.committed(5466, state=dict(iteration=5466, teacher_phase=dict(buffer='committed')))
        self.assertEqual(checked, [4970 + 974])  # one stat of the retiring age, not 975 hashes


class RatingSlot(unittest.TestCase):
    def test_without_bootstrap_the_fit_keeps_its_points_and_reports_no_intervals(self):
        from brass_ai.recovery import plackett_luce as pl
        from test_strength_rating import AGENTS, RATING, simulated
        games = simulated(setups=40); config = dict(RATING, bootstrap=20)
        with_intervals, samples = pl.estimate(games, AGENTS, config)
        without, empty = pl.estimate(games, AGENTS, config, bootstrap=False)
        self.assertEqual({a: r['points'] for a, r in without['ratings'].items()},
                         {a: r['points'] for a, r in with_intervals['ratings'].items()})
        self.assertTrue(all(r['points_ci95'] is None and r['theta_ci95'] is None for r in without['ratings'].values()))
        self.assertEqual((without['bootstrap_refits'], without['differences'], empty.shape), (0, [], (0, len(AGENTS))))
        self.assertEqual(samples.shape[0], 20)

    def test_alerts_and_the_plot_read_a_fit_without_intervals(self):
        from brass_ai.recovery import plackett_luce as pl, rating_plot, ratings as r
        state = dict(agents={a: dict(checkpoint=dict(generation=int(a.split('-')[1]))) for a in ('generation-000010', 'generation-000020')})
        report = dict(ratings={pl.IL: dict(points=0, points_ci95=None), pl.V4: dict(points=20, points_ci95=None),
                               'generation-000010': dict(points=100, points_ci95=None),
                               'generation-000020': dict(points=20, points_ci95=None)})
        panel = lambda win: dict(panels={'v4-panel': dict(learner_win_share=win)})
        result = r.alerts(report, state, panel(.1), [panel(.3), panel(.2)])
        self.assertTrue(result['regression']); self.assertIsNone(result['highest_lower_bound'])
        import matplotlib; matplotlib.use('Agg')
        from test_rating_plot import fixture, labels
        report, agents = fixture()
        for row in report['ratings'].values(): row['points_ci95'] = None
        report.update(differences=[], bootstrap_refits=0)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            path = Path(tmp) / 'progress.svg'; rating_plot.render(report, path)
            text = labels(path)
        self.assertIn('No interval (bootstrap off)', text); self.assertIn('No paired interval (bootstrap off)', text)
        self.assertIn('+80.0', text)

    def test_rating_replays_are_read_for_their_outcomes(self):
        from brass_ai.recovery import il_replay_storage, rating_ingress, ratings
        game = dict(game_id='g1', setup_group='s1', players=4, split='validation', opponents=['a', 'b', 'c', 'd'],
                    result=dict(placement=[1, 2, 3, 4]), records=[dict(view=dict(history=0))])
        with tempfile.TemporaryDirectory() as tmp:
            compact = Path(tmp) / 'compact.jsonl.gz'; legacy = Path(tmp) / 'legacy.jsonl.gz'; native = Path(tmp) / 'native.jsonl.gz'
            for path, header, line in ((compact, il_replay_storage.FORMAT, dict(history=[], game=game)),
                                       (legacy, il_replay_storage.LEGACY_FORMAT, game),
                                       (native, 'four-seat-native-replay/1.0.0', game)):
                with gzip.open(path, 'wt', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(format=header, compatibility={}, provenance={})) + '\n')
                    stream.write(json.dumps(line) + '\n')
            expected = [dict(game_id='g1', setup_group='s1', opponents=['a', 'b', 'c', 'd'], result=dict(placement=[1, 2, 3, 4]))]
            self.assertEqual(rating_ingress.outcomes(compact), expected)
            self.assertEqual(rating_ingress.outcomes(legacy), expected)
            self.assertIsNone(rating_ingress.outcomes(native))  # the full read handles native shards
        self.assertFalse(ratings.rating_replay_checks() or ratings.rating_bootstrap())
        with patch.dict(os.environ, {'BRASS_RATING_BOOTSTRAP': '1', 'BRASS_RATING_REPLAY_CHECKS': '1'}):
            self.assertTrue(ratings.rating_replay_checks() and ratings.rating_bootstrap())


if __name__ == '__main__':
    unittest.main()
