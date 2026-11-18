"""g3318 ownership stop: the era heads' own held-out owner-CE early stop (recovery/ownership_stop.py)."""
import copy
import unittest
from unittest import mock

import torch

from brass_ai.recovery import training_revision_g3140 as G
from brass_ai.recovery import local_scoring_v3 as v3
from brass_ai.recovery import local_scoring_auxiliary as auxiliary
from brass_ai.recovery import ownership_stop as S
from brass_ai.recovery import updates


class Recipe(unittest.TestCase):
    def test_item_needs_heads_persists_and_only_moves_forward(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from test_training_g3140 import g2880, signature
        base = g2880()
        heads = validate_config(G.configure(base, G.parse('time,validation,heads,targets'), 3275))
        self.assertIsNone(G.ownership_stop(heads))
        with self.assertRaisesRegex(ValueError, 'era heads'):
            G.configure(base, G.parse('ownership-stop'), 3318)
        stopped = validate_config(G.configure(heads, G.parse('ownership-stop'), 3318))
        self.assertEqual(G.ownership_stop(stopped), G.OWNERSHIP_STOP)
        self.assertEqual(stopped['local_scoring_auxiliary'], heads['local_scoring_auxiliary'])
        self.assertEqual({k for k in stopped if stopped[k] != heads[k]}, {G.KEY})
        self.assertEqual(G.configure(stopped, G.parse('ownership-stop,heads'), 3319), stopped)  # idempotent
        self.assertEqual(G.tag(stopped), G.tag(heads) + '-cestop16')
        self.assertTrue(G.changed(heads, stopped))
        self.assertTrue(signature_transition(signature(heads), signature(stopped)))
        self.assertFalse(signature_transition(signature(stopped), signature(heads)))
        later = validate_config(G.configure(stopped, G.parse('weight=0.15'), 3330))
        self.assertTrue(signature_transition(signature(stopped), signature(later)))
        self.assertEqual(G.ownership_stop(later), G.OWNERSHIP_STOP)
        odd = copy.deepcopy(stopped); odd[G.KEY]['ownership_stop']['interval_steps'] = 32
        with self.assertRaisesRegex(ValueError, 'Invalid g3140'):
            G.validate(odd[G.KEY])
        other = copy.deepcopy(heads); other['value_phase']['acceptance_steps'] = 32
        with self.assertRaisesRegex(ValueError, '16-step'):
            G.configure(other, G.parse('ownership-stop'), 3318)


class Stop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from test_gae_value_phase import fixture
        from test_local_scoring import annotate
        cls.model, _, cls.cfg, (_, data) = fixture(games=2); annotate(data)
        cls.ds = updates.Dataset(data, 'cpu'); cls.ids = list(range(len(data)))

    def companion(self, generation=5):
        return auxiliary.Companion(self.model, v3.default(generation), raw=dict(model_id='m'), generation=generation)

    def step(self, a, sign=1.):
        a.begin('ppg'); ix, x, _ = self.ds.batch(self.ids)
        self.model.zero_grad(set_to_none=True)
        loss = a.head_loss(self.model, x, self.ds, ix)
        (sign * loss.mean()).backward(); a.step('ppg')
        self.assertTrue(all(p.grad is None for p in self.model.parameters()), 'heads-only step')

    def test_head_loss_equals_the_joint_loss(self):
        a = self.companion(); ix, x, _ = self.ds.batch(self.ids)
        a.begin('ppg'); _, joint = a.forward(self.model, x, self.ds, ix)
        a.begin('ppg'); alone = a.head_loss(self.model, x, self.ds, ix)
        self.assertTrue(torch.allclose(joint, alone, atol=1e-6))

    def test_an_increase_keeps_zero_steps_and_ends_fitting(self):
        a = self.companion(); initial = copy.deepcopy(a.heads.state_dict())
        stop = S.OwnershipStop(a, G.OWNERSHIP_STOP, self.model, self.ds, self.ids, self.cfg['update'])
        self.assertEqual(stop.rows, len(self.ids)); self.assertTrue(stop.running)
        for _ in range(3): self.step(a, -1.)  # ascend the loss: held-out CE rises
        stop.boundary(3, 'restored')
        self.assertFalse(stop.running); self.assertTrue(a.stopped); self.assertEqual(stop.reason, 'heldout_ce_increased')
        for k, v in initial.items(): self.assertTrue(torch.equal(v, a.heads.state_dict()[k]), k)
        self.assertEqual((a.retained_steps['ppg'], a.fitting_steps, a.steps['ppg']), (0, 0, 0))
        self.assertEqual(stop.intervals[0]['kept_head_steps'], 0); self.assertEqual(stop.intervals[0]['head_steps'], 3)
        ix, x, _ = self.ds.batch(self.ids); a.begin('ppg')
        self.assertIsNone(a.forward(self.model, x, self.ds, ix)[1]); self.assertIsNone(a.head_loss(self.model, x, self.ds, ix))
        record = stop.finish()
        self.assertFalse(a.stopped)
        self.assertEqual((record['kept_head_steps'], record['attempted_head_steps']), (0, 3))
        self.assertEqual(record['after'], record['before'])
        report = a.report()
        self.assertFalse(report['head_steps_kept_on_rejection']); self.assertEqual(report['ownership_stop'], record)
        self.assertEqual(a.payload(6)['warmup']['fitting_steps'], 0)

    def test_decreases_are_kept_until_the_first_increase(self):
        a = self.companion()
        stop = S.OwnershipStop(a, G.OWNERSHIP_STOP, self.model, self.ds, self.ids, self.cfg['update'])
        for _ in range(3): self.step(a)
        stop.boundary(3, 'kept')
        self.assertTrue(stop.running); self.assertLess(stop.current['owner_ce'], stop.before['owner_ce'])
        kept = copy.deepcopy(a.heads.state_dict())
        for _ in range(10): self.step(a, -1.)  # Adam's momentum carries the first few downhill
        stop.boundary(13, 'kept')
        for k, v in kept.items(): self.assertTrue(torch.equal(v, a.heads.state_dict()[k]), k)
        record = stop.finish()
        self.assertEqual([i['kept_head_steps'] for i in record['intervals']], [3, 0])
        self.assertEqual((record['kept_head_steps'], a.retained_steps['ppg'], a.fitting_steps), (3, 3, 3))
        self.assertAlmostEqual(record['after']['owner_ce'], stop.current['owner_ce'], places=12)  # restored heads, same trunk
        self.assertEqual(a.payload(6)['warmup']['fitting_steps'], 3)


class ValuePhase(unittest.TestCase):
    """Through ppg_g210.fit: the value phase decides the trunk, the stop decides the heads."""

    def run_phase(self, patch_main=False, patch_ce=False):
        import tempfile
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery import ppg_g210, value_phase
        from brass_ai.recovery.actor_value import TrainingPhases
        from test_ppg_g210 import config, prepare
        torch.set_num_threads(2)
        m, opt, c = config()
        c['value_phase'] = dict(ppg_g210.KL_STOP_DEFAULT, batch_size=4, clone_rows=64, clone_batch_size=8)
        c[G.KEY] = dict(G.OFF, validation=copy.deepcopy(G.VALIDATION), ownership_stop=copy.deepcopy(G.OWNERSHIP_STOP))
        c['local_scoring_auxiliary'] = v3.default(1)
        original_metrics, original_measure = value_phase.metrics, S.OwnershipStop.measure
        calls = []

        def metrics(*args, **kwargs):
            result = original_metrics(*args, **kwargs)
            if kwargs.get('evidence') and 'group_cache' in kwargs:
                calls.append(1); result = dict(result, loss=result['loss'] + len(calls))  # every value check fails
            return result

        def measure(self):
            result = original_measure(self); calls.append(2)
            return dict(result, owner_ce=1. - .01 * len(calls))  # the heads always improve

        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp, \
                mock.patch.object(value_phase, 'metrics', metrics if patch_main else original_metrics), \
                mock.patch.object(S.OwnershipStop, 'measure', measure if patch_ce else original_measure):
            p = TrainingPhases(m, c, root=SOURCE_ROOT, run=Path(tmp))
            p.local_scoring = auxiliary.Companion(m, c['local_scoring_auxiliary'], raw=dict(model_id='m'), generation=1)
            prepare(m, p, 1, companion=True)
            trunk = copy.deepcopy(m.state_dict())
            r = p.fit(m)
            local = p.local_scoring.report()
            same = all(torch.equal(v, m.state_dict()[k]) for k, v in trunk.items())
        return r, local, same

    def check(self, r, local):
        stop = r['ownership_stop']
        self.assertEqual(local['ownership_stop'], stop)
        self.assertEqual(stop['kept_head_steps'], sum(i['kept_head_steps'] for i in stop['intervals']))
        self.assertEqual(stop['attempted_head_steps'], sum(i['head_steps'] for i in stop['intervals']))
        self.assertEqual(local['companion_steps']['ppg'], stop['kept_head_steps'])
        self.assertLessEqual(stop['attempted_head_steps'], r['planned_steps'])
        self.assertEqual(local['warmup']['retained_fitting_steps'], stop['kept_head_steps'])
        self.assertGreater(stop['rows'], 0); self.assertLess(stop['rows'], r['train_rows'])
        return stop

    def test_real_checks(self):
        r, local, _ = self.run_phase()
        stop = self.check(r, local)
        self.assertGreater(r['attempted_steps'], 0)
        self.assertEqual(stop['intervals'][0]['end_step'], r['interval_reports'][0]['steps'])

    def test_heads_fit_alone_after_the_value_stop_and_the_trunk_stays(self):
        r, local, same = self.run_phase(patch_main=True, patch_ce=True)
        stop = self.check(r, local)
        self.assertEqual(r['optimizer_steps'], 0); self.assertEqual(r['attempted_steps'], 16)
        self.assertTrue(same, 'heads-only steps must not move the trunk')
        self.assertGreater(r['planned_steps'], 32)  # the rest of epoch 1 and all of epoch 2 fit heads only
        self.assertEqual(stop['intervals'][0]['trunk'], 'restored')
        self.assertTrue(all(i['trunk'] == 'frozen' for i in stop['intervals'][1:]))
        self.assertEqual(stop['head_only_steps'], r['planned_steps'] - 16)
        self.assertEqual(stop['kept_head_steps'], r['planned_steps']); self.assertEqual(stop['stop_reason'], 'phase_steps_exhausted')


if __name__ == '__main__':
    unittest.main()
