"""g3140 value-phase revision: recipe ladder, era heads, lambda-returns, staged teacher rows, background purge."""
import copy
import math
import types
import unittest

import numpy as np
import torch

from brass_ai.recovery import training_revision_g3140 as G
from brass_ai.recovery import local_scoring_v3 as v3
from brass_ai.recovery import local_scoring_auxiliary as auxiliary
from brass_ai.recovery import updates


_BASE = None


def g2880():
    global _BASE
    if _BASE is None:
        from test_training_g2880 import g2654
        from brass_ai.recovery import training_revision_g2880 as lam
        _BASE = lam.configure(g2654())
    return copy.deepcopy(_BASE)


def signature(config):
    from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
    return dict(config=config, implementation=QUALIFIED)


class Recipe(unittest.TestCase):
    def test_items_validate_persist_and_only_move_forward(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        before = g2880()
        self.assertFalse(G.applied(before))
        first = validate_config(G.configure(before, G.parse('time,validation'), 3140))
        self.assertEqual(G.block(first)['train_metrics_every'], 10)
        self.assertEqual(G.validation_fraction(first), .25)
        self.assertIsNone(G.bootstrap(first)); self.assertEqual(G.value_weights(first), (.25, .1))
        self.assertNotIn('pause_search_value', G.block(first))
        self.assertEqual(first['local_scoring_auxiliary'], before['local_scoring_auxiliary'])
        second = validate_config(G.configure(first, G.parse('heads,targets'), 3150))
        local = second['local_scoring_auxiliary']
        self.assertTrue(v3.is_v3(local)); self.assertEqual(local['replacement']['generation'], 3150)
        self.assertEqual(local['weights'], dict(canal=.05, rail=.05, rail_end=0.))
        third = validate_config(G.configure(second, G.parse('weight=0.15,vp,rail-end'), 3160))
        self.assertEqual(third['local_scoring_auxiliary']['weights'], dict(canal=.15, rail=.15, rail_end=.075))
        self.assertEqual(G.value_weights(third), (.25, .5))
        fourth = validate_config(G.configure(third, G.parse('weight=0.5'), 3170))
        self.assertEqual(fourth['local_scoring_auxiliary']['weights'], dict(canal=.5, rail=.5, rail_end=.25))
        # Idempotent within one command, which can call the continuation more than once.
        self.assertEqual(G.configure(fourth, G.parse('heads,weight=0.5,time'), 3171), fourth)
        self.assertEqual(G.tag(fourth), 'scan10-newval25-eraheads-w05-railend-blend50-vp05')
        chain = (before, first, second, third, fourth)
        for a, b in zip(chain, chain[1:]):
            self.assertTrue(signature_transition(signature(a), signature(b)))
            self.assertFalse(signature_transition(signature(b), signature(a)))
        self.assertTrue(signature_transition(signature(before), signature(fourth)))
        self.assertTrue(signature_transition(signature(fourth), signature(fourth)))
        lower = copy.deepcopy(fourth); lower['local_scoring_auxiliary']['weights'].update(canal=.15, rail=.15, rail_end=.075)
        self.assertFalse(signature_transition(signature(fourth), signature(lower)))
        with self.assertRaisesRegex(ValueError, 'era heads'):
            G.configure(first, ('weight=0.15',), 3141)
        with self.assertRaisesRegex(ValueError, 'only moves up'):
            G.configure(fourth, ('weight=0.15',), 3172)
        with self.assertRaises(ValueError):G.parse('pause')

    def test_value_losses_default_weights_unchanged(self):
        torch.manual_seed(0)
        out = (None, torch.softmax(torch.randn(5, 4), -1), 1 + 3 * torch.rand(5, 4), 100 * torch.rand(5, 4))
        targets = (torch.softmax(torch.randn(5, 4), -1), 1 + 3 * torch.rand(5, 4), 100 * torch.rand(5, 4))
        default = updates.value_losses(out, targets)
        explicit = updates.value_losses(out, targets, (.25, .1))
        self.assertTrue(torch.equal(default[0], explicit[0]))
        heavier = updates.value_losses(out, targets, (.25, .5))
        self.assertTrue(torch.allclose(heavier[0] - default[0], .4 * default[3]))


class Bootstrap(unittest.TestCase):
    def test_lambda_returns_match_the_recursion_per_seat_chain(self):
        from brass_ai.recovery.critic_bootstrap import lambda_returns
        rng = np.random.default_rng(3140); lam = .95
        identities = [('g', 0), ('g', 1), ('g', 0), ('h', 2), ('g', 1), ('g', 0), ('h', 2)]
        win = {('g', 0): [1, 0, 0, 0], ('g', 1): [1, 0, 0, 0], ('h', 2): [0, 0, .5, .5]}
        z_win = torch.tensor([win[i] for i in identities], dtype=torch.float32)
        z_utility = torch.tensor(rng.normal(size=(7, 4)), dtype=torch.float32)
        ds = types.SimpleNamespace(data=[(None, None, i, None) for i in identities], targets=(z_win,), utility=z_utility)
        frozen = dict(win=torch.softmax(torch.tensor(rng.normal(size=(7, 4)), dtype=torch.float32), -1),
                      utility=torch.tensor(rng.normal(size=(7, 4)), dtype=torch.float32))
        got = lambda_returns(ds, frozen, lam)
        for identity in set(identities):
            chain = [j for j, i in enumerate(identities) if i == identity]
            for z, v, column in ((z_win, frozen['win'], 0), (z_utility, frozen['utility'], 1)):
                following = None
                for j in reversed(chain):
                    # Reference: G_t = r_t + (1-lam) V_{t+1} + lam G_{t+1}, r_t = Z_t - Z_{t+1}; G_last = Z_last.
                    expected = z[j].double() if following is None else (z[j] - z[following]).double() + (1 - lam) * v[following].double() + lam * following_g
                    np.testing.assert_allclose(got[j, column], expected.numpy(), atol=1e-5)
                    following, following_g = j, expected
        np.testing.assert_allclose(got[:, 0].sum(-1), 1., atol=1e-5)
        last = [max(j for j, i in enumerate(identities) if i == identity) for identity in set(identities)]
        np.testing.assert_allclose(got[last, 1], z_utility[last].numpy(), atol=1e-6)


class EraHeads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from test_gae_value_phase import fixture
        from test_local_scoring import annotate
        cls.model, cls.opt, _, (_, data) = fixture(games=2); annotate(data)
        cls.ds = updates.Dataset(data, 'cpu')

    def companion(self, generation=5, **weights):
        config = v3.default(generation)
        config['weights'].update(weights)
        return auxiliary.Companion(self.model, config, raw=dict(model_id='m'), generation=generation)

    def test_heads_shape_initialization_and_rng(self):
        from brass_ai.learning.local_scoring_heads_v2 import EraScoringHeads, output_names
        h = EraScoringHeads(); self.assertEqual(sum(p.numel() for p in h.parameters()), 670102)
        state = torch.get_rng_state().clone()
        a = self.companion()
        self.assertEqual(type(a).__name__, 'CompanionV3'); self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(a.scale, 0.)
        for name in output_names():
            self.assertEqual(float(a.heads.owner[name].weight.abs().sum() + a.heads.vp[name].weight.abs().sum()), 0.)

    def test_loss_by_era_head_steps_kept_and_checkpoint_round_trip(self):
        a = self.companion(rail_end=.025); ix, x, _ = self.ds.batch(range(len(self.ds.data)))
        before = copy.deepcopy(a.heads.state_dict()); saved = a.snapshot()
        a.begin('ppg'); out, loss = a.forward(self.model, x, self.ds, ix)
        self.assertTrue(torch.isfinite(loss).all() and (loss > 0).all())
        expected = updates.value_output(self.model, x, self.ds, ix)
        for original, new in zip(expected[1:], out[1:]): self.assertTrue(torch.allclose(original, new, atol=1e-6))
        self.model.zero_grad(); loss.mean().backward(); a.step('ppg')
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in self.model.parameters()), 'head-only warm-up')
        a.restore(saved)  # a rejected interval keeps the head step
        changed = any(not torch.equal(v, a.heads.state_dict()[k]) for k, v in before.items())
        self.assertTrue(changed); self.assertEqual(a.retained_steps['ppg'], 1); self.assertEqual(a.fitting_steps, 1)
        report = a.report()
        self.assertEqual(set(report['by_era']), {'CanalEra', 'RailEra', 'CanalEra->RailEnd'})
        self.assertTrue(report['head_steps_kept_on_rejection'])
        payload = a.payload(6)
        self.assertEqual(payload['warmup']['fitting_steps'], 1)
        loaded = auxiliary.Companion(self.model, a.config, raw=dict(training={auxiliary.KEY: payload}, model_id='m'), generation=6)
        for k, v in a.heads.state_dict().items(): self.assertTrue(torch.equal(v, loaded.heads.state_dict()[k]))
        raised = copy.deepcopy(a.config); raised['weights'].update(canal=.15, rail=.15, rail_end=.075)
        auxiliary.Companion(self.model, raised, raw=dict(training={auxiliary.KEY: payload}, model_id='m'), generation=6)
        lowered = copy.deepcopy(a.config); lowered['weights']['rail_end'] = 0.
        with self.assertRaisesRegex(ValueError, 'config/progress'):
            auxiliary.Companion(self.model, lowered, raw=dict(training={auxiliary.KEY: payload}, model_id='m'), generation=6)

    def test_replaces_a_2_0_0_bundle_only_at_its_generation(self):
        old = auxiliary.Companion(self.model, auxiliary.fresh_config(auxiliary.DEFAULT), generation=3, adopting=True)
        payload = old.payload(3); raw = dict(training={auxiliary.KEY: payload}, model_id='m')
        new = auxiliary.Companion(self.model, v3.default(3), raw=raw, generation=3)
        self.assertEqual(new.replacement['previous']['identity'], payload['identity'])
        self.assertEqual(new.origin, dict(generation=3, model_id='m'))
        with self.assertRaisesRegex(ValueError, 'committed generation'):
            auxiliary.Companion(self.model, v3.default(4), raw=raw, generation=3)

    def test_preservation_and_heldout_check(self):
        from brass_ai.recovery.value_preservation import outputs, divergences
        a = self.companion(); ix, x, _ = self.ds.batch(range(len(self.ds.data)))
        out = outputs(self.model, x, self.ds, ix, a)
        self.assertEqual(float(divergences(out, out, a)['local'].abs().sum()), 0.)
        check = a.diagnostics(self.model, self.ds, list(range(len(self.ds.data))))
        self.assertGreater(check['rows'], 0)
        for group in check['groups'].values():
            self.assertGreaterEqual(group['owner_ce'], 0.); self.assertIsNotNone(group['reference_owner_ce'])


class ValuePhase(unittest.TestCase):
    def test_two_generations_new_game_validation_blended_targets_and_era_heads(self):
        import tempfile
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery import ppg_g210
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.value_preservation import FrozenValues
        from test_ppg_g210 import config, prepare
        from test_gae_value_phase import finish
        from test_value_buffer import commit
        torch.set_num_threads(2)
        m, opt, c = config()
        c['value_phase'] = dict(ppg_g210.KL_STOP_DEFAULT, batch_size=8, clone_rows=64, clone_batch_size=8)
        c[G.KEY] = dict(G.OFF, train_metrics_every=10, validation=copy.deepcopy(G.VALIDATION),
                        bootstrap=copy.deepcopy(G.BOOTSTRAP), value_weights=dict(rank=.25, vp=G.VP_WEIGHT))
        c['local_scoring_auxiliary'] = v3.default(1)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            run = Path(tmp); p = TrainingPhases(m, c, root=SOURCE_ROOT, run=run)
            p.local_scoring = auxiliary.Companion(m, c['local_scoring_auxiliary'], raw=dict(model_id='m'), generation=1)
            reports = []
            for generation in (1, 2):
                prepare(m, p, generation, companion=True)
                held = {p.critic_dataset.data[i][2][0] for i in p.heldout_ids}
                games = {d[2][0] for d in p.critic_dataset.data}
                self.assertGreaterEqual(len(held), round(.25 * len(games)) - 2)
                p.record_frozen(FrozenValues(m, p.critic_dataset, c['update']).values)
                r = p.fit(m); reports.append(r)
                self.assertEqual(r['training_targets']['bootstrap_share'], .5)
                self.assertEqual(r['training_targets']['loss_weights'], [.25, .5])
                self.assertFalse(r['train_metrics_scheduled'])
                self.assertIn('bootstrap', r['buffer'])
                self.assertGreater(r['attempted_steps'], 0)
                local = p.local_scoring.report()
                self.assertEqual(set(local['by_era']), {'CanalEra', 'RailEra'})
                self.assertEqual(local['companion_steps']['ppg'], r['attempted_steps'])  # head steps kept on rejection
                update = dict(categorical_probe=dict(categorical_kl=0.), controller_warmup_done=True, optimizer_steps=1,
                              kl_early_stop=False, gae={})
                finish(p, update, generation); commit(run, p, generation)
                p.local_scoring.committed(p.local_scoring.generation + 1)
            current, previous = reports[1]['acceptance_support']
            self.assertGreater(current['rows'], 0); self.assertEqual(previous['rows'], 0)
            self.assertGreater(previous['trained_validation_games'], 0)
            self.assertEqual(reports[1]['bootstrap_ages'], [True, True])
            ages = reports[1]['buffer']['rows_by_age']
            self.assertEqual(reports[1]['train_rows'], ages['0'] - current['rows'] + ages['1'])


class Staged(unittest.TestCase):
    def test_staged_batch_equals_host_collate(self):
        from test_gae_value_phase import fixture
        from brass_ai.learning.batching import collate
        from brass_ai.recovery.batch_cache import InputAudit
        from brass_ai.recovery.teacher_term import StagedRows, target_tensor
        _, _, _, (_, data) = fixture(games=2); rng = np.random.default_rng(1)
        rows = []
        for _, _, _, e in data:
            n = len(e.choice_ids); p = rng.random(n); p /= p.sum()
            rows.append(types.SimpleNamespace(encoded=e, label=dict(target=p.tolist(), legal_ids=list(e.choice_ids))))
        staged = StagedRows.build(rows, 'cpu', InputAudit())
        for indices in ([0], [3, 1, 2], list(range(len(rows)))[::-1]):
            states, actions, mask, target = staged.batch(indices)
            s, a, m, _ = collate([rows[i].encoded for i in indices], 'cpu')
            self.assertTrue(torch.equal(states, s) and torch.equal(actions, a) and torch.equal(mask, m))
            self.assertTrue(torch.equal(target, target_tensor([rows[i] for i in indices], torch.zeros(m.shape))))
        rows[0].label['target'] = rows[0].label['target'][:-1]
        self.assertIsNone(StagedRows.build(rows, 'cpu', InputAudit()))


class Purge(unittest.TestCase):
    def test_trainer_has_no_unresolved_global_references(self):
        import builtins
        import inspect
        import symtable
        from brass_ai.recovery import league

        scope = symtable.symtable(inspect.getsource(league._train), league.__file__, 'exec').get_children()[0]
        globals_used = {s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced()}
        self.assertEqual(globals_used - vars(league).keys() - vars(builtins).keys(), set())

    def test_background_purge_joins_and_raises(self):
        from unittest import mock
        from brass_ai.recovery import retention
        with mock.patch.object(retention, 'purge', return_value=dict(permanently_deleted_files=2)):
            b = retention.BackgroundPurge(); self.assertIsNone(b.join())
            b.start('root', 'run', {}, 3145); result = b.join()
            self.assertEqual((result['permanently_deleted_files'], result['generation']), (2, 3145))
        with mock.patch.object(retention, 'purge', side_effect=ValueError('Quarantine changed')):
            b = retention.BackgroundPurge(); b.start('root', 'run', {}, 3150)
            with self.assertRaisesRegex(RuntimeError, 'journal'): b.join()


if __name__ == '__main__':
    unittest.main()
