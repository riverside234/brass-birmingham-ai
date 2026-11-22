"""g4770 era-head revision: the Rail-end heads restored (PPG only, same batches as Canal/Rail), PPO-game rows on
actual maps for every era head, and the search pool a 4-iteration-sized sample of the last 8 iterations."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brass_ai.recovery import training_revision_g3140 as G
from brass_ai.recovery import training_revision_g4770 as R


def dropped():
    """The run's recipe shape before adoption: era heads, ownership stop, search-aux and search-rows, Rail-end
    heads removed (drop-rail-end)."""
    from brass_ai.recovery.config import validate_config
    from test_rail_end_g3528 import Recipe as RailEnd
    rows = RailEnd().adopted()
    return validate_config(G.configure(rows, G.parse('drop-rail-end'), 3528))


def signature(config):
    from test_training_g3140 import signature as sign
    return sign(config)


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = dropped(); cls.after = R.configure(cls.before, 4770)

    def test_it_restores_the_rail_end_heads_at_005_with_search_returns_after_a_warm_up(self):
        from brass_ai.recovery import local_scoring_v3 as v3
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        local = after['local_scoring_auxiliary']
        self.assertEqual((local['rail_end_removed'], local['rail_end_restored']), (3528, 4770))
        self.assertEqual(local['weights'], dict(canal=.15, rail=.15, rail_end=.05))
        self.assertTrue(v3.rail_end_restored(local) and not v3.two_era(local))
        self.assertEqual(G.search_rows(after), G.rail_end_returns(4770 + G.RAIL_END_WARMUP))
        self.assertEqual([G.rail_end_returns_on(after, g) for g in (4770, 4773, 4774)], [False, False, True])
        self.assertTrue(R.ordinary_actual(after) and not R.ordinary_actual(before))
        self.assertEqual(R.search_pool(after), dict(window_iterations=8, sampled_iterations=4,
                                                    draw='uniform-tables-without-replacement'))
        self.assertIsNone(R.search_pool(before))
        self.assertEqual(R.configure(after, 4771), after)  # adopted once
        self.assertTrue(is_on(before) and is_on(after))
        self.assertIn('-railend005-', G.tag(after)); self.assertNotIn('norailend', G.tag(after))
        self.assertEqual(R.tag(after), 'ppoactual-pool4of8')
        with self.assertRaisesRegex(ValueError, 'g4770 era-head revision'):
            from test_rail_end_g3528 import Recipe as RailEnd
            R.configure(RailEnd().adopted(), 4770)  # the Rail-end heads were never removed there
        bad = copy.deepcopy(after); bad[R.KEY]['search_pool']['window_iterations'] = 16
        with self.assertRaisesRegex(ValueError, 'g4770 era-head revision'): validate_config(bad)
        unrestored = copy.deepcopy(after); del unrestored['local_scoring_auxiliary']['rail_end_restored']
        with self.assertRaises(ValueError): validate_config(unrestored)

    def test_it_is_adopted_forward_only_and_the_weight_moves_up_once(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        heavier = R.set_rail_end_weight(after, .25)
        self.assertEqual(heavier['local_scoring_auxiliary']['weights']['rail_end'], .25)
        self.assertIn('-railend025-', G.tag(heavier))
        self.assertTrue(signature_transition(signature(after), signature(heavier)))
        self.assertFalse(signature_transition(signature(heavier), signature(after)))
        self.assertEqual(R.set_rail_end_weight(heavier, .25), heavier)
        with self.assertRaisesRegex(ValueError, 'only moves up'): R.set_rail_end_weight(heavier, .05)
        with self.assertRaisesRegex(ValueError, 'Rail-end weight must be'): R.set_rail_end_weight(after, .1)
        with self.assertRaisesRegex(ValueError, 'follows the g4770'): R.set_rail_end_weight(before, .25)
        # A later local-scoring weight item leaves the restored head's own weight alone.
        later = validate_config(G.configure(after, G.parse('weight=0.5'), 4800))
        self.assertEqual(later['local_scoring_auxiliary']['weights'], dict(canal=.5, rail=.5, rail_end=.05))
        gone = copy.deepcopy(after); gone['local_scoring_auxiliary']['rail_end_restored'] = 4780
        self.assertFalse(signature_transition(signature(after), signature(gone)))  # a restoration is kept


class Restoration(unittest.TestCase):
    """The two-era checkpoint gains fresh Rail-end outputs once, at the restoration's generation."""

    def companion(self, config, generation, raw=None):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from test_gae_value_phase import fixture
        model = fixture(games=1)[0]
        return auxiliary.Companion(model, config, raw=dict(raw or {}, model_id='m'), generation=generation)

    def test_the_checkpoint_and_both_adams_gain_fresh_rail_end_slots_once(self):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        removed = dict(v3.default(5, weight=.15), rail_end_removed=5)
        a = self.companion(removed, 5)
        for optimizer in a.optimizers.values():  # one step, so both Adams have state to carry over
            for p in a.heads.parameters(): p.grad = torch.full_like(p, .1)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        saved = a.payload(6)
        restored = dict(removed, rail_end_restored=6, weights=dict(removed['weights'], rail_end=.05))
        b = self.companion(restored, 6, raw=dict(training={auxiliary.KEY: saved}))
        two = dict(a.heads.named_parameters()); full = dict(b.heads.named_parameters())
        added = sorted(set(full) - set(two))
        self.assertEqual(added, sorted(f'{m}.{h}.{p}' for m in ('owner', 'vp') for h in v3.RAIL_END for p in ('weight', 'bias')))
        for name, p in two.items(): self.assertTrue(torch.equal(full[name], p))
        for name in added: self.assertFalse(full[name].any())  # zero, as the heads' own initialisation
        names = list(full); kept = list(two)
        for phase in ('ppo', 'ppg'):
            old, new = a.optimizers[phase].state_dict(), b.optimizers[phase].state_dict()
            self.assertEqual(new['param_groups'][0]['params'], list(range(len(names))))
            for j, name in enumerate(kept):
                self.assertTrue(torch.equal(new['state'][names.index(name)]['exp_avg'], old['state'][j]['exp_avg']))
            self.assertFalse(any(names.index(n) in new['state'] for n in added))  # fresh Adam slots
        record = b.report()['rail_end_heads']
        self.assertEqual((record['removed_at'], record['restored_at'], record['weight']), (5, 6, .05))
        self.assertEqual((record['restoration']['parameters'], record['restoration']['max_abs'],
                          record['restoration']['generation']), (2 * (96 * 9 + 9), 0., 6))
        # PPO neither snapshots nor preserves the restored outputs; the stop reports their CE without deciding.
        self.assertEqual(sorted(b.preservation_weights()), sorted(n for n in v3.output_names() if not n.startswith('rail_end')))
        self.assertFalse(b.rail_end_decides_stop); self.assertTrue(a.rail_end_decides_stop)
        again = self.companion(restored, 6, raw=dict(training={auxiliary.KEY: b.payload(6)}))  # later loads: plain
        self.assertIsNone(again.rail_end_restoration)
        for name, p in again.heads.named_parameters(): self.assertTrue(torch.equal(p, full[name]))
        with self.assertRaisesRegex(ValueError, 'committed generation'):
            self.companion(dict(restored, rail_end_restored=7), 6, raw=dict(training={auxiliary.KEY: saved}))
        with self.assertRaises(ValueError):  # a restoration is never undone
            self.companion(removed, 6, raw=dict(training={auxiliary.KEY: b.payload(6)}))

    def test_frozen_ppo_outputs_leave_the_restored_heads_out(self):
        from brass_ai.recovery import local_scoring_v3 as v3
        from brass_ai.recovery.value_preservation import FrozenValues
        from test_gae_value_phase import fixture
        from test_local_scoring import annotate
        from brass_ai.recovery import updates, local_scoring_auxiliary as auxiliary
        model, _, cfg, (_, data) = fixture(games=1); annotate(data)
        ds = updates.Dataset(data, 'cpu')
        restored = dict(v3.default(6), rail_end_removed=6, rail_end_restored=7, weights=dict(canal=.05, rail=.05, rail_end=.05))
        companion = auxiliary.Companion(model, restored, raw=dict(model_id='m'), generation=6)
        values = FrozenValues(model, ds, cfg['update'], companion).values
        self.assertFalse(any(k.startswith('rail_end_') for k in values))
        self.assertTrue(any(k.startswith('canal_slots') for k in values) and any(k.startswith('rail_links') for k in values))


class Phase(unittest.TestCase):
    """Through ppg_g210.fit with the restored Rail-end head and the revision's actual PPO-game targets."""

    def run_phase(self, *, lam=.95, returns=True):
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        from brass_ai.recovery import ppg_g210
        from brass_ai.recovery import search_auxiliary as A
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.value_preservation import FrozenValues
        from test_ppg_g210 import config, prepare
        from test_search_auxiliary_g3423 import Buffer, roots, table as search_table
        torch.manual_seed(3); torch.set_num_threads(2)
        m, opt, c = config()
        c['value_phase'] = dict(ppg_g210.KL_STOP_DEFAULT, batch_size=4, clone_rows=64, clone_batch_size=8)
        settings = copy.deepcopy(G.SEARCH_ROWS); settings['auxiliary']['lam'] = lam
        with mock.patch.object(G, 'SEARCH_ROWS', settings):
            rows = G.rail_end_returns(0) if returns else copy.deepcopy(settings)
        c[G.KEY] = dict(G.OFF, validation=copy.deepcopy(G.VALIDATION), bootstrap=copy.deepcopy(G.BOOTSTRAP),
                        ownership_stop=copy.deepcopy(G.OWNERSHIP_STOP), search_auxiliary=copy.deepcopy(G.SEARCH_AUX),
                        search_rows=rows)
        c['local_scoring_auxiliary'] = dict(v3.default(1), rail_end_removed=1, rail_end_restored=2,
                                            weights=dict(canal=.05, rail=.05, rail_end=.05))
        c[R.KEY] = dict(copy.deepcopy(R.SETTINGS), from_generation=2)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp, mock.patch.object(G, 'SEARCH_ROWS', settings):
            p = TrainingPhases(m, c, root=SOURCE_ROOT, run=Path(tmp))
            p.local_scoring = auxiliary.Companion(m, c['local_scoring_auxiliary'], raw=dict(model_id='m'), generation=1)
            start = {k: v.detach().clone() for k, v in p.local_scoring.heads.state_dict().items()}
            _, data = prepare(m, p, 1, companion=True)
            frozen = FrozenValues(m, p.critic_dataset, c['update'], p.local_scoring).values
            p.record_frozen(frozen)
            kept_local = p.frozen_local
            sources = [(e.state, int(row['seat']), row['view']['context']['era']) for row, _, _, e in data]
            def game(name, split):
                receipt, result, job = search_table(name)
                return roots(name, 1, split, sources[:12], A.table_envelope(receipt, result, job))
            buffer = Buffer([game(f'teacher/t{i}', 'train') for i in range(3)] + [game('teacher/h', 'heldout')])
            p.search_auxiliary = dict(settings=copy.deepcopy(G.SEARCH_AUX), tables=A.tables(buffer, 1),
                                      rows=G.search_rows(c), rail_end=G.rail_end_returns_on(c, p.state['generation']))
            r = p.fit(m)
            r['_training'] = p.local_scoring.report()['training']
            r['_frozen'] = sorted(frozen); r['_kept_local'] = kept_local
            r['_sidecars'] = [np.load(f, mmap_mode='r').shape for f in Path(tmp).rglob('*.local-bootstrap.npy')]
            after = p.local_scoring.heads.state_dict()
            r['_moved'] = {k: float((after[k] - start[k]).abs().max()) for k in after}
            r['_report'] = p.local_scoring.report()
        return r

    def test_rail_end_trains_in_ppg_with_actual_ppo_targets_and_search_returns(self):
        r = self.run_phase()
        self.assertFalse(any(k.startswith('rail_end_') for k in r['_frozen']))  # no PPO snapshot or transfer
        self.assertIsNone(r['_kept_local'])  # no lambda-return caching for PPO games
        self.assertEqual(r['_sidecars'], [])  # no local-bootstrap sidecar
        self.assertEqual(r['buffer'].get('era_head_targets'), 'actual-map'); self.assertNotIn('local_bootstrap', r['buffer'])
        self.assertGreater(r['_moved']['owner.rail_end_slots.weight'], 0.)  # trained in PPG, in the same steps
        self.assertGreater(r['_moved']['owner.canal_slots.weight'], 0.)
        self.assertIn('CanalEra->RailEnd', r['_report']['by_era'])
        search = r['search_auxiliary']['lambda_targets']
        self.assertGreater(search['mean_owner_l1'], 0.); self.assertGreater(search['rail_end']['rows'], 0)
        stop = r['ownership_stop']
        self.assertIsNotNone(stop['before']['rail_end_owner_ce'])

    def test_actual_ppo_targets_equal_lambda_one_and_rail_end_does_not_decide_the_stop(self):
        key = 'ppg_weighted_auxiliary_loss_sum'
        actual = self.run_phase(lam=1., returns=False)  # search rows at lambda 1 too: only the PPO-game rule differs
        with mock.patch.object(R, 'ordinary_actual', lambda config: False):
            one = self.run_phase(lam=1., returns=False)  # the g3471 rule at lambda 1: corrections of zero
        self.assertTrue(one['_sidecars'] and one['_kept_local'])
        self.assertIn('local_bootstrap', one['buffer'])
        self.assertAlmostEqual(actual['_training'][key], one['_training'][key], places=4)
        # The decisive owner CE is the Canal/Rail part alone: a mean of the Canal and Rail rows' CE. With the
        # Rail-end CE in it (weight .05 = the Canal weight here) it would lie above both.
        for measured in (actual['ownership_stop']['before'], actual['ownership_stop']['after']):
            low, high = sorted((measured['canal_owner_ce'], measured['rail_owner_ce']))
            self.assertIsNotNone(measured['rail_end_owner_ce'])
            self.assertTrue(low - 1e-9 <= measured['owner_ce'] <= high + 1e-9, measured)


class Pool(unittest.TestCase):
    """The search tables are a uniform sample, as large as the last 4 iterations admitted, of the last 8."""

    @classmethod
    def setUpClass(cls):
        from test_search_auxiliary_g3423 import Buffer, roots, table
        from brass_ai.recovery import search_auxiliary as A
        cls.A = A; cls.G0 = 4800
        def game(name, generation, split):
            receipt, result, job = table(name)
            return roots(name, generation, split, [(np.zeros(3128, np.float32), 0, 'CanalEra')], A.table_envelope(receipt, result, job))
        games = [game(f'teacher/h{g}-{t}', g, 'heldout') for g in range(cls.G0 - 19, cls.G0 + 1) for t in range(4)]
        games += [game(f'teacher/t{g}-{t}', g, 'train') for g in range(cls.G0 - 11, cls.G0 + 1) for t in range(5)]
        cls.buffer = Buffer(games)

    def test_the_sample_has_the_four_iteration_size_from_the_last_eight(self):
        A, G0, pool = self.A, self.G0, R.SETTINGS['search_pool']
        recent = A.tables(self.buffer, G0, heldout_games=32)  # the 2.0.0 pool: the last four iterations
        sampled = A.tables(self.buffer, G0, heldout_games=32, pool=pool)
        names = lambda chosen: [g for g, _, _ in chosen['train']]
        self.assertEqual(len(names(sampled)), len(names(recent)))  # 4 x 5 train + 16 rotated
        self.assertEqual(len(names(recent)), 36)
        report = sampled['report']['sampled']
        self.assertEqual((report['window_iterations'], report['sampled_iterations'], report['candidates'], report['size']),
                         (8, 4, 8 * 5 + 8 * 4, 36))
        generation = lambda name: int(name.split('/')[1][1:].split('-')[0])
        for name in names(sampled):
            g = generation(name)
            self.assertTrue(G0 - 7 <= g <= G0 if name.startswith('teacher/t') else G0 - 15 <= g <= G0 - 8, name)
        self.assertEqual(len(set(names(sampled))), 36)
        self.assertFalse(set(names(sampled)) & {g for g, _, _ in sampled['heldout']})
        self.assertEqual([g for g, _, _ in sampled['heldout']], [g for g, _, _ in recent['heldout']])
        self.assertNotEqual(set(names(sampled)), set(names(recent)))  # older tables come back in
        self.assertEqual(names(A.tables(self.buffer, G0, heldout_games=32, pool=pool)), names(sampled))  # reproducible
        other = A.tables(self.buffer, G0 - 1, heldout_games=32, pool=pool)
        self.assertNotEqual(set(names(other)), set(names(sampled)))  # a fresh sample each iteration
        trains = [generation(n) for n in names(sampled) if n.startswith('teacher/t')]
        self.assertTrue(any(g < G0 - 3 for g in trains) and any(g >= G0 - 3 for g in trains))
        self.assertEqual(sampled['report']['sampled']['by_generation'],
                         {str(k): v for k, v in sorted(__import__('collections').Counter(generation(n) for n in names(sampled)).items())})


if __name__ == '__main__':
    unittest.main()
