"""g3528: the Rail-end head's lambda-return targets, and soft policy weight changes beyond 8 -> 6."""
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from brass_ai.recovery import search_rows as R
from brass_ai.recovery import training_revision_g3140 as G
from test_search_rows_g3471 import envelope_maps


def frozen_with_end(rows, seed):
    """Frozen head outputs by critic row for all three horizons (value_preservation.outputs names)."""
    rng = np.random.default_rng(seed)
    out = {}
    for horizon in ('canal', 'rail', 'rail_end'):
        for kind, n in (('slots', 49), ('links', 69)):
            out[f'{horizon}_{kind}_owner'] = torch.as_tensor(rng.dirichlet(np.ones(5), size=(rows, n)), dtype=torch.float32)
            out[f'{horizon}_{kind}_vp'] = torch.as_tensor(rng.normal(size=(rows, n, 4)), dtype=torch.float32)
    return out


def chain_dataset(eras, seat=1, chain=('teacher/g', 1)):
    e, _ = envelope_maps()
    mapping = [(s - seat) % 4 for s in range(4)]
    rows = [{'_local_scoring': dict(envelope=e, era=era)} for era in eras]
    data = [(None, None, chain, SimpleNamespace(stable_to_relative=mapping)) for _ in eras]
    return SimpleNamespace(rows=rows, data=data), mapping


class OrdinaryRows(unittest.TestCase):
    def test_canal_rows_chain_on_through_the_rail_run_to_the_game_end_map(self):
        from brass_ai.recovery.local_scoring_targets import map_for
        ds, mapping = chain_dataset(['CanalEra'] * 3 + ['RailEra'] * 2)
        frozen = frozen_with_end(5, 1); lam = .95
        plain, _ = R.ordinary_corrections(ds, frozen, lam)
        out, report = R.ordinary_corrections(ds, frozen, lam, rail_end=True)
        self.assertEqual(out.shape, (5, R.ENTITIES, R.PAIRED))
        np.testing.assert_array_equal(out[..., :R.WIDTH], plain)  # the era corrections are unchanged
        self.assertEqual((report['rail_end']['canal_runs'], report['rail_end']['linked_runs'], report['rail_end']['rows']), (1, 1, 3))
        z = [R.map_arrays(map_for(ds.rows[i]), mapping)[0] for i in range(5)]
        rail = lambda i: R.head_outputs(frozen, [i], [False])[0]
        end = lambda i: R.end_outputs(frozen, [i])[0]
        c3 = (1 - lam) * (rail(4) - z[4])  # the Rail run's era correction at its first row
        e2 = (1 - lam) * (rail(3) - z[3]) + lam * c3  # the last Canal row bootstraps at the first Rail row
        e1 = (1 - lam) * (end(2) - z[3]) + lam * e2
        e0 = (1 - lam) * (end(1) - z[3]) + lam * e1
        for i, e in ((0, e0), (1, e1), (2, e2)):
            np.testing.assert_allclose(out[i, :, R.WIDTH:].astype(np.float32), e, atol=3e-3)
        self.assertFalse(out[3:, :, R.WIDTH:].any())  # Rail rows have no Rail-end target
        ones, _ = R.ordinary_corrections(ds, frozen, 1., rail_end=True)
        self.assertFalse(ones.any())  # lambda 1: the actual game-end map
        q = z[3][:, :5] + .5 * out[0, :, R.WIDTH:R.WIDTH + 5].astype(np.float64)  # still a distribution
        self.assertTrue((q >= -1e-3).all()); np.testing.assert_allclose(q.sum(-1), 1., atol=5e-3)

    def test_a_canal_run_without_its_rail_run_keeps_the_actual_map(self):
        ds, _ = chain_dataset(['CanalEra'] * 3)
        out, report = R.ordinary_corrections(ds, frozen_with_end(3, 2), .95, rail_end=True)
        self.assertEqual(report['rail_end']['linked_runs'], 0); self.assertFalse(out[..., R.WIDTH:].any())


class SearchRows(unittest.TestCase):
    def test_canal_roots_chain_on_through_the_first_rail_root_in_the_stable_seats(self):
        _, maps = envelope_maps(); lam = .95
        rows = [dict(index=t, seat=s, mapping=[(x - s) % 4 for x in range(4)], turn_key=[era, 1, 1, s])
                for t, s, era in ((1, 0, 'CanalEra'), (2, 1, 'CanalEra'), (3, 2, 'RailEra'), (4, 3, 'RailEra'))]
        rows.insert(1, dict(kind='second', index=1, revision=15, seat=2, mapping=[(x - 2) % 4 for x in range(4)],
                            turn_key=['CanalEra', 1, 1, 0]))
        rows.append(dict(kind='sampled', index=2, revision=25, seat=3, mapping=[(x - 3) % 4 for x in range(4)],
                         turn_key=['CanalEra', 1, 2, 1]))
        rng = np.random.default_rng(2)
        draw = lambda: np.concatenate((rng.dirichlet(np.ones(5), size=R.ENTITIES), rng.normal(size=(R.ENTITIES, 4))), -1)
        at = {r['index']: j for j, r in enumerate(rows) if R.kind(r) == 'turn'}
        frozen = {j: draw() for j in at.values()}
        ends = {at[1]: draw(), at[2]: draw()}
        plain = R.search_corrections([(rows, maps, frozen)], lam)
        out = R.search_corrections([(rows, maps, frozen, ends)], lam, rail_end=True)
        self.assertEqual(out.shape, (len(rows), R.ENTITIES, R.PAIRED))
        np.testing.assert_allclose(out[..., :R.WIDTH], plain)
        stable = {era: R.map_arrays(maps[era], [0, 1, 2, 3])[0] for era in maps}
        S = lambda values, t: R.to_stable(values, rows[at[t]]['mapping'])
        rail3 = (1 - lam) * (S(frozen[at[4]], 4) - stable['RailEra'])  # root 3's Rail-era return
        end2 = (1 - lam) * (S(frozen[at[3]], 3) - stable['RailEra']) + lam * rail3
        end1 = (1 - lam) * (S(ends[at[2]], 2) - stable['RailEra']) + lam * end2
        for j, e in ((at[1], end1), (1, end1), (at[2], end2), (len(rows) - 1, end2)):  # extra rows take their root's
            np.testing.assert_allclose(out[j, :, R.WIDTH:], R.to_relative(e, rows[j]['mapping']), atol=1e-6)
        self.assertFalse(out[at[3], :, R.WIDTH:].any() or out[at[4], :, R.WIDTH:].any())
        self.assertFalse(R.search_corrections([(rows, maps, frozen, ends)], 1., rail_end=True).any())


class Replay(unittest.TestCase):
    def test_ages_with_and_without_the_rail_end_channels_read_together(self):
        from brass_ai.recovery.ppg_replay import LocalBootstrapColumn
        rng = np.random.default_rng(4)
        old = rng.normal(size=(3, R.ENTITIES, R.WIDTH)).astype('<f2')
        new = rng.normal(size=(2, R.ENTITIES, R.PAIRED)).astype('<f2')
        parts = [SimpleNamespace(local_bootstrap=None), SimpleNamespace(local_bootstrap=old), SimpleNamespace(local_bootstrap=new)]
        column = LocalBootstrapColumn(SimpleNamespace(parts=parts, ends=np.array([2, 5, 7])))
        values = column(torch.tensor([0, 3, 6, 5]))
        self.assertEqual(values.shape, (4, R.ENTITIES, R.PAIRED))
        self.assertFalse(values[0].any())
        np.testing.assert_array_equal(values[1, :, :R.WIDTH], old[1].astype(np.float32)); self.assertFalse(values[1, :, R.WIDTH:].any())
        np.testing.assert_array_equal(values[2], new[1].astype(np.float32)); np.testing.assert_array_equal(values[3], new[0].astype(np.float32))
        only_old = LocalBootstrapColumn(SimpleNamespace(parts=parts[:2], ends=np.array([2, 5])))
        self.assertEqual(only_old(np.array([3])).shape, (1, R.ENTITIES, R.WIDTH))


class Phase(unittest.TestCase):
    """Through ppg_g210.fit with the Rail-end head on (weight canal/2) and search-rows."""

    def run_phase(self, lam=.95, returns=True, removed=False):
        import tempfile
        from pathlib import Path
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
        c['local_scoring_auxiliary'] = dict(v3.default(1), rail_end_removed=1) if removed else v3.default(1, rail_end=.025)
        if removed: c[G.KEY]['search_rows'] = copy.deepcopy(settings)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp, mock.patch.object(G, 'SEARCH_ROWS', settings):
            p = TrainingPhases(m, c, root=SOURCE_ROOT, run=Path(tmp))
            p.local_scoring = auxiliary.Companion(m, c['local_scoring_auxiliary'], raw=dict(model_id='m'), generation=1)
            _, data = prepare(m, p, 1, companion=True)
            p.record_frozen(FrozenValues(m, p.critic_dataset, c['update'], p.local_scoring).values)
            frozen_keys = sorted(p.frozen_local)
            sources = [(e.state, int(row['seat']), row['view']['context']['era']) for row, _, _, e in data]
            def game(name, split):
                receipt, result, job = search_table(name)
                return roots(name, 1, split, sources[:12], A.table_envelope(receipt, result, job))
            buffer = Buffer([game(f'teacher/t{i}', 'train') for i in range(3)] + [game('teacher/h', 'heldout')])
            p.search_auxiliary = dict(settings=copy.deepcopy(G.SEARCH_AUX), tables=A.tables(buffer, 1),
                                      rows=G.search_rows(c), rail_end=G.rail_end_returns_on(c, p.state['generation']))
            r = p.fit(m)
            r['_training'] = p.local_scoring.report()['training']
            r['_frozen_keys'] = frozen_keys
            r['_sidecars'] = [np.load(f, mmap_mode='r').shape for f in Path(tmp).rglob('*.local-bootstrap.npy')]
            r['_head_names'] = [n for n, _ in p.local_scoring.heads.named_parameters()]
            r['_report'] = p.local_scoring.report()
        return r

    def test_without_the_rail_end_heads_the_phase_runs_and_nothing_reads_them(self):
        r = self.run_phase(removed=True)
        self.assertFalse(any('rail_end' in n for n in r['_head_names']))
        self.assertFalse(any(k.startswith('rail_end_') for k in r['_frozen_keys']))
        self.assertGreater(r['buffer']['local_bootstrap']['mean_owner_l1'], 0.)
        self.assertIsNone(r['ownership_stop']['before'].get('rail_end_owner_ce'))
        self.assertEqual(r['_report']['rail_end_heads'], dict(removed_at=1, migration=None))  # fresh: nothing to migrate

    def test_the_rail_end_head_trains_on_its_returns_and_lambda_one_keeps_the_game_end_map(self):
        r = self.run_phase()
        boot = r['buffer']['local_bootstrap']['rail_end']; search = r['search_auxiliary']['lambda_targets']['rail_end']
        self.assertTrue(any(k.startswith('rail_end_') for k in r['_frozen_keys']))
        self.assertGreater(boot['linked_runs'], 0); self.assertGreater(boot['mean_owner_l1'], 0.)
        self.assertGreater(search['rows'], 0); self.assertGreater(search['mean_owner_l1'], 0.)
        self.assertTrue(r['_sidecars'] and all(s[1:] == (R.ENTITIES, R.PAIRED) for s in r['_sidecars']))
        actual = self.run_phase(returns=False)  # rail-end on, its targets the actual map (the g3471 rule)
        self.assertNotIn('rail_end', actual['buffer']['local_bootstrap'])
        self.assertFalse(any(k.startswith('rail_end_') for k in actual['_frozen_keys']))
        self.assertTrue(actual['_sidecars'] and all(s[1:] == (R.ENTITIES, R.WIDTH) for s in actual['_sidecars']))
        key = 'ppg_weighted_auxiliary_loss_sum'
        self.assertNotAlmostEqual(r['_training'][key], actual['_training'][key], places=6)
        one, one_actual = self.run_phase(lam=1.), self.run_phase(lam=1., returns=False)
        self.assertEqual(one['buffer']['local_bootstrap']['rail_end']['mean_owner_l1'], 0.)
        self.assertAlmostEqual(one['_training'][key], one_actual['_training'][key], places=4)
        self.assertEqual(r['ownership_stop']['before'], actual['ownership_stop']['before'])  # the stop scores actual maps


class Recipe(unittest.TestCase):
    def adopted(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery import training_revision_g3270 as g3270
        from test_teacher_revision_g3270 import before
        teacher = g3270.configure(before(), 3270)
        heads = validate_config(G.configure(teacher, G.parse('heads,targets,weight=0.15,ownership-stop'), 3318))
        aux = validate_config(G.configure(heads, G.parse('search-aux'), 3423))
        aux['teacher_phase']['search_value'].setdefault('target', {})['lambda'] = .95
        return validate_config(G.configure(aux, G.parse('search-rows'), 3471))

    def test_rail_end_on_a_continued_run_gets_returns_after_a_warm_up(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        from test_training_g3140 import signature
        rows = self.adopted()
        on = validate_config(G.configure(rows, G.parse('rail-end'), 3528))
        self.assertEqual(on['local_scoring_auxiliary']['weights'], dict(canal=.15, rail=.15, rail_end=.075))
        self.assertEqual(G.search_rows(on), G.rail_end_returns(3528 + G.RAIL_END_WARMUP))
        self.assertEqual([G.rail_end_returns_on(on, g) for g in (3528, 3531, 3532)], [False, False, True])
        self.assertFalse(G.rail_end_returns_on(rows, 3600))
        self.assertTrue(G.tag(on).endswith('-searchrows-railendlam') and '-railend-' in G.tag(on))
        self.assertTrue(is_on(on))
        self.assertTrue(signature_transition(signature(rows), signature(on)))
        self.assertFalse(signature_transition(signature(on), signature(rows)))
        self.assertEqual(validate_config(G.configure(on, G.parse('rail-end'), 3540)), on)  # passing it again changes nothing
        self.assertEqual(G.project(on, on), on)
        back = copy.deepcopy(on); back[G.KEY]['search_rows'] = copy.deepcopy(G.SEARCH_ROWS)
        self.assertIsNone(G.project(on, back))  # the returns are never switched off
        no_head = copy.deepcopy(on); no_head['local_scoring_auxiliary']['weights']['rail_end'] = 0.
        self.assertIsNone(G.project(rows, no_head))
        later = validate_config(G.configure(on, G.parse('weight=0.5'), 3600))
        self.assertEqual(later['local_scoring_auxiliary']['weights']['rail_end'], .25)
        self.assertEqual(G.search_rows(later), G.search_rows(on))

    def test_a_trained_rail_end_head_gets_returns_from_search_rows_adoption(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery import training_revision_g3270 as g3270
        from test_teacher_revision_g3270 import before
        teacher = g3270.configure(before(), 3270)
        heads = validate_config(G.configure(teacher, G.parse('heads,targets,rail-end,ownership-stop,search-aux'), 3318))
        heads['teacher_phase']['search_value'].setdefault('target', {})['lambda'] = .95
        both = validate_config(G.configure(heads, G.parse('search-rows'), 3471))
        self.assertEqual(G.search_rows(both), G.rail_end_returns(3471))
        bad = copy.deepcopy(both); bad[G.KEY]['search_rows']['auxiliary']['rail_end']['through'] = 'canal'
        with self.assertRaisesRegex(ValueError, 'g3140'): G.validate(bad[G.KEY])


class HeadRemoval(unittest.TestCase):
    """g3528 drop-rail-end: the era heads lose their untrained Rail-end outputs at a committed generation."""

    def test_the_two_era_heads_are_the_full_heads_without_rail_end(self):
        from brass_ai.learning.local_scoring_heads_v2 import EraScoringHeads
        from brass_ai.recovery import local_scoring_v3 as v3
        from test_gae_value_phase import fixture
        torch.manual_seed(7); full = EraScoringHeads(64)
        torch.manual_seed(7); two = v3.TwoEraHeads(64)
        kept = dict(two.named_parameters())
        self.assertEqual(sorted(set(two.full_names) - set(kept)),
                         sorted(f'{m}.{h}.{p}' for m in ('owner', 'vp') for h in v3.RAIL_END for p in ('weight', 'bias')))
        for name, value in kept.items(): self.assertTrue(torch.equal(value, dict(full.named_parameters())[name]))
        encoded = SimpleNamespace(context=torch.randn(3, 64), targets=torch.randn(3, 130, 64))
        descriptors = torch.randn(3, 4, 32)
        a, b = full(encoded, descriptors), two(encoded, descriptors)
        self.assertEqual(sorted(b), sorted(k for k in a if not k.startswith('rail_end_')))
        for k in b:
            for x, y in zip(a[k], b[k]): self.assertTrue(torch.equal(x, y))

    def companion(self, config, generation, raw=None):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from test_gae_value_phase import fixture
        model = fixture(games=1)[0]
        return auxiliary.Companion(model, config, raw=dict(raw or {}, model_id='m'), generation=generation)

    def test_the_checkpoint_and_both_adams_lose_the_rail_end_slots_once(self):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        config = v3.default(5, weight=.15)
        a = self.companion(config, 5)
        for phase, optimizer in a.optimizers.items():  # one step: Rail-end gradients are zero, as at weight 0
            for name, p in a.heads.named_parameters():
                p.grad = torch.zeros_like(p) if 'rail_end' in name else torch.full_like(p, .1)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        saved = a.payload(5)
        removed = dict(config, rail_end_removed=5)
        b = self.companion(removed, 5, raw=dict(training={auxiliary.KEY: saved}))
        self.assertFalse(any('rail_end' in n for n, _ in b.heads.named_parameters()))
        full = dict(a.heads.named_parameters())
        for name, p in b.heads.named_parameters(): self.assertTrue(torch.equal(p, full[name]))
        names = [n for n, _ in a.heads.named_parameters()]; kept = [n for n, _ in b.heads.named_parameters()]
        for phase in ('ppo', 'ppg'):
            old, new = a.optimizers[phase].state_dict(), b.optimizers[phase].state_dict()
            self.assertEqual(new['param_groups'][0]['params'], list(range(len(kept))))
            for j, name in enumerate(kept):
                self.assertTrue(torch.equal(new['state'][j]['exp_avg'], old['state'][names.index(name)]['exp_avg']))
        migration = b.report()['rail_end_heads']['migration']
        self.assertEqual((migration['parameters'], migration['max_abs'], migration['generation']), (2 * (96 * 9 + 9), 0., 5))
        self.assertEqual(sorted(b.preservation_weights()), sorted(n for n in v3.output_names() if not n.startswith('rail_end')))
        again = self.companion(removed, 5, raw=dict(training={auxiliary.KEY: b.payload(5)}))  # later loads: no migration
        self.assertIsNone(again.rail_end_removal)
        with self.assertRaisesRegex(ValueError, 'committed generation'):
            self.companion(dict(config, rail_end_removed=6), 5, raw=dict(training={auxiliary.KEY: saved}))
        with self.assertRaises(ValueError):  # a removal is never undone
            self.companion(config, 5, raw=dict(training={auxiliary.KEY: b.payload(5)}))

    def test_the_item_removes_untrained_heads_once_and_refuses_rail_end_after(self):
        from brass_ai.recovery import local_scoring_v3 as v3
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from test_training_g3140 import signature
        rows = Recipe().adopted()
        dropped = validate_config(G.configure(rows, G.parse('drop-rail-end'), 3528))
        local = dropped['local_scoring_auxiliary']
        self.assertEqual((local['rail_end_removed'], local['weights']['rail_end']), (3528, 0.))
        self.assertIn('-norailend', G.tag(dropped))
        self.assertEqual(validate_config(G.configure(dropped, G.parse('drop-rail-end'), 3540)), dropped)
        with self.assertRaisesRegex(ValueError, 'removed'):
            G.configure(dropped, G.parse('rail-end'), 3540)
        on = validate_config(G.configure(rows, G.parse('rail-end'), 3528))
        with self.assertRaisesRegex(ValueError, 'never trained'):
            G.configure(on, G.parse('drop-rail-end'), 3530)
        self.assertTrue(v3.config_forward(rows['local_scoring_auxiliary'], local))
        self.assertFalse(v3.config_forward(local, rows['local_scoring_auxiliary']))
        self.assertTrue(signature_transition(signature(rows), signature(dropped)))
        self.assertFalse(signature_transition(signature(dropped), signature(rows)))
        later = validate_config(G.configure(dropped, G.parse('weight=0.5'), 3600))
        self.assertEqual(later['local_scoring_auxiliary']['weights'], dict(canal=.5, rail=.5, rail_end=0.))


class SoftWeight(unittest.TestCase):
    def block6(self):
        from brass_ai.recovery import training_revision_g3423 as S
        return S, dict(copy.deepcopy(S.SETTINGS), weight=6., from_generation=3423,
                       weight_change=dict(generation=3471, before=8.))

    def test_each_change_nests_the_one_before(self):
        S, block = self.block6()
        config = {S.KEY: S.validate(block)}
        with mock.patch('brass_ai.recovery.config.validate_config', lambda c: c):
            twelve = S.set_weight(config, 12, 3528)
            self.assertEqual(S.set_weight(twelve, 12, 3540), twelve)
            eight = S.set_weight(twelve, 8, 3600)
            with self.assertRaisesRegex(ValueError, 'reviewed'): S.set_weight(config, 5, 3528)
            with self.assertRaisesRegex(ValueError, 'follows'): S.set_weight(config, 12, 3471)
        b = twelve[S.KEY]
        self.assertEqual((b['weight'], b['weight_change']),
                         (12., dict(generation=3528, before=6., previous=dict(generation=3471, before=8.))))
        self.assertEqual(S.previous_settings(b), config[S.KEY])
        self.assertEqual(S.previous_settings(eight[S.KEY]), b)
        adopted = {k: v for k, v in block.items() if k != 'weight_change'}; adopted['weight'] = 8.
        self.assertEqual(S.previous_settings(S.previous_settings(b)), adopted)
        self.assertEqual((S.tag(config), S.tag(twelve)), ('soft-t4-w6-s1', 'soft-t4-w12-s1'))
        self.assertEqual(S.project(config, twelve), twelve)
        self.assertIsNone(S.project(twelve, config))
        self.assertIsNone(S.project(config, eight))  # one change per boundary
        for broken in (dict(generation=3528, before=12.),  # a change to the same weight
                       dict(generation=3528, before=6., previous=dict(generation=3600, before=8.)),  # out of order
                       dict(generation=3528, before=6., previous=dict(generation=3471, before=10.))):  # not from 8
            with self.assertRaises(ValueError): S.validate(dict(b, weight_change=broken))

    def test_the_head_carries_over_at_the_second_change(self):
        from brass_ai.recovery.soft_policy import SoftPolicy
        from test_gae_value_phase import fixture
        S, block = self.block6()
        block['weight_change']['generation'] = 3423  # a fresh head starts at its adoption generation
        model = fixture(games=1)[0]
        six = S.settings({S.KEY: block})
        state = SoftPolicy(model, six, None, generation=3423).finish()  # stamped under weight 6, committed at 3424
        with mock.patch('brass_ai.recovery.config.validate_config', lambda c: c):
            twelve = S.settings(S.set_weight({S.KEY: block}, 12, 3424))
        head = SoftPolicy(model, twelve, state, generation=3424)
        self.assertEqual(head.weight, 12.)
        self.assertEqual(SoftPolicy(model, twelve, head.finish(), generation=3425).weight, 12.)
        with self.assertRaisesRegex(ValueError, 'settings/progress'):
            SoftPolicy(model, dict(twelve, weight_change=dict(twelve['weight_change'], generation=3423)), state, generation=3424)

    def test_the_flag_offers_the_reviewed_weights(self):
        import ast
        import re
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery import training_revision_g3423 as S
        text = (Path(SOURCE_ROOT) / 'scripts' / 'continue-rust-g1491.py').read_text()
        choices = re.search(r"'--soft-policy-weight', type=float, choices=(\([^)]*\))", text).group(1)
        self.assertEqual(ast.literal_eval(choices), S.WEIGHTS)


if __name__ == '__main__':
    unittest.main()
