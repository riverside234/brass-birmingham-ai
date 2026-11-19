"""g3471 expanded search rows: extra search-teacher states, turn-clock targets and era-head lambda targets."""
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from brass_ai.recovery import search_rows as R
from brass_ai.recovery import training_revision_g3140 as G

CAP = .025
FINAL = np.array([.9, .3, .1, -.2])


def original_targets(game, lam):
    """search_value_phase._targets before g3471, verbatim: the reference for root parity."""
    game = sorted(game, key=lambda r: r['index'])
    following = np.asarray(game[-1]['terminal_utility'])
    targets = []
    for i in reversed(range(len(game))):
        next_u = np.asarray(game[i + 1]['u'] if i + 1 < len(game) else game[i]['terminal_utility'])
        cost = next_u - game[i]['u']
        boot = np.asarray(game[i + 1]['r'] if i + 1 < len(game) else game[i]['terminal_utility'])
        following = -cost + (1 - lam) * boot + lam * following
        targets.append(dict(game[i], target=following.tolist(), target_lambda=lam))
    return list(reversed(targets))


def era_round(index):
    era = 'CanalEra' if index <= 32 else 'RailEra'
    return era, (index - 1) % 32 // 4 + 1, (index - 1) % 4 + 1


def table(seed=0, extras=True):
    """One table: 64 roots and extra rows with consistent costs, outcomes u and search values r."""
    rng = np.random.default_rng(seed)
    spent = np.cumsum(rng.choice([0., .0025], size=(66, 4), p=[.9, .1]), 0).clip(max=CAP)
    final_spent = spent[65]
    rows = []
    for t in range(1, 65):
        era, round_, turn = era_round(t); seat = (t - 1) % 4
        rows.append(dict(game_id='teacher/g', index=t, revision=10 * t, seat=seat, mapping=[(s - seat) % 4 for s in range(4)],
                         turn_key=[era, round_, turn, seat], spent=spent[t].tolist(), r=rng.normal(size=4).tolist(),
                         u=(FINAL - (final_spent - spent[t])).tolist(), terminal_utility=FINAL.tolist()))
    if extras:
        for t in range(1, 65):
            mid = (spent[t] + spent[t + 1]) / 2 if t < 64 else (spent[t] + final_spent) / 2
            for k, name in ((3, 'second'), (5, 'sampled')):
                if name == 'second' and rows[t - 1]['turn_key'][:2] == ['CanalEra', 1]: continue
                seat = (t + k) % 4
                rows.append(dict(game_id='teacher/g', kind=name, index=t, revision=10 * t + k, seat=seat,
                                 mapping=[(s - seat) % 4 for s in range(4)], turn_key=list(rows[t - 1]['turn_key']),
                                 spent=mid.tolist(), u=(FINAL - (final_spent - mid)).tolist(), terminal_utility=FINAL.tolist()))
    return rows, final_spent


class Stream(unittest.TestCase):
    def test_the_sample_key_is_the_native_twin_and_the_sample_its_smallest_keys(self):
        self.assertEqual(R.sample_key(77, 1234), 0xc0c9a80706989395)  # cpu_capture.rs the_stream_matches_its_python_twin
        self.assertEqual(R.sample_key((1 << 63) + 5, 9), 0x854aa48166fe6ecf)
        candidates = list(range(100, 400, 3))
        chosen = R.sample(77, candidates, 24)
        self.assertEqual(len(chosen), 24); self.assertEqual(chosen, sorted(chosen))
        worst = max(R.sample_key(77, r) for r in chosen)
        self.assertTrue(all(R.sample_key(77, r) > worst for r in candidates if r not in chosen))
        self.assertEqual(R.sample(77, candidates[:10], 24), candidates[:10])  # fewer candidates: all of them


class SearchValue(unittest.TestCase):
    def test_root_targets_are_unchanged_and_lambda_one_is_the_realized_utility_to_go(self):
        rows, _ = table()
        roots = R.roots(rows)
        for lam in (.8, .95, 1.):
            reference = original_targets(roots, lam)
            self.assertEqual(R.turn_targets(roots, lam), reference)  # an old 64-row table: exactly as before
            mixed = R.turn_targets(rows, lam)
            self.assertEqual(mixed[:64], reference)  # the roots of an expanded table: exactly as before
            self.assertEqual(len(mixed), len(rows))
        for row in R.turn_targets(rows, 1.):
            np.testing.assert_allclose(row['target'], row['u'], rtol=0, atol=1e-12)

    def test_an_extra_state_bootstraps_at_its_turn_s_next_root(self):
        rows, _ = table(); lam = .95
        out = {(R.kind(r), r['index'], r['revision']): r for r in R.turn_targets(rows, lam)}
        root = {r['index']: r for r in R.roots(rows)}
        for x in rows:
            if R.kind(x) == 'turn': continue
            t = x['index']; got = np.asarray(out[(R.kind(x), t, x['revision'])]['target'])
            if t < 64:
                nxt = root[t + 1]; following = np.asarray(out[('turn', t + 1, nxt['revision'])]['target'])
                expected = -(np.asarray(nxt['u']) - x['u']) + (1 - lam) * np.asarray(nxt['r']) + lam * following
            else:
                expected = -(FINAL - np.asarray(x['u'])) + FINAL
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_admission_targets_match_the_fit_targets(self):
        rows, final_spent = table(); reward = dict(insolvency=dict(max_cost=CAP))
        roots, extras = R.roots(rows), [r for r in rows if R.kind(r) != 'turn']
        final = lambda *a: FINAL.copy()
        with mock.patch('brass_ai.recovery.search_value_rows.terminal_utility', final), \
                mock.patch('brass_ai.learning.reward.terminal_utility', final):
            for lam in (.95, 1.):
                admitted = R.complete(roots, extras, {}, final_spent.tolist(), reward, lam)
                fitted = [r for r in R.turn_targets(roots + [dict(x, u=a['u']) for x, a in zip(extras, admitted)], lam)
                          if R.kind(r) != 'turn']
                for a, f, x in zip(admitted, fitted, extras):
                    np.testing.assert_allclose(a['u'], x['u'], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(a['target'], f['target'], rtol=0, atol=1e-12)
                    if lam == 1.: np.testing.assert_allclose(a['target'], x['u'], rtol=0, atol=1e-12)


class Admission(unittest.TestCase):
    def block(self, seed=77):
        from brass_ai.recovery.search_value_rows import state_width
        width = state_width(); rows, _ = table(extras=False)
        roots = [dict(r, generation=3471, model_id='m', reward_id='r', split='train', profile='fast') for r in R.roots(rows)]
        def extra(name, t, k, seat, decision, legal):
            era, round_, turn = era_round(t); root = roots[t - 1]
            return dict(kind=name, index=t, seat=seat, revision=10 * t + k, decision=decision,
                        phase=R.game_phase(era, round_), legal=legal, actions_remaining=1, state=[0.] * width,
                        mapping=[(s - seat) % 4 for s in range(4)], remaining=(CAP - np.asarray(root['spent'])).tolist(),
                        spent=list(root['spent']), turn_key=list(root['turn_key']))
        second = [extra('second', t, 3, roots[t - 1]['seat'], 'action', 4) for t in range(1, 65) if era_round(t)[:2] != ('CanalEra', 1)]
        candidates = {era: [10 * t + k for t in range(1, 65) for k in (5, 7) if era_round(t)[0] == era] for era in R.ERAS}
        sampled = [extra('sampled', r // 10, r % 10, (r // 10) % 4, 'payment', 2)
                   for era in R.ERAS for r in R.sample(seed, candidates[era], 24)]
        block = dict(contract=R.CONTRACT, sampled_per_era=24, second=second, sampled=sampled, candidates=candidates)
        job = dict(search=dict(reward=dict(insolvency=dict(max_cost=CAP))), sampling_seed=seed)
        return block, roots, job

    def test_a_table_s_extra_rows_pass_with_its_lineage(self):
        block, roots, job = self.block()
        rows = R.validate(block, roots, job)
        self.assertEqual(R.counts(rows), dict(second=60, sampled=48))
        self.assertTrue(all(r['game_id'] == 'teacher/g' and r['generation'] == 3471 and r['split'] == 'train' for r in rows))

    def test_a_changed_sample_or_misplaced_row_is_refused(self):
        def refused(edit, message):
            block, roots, job = self.block(); edit(block, roots, job)
            with self.assertRaisesRegex(ValueError, message): R.validate(block, roots, job)
        def swap(block, roots, job):  # the same turn's other candidate, which the stream did not keep
            kept = {x['revision'] for x in block['sampled']}
            sibling = lambda r: r + 2 if r % 10 == 5 else r - 2  # 10t+5 <-> 10t+7
            row = next(x for x in block['sampled'] if sibling(x['revision']) not in kept)
            row['revision'] = sibling(row['revision'])
        refused(swap, 'stream')
        refused(lambda b, r, j: j.update(sampling_seed=78), 'stream')
        refused(lambda b, r, j: b['second'][0].update(revision=b['second'][0]['revision'] + 10), 'outside its turn')
        refused(lambda b, r, j: b['second'].append(copy.deepcopy(b['second'][1])), 'outside its turn or repeated')
        refused(lambda b, r, j: b['second'][0].update(mapping=[0, 1, 2, 3], seat=1), 'mapping')
        refused(lambda b, r, j: b['sampled'][0].update(legal=1), 'two or more')
        refused(lambda b, r, j: b['second'][0].update(spent=[1., 0., 0., 0.]), 'costs')
        first = lambda b, r, j: b['second'].append(dict(copy.deepcopy(b['second'][0]), index=1, revision=13,
                                                        turn_key=list(r[0]['turn_key']), seat=0, mapping=[0, 1, 2, 3]))
        refused(first, 'Second-action row')  # the first Canal round has one action per turn

    def test_admission_appends_the_extra_rows_and_keeps_the_roots(self):
        from brass_ai.recovery import search_value_phase as P
        block, roots, job = self.block()
        job = dict(job, split='train', expected=dict(job_id='j', game_id='teacher/g'),
                   teacher_revision=dict(search_rows=R.JOB))
        published = {}

        class Shards:
            def publish(self, generation, split, rows, *, game_id): published[game_id] = rows
        phase = P.SearchValuePhase.__new__(P.SearchValuePhase)
        phase.buffer, phase.report, phase.label_ages, phase.options = Shards(), {}, [], dict(schedule={})
        phase.machine = mock.Mock(state=dict(lam=.95)); phase.reward_id = 'r'; phase.scoring_maps = False
        completed = [dict(r, u=r['u'], target=r['u']) for r in roots]
        results = [dict(job_id='j', value_rows=[], result={}, final_spent=list(roots[-1]['spent']), search_rows=block, rows=[])]
        with mock.patch.object(P, 'complete_game', lambda *a, **k: copy.deepcopy(completed)), \
                mock.patch('brass_ai.recovery.search_value_rows.terminal_utility', lambda *a: FINAL.copy()), \
                mock.patch('brass_ai.learning.reward.terminal_utility', lambda *a: FINAL.copy()):
            phase.admit(results, [job], 3471)
        stored = published['teacher/g']
        self.assertEqual(R.counts(stored), dict(turn=64, second=60, sampled=48))
        self.assertEqual([r['index'] for r in stored[:64]], list(range(1, 65)))
        self.assertEqual(phase.report['extra_rows'], dict(second=60, sampled=48))
        self.assertEqual(len({R.row_key(r) for r in stored}), len(stored))
        missing = [dict(results[0], search_rows=None)]
        with mock.patch.object(P, 'complete_game', lambda *a, **k: copy.deepcopy(completed)), \
                self.assertRaisesRegex(ValueError, 'lacks its extra rows'):
            phase.admit(missing, [job], 3471)


class Buffers(unittest.TestCase):
    def game(self, name, generation, extras):
        rows, _ = table(extras=extras)
        return [dict(r, game_id=name, generation=generation, split='heldout', reward_id='r', state=[0., 1.],
                     remaining=[CAP] * 4) for r in rows]

    def test_old_64_row_and_expanded_tables_rotate_and_validate_together(self):
        from brass_ai.recovery.search_value_retention import complete, rotating_holdout
        old = self.game('teacher/old', 3460, extras=False); new = self.game('teacher/new', 3471, extras=True)
        self.assertTrue(complete(old) and complete(new))
        self.assertFalse(complete(old[1:]))
        validation, retired, report = rotating_holdout(old + new, 3471, games=2, minimum=1)
        self.assertEqual(len(validation), len(old) + len(new)); self.assertEqual(retired, [])
        self.assertTrue(all(r['target'] == r['u'] for r in validation))  # every row scores its realized outcome
        validation, retired, _ = rotating_holdout(old + new, 3471, games=1, minimum=1)
        self.assertEqual({r['game_id'] for r in validation}, {'teacher/new'})
        self.assertEqual(len(retired), 64)  # the old table joins training whole

    def test_the_buffer_keys_extra_rows_by_revision(self):
        import tempfile
        from pathlib import Path
        from brass_ai.recovery.search_value_rows import ValueBuffer
        with tempfile.TemporaryDirectory() as tmp:
            buffer = ValueBuffer(Path(tmp), 'r')
            rows = [dict(r, split='train') for r in self.game('teacher/new', 3471, extras=True)]
            buffer.publish(3471, 'train', rows, game_id='teacher/new')
            self.assertEqual(len(buffer.rows(3471)), len(rows))
            again = ValueBuffer(Path(tmp), 'r', receipts=buffer.receipts)
            self.assertEqual(R.counts(again.rows(3471)), R.counts(rows))
            with self.assertRaisesRegex(ValueError, 'Duplicate value row'):
                buffer.publish(3471, 'train', rows + rows[-1:], game_id='teacher/new')


def envelope_maps():
    from test_local_scoring import envelope
    e, _ = envelope('teacher/g')
    return e, {era['era']: era for era in e['receipt']['eras']}


def frozen_like(rows, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for horizon in ('canal', 'rail'):
        for kind, n in (('slots', 49), ('links', 69)):
            out[f'{horizon}_{kind}_owner'] = torch.as_tensor(rng.dirichlet(np.ones(5), size=(rows, n)), dtype=torch.float32)
            out[f'{horizon}_{kind}_vp'] = torch.as_tensor(rng.normal(size=(rows, n, 4)), dtype=torch.float32)
    return out


class EraHeads(unittest.TestCase):
    def dataset(self, eras, seat=1, chain=('teacher/g', 1)):
        e, _ = envelope_maps()
        mapping = [(s - seat) % 4 for s in range(4)]
        rows = [{'_local_scoring': dict(envelope=e, era=era)} for era in eras]
        data = [(None, None, chain, SimpleNamespace(stable_to_relative=mapping)) for _ in eras]
        return SimpleNamespace(rows=rows, data=data), mapping

    def test_ordinary_rows_follow_their_seat_chain_and_end_at_the_era_map(self):
        from brass_ai.recovery.local_scoring_targets import map_for
        ds, mapping = self.dataset(['CanalEra'] * 3 + ['RailEra'] * 2)
        frozen = frozen_like(5, 1); lam = .95
        out, report = R.ordinary_corrections(ds, frozen, lam)
        self.assertEqual(out.shape, (5, R.ENTITIES, R.WIDTH)); self.assertEqual(report['runs'], 2)
        z = [R.map_arrays(map_for(ds.rows[i]), mapping)[0] for i in range(5)]
        p = lambda i: R.head_outputs(frozen, [i], [ds.rows[i]['_local_scoring']['era'] == 'CanalEra'])[0]
        c1 = (1 - lam) * (p(2) - z[2]); c0 = (1 - lam) * (p(1) - z[1]) + lam * c1
        np.testing.assert_allclose(out[0].astype(np.float32), c0, atol=2e-3)
        np.testing.assert_allclose(out[1].astype(np.float32), c1, atol=2e-3)
        self.assertFalse(out[2].any())  # the era's last row ends at its actual map
        np.testing.assert_allclose(out[3].astype(np.float32), (1 - lam) * (p(4) - z[4]), atol=2e-3)
        self.assertFalse(out[4].any())
        ones, _ = R.ordinary_corrections(ds, frozen, 1.)
        self.assertFalse(ones.any())  # lambda 1: the actual maps
        # A blended target stays a distribution: one-hot plus half the correction.
        q = z[0][:, :5] + .5 * out[0][:, :5].astype(np.float64)
        self.assertTrue((q >= -1e-3).all()); np.testing.assert_allclose(q.sum(-1), 1., atol=5e-3)

    def test_search_rows_bootstrap_at_the_next_turn_root_through_the_stable_seats(self):
        _, maps = envelope_maps(); lam = .95
        seats = [0, 1, 2, 3]
        rows = [dict(index=t, seat=s, mapping=[(x - s) % 4 for x in range(4)], turn_key=[era, 1, 1, s])
                for t, s, era in ((1, 0, 'CanalEra'), (2, 1, 'CanalEra'), (3, 2, 'RailEra'), (4, 3, 'RailEra'))]
        rows.insert(1, dict(kind='second', index=1, revision=15, seat=2, mapping=[(x - 2) % 4 for x in range(4)],
                            turn_key=['CanalEra', 1, 1, 0]))
        rows.append(dict(kind='sampled', index=2, revision=25, seat=3, mapping=[(x - 3) % 4 for x in range(4)],
                         turn_key=['CanalEra', 1, 2, 1]))
        rng = np.random.default_rng(2)
        frozen = {j: np.concatenate((rng.dirichlet(np.ones(5), size=R.ENTITIES), rng.normal(size=(R.ENTITIES, 4))), -1)
                  for j, r in enumerate(rows) if R.kind(r) == 'turn'}
        out = R.search_corrections([(rows, maps, frozen)], lam)
        stable = {era: R.map_arrays(maps[era], seats)[0] for era in maps}
        at = {r['index']: j for j, r in enumerate(rows) if R.kind(r) == 'turn'}
        root1 = (1 - lam) * (R.to_stable(frozen[at[2]], rows[at[2]]['mapping']) - stable['CanalEra'])
        np.testing.assert_allclose(out[at[1]], R.to_relative(root1, rows[at[1]]['mapping']), atol=1e-6)
        np.testing.assert_allclose(out[1], R.to_relative(root1, rows[1]['mapping']), atol=1e-6)  # its turn root's, own seat
        self.assertFalse(out[at[2]].any()); self.assertFalse(out[-1].any())  # the last Canal turn: its horizon
        rail = (1 - lam) * (R.to_stable(frozen[at[4]], rows[at[4]]['mapping']) - stable['RailEra'])
        np.testing.assert_allclose(out[at[3]], R.to_relative(rail, rows[at[3]]['mapping']), atol=1e-6)
        self.assertFalse(R.search_corrections([(rows, maps, frozen)], 1.).any())
        for j, r in enumerate(rows):  # the frames round-trip
            np.testing.assert_allclose(R.to_relative(R.to_stable(frozen.get(j, out[j]), r['mapping']), r['mapping']),
                                       frozen.get(j, out[j]))

    def test_the_loss_uses_the_blended_target_and_reports_the_actual_one(self):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        from test_gae_value_phase import fixture
        model = fixture(games=1)[0]
        a = auxiliary.Companion(model, v3.default(5), raw=dict(model_id='m'), generation=5)
        rng = np.random.default_rng(3); torch.manual_seed(3)
        owner = torch.randn(2, 49, 5); vp = torch.randn(2, 49, 4)
        classes = rng.integers(0, 5, size=(2, 49)); values = rng.normal(size=(2, 49, 4)).astype(np.float32)
        mask = rng.random((2, 49)) > .2
        plain = a._terms(owner, vp, classes, values, mask, 'cpu')
        zero = a._terms(owner, vp, classes, values, mask, 'cpu', torch.zeros(2, 49, 9))
        for x, y in zip(plain, zero): self.assertTrue(torch.allclose(x.float(), y.float(), atol=1e-6))
        correction = torch.as_tensor(np.concatenate((rng.dirichlet(np.ones(5), size=(2, 49)) - np.eye(5)[classes],
                                                     rng.normal(size=(2, 49, 4))), -1), dtype=torch.float32) * .5
        blended = a._terms(owner, vp, classes, values, mask, 'cpu', correction)
        self.assertTrue(torch.equal(blended[3], plain[3]) and torch.equal(blended[4], plain[4]))  # reported: actual map
        self.assertFalse(torch.allclose(blended[5], plain[5]))


class Phase(unittest.TestCase):
    """Through ppg_g210.fit: ordinary corrections staged from the outputs PPO froze, search rows (roots and extra
    states) with their turn-clock corrections; lambda 1 leaves every era-head target the actual map."""

    def run_phase(self, lam=.95, rows=True):
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
        c[G.KEY] = dict(G.OFF, validation=copy.deepcopy(G.VALIDATION), bootstrap=copy.deepcopy(G.BOOTSTRAP),
                        ownership_stop=copy.deepcopy(G.OWNERSHIP_STOP), search_auxiliary=copy.deepcopy(G.SEARCH_AUX),
                        **({'search_rows': settings} if rows else {}))
        c['local_scoring_auxiliary'] = v3.default(1)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp, mock.patch.object(G, 'SEARCH_ROWS', settings):
            p = TrainingPhases(m, c, root=SOURCE_ROOT, run=Path(tmp))
            p.local_scoring = auxiliary.Companion(m, c['local_scoring_auxiliary'], raw=dict(model_id='m'), generation=1)
            _, data = prepare(m, p, 1, companion=True)
            p.record_frozen(FrozenValues(m, p.critic_dataset, c['update'], p.local_scoring).values)
            sources = [(e.state, int(row['seat']), row['view']['context']['era']) for row, _, _, e in data]
            def game(name, split):
                receipt, result, job = search_table(name)
                turn = roots(name, 1, split, sources[:12], A.table_envelope(receipt, result, job))
                extra = []
                for t, (state, seat, era) in enumerate(sources[12:18], 1):  # extra states in the roots' own eras
                    root = turn[t - 1]; s = (seat + t) % 4
                    extra.append(dict(root, kind='second' if t % 2 else 'sampled', revision=100 + t, seat=s,
                                      mapping=[(x - s) % 4 for x in range(4)], state=np.asarray(state, np.float32)))
                    extra[-1].pop(A.FIELD, None)
                return turn + extra
            buffer = Buffer([game(f'teacher/t{i}', 'train') for i in range(3)] + [game('teacher/h', 'heldout')])
            p.search_auxiliary = dict(settings=copy.deepcopy(G.SEARCH_AUX), tables=A.tables(buffer, 1),
                                      rows=G.search_rows(c))
            r = p.fit(m)
            r['_training'] = p.local_scoring.report()['training']
        return r

    def test_both_row_kinds_train_on_lambda_returns_and_lambda_one_keeps_the_maps(self):
        r = self.run_phase()
        boot = r['buffer']['local_bootstrap']; s = r['search_auxiliary']
        self.assertGreater(boot['rows'], 0); self.assertGreater(boot['mean_owner_l1'], 0.)
        self.assertEqual(boot['share'], .5)
        self.assertEqual(s['train']['kinds'], dict(turn=36, second=9, sampled=9))
        self.assertEqual(s['heldout']['kinds'], dict(turn=12, second=3, sampled=3))
        self.assertGreater(s['lambda_targets']['mean_owner_l1'], 0.); self.assertEqual(s['lambda_targets']['share'], 1.)
        self.assertEqual(s['presented']['steps'], r['ownership_stop']['attempted_head_steps'])
        self.assertEqual(r['ownership_stop']['search']['rows'], 18)  # held-out search rows of every kind, actual maps
        one = self.run_phase(lam=1.)
        self.assertEqual((one['buffer']['local_bootstrap']['mean_owner_l1'], one['search_auxiliary']['lambda_targets']['mean_owner_l1']),
                         (0., 0.))
        # The corrections reach both losses: ordinary rows and search rows train on other targets than at lambda 1.
        key = 'ppg_weighted_auxiliary_loss_sum'
        self.assertNotAlmostEqual(r['_training'][key], one['_training'][key], places=6)
        self.assertNotAlmostEqual(s['weighted_loss'], one['search_auxiliary']['weighted_loss'], places=6)
        base = self.run_phase(rows=False)
        # At lambda 1 the targets are the actual maps: the same losses as without the item.
        self.assertAlmostEqual(one['_training'][key], base['_training'][key], places=4)
        self.assertNotIn('local_bootstrap', base['buffer']); self.assertIsNone(base['search_auxiliary']['lambda_targets'])
        # The ordinary rows and the stop's starting point are the same with or without the lambda targets.
        self.assertEqual(r['attempted_value_presentations'], base['attempted_value_presentations'])
        self.assertEqual(r['ownership_stop']['before'], base['ownership_stop']['before'])


class SoftWeight(unittest.TestCase):
    def config(self):
        from brass_ai.recovery import training_revision_g3423 as S
        block = dict(copy.deepcopy(S.SETTINGS), from_generation=3423)
        return S, {S.KEY: block}

    def test_the_weight_moves_once_from_eight_to_six(self):
        S, config = self.config()
        with mock.patch('brass_ai.recovery.config.validate_config', lambda c: c):
            changed = S.set_weight(config, 6, 3471)
            self.assertEqual(S.set_weight(changed, 6, 3480), changed)
            with self.assertRaisesRegex(ValueError, '8 and 6'): S.set_weight(config, 4, 3471)
        block = changed[S.KEY]
        self.assertEqual((block['weight'], block['weight_change']), (6., dict(generation=3471, before=8.)))
        self.assertEqual(S.previous_settings(block), config[S.KEY])
        self.assertEqual((S.tag(config), S.tag(changed)), ('soft-t4-w8-s1', 'soft-t4-w6-s1'))
        self.assertEqual(S.project(config, changed), changed)
        self.assertIsNone(S.project(changed, config))  # never back
        other = copy.deepcopy(changed); other[S.KEY]['temperature'] = 2.
        self.assertIsNone(S.project(config, other))

    def test_the_head_carries_over_at_its_change_generation_only(self):
        from brass_ai.recovery.soft_policy import SoftPolicy
        from test_gae_value_phase import fixture
        S, config = self.config()
        model = fixture(games=1)[0]
        old = S.settings(config)
        state = SoftPolicy(model, old, None, generation=3423).finish()  # stamped under weight 8, committed at 3424
        with mock.patch('brass_ai.recovery.config.validate_config', lambda c: c):
            new = S.settings(S.set_weight(config, 6, 3424))
        head = SoftPolicy(model, new, state, generation=3424)
        self.assertEqual(head.weight, 6.)
        later = S.settings(dict(config, **{S.KEY: dict(new, weight_change=dict(generation=3423, before=8.))}))
        with self.assertRaisesRegex(ValueError, 'settings/progress'):
            SoftPolicy(model, later, state, generation=3424)  # a weight-8 stamp is only read at the change
        self.assertEqual(SoftPolicy(model, new, head.finish(), generation=3425).weight, 6.)


class Recipe(unittest.TestCase):
    def test_the_item_extends_search_aux_and_the_teacher_jobs(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery import training_revision_g1491 as g1491
        from brass_ai.recovery import training_revision_g3270 as g3270
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from test_teacher_revision_g3270 import before
        from test_training_g3140 import signature
        teacher = g3270.configure(before(), 3270)
        heads = validate_config(G.configure(teacher, G.parse('heads,targets,ownership-stop'), 3318))
        with self.assertRaisesRegex(ValueError, 'search-aux'):
            G.configure(heads, G.parse('search-rows'), 3471)
        aux = validate_config(G.configure(heads, G.parse('search-aux'), 3423))
        if aux['teacher_phase']['search_value'].get('target', {}).get('lambda') != .95:
            aux['teacher_phase']['search_value'].setdefault('target', {})['lambda'] = .95
        adopted = validate_config(G.configure(aux, G.parse('search-rows'), 3471))
        self.assertEqual(G.search_rows(adopted), G.SEARCH_ROWS); self.assertIsNone(G.search_rows(aux))
        self.assertTrue(G.tag(adopted).endswith('-searchaux-searchrows'))
        self.assertTrue(signature_transition(signature(aux), signature(adopted)))
        self.assertFalse(signature_transition(signature(adopted), signature(aux)))
        state = dict(iteration=3471, history=[], signature=signature(adopted))
        jobs, _ = plan_jobs(adopted, state, model_id='m' * 64, export_hash='a' * 64, epoch=3471, ready=True)
        self.assertTrue(jobs and all(j['teacher_revision']['search_rows'] == R.JOB for j in jobs))
        for job in jobs: g1491.validate_job(job)
        bad = copy.deepcopy(jobs[0]); bad['teacher_revision']['search_rows']['sampled_per_era'] = 12
        with self.assertRaisesRegex(ValueError, 'search-row capture'): g1491.validate_job(bad)
        old, _ = plan_jobs(aux, dict(state, signature=signature(aux)), model_id='m' * 64, export_hash='a' * 64, epoch=3471, ready=True)
        self.assertTrue(all('search_rows' not in j['teacher_revision'] for j in old))
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        self.assertTrue(is_on(adopted))


class Native(unittest.TestCase):
    def test_the_cpu_teacher_captures_before_applying_and_sends_the_rows_at_finish(self):
        import tempfile
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.rust_backend.lab_cpu_overlay import apply
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'lab-native'
            apply(target)
            text = (target / 'src/teacher.rs').read_text(encoding='utf-8')
            lib = (target / 'src/lib.rs').read_text(encoding='utf-8')
            self.assertTrue((target / 'src/cpu_capture.rs').is_file())
        self.assertIn('pub mod cpu_capture;', lib)
        self.assertEqual(text.count('crate::cpu_capture::kind('), 1)
        self.assertEqual(text.count('"search_rows":crate::cpu_capture::finish()'), 1)
        # Encoded before the decision is applied (ledger and state are those the decision saw).
        self.assertLess(text.index('crate::cpu_capture::record('), text.index('ledger.chosen(&view, selected);'))
        self.assertNotIn('cpu_capture', (SOURCE_ROOT / 'lab-native/src/teacher.rs').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
