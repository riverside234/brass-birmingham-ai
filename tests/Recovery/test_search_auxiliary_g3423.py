"""g3423 search auxiliary: search-teacher tables train the era heads (recovery/search_auxiliary.py)."""
import contextlib
import copy
import unittest
from collections import defaultdict
from unittest import mock

import numpy as np
import torch

from brass_ai.recovery import local_scoring_auxiliary as auxiliary
from brass_ai.recovery import local_scoring_targets as targets
from brass_ai.recovery import local_scoring_v3 as v3
from brass_ai.recovery import ownership_stop as S
from brass_ai.recovery import search_auxiliary as A
from brass_ai.recovery import training_revision_g3140 as G
from brass_ai.recovery import updates


def table(game='teacher/x'):
    """A native-producer receipt of one whole table, its result and job."""
    from test_local_scoring import envelope
    e, g = envelope(game)
    receipt = dict(copy.deepcopy(e['receipt']), producer='rust-local-scoring/1.0.0')
    return receipt, g['result'], dict(expected=dict(game_id=game))


def roots(game, generation, split, sources, packed=None):
    """Played roots of one table from (state, seat, era) sources; the map, if any, on the first root."""
    rows = []
    for index, (state, seat, era) in enumerate(sources, 1):
        rows.append(dict(game_id=game, generation=generation, split=split, reward_id='r', index=index, seat=seat,
                         mapping=[(s - seat) % 4 for s in range(4)], turn_key=[era, 1, seat], state=np.asarray(state, np.float32)))
    if packed is not None: rows[0][A.FIELD] = packed
    return rows


class Buffer:
    """The parts of search_value_rows.ValueBuffer that `tables` reads."""
    def __init__(self, games):
        self.reward_id = 'r'; self.windows = dict(train=16, heldout=20); self.receipts = []; self.shards = {}
        for rows in games:
            r = rows[0]; name = f"{r['generation']}-{r['split']}-{r['game_id']}"
            self.receipts.append(dict(file=name, generation=r['generation'], split=r['split'], reward_id='r', game_id=r['game_id']))
            self.shards[name] = rows

    def _read(self, receipt): return self.shards[receipt['file']]


class Maps(unittest.TestCase):
    def test_the_finish_receipt_is_audited_sealed_and_bound_to_its_table(self):
        receipt, result, job = table()
        packed = A.table_envelope(receipt, result, job)
        envelope = A.read_envelope(packed, 'teacher/x')
        self.assertEqual(envelope['receipt'], receipt)
        self.assertEqual(envelope['sha256'], targets.seal(envelope))
        self.assertEqual([e['era'] for e in envelope['receipt']['eras']], ['CanalEra', 'RailEra'])
        with self.assertRaisesRegex(ValueError, 'differs from its table'):
            A.read_envelope(packed, 'teacher/other')
        for edit in ('total', 'era', 'producer', 'final'):
            bad = copy.deepcopy(receipt); other = copy.deepcopy(result)
            if edit == 'total': bad['eras'][0]['audit']['industry'][0] += 1
            elif edit == 'era': bad['eras'].pop(0)
            elif edit == 'producer': bad['producer'] = 'local-scoring-host/1.0.0'
            else: other['seats'][0]['vp'] += 1
            with self.assertRaises(ValueError, msg=edit): A.table_envelope(bad, other, job)
        changed = copy.deepcopy(packed); changed['receipt']['eras'][0]['slots'][0][0] = (changed['receipt']['eras'][0]['slots'][0][0] + 1) % 5
        with self.assertRaisesRegex(ValueError, 'changed'): A.read_envelope(changed, 'teacher/x')

    def test_the_ordinary_receipt_check_is_unchanged(self):
        from test_local_scoring import envelope
        e, g = envelope(); targets.validate(e, g)
        bad = copy.deepcopy(e); bad['receipt']['final_vp'] = [v + (s == 0) for s, v in enumerate(e['receipt']['final_vp'])]; bad['sha256'] = targets.seal(bad)  # a new list: the fixture shares it with the last era's audit
        with self.assertRaisesRegex(ValueError, 'Whole-game local scoring'): targets.validate(bad, g)

    def test_admission_keeps_the_map_once_on_the_first_root(self):
        from brass_ai.recovery import search_value_phase as P
        receipt, result, job = table('teacher/a')
        job = dict(job, split='train', search=dict(reward={}))
        job['expected']['job_id'] = 'j'
        published = {}

        class Shards:
            def publish(self, generation, split, rows, *, game_id): published[game_id] = copy.deepcopy(rows)

        phase = P.SearchValuePhase.__new__(P.SearchValuePhase)
        phase.buffer, phase.report, phase.label_ages, phase.options = Shards(), {}, [], dict(schedule={})
        phase.machine = mock.Mock(state=dict(lam=.95)); phase.reward_id = 'r'
        rows = [dict(index=i) for i in range(1, 5)]
        results = [dict(job_id='j', value_rows=[], result=result, final_spent=[0.] * 4, scoring=receipt, rows=[])]
        with mock.patch.object(P, 'complete_game', lambda *a, **k: copy.deepcopy(rows)), \
                mock.patch('brass_ai.learning.reward.terminal_utility', lambda *a: np.zeros(4)):
            phase.scoring_maps = False; phase.admit(results, [job], 3423)
            self.assertNotIn(A.FIELD, published['teacher/a'][0])
            self.assertNotIn('scoring_maps', phase.report)
            phase.scoring_maps = True; phase.admit(results, [job], 3423)
        stored = published['teacher/a']
        self.assertEqual([A.FIELD in r for r in stored], [True, False, False, False])
        self.assertEqual(A.read_envelope(stored[0][A.FIELD], 'teacher/a')['receipt'], receipt)
        self.assertEqual(phase.report['scoring_maps'], dict(tables=1, with_maps=1))


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from test_gae_value_phase import fixture
        from test_local_scoring import annotate
        cls.model, _, cls.cfg, (_, data) = fixture(games=2); annotate(data)
        cls.ds = updates.Dataset(data, 'cpu'); cls.data = data
        cls.sources = [(e.state, int(row['seat']), row['view']['context']['era']) for row, _, _, e in data]

    def companion(self, generation=5):
        return auxiliary.Companion(self.model, v3.default(generation), raw=dict(model_id='m'), generation=generation)

    def game(self, name, generation, split, *, maps=True, count=8):
        receipt, result, job = table(name)
        packed = A.table_envelope(receipt, result, job) if maps else None
        return roots(name, generation, split, self.sources[:count], packed)


class Selection(Fixture):
    def test_train_is_two_iterations_of_search_value_training_tables_and_held_out_its_validation_set(self):
        A0 = 3424; G0 = A0 + 8; games = []  # maps from adoption at A0 on
        for g in range(G0 - 19, G0 + 1):
            for t in range(4): games.append(self.game(f'teacher/h{g}-{t}', g, 'heldout', maps=g >= A0, count=2))
        for g in range(G0 - 4, G0 + 1):
            for t in range(6): games.append(self.game(f'teacher/t{g}-{t}', g, 'train', maps=t != 5, count=2))
        buffer = Buffer(games); names = lambda items: sorted(game for game, _, _ in items)
        chosen = A.tables(buffer, G0, heldout_games=32, iterations=2); report = chosen['report']
        # Train: the train tables of G-1 and G, and the held-out tables that left the latest 32 in those two
        # iterations (G-9 and G-8; G-9 was collected before maps existed).
        self.assertEqual(names(chosen['train']), sorted([f'teacher/t{g}-{t}' for g in (G0 - 1, G0) for t in range(5)]
                                                        + [f'teacher/h{G0 - 8}-{t}' for t in range(4)]))
        self.assertEqual((report['train_generations'], report['rotated_generations']), ([G0 - 1, G0], [G0 - 8]))
        self.assertEqual(report['candidates'], dict(train=12, rotated=8, heldout=32))
        self.assertEqual(report['without_maps'], dict(train=2, rotated=4))
        # Held-out: the search-value validation set, the latest 32 (G-7..G), never in training.
        self.assertEqual(names(chosen['heldout']), sorted(f'teacher/h{g}-{t}' for g in range(G0 - 7, G0 + 1) for t in range(4)))
        self.assertFalse(set(names(chosen['train'])) & set(names(chosen['heldout'])))
        # After adoption the held-out set grows by the 4 new tables with maps each iteration: 4, 8, ..., 32.
        grown = [A.tables(buffer, g, heldout_games=32)['report']['heldout_games'] for g in range(A0, G0 + 1)]
        self.assertEqual(grown, [4, 8, 12, 16, 20, 24, 28, 32, 32])
        # The recipe's four iterations: train tables of G-3..G, and the tables that left in those iterations
        # (G-11..G-8, of which only G-8 has maps).
        self.assertEqual(G.SEARCH_AUX['train_iterations'], 4)
        four = A.tables(buffer, G0, heldout_games=32)['report']
        self.assertEqual((four['iterations'], four['train_generations'], four['rotated_generations']),
                         (4, list(range(G0 - 3, G0 + 1)), [G0 - 8]))
        self.assertEqual((four['candidates'], four['without_maps'], four['train_games'], four['rotated_games']),
                         (dict(train=24, rotated=16, heldout=32), dict(train=4, rotated=12), 20, 4))
        one = A.tables(buffer, G0, heldout_games=32, iterations=1)['report']
        self.assertEqual((one['train_generations'], one['rotated_generations'], one['train_games']), ([G0], [G0 - 8], 5))

    def test_the_held_out_order_is_the_search_value_phase_s_own(self):
        from brass_ai.recovery.search_value_retention import rotating_holdout
        held = defaultdict(list)
        for g, name in [(5, 'b'), (5, 'a'), (3, 'z'), (4, 'c'), (6, 'd'), (6, 'a2')]:
            held[name] = [dict(game_id=name, generation=g, split='heldout', index=i, u=[0.] * 4) for i in range(1, 65)]
        rows = [r for game in held.values() for r in game]
        for count in (2, 3, 6):
            self.assertEqual(A.latest(held, 6, count), rotating_holdout(rows, 6, games=count, minimum=1)[2]['games'])


class Mixing(Fixture):
    def selection(self, rows):
        g = self.game('teacher/a', 1, 'train', count=rows); h = self.game('teacher/h', 1, 'heldout', count=4)
        pack = lambda x: (x[0]['game_id'], x, A.read_envelope(x[0][A.FIELD], x[0]['game_id']))
        return dict(train=[pack(g)], heldout=[pack(h)], report=dict(generation=7))

    def test_each_epoch_spreads_every_search_root_once_over_its_batches(self):
        search = A.SearchAuxiliary(self.selection(11), 'cpu', G.SEARCH_AUX, batches=4)
        before = torch.get_rng_state()
        epochs = [[search.share(epoch, p) for p in range(4)] for epoch in range(3)]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))  # the ordinary batches' generator is untouched
        for shares in epochs:
            self.assertEqual(sorted(i for s in shares for i in s), list(range(11)))  # each root once per epoch
            self.assertEqual(sorted(len(s) for s in shares), [2, 3, 3, 3])  # an even share per batch
        self.assertNotEqual(epochs[0], epochs[1])  # a fresh permutation each epoch
        self.assertEqual(epochs[0], [A.SearchAuxiliary(self.selection(11), 'cpu', G.SEARCH_AUX, batches=4).share(0, p) for p in range(4)])
        with self.assertRaisesRegex(ValueError, 'outside the epoch'): search.share(0, 4)

    def test_each_root_reads_its_era_s_end_map_through_its_own_seat_mapping(self):
        rows = A.SearchRows([(g[0]['game_id'], g, A.read_envelope(g[0][A.FIELD], g[0]['game_id']))
                             for g in [self.game('teacher/a', 1, 'train')]], 'cpu')
        local = self.companion()
        for i, (target, key, token, mapping) in enumerate(rows.meta):
            row = rows.envelopes[i]; seat = token[3]; era = token[1]
            self.assertEqual(target['era'], era); self.assertEqual(key[1:3], (era, seat))
            classes, _, mask = local.target_arrays([rows.meta[i]], 'slots', 49)
            expected = [0 if r['owner'] == 0 else (r['owner'] - 1 - seat) % 4 + 1 for r in target['slots']]
            self.assertEqual(classes[0].tolist(), expected)
            self.assertTrue(mask[0].all())
        bad = self.game('teacher/b', 1, 'train'); bad[0]['mapping'] = [0, 1, 2, 3] if bad[0]['seat'] else [1, 2, 3, 0]
        with self.assertRaisesRegex(ValueError, 'seat mapping'):
            A.SearchRows([('teacher/b', bad, A.read_envelope(bad[0][A.FIELD], 'teacher/b'))], 'cpu')


class Companion(Fixture):
    def rows(self, count=8):
        g = self.game('teacher/a', 1, 'train', count=count)
        return A.SearchRows([('teacher/a', g, A.read_envelope(g[0][A.FIELD], 'teacher/a'))], 'cpu')

    def test_search_rows_add_only_era_head_losses_and_are_counted_apart(self):
        a = self.companion(); rows = self.rows()
        ix, x, _ = self.ds.batch(list(range(len(self.data))))
        a.begin('ppg'); self.model.zero_grad(set_to_none=True)
        _, ordinary = a.forward(self.model, x, self.ds, ix)
        counters = copy.deepcopy((a.attempted, a.batch_used, a.batch_fitting, a.map_support, a.map_masked))
        era = {k: v.clone() for k, v in a.era_stats.items()}
        loss = a.search_loss(self.model, rows, list(range(rows.rows)))
        self.assertEqual(loss.shape, (rows.rows,)); self.assertTrue(bool((loss > 0).all()))
        self.assertEqual(counters, (a.attempted, a.batch_used, a.batch_fitting, a.map_support, a.map_masked))
        self.assertEqual(era.keys(), a.era_stats.keys())
        for k, v in era.items(): self.assertTrue(torch.equal(v, a.era_stats[k]), k)
        self.assertTrue(a.pending); self.assertIn('CanalEra', a.search_report())
        loss.sum().backward()
        self.assertTrue(any(p.grad is not None for p in a.heads.parameters()))
        for name, p in self.model.named_parameters():
            if name.startswith(('outcomes.', 'utility', 'policy_scorer.', 'action_encoder.')):
                self.assertIsNone(p.grad, name)  # no value, utility or policy term
        # The heads' own trunk-gradient scale applies: none reaches the trunk while it is 0 (head-only warm-up).
        trunk = [p.grad for n, p in self.model.named_parameters() if n.startswith('state_encoder.')]
        self.assertEqual(any(g is not None and bool(g.abs().sum() > 0) for g in trunk), a.scale > 0)
        # A heads-only step: the trunk gets nothing, the heads still do.
        self.model.zero_grad(set_to_none=True); a.heads.zero_grad(set_to_none=True)
        a.search_loss(self.model, rows, list(range(rows.rows)), frozen=True).sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(any(p.grad is not None for p in a.heads.parameters()))
        a.stopped = True
        self.assertIsNone(a.search_loss(self.model, rows, [0]))

    def test_the_stop_pools_the_search_roots_with_the_ordinary_rows_in_one_criterion(self):
        a = self.companion(); rows = self.rows(); ids = list(range(len(self.data)))
        plain = S.OwnershipStop(a, G.OWNERSHIP_STOP, self.model, self.ds, ids, self.cfg['update'])
        stop = S.OwnershipStop(a, G.OWNERSHIP_STOP, self.model, self.ds, ids, self.cfg['update'], search=rows)
        self.assertEqual(plain.before, stop.before)  # before/after stay the ordinary rows' CE
        self.assertIsNone(plain.search_before); self.assertGreater(stop.search_before['owner_ce'], 0)
        self.assertEqual(plain.criterion(plain.before, None), plain.before['owner_ce'])  # without search: unchanged
        W, Ws = stop.weight, stop.search_weight
        self.assertEqual((W, Ws), (float(self.ds.critic_weights[[i for p in stop.parts for i in p]].sum()), float(rows.rows)))
        pooled = (stop.before['owner_ce'] * W + stop.search_before['owner_ce'] * Ws) / (W + Ws)
        self.assertAlmostEqual(stop.criterion(stop.before, stop.search_before), pooled)
        initial = copy.deepcopy(a.heads.state_dict())
        # Ordinary CE up a little, search CE down more, weighted by rows: the pooled CE falls, so the interval stays.
        up, down = 1e-3, 2e-3 * W / Ws
        shift = lambda current, d: dict(current, owner_ce=current['owner_ce'] + d)
        with mock.patch.object(S.OwnershipStop, 'measure', lambda self: shift(self.current, up)), \
                mock.patch.object(S.OwnershipStop, 'measure_search', lambda self: shift(self.search_current, -down)):
            stop.boundary(16, 'kept')
        self.assertTrue(stop.running); self.assertTrue(stop.intervals[-1]['accepted'])
        self.assertLess(stop.intervals[-1]['criterion_attempted'], stop.intervals[-1]['criterion_before'])
        self.assertGreater(stop.intervals[-1]['owner_ce_attempted'], stop.intervals[-1]['owner_ce_before'])
        # Ordinary CE down a little, search CE up more: the pooled CE rises, so the heads go back.
        kept = copy.deepcopy(a.heads.state_dict())
        with mock.patch.object(S.OwnershipStop, 'measure', lambda self: shift(self.current, -up)), \
                mock.patch.object(S.OwnershipStop, 'measure_search', lambda self: shift(self.search_current, down)):
            stop.boundary(32, 'kept')
        self.assertFalse(stop.running); self.assertEqual(stop.reason, 'heldout_ce_increased'); self.assertTrue(a.stopped)
        for k, v in kept.items(): self.assertTrue(torch.equal(v, a.heads.state_dict()[k]), k)
        record = stop.finish()
        self.assertEqual(record['search']['rows'], rows.rows); self.assertIn('owner_ce', record['search']['before'])
        self.assertAlmostEqual(record['criterion']['before'], pooled); self.assertAlmostEqual(record['criterion']['search_share'], Ws / (W + Ws))
        self.assertNotIn('search', plain.finish()); self.assertNotIn('criterion', plain.finish())
        self.assertTrue(all(torch.equal(v, initial[k]) for k, v in a.heads.state_dict().items()))


class Phase(unittest.TestCase):
    """Through ppg_g210.fit: search roots mixed into every head step, their validation tables pooled in the stop."""

    def run_phase(self, with_search=True, maps=True, value_stop=False):
        import tempfile
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery import ppg_g210
        from brass_ai.recovery.actor_value import TrainingPhases
        from test_ppg_g210 import config, prepare
        torch.manual_seed(3); torch.set_num_threads(2)
        m, opt, c = config()
        c['value_phase'] = dict(ppg_g210.KL_STOP_DEFAULT, batch_size=4, clone_rows=64, clone_batch_size=8)
        c[G.KEY] = dict(G.OFF, validation=copy.deepcopy(G.VALIDATION), ownership_stop=copy.deepcopy(G.OWNERSHIP_STOP),
                        search_auxiliary=copy.deepcopy(G.SEARCH_AUX))
        c['local_scoring_auxiliary'] = v3.default(1)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            p = TrainingPhases(m, c, root=SOURCE_ROOT, run=Path(tmp))
            p.local_scoring = auxiliary.Companion(m, c['local_scoring_auxiliary'], raw=dict(model_id='m'), generation=1)
            _, data = prepare(m, p, 1, companion=True)
            sources = [(e.state, int(row['seat']), row['view']['context']['era']) for row, _, _, e in data]
            def game(name, split):
                receipt, result, job = table(name)
                return roots(name, 1, split, sources[:12], A.table_envelope(receipt, result, job) if maps else None)
            buffer = Buffer([game(f'teacher/t{i}', 'train') for i in range(3)] + [game('teacher/h', 'heldout')])
            p.search_auxiliary = dict(settings=copy.deepcopy(G.SEARCH_AUX), tables=A.tables(buffer, 1)) if with_search else None
            from brass_ai.recovery import value_phase as V
            metrics = V.metrics
            def stopping(*args, **kwargs):  # the value check rejects its first interval: the heads go on alone
                result = metrics(*args, **kwargs)
                if kwargs.get('evidence'): result['loss'] = 1. + .1 * stopping.calls; stopping.calls += 1
                return result
            stopping.calls = 0
            with mock.patch.object(V, 'metrics', stopping) if value_stop else contextlib.nullcontext():
                r = p.fit(m)
            local = p.local_scoring.report(); local['_games'] = {t[0] for t in p.local_scoring.attempted}
        return r, local

    def check_mixed(self, r):
        s = r['search_auxiliary']; stop = r['ownership_stop']; shown = s['presented']
        self.assertTrue(s['ready']); self.assertIsNone(s['reason'])
        self.assertEqual((s['train']['rows'], s['train']['games'], s['heldout']['rows'], s['heldout']['games']), (36, 3, 12, 1))
        # Every step the heads attempted carried its even share of the 36 search roots.
        self.assertEqual(shown['steps'], stop['attempted_head_steps'])
        n, b = 36, shown['batches_per_epoch']
        self.assertEqual((shown['rows_per_step']['min'], shown['rows_per_step']['max']), (n // b, -(-n // b)) if n % b else (n // b,) * 2)
        self.assertGreater(s['weighted_loss'], 0.)
        self.assertEqual(stop['search']['rows'], 12)
        self.assertTrue(all('criterion_attempted' in i for i in stop['intervals']))
        self.assertLessEqual(s['kept']['heads_rows'], shown['rows'])
        return s, stop

    def test_search_roots_join_every_head_step_and_the_ordinary_rows_are_unchanged(self):
        r, local = self.run_phase()
        s, stop = self.check_mixed(r)
        self.assertEqual(s['presented']['heads_only_rows'] > 0, stop['head_only_steps'] > 0)
        self.assertGreater(local['training']['ppg_search_weighted_auxiliary_loss_sum'], 0.)
        base, base_local = self.run_phase(with_search=False)
        self.assertNotIn('search_auxiliary', base)
        # The ordinary rows of every step are the same: same presentations and the same ordinary head accounting.
        self.assertEqual(r['attempted_value_presentations'], base['attempted_value_presentations'])
        self.assertEqual(r['attempted_steps'], base['attempted_steps'])
        self.assertFalse(any(g.startswith('teacher/') for g in local['_games']))  # search rows are counted apart
        self.assertEqual(local['_games'], base_local['_games'])
        self.assertNotIn('search', base['ownership_stop']); self.assertNotIn('criterion', base['ownership_stop'])

    def test_heads_only_steps_carry_search_roots_with_the_trunk_frozen(self):
        r, _ = self.run_phase(value_stop=True)
        s, stop = self.check_mixed(r)
        self.assertGreater(stop['head_only_steps'], 0)
        self.assertGreater(s['presented']['heads_only_rows'], 0)
        self.assertEqual(s['kept']['trunk_rows'], 0)  # the only joint interval was rejected by the value check

    def test_without_maps_the_rows_are_skipped(self):
        r, _ = self.run_phase(maps=False)
        s = r['search_auxiliary']
        self.assertEqual((s['ready'], s['reason'], s['presented']['steps']), (False, 'no_search_maps', 0))
        self.assertNotIn('search', r['ownership_stop'])
        self.assertEqual(s['selection']['without_maps'], dict(train=3, heldout=1))


class Recipe(unittest.TestCase):
    def test_the_item_needs_the_heads_the_stop_and_the_teacher_revision(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from brass_ai.recovery import training_revision_g3270 as g3270
        from test_teacher_revision_g3270 import before
        from test_training_g3140 import signature
        # The run's order: teacher revision (g3270), heads and targets (g3275), ownership stop (g3318).
        teacher = g3270.configure(before(), 3270)
        heads = validate_config(G.configure(teacher, G.parse('heads,targets'), 3275))
        with self.assertRaisesRegex(ValueError, 'ownership stop'):
            G.configure(heads, G.parse('search-aux'), 3423)
        revised = validate_config(G.configure(heads, G.parse('ownership-stop'), 3318))
        with self.assertRaisesRegex(ValueError, 'g3270'):
            G.configure(g3270.previous(revised), G.parse('search-aux'), 3423)
        adopted = validate_config(G.configure(revised, G.parse('search-aux'), 3423))
        self.assertEqual(G.search_auxiliary(adopted), G.SEARCH_AUX); self.assertIsNone(G.search_auxiliary(revised))
        self.assertEqual({k for k in adopted if adopted[k] != revised[k]}, {G.KEY})
        self.assertEqual(G.configure(adopted, G.parse('search-aux'), 3424), adopted)
        self.assertTrue(G.tag(adopted).endswith('-searchaux'))
        self.assertTrue(signature_transition(signature(revised), signature(adopted)))
        self.assertFalse(signature_transition(signature(adopted), signature(revised)))
        loose = copy.deepcopy(adopted); loose[G.KEY].pop('ownership_stop')
        with self.assertRaisesRegex(ValueError, 'Invalid g3140'): G.validate(loose[G.KEY])
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        self.assertTrue(is_on(adopted))


class Native(unittest.TestCase):
    def test_the_cpu_teacher_sends_the_scoring_maps_with_its_finish_message(self):
        import tempfile
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.rust_backend.lab_cpu_overlay import apply
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'lab-native'
            apply(target)
            text = (target / 'src/teacher.rs').read_text(encoding='utf-8')
        original = (SOURCE_ROOT / 'lab-native/src/teacher.rs').read_text(encoding='utf-8')
        self.assertNotIn('"scoring"', original)  # lab-native itself is unchanged
        self.assertEqual(text.count('"scoring":recorder.finish(&state.export())?'), 1)
        self.assertIn('let mut recorder = Recorder::new(&initial);', text)


if __name__ == '__main__':
    unittest.main()
