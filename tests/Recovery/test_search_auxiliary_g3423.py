"""g3423 search auxiliary: search-teacher tables train the era heads (recovery/search_auxiliary.py)."""
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
    def test_the_40_tables_are_this_iteration_s_train_tables_and_the_4_leaving_the_held_out_set(self):
        G0 = 3430; games = []
        for g in range(G0 - 12, G0 + 1):
            for t in range(4): games.append(self.game(f'teacher/h{g}-{t}', g, 'heldout', maps=g >= G0 - 8, count=2))
        for g in (G0 - 1, G0):
            for t in range(6): games.append(self.game(f'teacher/t{g}-{t}', g, 'train', maps=t != 5, count=2))
        chosen = A.tables(Buffer(games), G0, heldout_games=32)
        report = chosen['report']
        names = lambda items: sorted(game for game, _, _ in items)
        self.assertEqual(report['rotated_generations'], [G0 - 8])  # the latest 32 move from G-8..G-1 to G-7..G
        self.assertEqual(names(chosen['train']), sorted([f'teacher/t{G0}-{t}' for t in range(5)] + [f'teacher/h{G0 - 8}-{t}' for t in range(4)]))
        self.assertEqual(names(chosen['heldout']), [f'teacher/h{G0}-{t}' for t in range(4)])
        self.assertEqual((report['current_train_games'], report['rotated_games'], report['heldout_games']), (5, 4, 4))
        self.assertEqual(report['without_maps'], dict(current_train=1))
        early = A.tables(Buffer(games), G0 - 1, heldout_games=32)  # G-9 leaves, collected before its maps existed
        self.assertEqual((early['report']['rotated_games'], early['report']['without_maps'].get('rotated')), (0, 4))
        filling = A.tables(Buffer([g for g in games if g[0]['generation'] >= G0 - 3 or g[0]['split'] == 'train']), G0)
        self.assertEqual(filling['report']['candidates']['rotated'], 0)  # nothing leaves before the set holds 32

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
        a.stopped = True
        self.assertIsNone(a.search_loss(self.model, rows, [0]))

    def test_the_stop_scores_the_search_tables_and_judges_the_search_interval_with_them(self):
        a = self.companion(); rows = self.rows(); ids = list(range(len(self.data)))
        plain = S.OwnershipStop(a, G.OWNERSHIP_STOP, self.model, self.ds, ids, self.cfg['update'])
        stop = S.OwnershipStop(a, G.OWNERSHIP_STOP, self.model, self.ds, ids, self.cfg['update'], search=rows)
        self.assertEqual(plain.before, stop.before)  # the ordinary criterion is unchanged
        self.assertIsNone(plain.search_before); self.assertGreater(stop.search_before['owner_ce'], 0)
        initial = copy.deepcopy(a.heads.state_dict())
        worse = lambda value: dict(stop.search_current, owner_ce=stop.search_current['owner_ce'] + value)
        with mock.patch.object(S.OwnershipStop, 'measure', lambda self: dict(self.current, owner_ce=self.current['owner_ce'] - 1e-3)):
            with mock.patch.object(S.OwnershipStop, 'measure_search', lambda self: worse(1.)):
                stop.boundary(16, 'kept')  # no search batch in this interval: the search CE is reported only
                self.assertTrue(stop.running); self.assertFalse(stop.intervals[-1]['search_batch'])
                self.assertIsNone(stop.intervals[-1]['stopped_by'])
                stop.search_pending = True; kept = copy.deepcopy(a.heads.state_dict())
                stop.boundary(32, 'kept')
        self.assertFalse(stop.running); self.assertEqual(stop.reason, 'search_heldout_ce_increased')
        self.assertEqual(stop.intervals[-1]['stopped_by'], 'search'); self.assertTrue(a.stopped)
        for k, v in kept.items(): self.assertTrue(torch.equal(v, a.heads.state_dict()[k]), k)
        record = stop.finish()
        self.assertEqual(record['search']['rows'], rows.rows); self.assertIn('owner_ce', record['search']['before'])
        self.assertNotIn('search', plain.finish())
        self.assertTrue(all(torch.equal(v, initial[k]) for k, v in a.heads.state_dict().items()))


class Phase(unittest.TestCase):
    """Through ppg_g210.fit: one batch of search roots in the first step, its tables in the stop."""

    def run_phase(self, with_search=True, maps=True):
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
            r = p.fit(m)
            local = p.local_scoring.report(); local['_games'] = {t[0] for t in p.local_scoring.attempted}
        return r, local

    def test_the_first_step_adds_the_search_batch_and_keeps_the_ordinary_rows(self):
        r, local = self.run_phase()
        s = r['search_auxiliary']
        self.assertTrue(s['ready']); self.assertEqual(s['step'], 1); self.assertIsNone(s['reason'])
        self.assertEqual((s['train']['rows'], s['train']['games'], s['heldout']['rows'], s['heldout']['games']), (36, 3, 12, 1))
        self.assertGreater(s['weighted_loss'], 0.)
        self.assertGreater(local['training']['ppg_search_weighted_auxiliary_loss_sum'], 0.)
        stop = r['ownership_stop']
        self.assertTrue(stop['intervals'][0]['search_batch'])
        self.assertTrue(all(not i['search_batch'] for i in stop['intervals'][1:]))
        self.assertEqual(s['heads_kept'], stop['intervals'][0]['accepted'])
        self.assertEqual(s['trunk_kept'], r['interval_reports'][0]['accepted'])
        self.assertEqual(stop['search']['rows'], 12)
        base, base_local = self.run_phase(with_search=False)
        self.assertNotIn('search_auxiliary', base)
        # The ordinary rows of every step are the same: same presentations and the same ordinary head accounting.
        self.assertEqual(r['attempted_value_presentations'], base['attempted_value_presentations'])
        self.assertFalse(any(g.startswith('teacher/') for g in local['_games']))  # search rows are counted apart
        self.assertEqual(local['_games'], base_local['_games'])
        self.assertNotIn('search', base['ownership_stop'])

    def test_without_maps_the_batch_is_skipped(self):
        r, _ = self.run_phase(maps=False)
        s = r['search_auxiliary']
        self.assertEqual((s['ready'], s['reason'], s['step']), (False, 'no_search_maps', None))
        self.assertNotIn('search', r['ownership_stop'])
        self.assertEqual(s['selection']['without_maps'], dict(current_train=3, heldout=1))


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
