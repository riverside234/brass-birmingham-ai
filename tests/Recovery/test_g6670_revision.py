"""g6670 revision: the paused search-value head cut, the PL rating on the latest 64 checkpoints and 40,960 games, the
PPG beta floor back at .25, and PPG clip 3.0 with a one-time Adam rescale (PPO clip unchanged)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_g210 as P
from brass_ai.recovery import training_revision_g5466 as g5466
from brass_ai.recovery import training_revision_g5685 as g5685
from brass_ai.recovery import training_revision_g6230 as g6230
from brass_ai.recovery import training_revision_g6670 as R

# g6640-g6669 of the g6670 export (value_phase.gradient_clipping): iteration, steps, steps above 2.5, mean and P98.
LOGGED = [
    (6640, 54, 3, 1.340189, 2.625365), (6641, 16, 1, 1.264831, 2.446984), (6642, 54, 3, 1.420605, 2.796157),
    (6643, 16, 0, 1.276925, 1.927246), (6644, 16, 1, 1.66243, 3.007095), (6645, 16, 0, 1.594089, 2.310273),
    (6646, 16, 2, 1.948648, 2.760291), (6647, 48, 3, 1.395133, 2.599986), (6648, 32, 2, 1.444235, 2.762652),
    (6649, 16, 0, 1.507128, 2.440189), (6650, 32, 1, 1.324793, 2.461918), (6651, 16, 1, 1.622649, 3.290835),
    (6652, 16, 0, 1.654653, 2.180009), (6653, 32, 0, 1.324101, 2.33524), (6654, 32, 2, 1.520118, 2.935388),
    (6655, 32, 4, 1.732669, 3.282097), (6656, 54, 8, 1.734602, 3.771437), (6657, 16, 0, 1.358466, 2.32485),
    (6658, 16, 0, 1.199433, 2.034872), (6659, 54, 1, 1.323831, 2.453402), (6660, 48, 0, 1.176137, 1.992616),
    (6661, 32, 2, 1.57696, 2.820296), (6662, 48, 1, 1.331801, 2.488782), (6663, 54, 3, 1.472928, 3.086939),
    (6664, 16, 0, 1.311104, 2.068104), (6665, 54, 2, 1.40738, 2.955283), (6666, 54, 5, 1.539393, 2.860958),
    (6667, 16, 0, 1.363242, 1.910518), (6668, 32, 1, 1.610581, 2.684648), (6669, 16, 3, 1.783163, 2.885765),
]
NORMS = [dict(iteration=i, steps=n, clipped_steps=c, mean=m, p98=q) for i, n, c, m, q in LOGGED]


def before_recipe():
    """The run's recipe shape at g6670: g6230 on g5685, with the g3658 .125 floor and the g4475 head pause."""
    from brass_ai.recovery import training_revision_g3658 as g3658
    from brass_ai.recovery import training_revision_g4475 as g4475
    from brass_ai.recovery.config import validate_config
    from test_soft_head_revision_g6230 import NORMS as G6230_NORMS, before_recipe as g5685_recipe
    base = g4475.configure(validate_config(g3658.configure(g5685_recipe())), 4475)
    return g6230.configure(base, 6230, gradient_norms=G6230_NORMS)


def signature(config):
    from test_training_g3140 import signature as sign
    return sign(config)


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before_recipe(); cls.after = R.configure(cls.before, 6670, gradient_norms=NORMS)

    def test_one_block_the_ppg_floor_and_clip_and_nothing_else(self):
        from brass_ai.recovery import training_revision_g3658 as g3658
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        self.assertEqual(changed, ['g6670_revision', 'value_phase'])
        self.assertEqual({k for k in before['value_phase'] if before['value_phase'][k] != after['value_phase'][k]}, {'beta_clone_min'})
        self.assertEqual((before['value_phase']['beta_clone_min'], after['value_phase']['beta_clone_min']), (.125, .25))
        self.assertEqual((R.ppg_grad_clip(before), R.ppg_grad_clip(after)), (2.5, 3.))
        self.assertEqual(after['update']['grad_clip'], before['update']['grad_clip'])  # PPO keeps 6.5
        self.assertEqual(after['update']['grad_clip'], 6.5)
        self.assertTrue(g3658.applied(before) and g3658.applied(after))  # the same floor lineage
        self.assertEqual(R.previous(after), before)
        self.assertEqual(R.configure(after, 6671, gradient_norms=None), after)  # adopted once
        self.assertTrue(is_on(after)); self.assertEqual(R.tag(after), 'svcut-pl64x40960-ppgfloor025-ppgclip30')
        self.assertTrue(R.search_head_cut(after)); self.assertFalse(R.search_head_cut(before))
        self.assertEqual(R.rating_window(after), dict(checkpoints=64, games=40960)); self.assertIsNone(R.rating_window(before))
        with self.assertRaisesRegex(ValueError, 'follows the g6230'):
            R.configure(g6230.previous(before), 6670, gradient_norms=NORMS)
        with self.assertRaisesRegex(ValueError, 'trained at PPG clip 2.5'):
            R.configure(before, 6670, gradient_norms=[dict(r, iteration=r['iteration'] - 500) for r in NORMS])
        unpaused = copy.deepcopy(before); unpaused.pop('teacher_training_revision')
        with self.assertRaisesRegex(ValueError, 'only while paused'):
            R.configure(unpaused, 6670, gradient_norms=NORMS)
        bad = copy.deepcopy(after); bad[R.KEY]['rating']['checkpoints'] = 32
        with self.assertRaisesRegex(ValueError, 'Invalid g6670 revision'): validate_config(bad)
        bad = copy.deepcopy(after); bad['value_phase']['beta_clone_min'] = .125
        with self.assertRaisesRegex(ValueError, 'holds the PPG floor'): validate_config(bad)

    def test_it_is_adopted_forward_only(self):
        from brass_ai.rust_backend.lab_shadow_config import is_shadow, signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        other = R.configure(before, 6670, gradient_norms=[dict(r, mean=r['mean'] * 1.01) for r in NORMS])
        self.assertFalse(signature_transition(signature(after), signature(other)))
        self.assertEqual(is_shadow(after), is_shadow(before))


class PpgClip(unittest.TestCase):
    def test_the_rescale_fits_the_logged_norms(self):
        r = R.ppg_adam_rescale(NORMS)
        self.assertEqual((r['steps'], r['old_clip'], r['new_clip']), (954, 2.5, 3.))
        self.assertAlmostEqual(r['logged_clipped_percent'], 100. * 49 / 954)
        self.assertAlmostEqual(r['predicted_clipped_percent'], r['logged_clipped_percent'], delta=1.)
        self.assertAlmostEqual(r['first_moment'], 1.01063, places=4)
        self.assertAlmostEqual(r['second_moment'], 1.03393, places=4)

    def test_the_norms_must_have_clipped_at_2_5(self):
        from test_soft_head_revision_g6230 import clipping
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            state = dict(iteration=6670, recent_training=[dict(iteration=r['iteration'], value_phase=dict(
                gradient_clipping=clipping(r, 2.5))) for r in NORMS])
            self.assertEqual(R.committed_ppg_norms(run, state), NORMS)
            state['recent_training'][3]['value_phase']['gradient_clipping'] = clipping(NORMS[3], 2.4)
            with self.assertRaisesRegex(ValueError, 'clipped PPG at 2.5'): R.committed_ppg_norms(run, state)

    def test_the_ppg_adam_is_rescaled_once_after_the_earlier_receipts(self):
        config = R.configure(before_recipe(), 6670, gradient_norms=NORMS)
        torch.manual_seed(0); model = torch.nn.Linear(3, 2); optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        for _ in range(3):
            for p in model.parameters(): p.grad = torch.randn_like(p)
            optimizer.step()
        group = optimizer.param_groups[0]  # the run's PPG Adam carries the g5466, g5685 and g6230 receipts
        group[g5466.GROUP_KEY] = dict(generation=5466); group[g5685.PPG_GROUP_KEY] = dict(generation=5685)
        group[g6230.PPG_GROUP_KEY] = dict(generation=6230)
        for earlier in (g5466.rescale_ppg, g5685.rescale_ppg, g6230.rescale_ppg):
            self.assertIsNone(earlier(optimizer, config, 6670))
        moments = copy.deepcopy(optimizer.state_dict()['state'])
        receipt = R.rescale_ppg(optimizer, config, 6670)
        s = config[R.KEY]['ppg_adam_rescale']
        self.assertEqual((receipt['old_clip'], receipt['new_clip'], receipt['contract']), (2.5, 3., R.CONTRACT))
        for i, state in optimizer.state_dict()['state'].items():
            torch.testing.assert_close(state['exp_avg'], moments[i]['exp_avg'] * s['first_moment'])
            torch.testing.assert_close(state['exp_avg_sq'], moments[i]['exp_avg_sq'] * s['second_moment'])
        self.assertIsNone(R.rescale_ppg(optimizer, config, 6670))  # once
        self.assertIsNone(R.rescale_ppg(optimizer, R.previous(config), 6670))
        self.assertEqual(R.ppg_grad_clip(config), 3.)


class Floor(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_the_first_resume_raises_beta_to_a_quarter_and_keeps_the_trips(self):
        from brass_ai.recovery.actor_value import TrainingPhases, KEY
        from test_gae_value_phase import finish
        from test_ppg_g210 import config, prepare
        from test_value_buffer import commit
        m, _, c = config()
        c['value_phase'] = dict(P.KL_STOP_DEFAULT, max_epochs=1, batch_size=4, clone_rows=64, clone_batch_size=8,
                                beta_clone_min=.125)
        high = copy.deepcopy(c); high['value_phase']['beta_clone_min'] = .25
        high[R.KEY] = dict(R.configure(before_recipe(), 6670, gradient_norms=NORMS)[R.KEY], from_generation=1)
        update = dict(categorical_probe=dict(categorical_kl=0.), controller_warmup_done=True, optimizer_steps=1, kl_early_stop=False, gae={})
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            run = Path(tmp); p = TrainingPhases(m, c, root=SOURCE_ROOT, run=run)
            prepare(m, p, 1); p.fit(m); finish(p, update, 1)
            p.beta = .125; payload = p.payload(); commit(run, p, 1)
            q = TrainingPhases(m, high, raw=dict(training={KEY: payload}), generation=1, root=SOURCE_ROOT, run=run)
            migration = q.ppg_schedule_migration
            self.assertEqual((q.beta, q.trips), (.25, p.trips))
            self.assertEqual((migration['contract'], migration['beta_before'], migration['beta_after']), (R.FLOOR_MIGRATION, .125, .25))
            self.assertEqual(q.ppg_adam_rescale['new_clip'], 3.)  # the clip's one-time rescale at the same resume
            kept = TrainingPhases(m, c, raw=dict(training={KEY: payload}), generation=1, root=SOURCE_ROOT, run=run)
            self.assertEqual(kept.beta, .125)  # without the revision nothing changes
            above = dict(payload, beta_clone=.5)
            self.assertEqual(TrainingPhases(m, high, raw=dict(training={KEY: above}), generation=1, root=SOURCE_ROOT, run=run).beta, .5)

    def test_beta_halves_only_down_to_a_quarter(self):
        options = dict(P.KL_STOP_DEFAULT, max_epochs=1)
        P.validate(dict(options, batch_size=4096, clone_rows=8192, clone_batch_size=512))
        clean = dict(kl_guard_tripped=False, optimizer_steps=16, probe_kl=.0001)
        beta, _, feedback = P.beta_after_phase(.25, [], clean, options)
        self.assertEqual((options['beta_clone_min'], beta, feedback['reason']), (.25, .25, 'at_minimum'))


class SearchHead(unittest.TestCase):
    def test_the_cut_removes_the_head_and_its_adam_at_the_ppg_boundary(self):
        from brass_ai.learning.batching import model_id
        from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
        from brass_ai.learning.search_value import install
        from brass_ai.recovery.search_value_iteration import SearchValueIteration
        from brass_ai.recovery.search_value_training import Companion
        from test_gae_value_phase import fixture
        model = fixture(games=1)[0]
        ordinary = {n: p.detach().clone() for n, p in model.named_parameters()}
        install(model, hidden=32); companion = Companion(model, hidden=32); with_head = model_id(model)
        it = SearchValueIteration.__new__(SearchValueIteration)
        it.model, it.generation, it.pending_generation, it.report = model, 6670, 6671, None
        it.paused, it.cut, it._publication = True, True, None
        it.phase = SimpleNamespace(companion=companion, machine=SimpleNamespace(state=dict(stage='C', w=1.)),
                                   paused=lambda generation: dict(updated=False, paused=dict(generation=generation)))
        report = it.finish_ppg()
        cut = report['cut']
        self.assertEqual((cut['contract'], cut['generation'], cut['width'], cut['stage']), (R.CUT_CONTRACT, 6670, 32, 'C'))
        self.assertGreater(cut['parameters'], 0)
        self.assertFalse(hasattr(model, 'search_value_head')); self.assertIsNone(it.phase.companion)
        self.assertEqual({n: p for n, p in model.named_parameters()}.keys(), ordinary.keys())
        for n, p in model.named_parameters(): self.assertTrue(torch.equal(p, ordinary[n]), n)
        self.assertNotEqual(model_id(model), with_head)  # the next checkpoint's identity is the head-less model's
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'next.pt'
            save_checkpoint(path, model, None, step=6671, training={})
            loaded, raw = load_checkpoint(path)
            self.assertNotIn('search_value_head', raw['training']); self.assertFalse(hasattr(loaded, 'search_value_head'))
        # A later iteration without the head reports the pause only.
        it.report, it.pending_generation = None, 6672
        self.assertNotIn('cut', it.finish_ppg())


class Rating(unittest.TestCase):
    def registry(self, count):
        from brass_ai.recovery.plackett_luce import IL, V4
        from brass_ai.recovery.rating_coverage import SEED
        agents = {IL: dict(kind='model', checkpoint=dict(generation=0, model_id='il', path='il.pt')),
                  V4: dict(kind='v4', policy={}),
                  SEED: dict(kind='model', checkpoint=dict(generation=0, model_id='seed', path='seed.pt'))}
        for i in range(1, count + 1):
            agents[f'generation-{20 * i:06d}'] = dict(kind='model', checkpoint=dict(
                generation=20 * i, model_id=f'm{i}', sha256='x', weights_only=True,
                path=f'artifacts/step5/{"parent" if i <= 3 else "child"}/population/weights-m{i}.pt'))
        return dict(agents=agents, slots=[])

    def test_only_the_latest_64_checkpoints_stay_with_their_weights(self):
        from brass_ai.recovery import ratings as Q
        rating = self.registry(70)
        retired = Q.retire_checkpoints(rating, 64)
        self.assertEqual(retired, [f'generation-{20 * i:06d}' for i in range(1, 7)])
        self.assertEqual(len(rating['agents']), 64 + 3)  # the references stay
        self.assertEqual(min(e['checkpoint']['generation'] for a, e in rating['agents'].items() if a.startswith('generation-00')), 140)
        self.assertEqual(rating['retired_agents']['generation-000020'],
                         dict(kind='model', generation=20, model_id='m1', stored_in='artifacts/step5/parent'))
        self.assertEqual(rating['retired_agents']['generation-000080']['stored_in'], 'artifacts/step5/child')
        self.assertNotIn('.pt', json.dumps(rating['retired_agents']))  # retention no longer pins their files
        self.assertEqual(Q.retire_checkpoints(rating, 64), [])
        rating['agents']['generation-001420'] = dict(kind='model', checkpoint=dict(generation=1420, model_id='new', path='p.pt'))
        self.assertEqual(Q.retire_checkpoints(rating, 64), ['generation-000140'])  # one in, one out
        known = Q.known_agents(rating)
        self.assertEqual(len(known), len(rating['agents']) + len(rating['retired_agents']))
        self.assertEqual(known['generation-000020']['checkpoint'], dict(generation=20, model_id='m1'))

    def test_the_fit_reads_the_latest_games_and_keeps_retired_agents_in_them(self):
        from brass_ai.recovery import ratings as Q
        from brass_ai.recovery.plackett_luce import IL, V4, estimate
        from brass_ai.recovery.rating_coverage import SEED
        import random
        rng = random.Random(3); names = [f'generation-{20 * i:06d}' for i in range(1, 9)]; games = []
        for slot in range(20, 181, 20):
            for k in range(8):
                agents = [f'generation-{slot:06d}', SEED, *rng.sample(names, 2)]
                order = list(range(4)); rng.shuffle(order)
                games.append(dict(game_id=f'{slot}-{k}', setup_id=f's{k}', candidate_seat=k % 4, slot=slot, kind='rating',
                                  agents=agents, order=order))
        window = Q.latest_games(games, 40)
        self.assertEqual(len(window), 40); self.assertEqual({g['slot'] for g in window}, {100, 120, 140, 160, 180})
        fit_agents = sorted({a for g in window for a in g['agents']} | {IL, V4, SEED})
        config = dict(contract='plackett-luce-rating/1.0.0', theta_prior_sd=2., seat_prior_sd=.5, elo_scale=173.7,
                      seed=1, bootstrap=0)
        fitted, _ = estimate(window, fit_agents, config, baseline=SEED, bootstrap=False)
        self.assertEqual(set(fitted['ratings']), set(fit_agents)); self.assertEqual(fitted['games'], 40)
        old = {a for g in window for a in g['agents'] if a.startswith('generation-') and int(a[11:]) < 100}
        self.assertTrue(old and old <= set(fitted['ratings']))  # old opponents of the recent games are still fitted


class ParentCleanup(unittest.TestCase):
    """The retired checkpoints stored in the parent run are removed there once, where nothing still pins them."""

    def tree(self, tmp):
        from brass_ai.step5.league import write_json
        root = Path(tmp); step5 = root / 'artifacts' / 'step5'
        parent, child = step5 / 'parent', step5 / 'child'
        (parent / 'population').mkdir(parents=True); (child / 'population').mkdir(parents=True)

        def weights(run, i, size=10):
            path = step5 / run / 'population' / f'weights-m{i}.pt'; path.write_bytes(b'x' * size)
            return dict(generation=20 * i, model_id=f'm{i}', path=path.relative_to(root).as_posix(), sha256='x', weights_only=True)
        refs = {'generation-0': dict(kind='model', checkpoint=weights('parent', 0)), 'v4': dict(kind='v4', policy={})}
        shared = {f'generation-{20 * i:06d}': dict(kind='model', checkpoint=weights('parent', i, 100 + i)) for i in range(1, 6)}
        own = {'generation-000200': dict(kind='model', checkpoint=weights('parent', 10))}  # rated in the parent only
        pool = [shared['generation-000080']['checkpoint']]  # the parent's own opponent pool keeps one of them
        write_json(parent / 'state.json', dict(format='step5-recovery-league/2.2.0', iteration=200, population=pool,
                                               ratings=dict(agents=dict(refs, **shared, **own), slots=[])))
        kept = {f'generation-{20 * i:06d}': dict(kind='model', checkpoint=weights('child', i)) for i in range(6, 9)}
        child_rating = dict(agents=dict(refs, **copy.deepcopy(shared), **kept), slots=[])
        # The parent's registry differs for one name (another checkpoint under the same name): never removed.
        child_rating['agents']['generation-000100']['checkpoint']['model_id'] = 'other'
        from brass_ai.recovery import ratings as Q
        Q.retire_checkpoints(child_rating, 3)
        child_state = dict(format='step5-recovery-league/2.2.0', iteration=6671, ratings=child_rating)
        write_json(child / 'state.json', child_state)
        return root, parent, child, child_state

    def test_unpinned_parent_files_are_removed_once_and_pinned_ones_stay(self):
        from brass_ai.bank import read
        from brass_ai.recovery import rating_parent_cleanup as C
        with tempfile.TemporaryDirectory() as tmp:
            root, parent, child, state = self.tree(tmp)
            summary = C.cleanup(root, child, state, generation=6671)
            item = summary['parents']['artifacts/step5/parent']
            self.assertEqual((summary['status'], item['planned'], item['removed'], item['kept_pinned']), ('done', 4, 3, 1))
            self.assertEqual(item['removed_bytes'], 101 + 102 + 103)
            files = {p.name for p in (parent / 'population').iterdir()}
            self.assertEqual(files, {'weights-m0.pt', 'weights-m4.pt', 'weights-m5.pt', 'weights-m10.pt'})
            other = read(parent / 'state.json')
            self.assertEqual(sorted(other['ratings']['retired_agents']),
                             ['generation-000020', 'generation-000040', 'generation-000060', 'generation-000080'])
            self.assertEqual(other['ratings']['retired_agents']['generation-000020']['retired_by'], 'artifacts/step5/child')
            self.assertIn('generation-000100', other['ratings']['agents'])  # a different checkpoint under that name
            self.assertIn('generation-000200', other['ratings']['agents'])  # the parent's own later checkpoints stay
            backup = read(parent / 'configuration-history/g6670-rating-cleanup/state-before.json')
            self.assertEqual(set(backup['ratings']['agents']), {'generation-0', 'v4', 'generation-000200',
                             *(f'generation-{20 * i:06d}' for i in range(1, 6))})  # the parent's registry before
            receipt = read(child / '.retention' / C.RECEIPT)
            reason = receipt['parents']['artifacts/step5/parent']['kept_pinned']['generation-000080']
            self.assertTrue(reason.startswith('active state') and reason.endswith('state.json'), reason)  # its own pool
            self.assertEqual(receipt['parents']['artifacts/step5/parent']['skipped'], ['generation-000100'])
            self.assertIsNone(C.cleanup(root, child, state, generation=6672))  # once

    def test_an_active_parent_waits_and_an_interrupted_pass_resumes(self):
        from unittest import mock
        from brass_ai.bank import read
        from brass_ai.recovery import rating_parent_cleanup as C
        from brass_ai.recovery import retention
        with tempfile.TemporaryDirectory() as tmp:
            root, parent, child, state = self.tree(tmp)
            marker = parent / '.active-run.json'
            marker.write_text(json.dumps(dict(pid=1, host='another-host', token='t')))
            summary = C.cleanup(root, child, state, generation=6671)
            self.assertEqual((summary['status'], summary['parents']['artifacts/step5/parent']['removed']), ('pending', 0))
            self.assertNotIn('retired_agents', read(parent / 'state.json')['ratings'])
            marker.unlink()
            with mock.patch.object(retention, 'references', side_effect=RuntimeError('stopped')):
                with self.assertRaises(RuntimeError): C.cleanup(root, child, state, generation=6672)
            self.assertIn('retired_agents', read(parent / 'state.json')['ratings'])  # retired, nothing deleted yet
            self.assertEqual(len(list((parent / 'population').iterdir())), 7)
            summary = C.cleanup(root, child, state, generation=6673)  # resumes from the receipt's plan
            self.assertEqual(summary['parents']['artifacts/step5/parent']['removed'], 3)
            self.assertEqual(read(child / '.retention' / C.RECEIPT)['status'], 'done')


if __name__ == '__main__':
    unittest.main()
