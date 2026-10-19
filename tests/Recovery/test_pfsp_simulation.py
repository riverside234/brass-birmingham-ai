"""Supplied NumPy reference vs trainer arithmetic and reusable simulation fixes."""
import contextlib
import hashlib
import importlib.util
import io
import unittest
from unittest.mock import patch

import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import pfsp_pool as production, quality_pool as quality
from test_pfsp_pool import pool
from test_quality_pool import pool as legacy_pool


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, SOURCE_ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


reference = module('ai-research/simulations/pfsp_sim_original.py', 'pfsp_reference')
sim = module('scripts/pfsp_sim.py', 'pfsp_simulation')


class Simulation(unittest.TestCase):
    def test_original_is_preserved_exactly(self):
        path = SOURCE_ROOT / 'ai-research/simulations/pfsp_sim_original.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            'd4a3b527e6da783d8ee7da1fb1d577ccc47744021210297fa6ae6fffb2ab22d1')

    def test_corrected_generator_keeps_original_nonempty_seeded_outcomes(self):
        for rule in ('fixed', 'reciprocal'):
            for n in (1, 16, 23):
                args = (np.full(n, 1/n), np.linspace(-2., 0., n), .3, -3., sim.il_share(n, rule))
                old = reference.play_iteration(np.random.default_rng(501), *args)
                new = sim.play_iteration(np.random.default_rng(501), *args)
                for a, b in zip(old, new):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(len(new[0]), 144*4)

    def test_IL_only_and_duplicate_seats_count_all_four_pairs(self):
        for n, expected in ((0, -1), (1, 0)):
            mem, out = sim.play_iteration(np.random.default_rng(42), np.ones(n), np.zeros(n),
                0., -2., 1. if n == 0 else 0., tables=7)
            np.testing.assert_array_equal(mem, np.full(28, expected))
            self.assertTrue(set(out) <= {0., 1.})
        with self.assertRaises(ValueError):
            sim.play_iteration(np.random.default_rng(0), [], [], 0., 0., .2)
        self.assertEqual(len(sim.PFSP().probs()), 0)
        self.assertEqual(len(sim.Quality().probs()), 0)

    def test_original_dynamic_trajectory_is_reproducible_with_min_count_zero(self):
        for method in ('pfsp', 'quality'):
            for cap in (16, 23):
                old = reference.dynamic(method, cap=cap, seed=3)
                new = sim.dynamic(method, cap=cap, seed=3, min_count=0.)
                self.assertEqual(len(old), len(new))
                for a, b in zip(old, new):
                    for key in a:
                        if isinstance(a[key], np.ndarray):
                            np.testing.assert_array_equal(a[key], b[key])
                        else:
                            self.assertEqual(a[key], b[key])

    def test_reference_estimator_and_probabilities_match_trainer(self):
        for n in (1, 16, 23):
            value = pool(n+1, games=100000)
            ref = reference.PFSP()
            for i in range(n): ref.admit(i)
            rng = np.random.default_rng(514+n)
            for iteration in range(12):
                plan = production.distribution(value)
                self.assertEqual(plan['il_share'], reference.il_share(n, 'reciprocal'))
                np.testing.assert_allclose([r['p'] for r in plan['members']], ref.probs(), rtol=2e-15, atol=1e-16)
                mem, out = reference.play_iteration(rng, ref.probs(), -sim.logit(np.linspace(.55, .92, n)),
                    0., -sim.logit(.95), plan['il_share'])
                # Include a zero-game iteration and residual ties; neither is
                # present in the original continuous Gumbel experiment.
                if iteration == 3: mem, out = mem[:0], out[:0]
                if iteration == 5: out[::3] = .5
                rows = {m['model_id']: dict(seats=int((mem == j-1).sum())//2,
                    outcomes=out[mem == j-1].tolist()) for j, m in enumerate(value['members'])}
                scored = mem >= 0
                ref.update(np.bincount(mem[scored], weights=out[scored], minlength=n),
                    np.bincount(mem[scored], minlength=n))
                value, report = production.update_counts(value, rows, plan,
                    generation=value['generation']+1, games_done=value['games_done']+320)
                np.testing.assert_allclose([m['S'] for m in value['members'][1:]], ref.S, rtol=2e-15)
                np.testing.assert_allclose([m['C'] for m in value['members'][1:]], ref.C, rtol=2e-15)
                self.assertEqual(sum(m['comparisons'] for m in report['members']), len(out))
                self.assertEqual(sum(m['seats_played'] for m in report['members']), len(out)//2)

    def test_reference_quality_update_matches_legacy_trainer(self):
        value = legacy_pool(4)
        ref = reference.Quality()
        for i in range(3): ref.admit(i)
        for ranks in ([1, 2, 3, 4], [3, 4, 1, 2], [1, 2, 2, 4]):
            plan = quality.distribution(value)
            np.testing.assert_allclose([r['p'] for r in plan['members']], ref.probs(), atol=1e-15)
            ids = ['learner', 'learner', 'model-1', 'model-2']
            opponents = [dict(model_id=f'model-{j+1}', seat=j+2, source='scored', N=3,
                p=plan['members'][j]['p']) for j in range(2)]
            table = dict(index=0, learner_seats=[0, 1], opponents=opponents, policy_ids=ids)
            outcomes = [1. if ranks[l] < ranks[s] else .5 if ranks[l] == ranks[s] else 0.
                for s in (2, 3) for l in (0, 1)]
            ref.update_with(np.array([0, 0, 1, 1]), np.array(outcomes), ref.probs())
            with patch('brass_ai.recovery.quality_pool.terminal_targets', return_value=dict(placement=ranks)):
                value, _ = quality.update_scores(value, [dict(setup_group='h', opponents=ids, result={})],
                    {'h': table}, plan, generation=value['generation']+1)
            np.testing.assert_allclose([m['q'] for m in value['members'][1:]], ref.q, atol=1e-15)

    def test_shared_eviction_case_matches_with_production_support(self):
        from test_quality_pool import entry
        value = pool(17, generation=100)
        ref = reference.PFSP(min_count=15.)
        for i, member in enumerate(value['members'][1:]):
            ref.admit(member['model_id'])
            ref.age[i] = 100-member['admitted_iteration']
            x = .95 if i == 2 else .6
            ref.C[i] = 40.; ref.S[i] = 40*x
            member.update(S=ref.S[i], C=ref.C[i])
        production.seal(value)
        ref.evict()
        expected = next(m['model_id'] for m in value['members'][1:] if m['model_id'] not in ref.ids)
        _, event = production.admission(value, entry(100), dict(ok=True),
            {m['model_id']: .1 for m in value['members']}, lambda a, b: .1)
        self.assertEqual(event['evicted']['model_id'], expected)

    def test_response_crossings_are_trailing_and_unreached_is_not_one(self):
        self.assertEqual(sim.response_times(np.r_[0., np.ones(79)]), (2, 3))
        self.assertEqual(sim.response_times(np.ones(80)), (None, None))
        self.assertEqual(sim.response_times(np.r_[0., np.full(79, np.nan)]), (None, None))
        sparse = np.full(80, np.nan); sparse[0] = 0.; sparse[30:80:3] = 1.
        self.assertEqual(sim.response_times(sparse), (None, None))
        self.assertEqual(sim.response_times([]), (None, None))

    def test_cli_share_validation_and_precision_volume(self):
        self.assertEqual(sim.parse_args([]).tables, 144)
        args = sim.parse_args(['--history-share', '.3'])
        self.assertEqual(args.tables, 96)
        self.assertEqual(sim.parse_args(['--history-share', '.301', '--tables', '97']).tables, 97)
        for argv in (['--history-share', '.301'], ['--tables', '0'], ['--half-life', 'nan'], ['--seeds', '0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): sim.parse_args(argv)
        result = sim.static_precision(4, iters=20, burn=5, tables=args.tables)
        self.assertAlmostEqual(result['scored']+result['il'], 4*96)


if __name__ == '__main__':
    unittest.main()
