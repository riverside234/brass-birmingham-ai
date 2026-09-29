import copy
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from brass_ai.bank import load_bank, read
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import split_for
from brass_ai.step5.policy import NeuralPolicy, context_for, exploration_distribution, INDUSTRIES
from brass_ai.step5.search import confirmed_teacher, first_choices, turn_key
from brass_ai.step5.updates import ppo_update, reptile_step, distill_update
from brass_ai.step5.league import seeds_for, table_kind, validate_config, pottery_metrics, checked_snapshot
from brass_ai.step5.evaluation import summarize_outcomes
from fixtures import position, game


class Step5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(781)
        self.model = FourSeatNet(ModelConfig.preset('small'))

    def onpolicy_game(self, policy):
        g = game()
        seed = seeds_for(551000, 1)[0][0]
        group = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed))
        g.update(setup_group=group, split=split_for(group), opponents=[policy.policy_id] * 4)
        for row in g['records']:
            rng = random.Random(row['seat'])
            selected, probability, _ = policy.select_batch([(row['view'], row['choices'], rng)])[0]
            row.update(chosen=selected, style=policy.context_features(row['view'], rng)[0],
                       intent=policy.context_features(row['view'], rng)[1],
                       behavior=dict(kind='model', policy_id=policy.policy_id, probability=probability, temperature=1.0))
        return g

    def test_industry_exploration_keeps_pottery_equal(self):
        choices = [dict(id=i, kind='build', industry=i) for i in INDUSTRIES]
        q = exploration_distribution(choices)
        np.testing.assert_allclose(q, np.ones(6) / 6)
        duplicate = choices + [dict(id='pottery-second', kind='build', industry='Pottery')]
        q = exploration_distribution(duplicate)
        self.assertAlmostEqual(q[2] + q[-1], q[0])

    def test_pottery_context_has_real_input_and_full_support(self):
        style, intent = context_for('Pottery')
        self.assertEqual(intent[2], 1)
        choices = [dict(id=i, kind='build', industry=i) for i in INDUSTRIES]
        q = exploration_distribution(choices, intent)
        self.assertGreater(q[2], q[0]); self.assertTrue((q > 0).all())

    def test_neural_context_is_per_seat_and_persistent(self):
        p = NeuralPolicy(self.model, focus='mixed')
        view, _ = position()
        rng = random.Random(62)
        first = p.context_features(view, rng)
        rng.random()
        self.assertEqual(first, p.context_features(view, rng))
        other = random.Random(17)
        p.context_features(view, other)
        self.assertEqual(len(p.contexts), 2)

    def test_default_tables_are_95_percent_neural(self):
        kinds = [table_kind(i) for i in range(100)]
        self.assertEqual(kinds.count('legacy'), 5)
        self.assertEqual(kinds.count('historical'), 25)
        self.assertNotIn('legacy', [table_kind(i, 0) for i in range(100)])
        cfg = read(SOURCE_ROOT / 'configs/step5/lab.json'); cfg['legacy_every'] = 2
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_twelve_training_seats_include_all_six_industry_focuses(self):
        policy = NeuralPolicy(self.model)
        rngs = [random.Random(i) for i in range(12)]
        features = [policy.context_features(position()[0], rng)[1] for rng in rngs]
        self.assertEqual(sum(sum(v) == 0 for v in features), 6)
        for i in range(6):
            self.assertEqual(sum(v[i] == 1 for v in features), 1)

    def test_train_validation_test_setup_streams_disjoint(self):
        streams = [set(seeds_for(500000, 30, p)[0]) for p in ('train', 'validation', 'test')]
        self.assertFalse(streams[0] & streams[1] or streams[0] & streams[2] or streams[1] & streams[2])

    def test_ppo_changes_policy_and_keeps_losing_trajectories(self):
        policy = NeuralPolicy(self.model, exploration=.1)
        g = self.onpolicy_game(policy)
        old = self.model.policy_scorer[0].weight.detach().clone()
        report = ppo_update(self.model, torch.optim.AdamW(self.model.parameters(), lr=.001), [g], policy, epochs=2)
        self.assertEqual(report['learner_trajectories'], 4)
        self.assertFalse(torch.equal(old, self.model.policy_scorer[0].weight))

    def test_ppo_excludes_external_and_historical_actor_rows(self):
        policy = NeuralPolicy(self.model)
        g = self.onpolicy_game(policy)
        for seat in (1, 2, 3):
            g['opponents'][seat] = 'old-model'
            g['records'][seat]['behavior'].update(kind='external', policy_id='old-model', probability=.99)
        report = ppo_update(self.model, torch.optim.AdamW(self.model.parameters()), [g], policy, epochs=1)
        self.assertEqual(report['learner_trajectories'], 1)
        self.assertEqual(report['learner_decisions'], 1)

    def test_ppo_rejects_stale_likelihood_or_evaluation(self):
        policy = NeuralPolicy(self.model)
        g = self.onpolicy_game(policy); g['records'][0]['behavior']['probability'] = .999
        with self.assertRaises(ValueError):
            ppo_update(self.model, torch.optim.AdamW(self.model.parameters()), [g], policy)
        g = self.onpolicy_game(policy); g['split'] = 'test'
        with self.assertRaises(ValueError):
            ppo_update(self.model, torch.optim.AdamW(self.model.parameters()), [g], policy)

    def test_proposals_keep_setup_actions_and_pottery(self):
        choices = [dict(id='pottery', kind='build', industry='Pottery'),
                   dict(id='cotton', kind='build', industry='CottonMill'),
                   dict(id='loan', kind='loan'), dict(id='develop', kind='develop')]
        indices = first_choices(choices, [.000001, .99, .001, .001], 2, random.Random(1))
        self.assertEqual(set(indices), set(range(4)))

    def test_complete_plan_teacher_uses_fresh_scores_and_prefix(self):
        choices = [{'id': x} for x in ('a', 'b', 'unseen')]
        plans = [['a', 'x'], ['a', 'y'], ['b', 'z']]
        t = confirmed_teacher(choices, plans, [.8, .4, .2], [], .7, 4, .1)
        self.assertEqual(t['coverage'], [True, True, False])
        self.assertGreater(t['probabilities'][0], .99)
        self.assertEqual(t['probabilities'][2], 0)
        t = confirmed_teacher([{'id': 'x'}, {'id': 'y'}], plans, [.8, .4, .2], ['a'], .7, 4, .1)
        self.assertEqual(t['coverage'], [True, True])
        self.assertIsNone(confirmed_teacher([{'id': 'z'}], plans, [.8, .4, .2], ['b'], .7, 4, .1))

    def test_no_teacher_is_not_fabricated_as_distillation(self):
        p = NeuralPolicy(self.model); g = self.onpolicy_game(p)
        result = distill_update(self.model, torch.optim.AdamW(self.model.parameters()), [g])
        self.assertFalse(result['updated'])

    def test_turn_identity_includes_era_round_and_spending_order_turn(self):
        view, _ = position(); first = turn_key(view['context'])
        for field in ('round', 'turn', 'turn_owner'):
            c = dict(view['context']); c[field] += 1
            self.assertNotEqual(first, turn_key(c))

    def test_reptile_moves_to_mean_adapted_weights(self):
        a, b = copy.deepcopy(self.model), copy.deepcopy(self.model)
        original = next(self.model.parameters()).detach().clone()
        with torch.no_grad():
            for p in a.parameters(): p.add_(1)
            for p in b.parameters(): p.add_(3)
        reptile_step(self.model, [a, b], .25)
        torch.testing.assert_close(next(self.model.parameters()), original + .5)

    def test_frozen_neural_snapshot_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pt'; save_checkpoint(path, self.model)
            with self.assertRaises(ValueError):
                checked_snapshot(Path(temp), dict(path='model.pt', sha256='bad', model_id='bad'), 'cpu')

    def test_v3_holdout_is_policy_identity_only_and_still_loads(self):
        pool = read(SOURCE_ROOT / 'training/opponents/pool.json')
        self.assertFalse(any('pottery-flex' in p for p in pool['training']))
        self.assertTrue(any('pottery-flex' in p for p in pool['evaluation']))
        bank = read(SOURCE_ROOT / 'training/opponents/current.json')['bank']
        manifest = load_bank(SOURCE_ROOT, bank)
        self.assertIn('pottery-flex-v3', manifest['payload']['profiles'])

    def test_clustered_results_and_pottery_warning_do_not_claim_strength(self):
        rows = [dict(seed=seed, seat=seat, win=1., rank=1, vp=150, margin=40, margin_to_best=20)
                for seed in range(2) for seat in range(4)]
        report = summarize_outcomes(rows)
        self.assertEqual(report['setups'], 2)
        self.assertEqual(report['seat_counts'], {str(i): 2 for i in range(4)})
        self.assertNotIn('promotion_eligible', report)


if __name__ == '__main__':
    unittest.main()
