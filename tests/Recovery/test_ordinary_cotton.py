"""The ordinary cotton mixture survives sampling, replay and PPO preparation."""
from collections import Counter
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from brass_ai.recovery import ordinary_cotton as cotton
from brass_ai.recovery.exploration import check_behavior
from brass_ai.recovery.ppo_preparation import Preparation, prepare, view
from brass_ai.recovery.updates import Dataset, ExplorationMixture, row_probabilities
from brass_ai.step5.policy import NeuralPolicy, cotton_candidates, exploration_distribution


class OrdinaryCotton(unittest.TestCase):
    def test_quota_and_frozen_seats(self):
        start = 2080*320
        assignments = [cotton.assignment(i, 9187+i) for i in range(start,start+320)]
        self.assertEqual(Counter(len(a['seats']) if a else 0 for a in assignments), {0:256, 1:32, 2:32})
        for index, value in enumerate(assignments, start):
            if value:
                proof = dict(kind='selfplay', purpose='training', ordinary_game_index=index,
                             sampling_seed=9187+index, **{cotton.KEY: value})
                self.assertEqual(cotton.validate_assignment(proof), tuple(value['seats']))
                with self.assertRaisesRegex(ValueError, 'seat assignment'):
                    cotton.validate_assignment(dict(proof, kind='historical'))

    def test_full_marginal_probability_and_develop_continuation(self):
        choices = [dict(kind='develop',industry='CottonMill'),dict(kind='develop',industry='Pottery'),dict(kind='finish')]
        logits = torch.tensor([-1.5,.7,.3])
        raw = logits.double().softmax(-1).numpy()
        for exploration in (0., .08):
            policy = NeuralPolicy(None, evaluator=SimpleNamespace(model_id='learner'), exploration=exploration,
                                  exploration_version=3, opening=dict(temperature=1.3,action_decisions_per_seat=3))
            rng = random.Random(2080)
            for phase, legal in (('action', [choices[0], choices[1], dict(kind='build',industry='CottonMill')]),
                                 ('continuation', choices), ('continuation', [choices[1], choices[1], choices[2]])):
                view = dict(context=dict(phase=phase))
                chosen, p, _ = policy.sample_training(view, legal, rng, raw, None, cotton=True)
                b = dict(policy.last_behavior[rng], kind='model', policy_id=policy.policy_id, probability=p)
                row = dict(view=view, choices=legal, chosen=chosen, behavior=b, intent=None)
                check_behavior(b, phase)
                candidates = cotton.check_row(row, True)
                self.assertEqual(candidates, cotton_candidates(legal, phase))
                self.assertEqual(bool(candidates), cotton.KEY in b)
                temperature = b['temperature']
                base = (1-exploration)*(logits.double()/temperature).softmax(-1).numpy()
                base += exploration*exploration_distribution(legal, None, 3, phase)
                expected = base.copy()
                if candidates:
                    expected *= .85
                    expected[candidates] += .15/len(candidates)
                    with self.assertRaisesRegex(ValueError, 'eligibility'):
                        cotton.check_row(row, False)
                q = torch.tensor(ExplorationMixture([row])[0])[None]
                dataset = SimpleNamespace(temperatures=torch.tensor([[temperature]]),
                                          cotton_rates=torch.tensor([[b.get(cotton.KEY,0.)]]))
                actual = row_probabilities(dataset, [0], logits[None], q, policy)
                np.testing.assert_allclose(actual[0].numpy(), expected, rtol=1e-6)
                self.assertAlmostEqual(p, expected[chosen], places=12)

    def test_cpu_preparation_keeps_the_cotton_mixture_with_zero_base_exploration(self):
        from test_training_speedups import Staging
        e = Staging().encoded()
        row = dict(view=dict(context=dict(phase='continuation')), chosen=0, seat=0, intent=None,
                   choices=[dict(kind='develop',industry='CottonMill'), dict(kind='finish')],
                   behavior=dict(exploration=0., exploration_version=3, probability=.575,
                                 temperature=1., **{cotton.KEY: .15}))
        target = dict(win_share=[.25]*4, placement=[2.5]*4, vp=[100.]*4)
        dataset = Dataset([(row, target, ('game',0), e)], 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'inputs.bin'; output=Path(folder)/'batch.bin'
            stage=Preparation.__new__(Preparation)
            stage.cache={};stage.metrics=dict(input_bytes=0,staging_seconds=0.)
            with source.open('wb') as stage.stream:
                specs=stage._rows(dataset,[0],0.,False)
                result=prepare(source,stage.stream.tell(),specs,[0],output)
            with output.open('rb') as stream:
                staged=view(stream.read(),result['layout'][-1])
                np.testing.assert_array_equal(staged,dataset.batch([0])[2].numpy())
                np.testing.assert_array_equal(staged,[[1.,0.]])
