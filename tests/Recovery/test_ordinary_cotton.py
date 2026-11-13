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
        assignments = [cotton.assignment(i, 9187+i, probability=.10) for i in range(start,start+320)]
        self.assertEqual(Counter(len(a['seats']) if a else 0 for a in assignments), {0:256, 1:32, 2:32})
        for index, value in enumerate(assignments, start):
            if value:
                proof = dict(kind='selfplay', purpose='training', ordinary_game_index=index,
                             sampling_seed=9187+index, **{cotton.KEY: value})
                self.assertEqual(cotton.validate_assignment(proof), (tuple(value['seats']), .10))
                self.assertEqual(value['seats'], cotton.assignment(index, 9187+index)['seats'])
                with self.assertRaisesRegex(ValueError, 'seat assignment'):
                    cotton.validate_assignment(dict(proof, kind='historical'))

    def test_full_marginal_probability_and_develop_continuation(self):
        choices = [dict(kind='develop',industry='CottonMill'),dict(kind='develop',industry='Pottery'),dict(kind='finish')]
        logits = torch.tensor([-1.5,.7,.3])
        raw = logits.double().softmax(-1).numpy()
        for exploration in (0., .08):
            rate, opening = (.10, 2.) if exploration == 0. else (.15, 1.3)
            policy = NeuralPolicy(None, evaluator=SimpleNamespace(model_id='learner'), exploration=exploration,
                                  exploration_version=3, opening=dict(temperature=opening,action_decisions_per_seat=3))
            rng = random.Random(2080)
            for phase, legal in (('action', [choices[0], choices[1], dict(kind='build',industry='CottonMill')]),
                                 ('continuation', choices), ('continuation', [choices[1], choices[1], choices[2]])):
                view = dict(context=dict(phase=phase))
                chosen, p, _ = policy.sample_training(view, legal, rng, raw, None, cotton_probability=rate)
                b = dict(policy.last_behavior[rng], kind='model', policy_id=policy.policy_id, probability=p)
                row = dict(view=view, choices=legal, chosen=chosen, behavior=b, intent=None)
                check_behavior(b, phase)
                candidates = cotton.check_row(row, rate)
                self.assertEqual(candidates, cotton_candidates(legal, phase))
                self.assertEqual(bool(candidates), cotton.KEY in b)
                temperature = b['temperature']
                self.assertEqual(temperature, opening if phase=='action' else 1.)
                base = (1-exploration)*(logits.double()/temperature).softmax(-1).numpy()
                base += exploration*exploration_distribution(legal, None, 3, phase)
                expected = base.copy()
                if candidates:
                    expected *= 1-rate
                    expected[candidates] += rate/len(candidates)
                    with self.assertRaisesRegex(ValueError, 'eligibility'):
                        cotton.check_row(row, 0.)
                    with self.assertRaisesRegex(ValueError, 'eligibility'):
                        cotton.check_row(row, .25-rate)
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

    def test_diagnostics_split_selfplay_builds_and_unique_flips_across_shards(self):
        from brass_ai.recovery import diagnostics
        from brass_ai.recovery.history import compact
        from brass_ai.recovery.replay import CheckedGame
        from unittest.mock import patch

        def game(name, *, cotton_table=False, flipped=False):
            player = dict(cash=10, income=0, mat=[])
            tile = dict(id='cotton-1', seat=0, type='CottonMill', level=1, resource=0, flipped=flipped)
            view = dict(context=dict(era='CanalEra', phase='action'), players=[player]*4, slots=[dict(tile=tile)])
            build = dict(seat=0, chosen=0, choices=[dict(kind='build', industry='CottonMill')],
                         view=dict(view, slots=[]))
            later = dict(seat=0, chosen=0, choices=[dict(kind='loan')], view=view)
            g = CheckedGame(dict(game_id=name, setup_group=name, opponents=['learner']*4, records=[build,later,later]))
            if cotton_table:
                g.cotton_seats = (0, 2)
            return g

        with (
            patch.object(diagnostics, 'match_report', return_value=dict(policies={}, games=1)),
            patch.object(diagnostics, 'learner_seats', return_value={}),
            patch.object(diagnostics, 'health_counts', return_value={}),
            patch.object(diagnostics, 'pottery_metrics', return_value=dict(builds={'Pottery':0},
                legal_build_opportunities={'Pottery':0}, builds_matching_assigned_focus={}, seat_trajectory_contexts={})),
        ):
            parts = [diagnostics.summarize([g], {'learner'}, {g['game_id']: kind}) for g,kind in (
                (game('cotton', cotton_table=True, flipped=True), 'selfplay'),
                (game('standard'), 'selfplay'), (game('historical', flipped=True), 'historical'))]
            summary = diagnostics.merge_summaries(parts)
        expected = dict(cotton_exploration=dict(tables=1,builds=1,flips=1),
                        standard_exploration=dict(tables=1,builds=1,flips=0))
        self.assertEqual(summary['economy']['cotton_by_selfplay'], expected)
        self.assertEqual(summary['economy']['observed_flips']['CottonMill'], 2)
        self.assertEqual(compact(summary)['economy']['cotton_by_selfplay'], expected)
