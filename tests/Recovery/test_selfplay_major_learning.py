"""Focused remaining-bug regressions; authored replays and bounded CPU fits."""
import copy
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.recovery import utility
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.replay import check_game
from brass_ai.recovery.updates import Dataset, prepared


class SelfplayMajorLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_checked_replay_rejects_reordered_or_duplicated_records(self):
        spec = specification()
        source = game()
        expected = prepared([source], insolvency=spec['insolvency'], reward=spec)
        checked = check_game(copy.deepcopy(source))
        actual = prepared([checked], insolvency=spec['insolvency'], reward=spec)
        self.assertEqual([d[:3] for d in actual], [d[:3] for d in expected])
        for mutation in ('reorder', 'duplicate'):
            with self.subTest(mutation=mutation):
                changed = check_game(copy.deepcopy(source))
                if mutation == 'reorder':
                    changed['records'].reverse()
                else:
                    changed['records'][1] = changed['records'][0]
                with self.assertRaisesRegex(ValueError, 'Validated game.*changed'):
                    prepared([changed], insolvency=spec['insolvency'], reward=spec)

    def test_filtered_opponent_mutation_cannot_change_learner_cost(self):
        spec = specification()
        source = game()
        # The final opponent observation is the first public debt view. Seat 0's
        # preceding action owes one event even though only its row is learned.
        last = source['records'][-1]
        last['view']['context']['phase'] = 'debt'
        last['view']['players'][0]['debt'] = 4
        slot = last['view']['slots'][0]
        slot['tile'] = copy.deepcopy(last['view']['players'][3]['mat'][0]['tiles'][0])
        last['choices'] = [dict(id='liquidate', kind='liquidate', target=slot['id'])]
        last['behavior']['probability'] = 1.
        checked = check_game(source)
        predicate = lambda row: row['seat'] == 0
        before = prepared([checked], predicate, insolvency=spec['insolvency'], reward=spec)
        self.assertEqual(before[0][0]['_insolvency_cost_to_go'], spec['insolvency']['event_cost'])
        checked['records'][-1]['view']['players'][0]['debt'] = 0
        with self.assertRaisesRegex(ValueError, 'Validated replay record changed'):
            prepared([checked], predicate, insolvency=spec['insolvency'], reward=spec)

    def test_initial_calibration_failure_restores_learning_rates_and_gradient_flags(self):
        spec = specification()
        torch.manual_seed(771)
        model = FourSeatNet(ModelConfig.preset('small'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        utility.migrate(model, optimizer, spec)
        next(model.parameters()).requires_grad_(False)
        flags = [p.requires_grad for p in model.parameters()]
        weights = copy.deepcopy(model.state_dict())
        data = Dataset(prepared([game()], insolvency=spec['insolvency'], reward=spec), 'cpu')
        config = dict(UPDATE_V21, effective_batch_size=4, microbatch_size=2)
        with patch.object(model, 'forward', side_effect=RuntimeError('initial calibration failure')):
            with self.assertRaisesRegex(RuntimeError, 'initial calibration failure'):
                utility.calibrate(model, optimizer, data, config)
        self.assertEqual(optimizer.param_groups[0]['lr'], 3e-5)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        self.assertEqual(len(optimizer.state), 0)
        for name, value in weights.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
