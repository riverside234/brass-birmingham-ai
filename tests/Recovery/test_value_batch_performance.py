"""State-only critic batching preserves losses, optimization and policy inputs."""
import copy
from concurrent.futures import ThreadPoolExecutor
import random
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery import updates, utility
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.step5.policy import NeuralPolicy


class ActionDependentNet(FourSeatNet):
    def forward(self, states, actions, legal_mask, stable_to_relative):
        logits, win, rank, vp = super().forward(states, actions, legal_mask, stable_to_relative)
        return logits, win, rank, vp + actions.sum((1, 2))[:, None]


class ValueBatchPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, combined=False, device='cpu'):
        torch.manual_seed(481)
        config = ModelConfig.preset('small')
        model = UtilityNet(utility_config(config, specification())) if combined else FourSeatNet(config)
        return model.to(device)

    def data(self, combined=False):
        episode = game()
        for row, count in zip(episode['records'], (3, 1, 2, 3)):
            row['choices'] = row['choices'][:count]
            row['behavior']['probability'] = 1 / count
        spec = specification() if combined else None
        return updates.prepared([episode], insolvency=spec['insolvency'] if spec else None, reward=spec)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_heads_gradients_and_rng_match_full_candidates_for_both_models(self):
        for combined in (False, True):
            with self.subTest(combined=combined):
                model = self.model(combined)
                ds = updates.Dataset(self.data(combined), 'cpu')
                indices = [3, 1, 0, 2]
                ix, full, _ = ds.batch(indices)
                narrowed_ix, narrowed, _ = ds.value_batch(indices, model)
                self.assertEqual(full[1].shape[1], 3)
                self.assertEqual(narrowed[1].shape[1], 1)
                torch.testing.assert_close(ix, narrowed_ix, rtol=0, atol=0)
                for position in (0, 3):
                    torch.testing.assert_close(full[position], narrowed[position], rtol=0, atol=0)
                torch.testing.assert_close(full[1][:, :1], narrowed[1], rtol=0, atol=0)
                self.assertTrue(bool(narrowed[2].all()))
                outputs, gradients = [], []
                rng = torch.get_rng_state().clone()
                for inputs in (full, narrowed):
                    model.zero_grad(set_to_none=True)
                    kwargs = dict(remaining=ds.remaining[ix]) if combined else {}
                    out = model(*inputs, **kwargs)
                    loss = updates.value_losses(out, tuple(t[ix] for t in ds.targets))[0].sum()
                    if combined:
                        loss = loss + (out[4] - ds.utility[ix]).square().sum()
                    loss.backward()
                    outputs.append(tuple(t.detach().clone() for t in out[1:]))
                    gradients.append({k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()})
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assert_tree_equal(outputs[0], outputs[1])
                self.assert_tree_equal(gradients[0], gradients[1])
                self.assertTrue(all(value is None for key, value in gradients[1].items()
                                    if key.startswith(('action_encoder.', 'policy_scorer.'))))

    def test_policy_supervised_paths_and_custom_values_keep_all_actions(self):
        ds = updates.Dataset(self.data(), 'cpu')
        model = self.model()
        parts = list(ds.microbatches(range(4), 2, 6))
        original_parts = [list(part) for part in parts]
        narrowed = list(ds.value_batches(parts, model))
        self.assertEqual([part for part, _, _ in narrowed], original_parts)
        self.assertTrue(all(inputs[1].shape[1] == 1 for _, _, inputs in narrowed))
        self.assertEqual(next(ds.policy_batches([[0, 1]], .05))[2][1].shape[1], 3)
        self.assertEqual(next(ds.supervised_batches([[0, 1]]))[2][1].shape[1], 3)
        custom = ActionDependentNet(model.config)
        custom.load_state_dict(model.state_dict())
        self.assertFalse(updates.state_value_model(custom))
        ix, inputs, _ = ds.value_batch([0, 1], custom)
        self.assertEqual(inputs[1].shape[1], 3)
        full = ds.batch([0, 1])[1]
        with torch.no_grad():
            self.assert_tree_equal(custom(*inputs), custom(*full))
        self.assertEqual(next(ds.value_batches([[0, 1]], custom))[2][1].shape[1], 3)

    def test_original_shape_and_version_checks_cannot_be_hidden_by_slicing(self):
        model = self.model()
        for field in ('actions', 'choice_ids', 'feature_version'):
            with self.subTest(field=field):
                ds = updates.Dataset(self.data(), 'cpu')
                row = copy.copy(ds.encoded[0]); ds.encoded[0] = row
                if field == 'actions':
                    row.actions = row.actions[:2]
                elif field == 'choice_ids':
                    row.choice_ids = row.choice_ids[:2]
                else:
                    row.feature_version = 'old-layout'
                with self.assertRaisesRegex(ValueError, 'encoded features'):
                    ds.value_batch([0], model)

    def onpolicy(self, model):
        episode = game(); episode['split'] = 'train'
        policy = NeuralPolicy(model, focus='neutral', exploration=.05)
        self.addCleanup(policy.close)
        episode['opponents'] = [policy.policy_id] * 4
        for row, count in zip(episode['records'], (3, 1, 2, 3)):
            row['choices'] = row['choices'][:count]
            index, probability, _ = policy.select_batch([(row['view'], row['choices'], random.Random(row['seat']))])[0]
            row.update(chosen=index, behavior=dict(kind='model', policy_id=policy.policy_id,
                                                  probability=probability, temperature=1.))
        return policy, episode

    def test_complete_ppo_matches_full_candidate_weights_adam_rng_and_metrics(self):
        config = dict(UPDATE_V21, epochs=2, effective_batch_size=2, critic_batch_size=3,
                      microbatch_size=2, max_candidate_cells=6)
        for combined in (False, True):
            with self.subTest(combined=combined):
                initial = self.model(combined)
                policy, episode = self.onpolicy(initial)
                results = []
                for narrow in (False, True):
                    model = copy.deepcopy(initial)
                    adam = torch.optim.AdamW(model.parameters(), lr=1e-6)
                    torch.manual_seed(884)
                    gate = updates.state_value_model if narrow else lambda _: False
                    with patch.object(updates, 'state_value_model', gate):
                        spec = model.config.utility if combined else None
                        report = updates.ppo_update(model, adam, [episode], policy, config,
                            insolvency=spec['insolvency'] if spec else None, reward=spec)
                    report = {k: v for k, v in report.items() if k not in ('timings', 'memory')}
                    results.append((copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict()),
                                    torch.get_rng_state().clone(), report))
                self.assert_tree_equal(results[0], results[1])

    def test_calibration_matches_full_candidate_receipt_adam_rng_and_flags(self):
        initial = self.model(True)
        initial_adam = torch.optim.AdamW(initial.parameters(), lr=3e-5)
        sum(p.square().mean() for p in initial.parameters()).backward(); initial_adam.step()
        next(initial.parameters()).requires_grad_(False)
        ds = updates.Dataset(self.data(True), 'cpu')
        config = dict(UPDATE_V21, effective_batch_size=3, microbatch_size=2, max_candidate_cells=6)
        results = []
        for narrow in (False, True):
            model = copy.deepcopy(initial)
            adam = torch.optim.AdamW(model.parameters(), lr=3e-5)
            adam.load_state_dict(copy.deepcopy(initial_adam.state_dict()))
            torch.manual_seed(885)
            gate = updates.state_value_model if narrow else lambda _: False
            with patch.object(updates, 'state_value_model', gate):
                report = utility.calibrate(model, adam, ds, config)
            results.append((copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict()),
                            torch.get_rng_state().clone(), report,
                            [p.requires_grad for p in model.parameters()]))
        self.assert_tree_equal(results[0], results[1])
        self.assertEqual(results[1][-1], [p.requires_grad for p in initial.parameters()])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_pinned_value_batches_preserve_heads_and_microbatch_order(self):
        model = self.model(True, 'cuda').eval()
        ds = updates.Dataset(self.data(True), 'cuda')
        parts = list(ds.microbatches(range(4), 2, 6))
        with ThreadPoolExecutor(max_workers=2) as executor, torch.no_grad():
            narrowed = list(ds.value_batches(parts, model, executor))
            self.assertEqual([part for part, _, _ in narrowed], parts)
            for part, ix, inputs in narrowed:
                self.assertEqual(inputs[1].shape[1], 1)
                self.assertTrue(all(t.device.type == 'cuda' for t in inputs))
                full = ds.batch(part)[1]
                a = model(*full, remaining=ds.remaining[ix])
                b = model(*inputs, remaining=ds.remaining[ix])
                for before, after in zip(a[1:], b[1:]):
                    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
