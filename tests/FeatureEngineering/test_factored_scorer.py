"""Same scorer tensors and algebra across training, chunked inference and export."""
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.batching import model_id, predict_encoded
from brass_ai.learning.contracts import (SOURCE_ROOT, CONFIG_DIR, compatibility, digest,
    read_json, architecture_fingerprint, assert_compatible)
from brass_ai.learning.features import Encoded
from brass_ai.learning.model import FourSeatNet, ModelConfig, dense_multiply_adds
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.learning.reward import specification
from brass_ai.personality.model import PersonalityNet, PersonalityConfig
from brass_ai.personality.contracts import CONTEXT_DIM


@contextmanager
def unfactored(model):
    """Pre-revision concatenated scorer, using the very same Parameter objects."""
    core = model.core if isinstance(model, PersonalityNet) else model
    def score(candidates, context):
        expanded = context[:, None].expand(-1, candidates.shape[1], -1)
        return core.policy_scorer(torch.cat((expanded, candidates), -1)).squeeze(-1)
    with patch.object(core, 'project_policy_context', side_effect=lambda x: x), \
         patch.object(core, 'score_candidates', side_effect=score):
        yield


def inputs(config, batch=3, candidates=17):
    generator = torch.Generator().manual_seed(810 + batch + candidates)
    states = torch.randn(batch, config.state_dim, generator=generator) * .2
    actions = torch.randn(batch, candidates, config.action_dim, generator=generator) * .2
    from brass_ai.learning.local_entities import entity_layout
    for field in entity_layout()['action_fields'].values():
        actions[...,field['start']:field['stop']]=0
        for i in range(candidates):
            if i%5:actions[:,i,field['start']+i%len(field['identities'])]=1
    mask = torch.ones(batch, candidates, dtype=torch.bool)
    mask[-1, 1:] = False
    mapping = torch.tensor([[(s-i) % 4 for s in range(4)] for i in range(batch)])
    return states, actions, mask, mapping


def forward(model, data):
    kwargs = {}
    core = model.core if isinstance(model, PersonalityNet) else model
    if getattr(core.config, 'utility', None) is not None:
        kwargs['remaining'] = data[0].new_full((len(data[0]), 4), .025)
    if isinstance(model, PersonalityNet):
        kwargs['personality'] = data[0].new_full((len(data[0]), CONTEXT_DIM), .2)
        out = model(*data, **kwargs)
        return tuple(out[k] for k in ('policy_logits', 'win_share_4', 'placement_4', 'vp_4', 'remaining_utility_4')) + tuple(out['goals'].values()) + (out['confidence_logits'],)
    return model(*data, **kwargs)


class FactoredScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def models(self):
        torch.manual_seed(724)
        config = ModelConfig.preset('medium')
        utility = utility_config(config, specification())
        actor = PersonalityNet(PersonalityConfig(utility.to_dict()))
        # Exercise trained, nonzero context conditioning, not only its zero init.
        torch.nn.init.normal_(actor.context_projection.weight, std=.02)
        return [FourSeatNet(config), UtilityNet(utility), actor]

    def test_forward_masks_permutations_and_all_heads(self):
        for model in self.models():
            core = model.core if isinstance(model, PersonalityNet) else model
            with self.subTest(model=type(model).__name__), torch.inference_mode():
                for batch, count in ((1,1), (3,7), (2,129)):
                    data = inputs(core.config, batch, count)
                    with unfactored(model): reference = forward(model, data)
                    actual = forward(model, data)
                    torch.testing.assert_close(actual[0], reference[0], rtol=2e-5, atol=2e-6)
                    for before, after in zip(reference[1:], actual[1:]):
                        torch.testing.assert_close(after, before, rtol=0, atol=0)
                    self.assertTrue(torch.isneginf(actual[0][~data[2]]).all())
                    p = actual[0].softmax(-1)
                    torch.testing.assert_close(p.sum(-1), torch.ones(batch))
                    self.assertEqual(float(p[-1,0]), 1.)
                    order = torch.arange(count-1, -1, -1)
                    permuted = forward(model, (data[0], data[1][:,order], data[2][:,order], data[3]))
                    torch.testing.assert_close(permuted[0], actual[0][:,order], rtol=2e-5, atol=2e-6)

    def test_scorer_gradients_include_both_weight_slices_bias_and_inputs(self):
        for dtype, tolerance in ((torch.float64, 2e-12), (torch.float32, 2e-5)):
            for size in ('small', 'medium'):
                for count in (1, 31):
                    with self.subTest(dtype=dtype, size=size, count=count):
                        torch.manual_seed(27)
                        model = FourSeatNet(ModelConfig.preset(size)).to(dtype)
                        state = torch.randn(2, model.config.hidden, dtype=dtype, requires_grad=True)
                        actions = torch.randn(2, count, model.config.action_hidden, dtype=dtype, requires_grad=True)
                        scale = torch.randn(2, count, dtype=dtype)
                        params = [state, actions, *model.policy_scorer.parameters()]
                        with unfactored(model):
                            before = model.score_candidates(actions, model.project_policy_context(state))
                            expected = torch.autograd.grad((before*scale).sum(), params)
                        after = model.score_candidates(actions, model.project_policy_context(state))
                        actual = torch.autograd.grad((after*scale).sum(), params)
                        torch.testing.assert_close(after, before, rtol=tolerance, atol=tolerance)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)

    def check_autocast(self, device):
        model = FourSeatNet().to(device)
        data = tuple(t.to(device) for t in inputs(model.config, 3, 65))
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            with unfactored(model): before = model(*data)[0]
            after = model(*data)[0]
        # Two rounded GEMMs plus an add are not bit-identical to one GEMM.
        self.assertEqual(after.dtype, torch.bfloat16)
        self.assertLess(float((after[data[2]]-before[data[2]]).abs().max().detach()), .01)
        torch.testing.assert_close(after.float().softmax(-1), before.float().softmax(-1), rtol=.02, atol=2e-4)
        (-after.float().log_softmax(-1)[:,0].mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cpu_bfloat16_autocast(self):
        self.check_autocast('cpu')

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 unavailable')
    def test_cuda_bfloat16_autocast(self):
        self.check_autocast('cuda')

    def test_chunked_and_packed_inference_project_state_only_once(self):
        from brass_ai.recovery.inference import predict_packed
        for model in self.models()[:2]:
            data = inputs(model.config, 1, 129)
            # The chunk path receives legal candidates only.
            row = Encoded(data[0][0].numpy(), data[1][0].numpy(), data[3][0].numpy(), tuple(map(str,range(129))))
            with torch.inference_mode():
                expected = predict_encoded(model, [row], 'cpu', 1024)
                for limit in (1,17,64):
                    with patch.object(model, 'project_policy_context', wraps=model.project_policy_context) as projected:
                        actual = predict_encoded(model, [row], 'cpu', limit)
                        self.assertEqual(projected.call_count, 1)
                    for a, b in zip(actual, expected): np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
                packed = predict_packed(model, [row], 'cpu', 1024)
                for a, b in zip(packed, expected): np.testing.assert_array_equal(a,b)

    def test_v11_is_not_v9_source_equivalence_and_unknown_edits_fail(self):
        original = architecture_fingerprint()
        self.assertNotEqual(original, '9c5ad1e2a20f7dfb0483e04ecbf7b77b6d1c6808641283e1a5e72b23a3bb2173')
        known = compatibility()
        real_read = Path.read_text
        def edited(path, *args, **kwargs):
            content = real_read(path, *args, **kwargs)
            return content+'\n# unreviewed change\n' if path.name=='local_entities.py' else content
        with patch.object(Path, 'read_text', edited):
            self.assertNotEqual(architecture_fingerprint(), original)
            with self.assertRaisesRegex(ValueError, 'Incompatible'): assert_compatible(known)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            assert_compatible(dict(known, architecture_sha256='0'*64))

    def test_parameters_identity_and_mac_count(self):
        for model, count, fixed in zip(self.models(), (5904644,5921320,6023137), (8235232,8251872,8353504)):
            core = model.core if isinstance(model, PersonalityNet) else model
            params = [(name,id(t),tuple(t.shape)) for name,t in model.named_parameters()]
            core_id = model_id(core)
            with torch.no_grad(): forward(model, inputs(core.config))
            self.assertEqual(params, [(name,id(t),tuple(t.shape)) for name,t in model.named_parameters()])
            self.assertEqual(model_id(core), core_id)
            self.assertEqual(model.parameter_count, count)
            self.assertEqual(tuple(core.policy_scorer[0].weight.shape), (192,672))
            mac = dense_multiply_adds(model)
            self.assertEqual((mac['fixed_per_position'],mac['per_padded_candidate']), (fixed,83232))

    def test_native_fixture_behavior_verification_and_bounded_ppo(self):
        from test_teacher_value import TeacherValueTests
        from brass_ai.recovery.updates import Dataset, probabilities, policy_measure, ppo_update
        from brass_ai.recovery.telemetry import Timings
        TeacherValueTests.setUpClass()
        model, opt, _, ordinary, _, cfg, _, _ = TeacherValueTests().fixture()
        dataset = Dataset(ordinary, 'cpu')
        _, data, q = dataset.batch(list(range(len(ordinary))), .08, 1.)
        # Collect using the new forward path, then reconstruct in smaller PPO batches.
        with torch.no_grad(): p = probabilities(model(*data)[0], q, .08, 1.)
        for i, row in enumerate(dataset.rows): row['behavior']['probability'] = float(p[i,row['chosen']])
        dataset = Dataset(ordinary, 'cpu')
        behavior = SimpleNamespace(exploration=.08, temperature=1., policy_id='fixture', evaluator=SimpleNamespace(model_id=model_id(model)))
        self.assertLess(abs(policy_measure(model,dataset,range(len(ordinary)),behavior,cfg,Timings(),True)), 1e-6)
        cfg = dict(cfg, epochs=1)
        report = ppo_update(model,opt,[],behavior,cfg,learner_data=ordinary,
                            insolvency=model.config.utility['insolvency'],reward=model.config.utility)
        self.assertGreater(report['optimizer_steps'], 0)
        dataset.old[0] *= .5
        with self.assertRaisesRegex(ValueError, 'Wrong/stale behavior'):
            policy_measure(model,dataset,range(len(ordinary)),behavior,cfg,Timings(),True)


if __name__=='__main__': unittest.main()
