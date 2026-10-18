"""Pre-GELU local conditioning, static-weight export, and fresh-IL boundaries."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import torch
from torch import nn

from brass_ai.learning.batching import collate, predict_encoded
from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint
from brass_ai.learning.contracts import (SOURCE_ROOT, architecture_fingerprint, compatibility,
    assert_compatible, assert_heuristic_data_compatible, read_json)
from brass_ai.learning.features import Encoded
from brass_ai.learning.local_entities import entity_layout, check_action_selectors
from brass_ai.learning.model import ARCHITECTURE, FourSeatNet, ModelConfig
from brass_ai.learning.onnx_tools import prepare_export, SelectorMatmulExport, cpu_options
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.learning.reward import specification
from brass_ai.personality.model import import_parent
from brass_ai.personality.laptop.export import export, Graph, samples, compare, BASE_INPUTS, check_backend
from brass_ai.personality.contracts import CONTEXT_DIM
from brass_ai.recovery.config import latest_model_config
from test_mlp_v11 import inputs
from test_v11_selectors import trained_branches


def varied_inputs(batch, candidates):
    x, a, mask, mapping = inputs(batch, candidates)
    mask[:] = True
    if candidates > 1: mask[-1, -1] = False
    # Distinct reference rows across observations; include each absent/present
    # combination, all three roles together, and the first/last bank entries.
    for role, field in enumerate(entity_layout()['action_fields'].values()):
        block = a[..., field['start']:field['stop']]
        block.zero_()
        for b in range(batch):
            for i in range(candidates):
                if (i % 8) & (1 << role):
                    block[b, i, (i + b * 7) % block.shape[-1]] = 1
    return x, a, mask, mapping


class V111Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(113)
        model = UtilityNet(utility_config(ModelConfig.preset('small'), specification())).eval()
        trained_branches(model)
        return model

    def test_training_uses_pre_gelu_context_and_keeps_zero_initialization(self):
        model = FourSeatNet(ModelConfig.preset('small'))
        x, a, _, _ = varied_inputs(2, 16)
        e = model.encode_state(x); p = model.prepare_candidates(e)
        torch.testing.assert_close(model.score_prepared(a, p),
            model.score_candidates(model.action_encoder(a), p.context), rtol=0, atol=0)
        self.assertTrue(all(torch.count_nonzero(m.weight) == 0 for m in model.entity_projections.values()))
        trained_branches(model)
        p = model.prepare_candidates(model.encode_state(x))
        local = model.reference_context(a, p)
        expected = model.score_candidates(model.action_encoder[2](
            model.action_encoder[1](model.action_encoder[0](a) + local)), p.context)
        torch.testing.assert_close(model.score_prepared(a, p), expected, rtol=0, atol=0)
        v11 = model.score_candidates(model.action_encoder(a) + local, p.context)
        self.assertGreater(float((expected - v11).abs().max().detach()), 1e-4)

    def test_training_parameter_and_state_gradients_match_explicit_row_reference(self):
        model = self.model().double(); reference = copy.deepcopy(model)
        x, a, mask, mapping = varied_inputs(2, 16)
        x = x.double().requires_grad_(); other = x.detach().clone().requires_grad_(); a = a.double()
        actual = model(x, a, mask, mapping)
        e = reference.encode_state(other); p = reference.prepare_candidates(e)
        rows = []
        for b in range(2):
            values = []
            for i in range(16):
                pre = reference.action_encoder[0](a[b, i])
                for field, bank in zip(entity_layout()['action_fields'].values(), (p.targets, p.merchants, p.industries)):
                    selected = torch.nonzero(a[b, i, field['start']:field['stop']]).flatten()
                    if len(selected):pre = pre + bank[b, int(selected[0]) + 1]
                values.append(reference.action_encoder[2](reference.action_encoder[1](pre)))
            rows.append(torch.stack(values))
        logits = reference.score_candidates(torch.stack(rows), p.context).masked_fill(~mask, -torch.inf)
        expected = (logits,) + reference.heads_from_encoded(e, mapping)
        def loss(out):return out[0][mask].square().mean() + sum(v.square().mean() * .0001 for v in out[1:])
        loss(actual).backward(); loss(expected).backward()
        torch.testing.assert_close(x.grad, other.grad, rtol=1e-8, atol=1e-9)
        for (name, p), (other_name, q) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-9, msg=name)
        for layer in model.entity_projections.values():self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_export_preparation_once_reused_across_candidate_chunks(self):
        model = self.model(); frozen = prepare_export(model)
        for batch in (1, 2):
            x, a, mask, mapping = varied_inputs(batch, 772)
            with torch.inference_mode():
                encoded = frozen.encode_state(x)
                with patch.object(frozen, 'prepare_candidates', wraps=frozen.prepare_candidates) as prepare:
                    prepared = frozen.prepare_candidates(encoded)
                    self.assertEqual(tuple(prepared.banks.shape), (batch, 143, 160))
                    for n in (1, 2, 7, 127, 128, 129, 257, 771, 772):
                        expected = model(x, a[:, :n], mask[:, :n], mapping)[0]
                        observed = frozen.score_prepared(a[:, :n], prepared).masked_fill(~mask[:, :n], -torch.inf)
                        torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-5)
                    expected = frozen.score_prepared(a, prepared)
                    for cap in (1, 127, 256):
                        chunked = torch.cat([frozen.score_prepared(a[:, start:start+cap], prepared)
                            for start in range(0, 772, cap)], dim=1)
                        torch.testing.assert_close(chunked, expected, rtol=3e-5, atol=3e-5)
                    self.assertEqual(prepare.call_count, 1)
        row = Encoded(x[0].numpy(), a[0].numpy(), mapping[0].numpy(), tuple(map(str, range(772))))
        with torch.inference_mode():
            expected = predict_encoded(model, [row], 'cpu', 1000)
            with patch.object(frozen, 'prepare_candidates', wraps=frozen.prepare_candidates) as prepare:
                actual = predict_encoded(frozen, [row], 'cpu', 127)
            self.assertEqual(prepare.call_count, 1)
        for u, v in zip(actual, expected):np.testing.assert_allclose(u, v, rtol=3e-5, atol=3e-5)

    def test_selector_admission_strict_including_masked_rows(self):
        x, a, _, mapping = varied_inputs(1, 8)
        row = Encoded(x[0].numpy(), a[0].numpy(), mapping[0].numpy(), tuple(map(str, range(8))))
        for field in entity_layout()['action_fields'].values():
            for invalid in ((1, 1), (.5, 0), (-1, 0), (float('inf'), 0), (float('nan'), 0)):
                bad = copy.deepcopy(row); bad.actions[-1, field['start']:field['stop']] = 0
                bad.actions[-1, field['start']:field['start']+2] = invalid
                with self.assertRaisesRegex(ValueError, 'one-hot or absent'):collate([bad])
                with self.assertRaisesRegex(ValueError, 'one-hot or absent'):check_action_selectors(bad.actions)

    def test_new_identity_rejects_v11_weights_but_reuses_heuristic_data_and_recipe(self):
        self.assertEqual(ARCHITECTURE, 'brass-four-seat-mlp-resnet/11.2.0')
        self.assertNotEqual(architecture_fingerprint(), 'c7df5bf97fbe02f605aabeea89c9cc24d74981c5a4355a15f050f6b97f0cae39')
        old = dict(compatibility(), architecture_sha256='c7df5bf97fbe02f605aabeea89c9cc24d74981c5a4355a15f050f6b97f0cae39')
        assert_heuristic_data_compatible(old)
        with self.assertRaises(ValueError):assert_compatible(old)
        current = read_json(SOURCE_ROOT / 'configs/step5/joint7-latest.json')
        previous = dict(current, architecture='brass-four-seat-mlp-resnet/11.0.0')
        self.assertEqual(latest_model_config(previous, previous['imitation_checkpoint']), current)
        self.assertEqual(latest_model_config(current, current['imitation_checkpoint']), current)
        for model in (FourSeatNet(ModelConfig.preset('small')), self.model()):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
                path = Path(directory) / 'old.pt'; save_checkpoint(path, model)
                raw = torch.load(path, weights_only=True)
                raw['model_config']['architecture'] = 'brass-four-seat-mlp-resnet/11.0.0'
                raw['compatibility'] = old
                torch.save(raw, path); before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'Architecture 11.2 required'):load_checkpoint(path)
                self.assertEqual(path.read_bytes(), before)

    def test_onnx_static_weights_one_local_multiply_dynamic_shapes_and_fusions(self):
        utility = self.model(); actor, _ = import_parent(utility)
        torch.nn.init.normal_(actor.context_projection.weight, std=.02)
        for name, model in (('utility', utility), ('actor', actor)):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
                path = Path(directory) / f'{name}.onnx'; meta = export(model, path)
                graph = onnx.load(str(path)); initializers = {x.name for x in graph.graph.initializer}
                self.assertFalse(any(node.op_type == 'ArgMax' for node in graph.graph.node))
                # State/entity and stable-seat output gathers remain; no gather
                # may depend on candidate actions in the optimized graph.
                depends_on_actions = {'actions'}
                for node in graph.graph.node:
                    if depends_on_actions.intersection(node.input):
                        self.assertNotIn(node.op_type, ('Gather', 'GatherElements', 'GatherND'))
                        depends_on_actions.update(node.output)
                action_weights = {v.name for v in graph.graph.initializer if tuple(v.dims) == (327, 160)}
                # onnxsim folds MatMul+bias into a reshape/Gemm/reshape; the
                # weight must stay constant under either standard spelling.
                action_linear = [node for node in graph.graph.node if node.op_type in ('MatMul', 'Gemm')
                                 and node.input[1] in action_weights]
                self.assertEqual(len(action_linear), 1)
                self.assertIn(action_linear[0].input[1], initializers)
                local = [node for node in graph.graph.node if node.op_type == 'MatMul'
                         and node.input[0] in depends_on_actions and node.input[1] not in initializers]
                self.assertEqual(len(local), 1)
                execution = meta['optimization']['execution']
                self.assertGreater(execution['fused_layernorm'], 0); self.assertGreater(execution['fused_gelu'], 0)
                self.assertEqual(execution['unfused_erf'], 0)
                self.assertEqual(meta['optimization']['candidate_encoding']['dynamic_values_per_position'], 22880)
                runtime = Graph(path)
                self.assertTrue(check_backend(runtime)['passed'])
                for b in (1, 2):
                    for n in (1, 2, 7, 127, 128, 129, 257, 771, 772):
                        x, a, mask, mapping = varied_inputs(b, n)
                        remaining = torch.full((b, 4), .075)
                        data = dict(zip(BASE_INPUTS, [t.numpy() for t in (x, a, mask, mapping, remaining)]))
                        from brass_ai.learning.seat_descriptors import unknown_numpy
                        data['descriptors'] = np.stack([unknown_numpy() for _ in range(b)])
                        with torch.inference_mode():
                            if name == 'actor':
                                from brass_ai.personality.laptop.export import Named
                                data['personality'] = np.full((b, CONTEXT_DIM), .2, dtype=np.float32)
                                values = Named(model)(*(torch.from_numpy(data[k]) for k in meta['inputs']))
                            else:values = model(x, a, mask, mapping, remaining)
                        compare(dict(zip(meta['outputs'], [t.numpy() for t in values])), runtime.run(data), mask.numpy())
                bad = samples(meta['config'], name == 'actor')[1]
                field = entity_layout()['action_fields']['target']
                bad['actions'][-1, -1, field['start']:field['stop']] = .5
                with self.assertRaisesRegex(ValueError, 'one-hot or absent'):runtime.run(bad)
                del runtime


if __name__ == '__main__':unittest.main()
