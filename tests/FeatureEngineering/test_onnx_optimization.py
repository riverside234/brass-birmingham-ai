"""Export-only seat batching, dynamic graph parity and real ORT trace accounting."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT, architecture_fingerprint
from brass_ai.learning.features import Encoded
from brass_ai.learning.model import ModelConfig, FourSeatNet
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.learning.reward import specification
from brass_ai.learning.onnx_tools import BatchedSeatEncoder, prepare_export, cpu_options
from brass_ai.learning.onnx_profile import profile_model, summarize_trace
from brass_ai.learning.export import export_onnx, load_export
from brass_ai.personality.model import import_parent
from brass_ai.personality.laptop.export import export, Graph, samples, compare, BASE_INPUTS
from test_mlp_v11 import inputs
from test_v11_selectors import trained_branches


class OnnxOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, dtype=torch.float32):
        torch.manual_seed(817)
        model = UtilityNet(utility_config(ModelConfig.preset('small'), specification())).to(dtype).eval()
        trained_branches(model)
        with torch.no_grad():
            for role in ('own_player', 'other_player'):
                norm = model.entity_encoder.encoders[role][2]
                norm.weight.uniform_(.5, 1.5)
                norm.bias.uniform_(-.2, .2)
        return model

    def test_batched_seats_preserve_distinct_weights_affines_outputs_and_original_model(self):
        for dtype, tolerance in ((torch.float32, 3e-5), (torch.float64, 2e-10)):
            model = self.model(dtype)
            before = copy.deepcopy(model.state_dict())
            config = copy.deepcopy(model.config)
            fingerprint = architecture_fingerprint()
            frozen = prepare_export(model)
            self.assertIsInstance(frozen.entity_encoder, BatchedSeatEncoder)
            seen = []
            original_matmul = torch.matmul
            def record(left, right):
                seen.append((tuple(left.shape), tuple(right.shape)))
                return original_matmul(left, right)
            x, a, mask, mapping = inputs(4, 31)
            with torch.inference_mode(), patch('torch.matmul', side_effect=record):
                expected = model(x.to(dtype), a.to(dtype), mask, mapping)
                actual = frozen(x.to(dtype), a.to(dtype), mask, mapping)
            self.assertEqual(seen, [((4, 4, 287), (4, 287, 64)), ((4, 31, 143), (4, 143, 160))])
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
            self.assertEqual(model.config, config)
            self.assertEqual(architecture_fingerprint(), fingerprint)
            self.assertEqual(model.state_dict().keys(), before.keys())
            for key, value in before.items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_named_utility_and_actor_export_parity_fusions_and_real_profile(self):
        utility = self.model()
        actor, _ = import_parent(utility)
        torch.nn.init.normal_(actor.context_projection.weight, std=.02)
        for name, model in (('utility', utility), ('actor', actor)):
            before = copy.deepcopy(model.state_dict())
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
                path = Path(directory) / f'{name}.onnx'
                meta = export(model, path)
                self.assertTrue(meta['parity']['passed'])
                self.assertEqual(meta['parity']['shapes'], [[1, 1], [2, 7], [4, 31], [1, 257], [1, 772], [2, 772]])
                execution = meta['optimization']['execution']
                self.assertGreater(execution['fused_gelu'], 0)
                self.assertGreater(execution['fused_layernorm'], 0)
                self.assertEqual(execution['unfused_erf'], 0)
                graph = Graph(path, threads=1)
                self.assertEqual(str(graph.engine.get_session_options().graph_optimization_level),
                                 'GraphOptimizationLevel.ORT_ENABLE_ALL')
                feed = samples(meta['config'], name == 'actor')[2]
                result = profile_model(path, [('four-positions', feed)], Path(directory) / 'profile',
                                       threads=1, warmups=1, repeats=3)
                case = result['cases'][0]
                self.assertEqual(case['measured_runs'], 3)
                self.assertEqual(case['excluded_warmup_runs'], 1)
                self.assertTrue(any(row['operation'] == 'Gelu' for row in case['operations']))
                self.assertTrue(all(row['provider'] == 'CPUExecutionProvider' for row in case['operations']))
                self.assertAlmostEqual(sum(row['percent_node_time'] for row in case['operations']), 100.)
                self.assertTrue((Path(directory) / 'profile' / case['trace']).exists())
                del graph
            for key, value in before.items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_base_export_keeps_dynamic_axes_and_saves_profile_inputs(self):
        model = FourSeatNet(ModelConfig.preset('small')).eval()
        trained_branches(model)
        x, a, _, mapping = inputs(3, 7)
        encoded = [Encoded(x[i].numpy(), a[i].numpy(), mapping[i].numpy(), tuple(map(str, range(7)))) for i in range(3)]
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
            path = Path(directory) / 'base.onnx'
            meta = export_onnx(model, path, [encoded[:1], encoded[1:]])
            self.assertLess(meta['parity_max_absolute_error'], 2e-4)
            session, _ = load_export(path)
            self.assertEqual(session.get_inputs()[0].shape[0], 'positions')
            self.assertEqual(session.get_inputs()[1].shape[1], 'candidates')
            with np.load(path.with_suffix('.parity.npz'), allow_pickle=False) as data:
                self.assertEqual(data['0:input:states'].shape, (1, 3128))
                self.assertEqual(data['1:input:actions'].shape, (2, 7, 327))
            del session

    def test_profile_excludes_warmups_fences_and_unrelated_events(self):
        events = [dict(name='model_run', ts=0, dur=10), dict(name='model_run', ts=20, dur=10)]
        for start, duration in ((1, 9), (21, 3)):
            events.append(dict(cat='Node', name='linear_kernel_time', ts=start, dur=duration,
                               args=dict(op_name='Gemm', provider='CPUExecutionProvider')))
        events += [dict(cat='Node', name='linear_fence_after', ts=22, dur=100),
                   dict(cat='Node', name='outside_kernel_time', ts=40, dur=100)]
        report = summarize_trace(events, warmups=1)
        self.assertEqual(report['measured_runs'], 1)
        self.assertEqual(report['total_kernel_us'], 3.)
        self.assertEqual(report['nodes'][0]['calls'], 1)


if __name__ == '__main__':
    unittest.main()
