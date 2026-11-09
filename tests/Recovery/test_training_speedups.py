"""Faster data paths that must not change what the trainer computes."""
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brass_ai.learning.contracts import FEATURE_VERSION
from brass_ai.learning.features import Encoded
from brass_ai.recovery import ordinary, ppo_preparation
from brass_ai.recovery.replay_binding import seal_encoded
from brass_ai.recovery.updates import ExplorationMixture
from brass_ai.recovery.value_preservation import GRADIENT_SAMPLE_EVERY, TrunkGradients, measure_actor_gradients


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = torch.nn.Linear(4, 3)
        self.entity_encoder = torch.nn.Linear(3, 2)
        self.head = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.head(self.entity_encoder(self.state_encoder(x)))


def losses(net):
    x = torch.tensor([[1., 2., 3., 4.], [0., 1., 0., 1.]])
    output = net(x).squeeze(1)
    return output.square().sum(), output.sum()*.25


class Gradients(unittest.TestCase):
    def accumulate(self, measure):
        torch.manual_seed(7)
        net = Net()
        terms = TrunkGradients(net, measure=measure)
        actor, value = losses(net)
        terms.add_actor(actor)
        actor.backward(retain_graph=True)
        terms.before_value()
        value.backward()
        report = terms.finish(1)
        return {name: p.grad.clone() for name, p in net.named_parameters()}, report

    def test_skipping_the_actor_pass_leaves_the_gradients_byte_identical(self):
        measured, report = self.accumulate(True)
        skipped, none = self.accumulate(False)
        self.assertIsNone(none)
        self.assertIsNotNone(report['actor_norm'])
        for name in measured:
            self.assertTrue(torch.equal(measured[name], skipped[name]), name)

    def test_actor_norms_are_sampled_once_beta_is_fixed_and_always_while_qualifying(self):
        on = mock.Mock(mode='on')
        shadow = mock.Mock(mode='shadow')
        self.assertEqual([s for s in range(16) if measure_actor_gradients(s, on)], list(range(0, 16, GRADIENT_SAMPLE_EVERY)))
        self.assertTrue(all(measure_actor_gradients(s, shadow) for s in range(16)))
        self.assertEqual([s for s in range(8) if measure_actor_gradients(s, None)], [0, 4])


class Exploration(unittest.TestCase):
    def rows(self):
        return [dict(choices=[dict(kind='pass', id='a'), dict(kind='network', id='b')], intent=None,
                     behavior=dict(exploration_version=3), view=dict(context=dict(phase='action'))) for _ in range(3)]

    def test_each_row_is_computed_once_and_only_when_used(self):
        rows = self.rows()
        with mock.patch('brass_ai.recovery.updates.exploration_distribution', wraps=ordinary_distribution) as computed:
            mixture = ExplorationMixture(rows)
            self.assertEqual(computed.call_count, 0)
            first = mixture[1]
            self.assertIs(mixture[1], first)
            self.assertEqual(computed.call_count, 1)
            self.assertEqual(len(mixture), 3)
            self.assertEqual(first.dtype, np.float32)

    def test_values_match_the_eager_computation(self):
        from brass_ai.step5.policy import exploration_distribution
        rows = self.rows()
        mixture = ExplorationMixture(rows)
        expected = exploration_distribution(rows[0]['choices'], rows[0]['intent'], 3, 'action').astype(np.float32)
        np.testing.assert_array_equal(mixture[0], expected)


def ordinary_distribution(choices, intent, version, phase):
    from brass_ai.step5.policy import exploration_distribution
    return exploration_distribution(choices, intent, version, phase)


class Staging(unittest.TestCase):
    def preparation(self):
        target = ppo_preparation.Preparation.__new__(ppo_preparation.Preparation)
        target.cache = {}
        target.metrics = dict(input_bytes=0, staging_seconds=0.)
        target.stream = _Stream()
        return target

    def dataset(self, rows, q=None):
        class Data:
            pass
        data = Data()
        data.q = q
        data._batch_rows = lambda part, value_only: [rows[i] for i in part]
        return data

    def encoded(self):
        state = np.ones(3128, dtype=np.float32)
        actions = np.zeros((2, 327), dtype=np.float32)
        actions[:, 0] = 1.
        return Encoded(state, actions, np.arange(4, dtype=np.int64), ('a', 'b'), FEATURE_VERSION)

    def test_sealed_rows_are_never_hashed_while_staging(self):
        with mock.patch('brass_ai.recovery.replay_binding.check_action_selectors', create=True), \
             mock.patch('brass_ai.learning.local_entities.check_action_selectors'):
            row = seal_encoded(self.encoded(), _checked=True)
        target = self.preparation()
        data = self.dataset([row])
        with mock.patch.object(ppo_preparation, 'array_digest', side_effect=AssertionError('hashed a sealed row')):
            first = target._rows(data, [0], 0, False)
            again = target._rows(data, [0], 0, False)
        self.assertEqual(first, again)

    def test_plain_rows_are_still_hashed_and_a_changed_row_is_refused(self):
        row = self.encoded()
        target = self.preparation()
        data = self.dataset([row])
        target._rows(data, [0], 0, False)
        row.state[0] = 2.
        with self.assertRaises(ValueError):
            target._rows(data, [0], 0, False)

    def test_exploration_vectors_become_read_only_once_staged(self):
        with mock.patch('brass_ai.learning.local_entities.check_action_selectors'):
            row = seal_encoded(self.encoded(), _checked=True)
        q = np.array([.5, .5], dtype=np.float32)
        target = self.preparation()
        target._rows(self.dataset([row], q=[q]), [0], 1, False)
        self.assertFalse(q.flags.writeable)
        with self.assertRaises(ValueError):
            q[0] = 1.


class _Stream:
    def __init__(self):
        self.position = 0

    def tell(self):
        return self.position

    def write(self, data):
        self.position += len(bytes(data))

    def flush(self):
        pass


class Ingress(unittest.TestCase):
    def test_shards_are_sealed_as_they_arrive_and_returned_sealed(self):
        from brass_ai.recovery.ingress import StreamingCollection
        from brass_ai.recovery.replay_binding import checked_encoded
        from brass_ai.recovery import collector_runtime as R
        encoded = Encoded(np.ones(3128, dtype=np.float32), np.ones((1, 327), dtype=np.float32), np.arange(4), ('a',), FEATURE_VERSION)
        class Executor:
            def submit(self, function, task):
                future = Future()
                future.set_result((['h'], [('row', 'target', 'id', encoded)], {}, {}))
                return future
        record = dict(path='shard', job=1, sha256='s')
        with R.scope(dict(R.DEFAULTS, replay_bindings=True)), mock.patch('brass_ai.learning.local_entities.check_action_selectors'), \
                StreamingCollection(Executor(), 2, 'policy', {}) as stream:
            stream.submit([record])
            headers, data, diagnostics, metrics = stream.result(([record], 'policy', {}, None, None))
        self.assertTrue(checked_encoded(data[0][3]))
        self.assertEqual((headers, data[0][:3], diagnostics, metrics), (['h'], ('row', 'target', 'id'), {}, {}))

    def test_a_reader_failure_reaches_the_learner(self):
        from brass_ai.recovery.ingress import StreamingCollection
        class Executor:
            def submit(self, function, task):
                future = Future()
                future.set_exception(ValueError('reader failed'))
                return future
        record = dict(path='shard', job=1, sha256='s')
        with self.assertRaises(ValueError), StreamingCollection(Executor(), 2, 'policy', {}) as stream:
            stream.submit([record])
            stream.result(([record], 'policy', {}, None, None))


class Planning(unittest.TestCase):
    def members(self, root):
        members = []
        for index in range(3):
            path = root/f'model-{index}.pt'
            path.write_bytes(bytes([index])*1000)
            from brass_ai.bank import sha
            members.append(dict(path=path.relative_to(root).as_posix(), sha256=sha(path), model_id=f'm{index}'))
        members.append(dict(kind='v4-history', model_id='bot'))
        return members

    def test_the_whole_pool_is_hashed_once_and_bots_are_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=str(Path.cwd()/'artifacts')) as temp:
            root = Path(temp).resolve()
            members = self.members(root)
            verified = set()
            with mock.patch.object(ordinary, 'project_path', lambda r, p: Path(r)/p):
                ordinary.verify_pool_checkpoints(root, members, verified)
            self.assertEqual(verified, {'m0', 'm1', 'm2'})

    def test_a_changed_checkpoint_stops_the_plan(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=str(Path.cwd()/'artifacts')) as temp:
            root = Path(temp).resolve()
            members = self.members(root)
            (root/members[1]['path']).write_bytes(b'tampered')
            with mock.patch.object(ordinary, 'project_path', lambda r, p: Path(r)/p), self.assertRaises(ValueError):
                ordinary.verify_pool_checkpoints(root, members, set())

    def test_cpu_workers_do_not_need_opponents_loaded_when_a_bundle_exists(self):
        from brass_ai.recovery import collector_runtime as R
        service = mock.Mock(models={})
        entry = dict(model_id='m1')
        with R.scope(dict(R.DEFAULTS, native_inference=R.cpu_block())), \
                mock.patch('brass_ai.rust_backend.lab_cpu_models.bundle_ready', return_value=True):
            ordinary.register_opponent('root', service, entry, 'checkpoint.pt')
        service.defer_checkpoint.assert_called_once_with('m1', 'checkpoint.pt')
        service.register_checkpoint.assert_not_called()

    def test_a_missing_bundle_or_the_gpu_path_loads_the_checkpoint(self):
        from brass_ai.recovery import collector_runtime as R
        for settings, ready in ((dict(R.DEFAULTS, native_inference=R.cpu_block()), False), (dict(R.DEFAULTS), True)):
            service = mock.Mock(models={})
            with R.scope(settings), mock.patch('brass_ai.rust_backend.lab_cpu_models.bundle_ready', return_value=ready):
                ordinary.register_opponent('root', service, dict(model_id='m1'), 'checkpoint.pt')
            service.register_checkpoint.assert_called_once_with('checkpoint.pt')
            service.defer_checkpoint.assert_not_called()

    def test_a_loaded_opponent_is_left_alone(self):
        service = mock.Mock(models=dict(m1=object()))
        ordinary.register_opponent('root', service, dict(model_id='m1'), 'checkpoint.pt')
        service.register_checkpoint.assert_not_called()
        service.defer_checkpoint.assert_not_called()


class Deferred(unittest.TestCase):
    def service(self):
        from brass_ai.recovery.inference import InferenceService
        service = InferenceService.__new__(InferenceService, None)
        service.models = {}; service.deferred = {}; service.cache = {}; service.lock = threading.RLock()
        service.loaded = []
        service.register_checkpoint = service.loaded.append
        return service

    def test_a_deferred_checkpoint_loads_once_when_needed(self):
        service = self.service()
        service.defer_checkpoint('m', 'p.pt')
        service.materialize('m')
        service.materialize('m')
        self.assertEqual(service.loaded, ['p.pt'])

    def test_a_loaded_model_is_never_deferred(self):
        service = self.service()
        service.models['m'] = object()
        service.defer_checkpoint('m', 'p.pt')
        service.materialize('m')
        self.assertEqual(service.loaded, [])

    def test_a_retired_model_is_forgotten(self):
        service = self.service()
        service.defer_checkpoint('m', 'p.pt')
        service.trim(['other'])
        service.materialize_all()
        self.assertEqual(service.loaded, [])


class BundleFallback(unittest.TestCase):
    def test_ensure_materializes_a_deferred_model_before_exporting(self):
        import json
        import tempfile
        from brass_ai.rust_backend import lab_cpu_models as M
        with tempfile.TemporaryDirectory(dir=str(Path.cwd()/'artifacts')) as temp:
            root = Path(temp).resolve()
            models = {}
            def materialize(identity):
                models[identity] = 'loaded-model'
            def export(model, directory):
                self.assertEqual(model, 'loaded-model')
                directory.mkdir(parents=True)
                value = dict(contract=M.CONTRACT, model_id='m')
                (directory/'manifest.json').write_text(json.dumps(value), encoding='utf-8')
                return value
            with mock.patch.object(M, 'export_bundle', export):
                bundles, exported = M.ensure(root, models, ['m'], materialize=materialize)
            self.assertEqual(exported, 1)
            self.assertIn('m', bundles)


if __name__ == '__main__':
    unittest.main()
