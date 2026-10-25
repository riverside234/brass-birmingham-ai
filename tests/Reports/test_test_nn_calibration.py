"""Read-only metrics, unchanged weights and evidence-bearing checkpoint transport."""
import copy
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


calibration = module('calibration_adapter', 'scripts/test_nn_calibration.py')
exporter = module('checkpoint_export', 'scripts/export-test-nn.py')
TORCH = importlib.util.find_spec('torch') is not None


class OptionsTests(unittest.TestCase):
    def test_bounded_validation_cannot_be_disabled(self):
        for option in ({'games': 0}, {'heldout_games': 4}, {'rows_per_game': 100000},
                       {'threads': 128}, {'device': 'bad'}, {'seed': -1}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                calibration.options(option)
        self.assertEqual(calibration.options()['device'], 'cpu')
        self.assertEqual(calibration.options()['games'], 4)

    def test_sampling_is_deterministic_bounded_and_covers_groups(self):
        game = dict(game_id='authored', records=[dict(step=i, seat=i % 4,
            view=dict(context=dict(era='CanalEra' if i < 32 else 'RailEra', phase='action')),
            choices=[{}] if i % 3 else [{}, {}]) for i in range(64)])
        a = calibration.sample_steps(game, 32)
        self.assertEqual(a, calibration.sample_steps(game, 32))
        self.assertEqual(len(a), 32)
        self.assertEqual({game['records'][i]['seat'] for i in a}, set(range(4)))


@unittest.skipUnless(TORCH, 'PyTorch required for read-only utility tests')
class RealCalibrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.reward import specification
        from brass_ai.learning.utility_model import UtilityNet, utility_config
        torch.set_num_threads(1)
        torch.manual_seed(340)
        self.model = UtilityNet(utility_config(ModelConfig.preset('small'), specification())).eval()
        self.update = dict(effective_batch_size=16, microbatch_size=8,
                           max_candidate_cells=4096, bucket=True, grad_clip=1.)
        self.validation = self.data('validation')

    def data(self, prefix):
        import torch
        from brass_ai.learning.features import Encoded
        from brass_ai.recovery.updates import Dataset
        states = torch.zeros(16, self.model.config.state_dim)
        states[:, 0] = torch.arange(16) * .001
        actions = torch.zeros(16, 1, self.model.config.action_dim)
        mapping = torch.arange(4).repeat(16, 1)
        with torch.no_grad():
            out = self.model(states, actions, torch.ones(16, 1, dtype=torch.bool), mapping)
        rows = []
        cap = self.model.config.utility['insolvency']['max_cost']
        for i in range(16):
            record = dict(chosen=0, seat=i % 4, choices=[dict(id='one', kind='loan')],
                          behavior=dict(probability=1., temperature=1.),
                          view=dict(context=dict(era='CanalEra' if i % 8 < 4 else 'RailEra')),
                          _reward_return=(out[-1][i] + (.25 if i % 8 < 4 else -.5)).tolist(),
                          _reward_remaining=[cap] * 4)
            target = {name: values[i].tolist() for name, values in zip(('win_share', 'placement', 'vp'), out[1:4])}
            encoded = Encoded(states[i].numpy(), actions[i].numpy(), mapping[i].numpy(), ('one',))
            rows.append((record, target, (f'{prefix}-{i // 8}', i % 4), encoded))
        return Dataset(rows, 'cpu', policy=False)

    def test_exact_metrics_with_zero_training_and_unchanged_weights(self):
        import torch
        from brass_ai.learning.contracts import digest
        before = copy.deepcopy(self.model.state_dict())
        self.model.train()
        with patch('torch.optim.Adam', side_effect=AssertionError('must not create an optimizer')), \
             patch('torch.Tensor.backward', side_effect=AssertionError('must not backpropagate')):
            report = calibration.evaluate(self.model, self.validation, self.update)
        self.assertTrue(report['ready'])
        receipt = report['calibration_receipt']
        self.assertIs(receipt['ready'], True)
        self.assertEqual(receipt['steps'], 0)
        self.assertEqual(receipt['reward_id'], digest(self.model.config.utility))
        self.assertIsNone(self.model.utility_calibration)
        self.assertTrue(self.model.training)
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        metrics = report['metrics']
        self.assertEqual(metrics['games'], 2)
        for values, mse, bias, count in ((metrics['overall'], .15625, .125, 64),
            (metrics['by_era']['CanalEra'], .0625, -.25, 32),
            (metrics['by_era']['RailEra'], .25, .5, 32)):
            self.assertAlmostEqual(values['mse'], mse, places=6)
            self.assertAlmostEqual(values['bias'], bias, places=6)
            self.assertEqual(values['values'], count)

    def test_high_finite_error_is_reported_without_a_performance_gate(self):
        self.validation.utility.add_(100.)
        report = calibration.evaluate(self.model, self.validation, self.update)
        self.assertTrue(report['ready'])
        self.assertGreater(report['metrics']['overall']['mse'], 9000)

    def test_missing_era_or_nonfinite_targets_cannot_issue_a_receipt(self):
        for row in self.validation.rows:
            row['view']['context']['era'] = 'CanalEra'
        with self.assertRaisesRegex(ValueError, 'No utility validation targets'):
            calibration.evaluate(self.model, self.validation, self.update)
        self.validation = self.data('validation')
        self.validation.utility[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            calibration.evaluate(self.model, self.validation, self.update)
        self.assertIsNone(self.model.utility_calibration)

    def test_real_checkpoint_export_restore_cache_and_source_preservation(self):
        import torch
        from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory(prefix='nn-calib-', dir=ROOT / 'artifacts') as temporary:
            root = Path(temporary).resolve()
            run = root / exporter.DEFAULT_RUN
            run.mkdir(parents=True)
            source = run / 'generation-0012.pt'
            identity = save_checkpoint(source, self.model, optimizer=torch.optim.Adam(self.model.parameters()),
                                       step=12, training=dict(preserved='original metadata'))
            original = exporter.receipt(source)
            entry = dict(path=source.relative_to(root).as_posix(), generation=12,
                         model_id=identity, sha256=original['sha256'])
            state = dict(iteration=12, history=[entry], signature=dict(config=dict(host='unused')))
            (run / 'state.json').write_bytes(exporter.encode(state))
            raw_state = (run / 'state.json').read_bytes()
            bundle = exporter.export(root)['latest']
            for name in ('src/brass_ai/recovery/utility.py', 'src/brass_ai/recovery/updates.py',
                         'src/brass_ai/learning/model.py', 'src/brass_ai/learning/utility_model.py',
                         'src/brass_ai/recovery/execution.py', 'src/brass_ai/learning/features.py'):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            opt = calibration.options(dict(games=2))
            games = [dict(game_id=f'validation-{i}', setup_group=f'setup-{i}', complete=True,
                          path=f'validation-games/{i}.jsonl.gz') for i in range(2)]

            def collect_fixture(root, stage, *args):
                # Small opaque shards here; native complete-game collection is checked separately.
                for game in games:
                    path = stage / game['path']
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b'authored fixture')
                return self.validation.data, games, dict(authored_fixture=True)

            with patch.object(calibration, 'collect_data', side_effect=collect_fixture) as collect:
                candidate, derived = calibration.prepare(root, bundle, opt, 'latest', part_bytes=1024 * 1024)
                self.assertEqual(collect.call_count, 1)
            self.assertEqual(derived['model_id'], identity)
            self.assertNotEqual(derived['sha256'], original['sha256'])
            manifest = exporter.verify_bundle(candidate)
            self.assertEqual(manifest['recorded_checkpoint'], entry)
            self.assertTrue(manifest['calibration']['ready'])
            exporter.write_pointer(root / 'test_nn', 'latest', candidate, derived, 'read-only validation')
            restored = exporter.materialize(root, 'latest')
            checked, raw = load_checkpoint(restored)
            self.assertIs(raw['training']['utility_calibration']['ready'], True)
            self.assertEqual(exporter.receipt(source), original)
            self.assertEqual((run / 'state.json').read_bytes(), raw_state)
            _, parent_raw = load_checkpoint(source)
            self.assertIsNone(parent_raw['training']['utility_calibration'])
            self.assertEqual(raw['training']['preserved'], 'original metadata')
            self.assertEqual(raw['optimizer'], parent_raw['optimizer'])
            self.assertTrue(torch.equal(raw['rng'], parent_raw['rng']))
            self.assertTrue(all(torch.equal(v, raw['state_dict'][k]) for k, v in parent_raw['state_dict'].items()))
            with patch.object(calibration, 'collect_data', side_effect=AssertionError('must reuse accepted bundle')):
                reused, _ = calibration.prepare(root, bundle, opt, 'latest', part_bytes=1024 * 1024)
            self.assertEqual(reused, candidate)

    def test_failed_validation_keeps_original_pointer(self):
        # The orchestration must never publish an unqualified derivative.
        import test_export_test_nn
        case = test_export_test_nn.CheckpointExportTests()
        case.setUp()
        try:
            case.export()
            pointer = (case.root / 'test_nn/latest.json').read_bytes()
            with patch.object(test_export_test_nn.exporter, 'calibrate_bundle', side_effect=ValueError('incomplete game')):
                with self.assertRaisesRegex(ValueError, 'incomplete game'):
                    test_export_test_nn.exporter.export(case.root, calibration={})
            self.assertEqual((case.root / 'test_nn/latest.json').read_bytes(), pointer)
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main()
