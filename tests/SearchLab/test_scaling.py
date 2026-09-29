"""Focused scheduling/config/cache checks; no qualification tournament."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import split_for
from brass_ai.searchlab import experiment as experiment
from brass_ai.searchlab.config import validate_config
from brass_ai.searchlab.runtime import sha
from brass_ai.step5.league import seeds_for

sys.path.insert(0, str(SOURCE_ROOT / 'tests/R2'))
from fixtures import game


class PolicyStub:
    kind = 'external'
    policy_id = 'search-fixture'
    telemetry = {'confirmed': 0, 'incomplete': 1}
    def close(self):
        pass


def episode(seed, policies):
    value = game()
    group = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed))
    value.update(game_id='fixture-'+str(seed),setup_group=group, split=split_for(group), opponents=[p.policy_id for p in policies])
    for row in value['records']:
        row['behavior'].update(kind=policies[row['seat']].kind, policy_id=policies[row['seat']].policy_id)
    return value


class SearchScalingTests(unittest.TestCase):
    def config(self, **overrides):
        config = read_json(SOURCE_ROOT / 'configs/searchlab/experiment.json')
        config.update(overrides)
        return validate_config(config)

    def temporary(self):
        parent = SOURCE_ROOT / 'artifacts/searchlab'
        parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)

    def test_larger_and_small_presets_and_cli(self):
        config = self.config()
        self.assertEqual((config['setups'], config['workers'], config['distill_games'], config['distill_workers']), (64, 8, 64, 8))
        self.assertEqual((config['batch_size'], config['minibatch_size'], config['simulations']), (64, 512, 128))
        small = validate_config(read_json(SOURCE_ROOT / 'configs/searchlab/small.json'))
        self.assertEqual((small['setups'], small['workers'], small['distill_games'], small['minibatch_size']), (1, 1, 2, 32))
        for field, value in [('workers', 0), ('distill_workers', True), ('minibatch_size', -1),
                              ('seconds', float('nan')), ('unused_workers', 32)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.config(**{field: value})
        spec = importlib.util.spec_from_file_location('searchlab_cli', SOURCE_ROOT / 'scripts/r4r5.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        args = cli.parser().parse_args(['compare', '--checkpoint', 'a', '--references', 'b', '--output', 'c'])
        self.assertIsNone(args.setups)
        self.assertIsNone(args.workers)
        args = cli.parser().parse_args(['distill-s3', '--checkpoint', 'a', '--output', 'c', '--workers', '3', '--minibatch-size', '256'])
        self.assertEqual((args.workers, args.minibatch_size), (3, 256))

    def test_all_paired_jobs_are_stable_and_parallel_results_cannot_promote(self):
        seeds = seeds_for(3910000, 64, 'validation')[0]
        jobs = experiment.comparison_jobs(seeds, ['s2', 'r4', 's3'])
        self.assertEqual(jobs, experiment.comparison_jobs(seeds, ['s2', 'r4', 's3']))
        self.assertEqual(len(jobs), 768)
        self.assertEqual(len({(j['seed'], j['seat'], j['variant']) for j in jobs}), 768)
        rows = [dict(variant=v, seed=s, seat=p, win=float(v=='s3'), rank=1, vp=100,
            over_budget=0, max_turn_seconds=.5,
            pottery=dict(builds=dict(Pottery=1), legal_build_opportunities=dict(Pottery=2)))
            for s in seeds[:30] for p in range(4) for v in ['s2', 's3']]
        self.assertTrue(experiment.paired_summary(rows, ['s2','s3'], 30, 'validation')['s3']['eligible'])
        self.assertFalse(experiment.paired_summary(rows[::-1], ['s2','s3'], 30, 'validation', workers=8)['s3']['eligible'])

    def test_comparison_worker_cache_and_failure_evidence(self):
        with self.temporary() as temporary:
            directory = Path(temporary)
            config = self.config(workers=1, threads=2)
            experiment._WORKER.update(root=SOURCE_ROOT, model=None, references=[None], config=config, device='cpu', mono=None)
            job = dict(variant='s3', seed=seeds_for(3910000,1,'validation')[0][0], seat=2,
                       directory=str(directory), spec='test-spec')
            def fake_match(root, seed, table, seat, cap, **kwargs):
                value = episode(seed, table)
                return value, experiment.result_from_game(value, seed, seat, cap)
            with patch.object(experiment, 'make_policy', return_value=PolicyStub()), \
                 patch.object(experiment, 'neural_policy', return_value=SimpleNamespace(policy_id='neural-fixture',kind='model')), \
                 patch.object(experiment, 'match', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    experiment.comparison_job(job)
            self.assertEqual(len(list(directory.glob('*.failure.json'))), 1)
            with patch.object(experiment, 'make_policy', return_value=PolicyStub()), \
                 patch.object(experiment, 'neural_policy', return_value=SimpleNamespace(policy_id='neural-fixture',kind='model')), \
                 patch.object(experiment, 'match', side_effect=fake_match):
                first = experiment.comparison_job(job)
            with patch.object(experiment, 'match', side_effect=AssertionError('Already cached')):
                cached = experiment.comparison_job(job)
            self.assertEqual(first, cached)
            self.assertEqual(len(list(directory.glob('*.failure.json'))), 1)
            with self.assertRaisesRegex(ValueError, 'Cached evidence changed'):
                experiment.comparison_job(dict(job, spec='changed'))
        experiment._WORKER.clear()

    def test_distillation_uses_scaled_update_settings_and_resumes_without_learning_twice(self):
        torch.set_num_threads(2)
        with self.temporary() as temporary:
            directory=Path(temporary);source=directory/'source.pt';output=directory/'result'
            torch.manual_seed(142)
            identity=save_checkpoint(source,FourSeatNet(ModelConfig.preset('small')))
            initial=sha(source)
            config=self.config(distill_games=2,distill_workers=1,threads=2)
            sampling=[]
            def collect(host, seeds, policies, **kwargs):
                sampling.append(kwargs['sampling_seed'])
                return [episode(s,policies) for s in seeds]
            with patch.object(experiment,'require_gate',return_value={'identity':'fixture-gate'}), \
                 patch.object(experiment,'make_policy',return_value=PolicyStub()), \
                 patch('brass_ai.learning.runner.collect_games',side_effect=collect), \
                 patch.object(experiment,'distill_update',wraps=experiment.distill_update) as update:
                report=experiment.distill_s3(SOURCE_ROOT,source,output,config)
            self.assertEqual(len(set(sampling)),2)
            self.assertEqual(update.call_args.kwargs['batch_size'],512)
            self.assertEqual(update.call_args.kwargs['max_candidate_cells'],131072)
            self.assertEqual(update.call_args.kwargs['steps'],8)
            self.assertFalse(report['updated'])  # No confirmed targets in this fixture.
            self.assertFalse(report['promoted'])
            self.assertEqual(sha(source),initial)
            self.assertEqual(load_checkpoint(output/'candidate.pt')[1]['model_id'],identity)
            with patch.object(experiment,'require_gate',return_value={'identity':'fixture-gate'}), \
                 patch.object(experiment,'run_jobs',side_effect=AssertionError('Already cached')), \
                 patch.object(experiment,'distill_update',side_effect=AssertionError('Already updated')):
                self.assertEqual(experiment.distill_s3(SOURCE_ROOT,source,output,config),report)
                with self.assertRaisesRegex(ValueError,'spec changed'):
                    experiment.distill_s3(SOURCE_ROOT,source,output,dict(config,minibatch_size=256))


if __name__ == '__main__':
    unittest.main()
