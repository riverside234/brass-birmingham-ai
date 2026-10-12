"""Evaluation scale, explicit rotations, cache isolation and worker ownership."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import read_shards, split_for
from brass_ai.learning.runner import RandomPolicy
from brass_ai.recovery.replay import write_shard, cache_receipt
from brass_ai.step5.collector import collect_games
from brass_ai.step5.cli import parser
from brass_ai.step5.evaluation import (evaluate, evaluation_config, evaluation_jobs,
                                     validate_evaluation_config)
from brass_ai.step5.league import seeds_for
from brass_ai.learning.contracts import digest
from fixtures import game, position, result


class TinyHost:
    """Authored four-decision host to compare old/new collector routing."""
    def __init__(self, *args, **kwargs):
        self.step = 0
    def context(self):
        c = position(self.step % 4)[0]['context']
        c.update(revision=self.step, terminal=self.step == 4)
        return c
    def reset(self, seed):
        self.step = 0
        return self.context()
    def observe(self, seat):
        view = position(seat)[0]
        view['context'] = self.context()
        return view
    def choices(self):
        return position(self.step % 4)[1]
    def apply(self, context, choice):
        self.step += 1
        return self.context()
    def call(self, operation):
        assert operation == 'result'
        return result()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def fixture_collect(actor, jobs):
    records = []
    for job in jobs:
        seed,rotation,policies=job['seed'],job['rotation'],job['policies']
        path=Path(job['path'])
        episode = game()
        group = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed))
        assigned = [policies[(seat - rotation) % 4] for seat in range(4)]
        episode.update(game_id=uuid.uuid4().hex, setup_group=group, split=split_for(group),
                       opponents=[p['policy_id'] for p in assigned])
        for row in episode['records']:
            row['behavior'].update(kind='model', policy_id=assigned[row['seat']]['policy_id'])
        cached=path.exists()
        if not cached:write_shard(path,[episode],job['provenance'])
        else:episode=read_shards([path])[0]
        records.append(dict(job=job['index'],game_id=episode['game_id'],path=str(path),sha256=sha(path),features=cache_receipt(path),cached=cached))
    return records[::-1]  # Completion order differs from scheduling.


class EvaluationScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_defaults_cli_overrides_and_strict_config(self):
        config = read(SOURCE_ROOT / 'configs/step5/evaluation.json')
        self.assertEqual((config['setups'], config['workers'], config['batch_size']), (30, 28, 64))
        self.assertEqual(config['legacy_workers'], 24)
        for key, value in [('setups', 0), ('workers', True), ('legacy_workers', -1), ('partition', 'train'),
                           ('include_legacy', 1), ('focus', 'mixed')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation_config(dict(config, **{key: value}))
        args = parser().parse_args(['evaluate', '--candidate', 'a', '--references', 'b', '--output', 'c',
                                   '--setups', '2', '--no-include-legacy', '--no-search'])
        resolved = evaluation_config(config, setups=args.setups, include_legacy=args.include_legacy,
                                     use_search=args.search)
        self.assertEqual(resolved['setups'], 2)
        self.assertFalse(resolved['include_legacy'] or resolved['use_search'])
        self.assertIsNone(args.config)
        self.assertNotIn('minibatch_size', config)

    def test_complete_heldout_matrix_and_stable_sampling_seeds(self):
        seeds = seeds_for(910000, 64, 'validation')[0]
        jobs = evaluation_jobs(seeds)
        self.assertEqual(len(jobs), 256)
        self.assertEqual(len({(j['seed'], j['seat']) for j in jobs}), 256)
        self.assertTrue(all(j['sampling_seed'] == j['seed'] * 7 + j['seat'] for j in jobs))
        for seed in seeds:
            self.assertEqual({j['seat'] for j in jobs if j['seed'] == seed}, set(range(4)))

    def test_batched_rotations_match_original_individual_rng_routing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'BrassHost.exe').write_bytes(b'fake-host')
            pins = dict(rules='steam-brass-patch-1.1.0', host_sha256=hashlib.sha256(b'fake-host').hexdigest())
            (directory / 'host-1.0.0.json').write_text(json.dumps(pins))
            policies = [RandomPolicy() for _ in range(4)]
            for seat, policy in enumerate(policies):
                policy.policy_id = 'seat-policy-' + str(seat)
            jobs = evaluation_jobs([913701])
            with patch('brass_ai.step5.collector.CONFIG_DIR', directory), patch('brass_ai.step5.collector.RulesHost', TinyHost), \
                 patch('brass_ai.learning.runner.CONFIG_DIR', directory), patch('brass_ai.learning.runner.RulesHost', TinyHost):
                batched = collect_games(directory, [j['seed'] for j in jobs], policies, workers=4, parallel_hosts=True,
                                        rotations=[j['seat'] for j in jobs], sampling_seeds=[j['sampling_seed'] for j in jobs])
                singles = []
                for job in jobs:
                    table = [policies[(s - job['seat']) % 4] for s in range(4)]
                    singles += collect_games(directory, [job['seed']], table, workers=1, sampling_seed=job['sampling_seed'])
            def semantic(g):
                return (g['opponents'], g['result'], [{k: v for k, v in row.items() if k != 'seconds'} for row in g['records']])
            key = lambda g: g['opponents'].index('seat-policy-0')
            self.assertEqual([semantic(g) for g in sorted(batched, key=key)], [semantic(g) for g in sorted(singles, key=key)])

    def assets(self, directory):
        paths = []
        for seed in (3901, 3902, 3903):
            torch.manual_seed(seed)
            path = directory / (str(seed) + '.pt')
            save_checkpoint(path, FourSeatNet(ModelConfig.preset('small')), training=dict(mode='evaluation-fixture',fresh_run=directory.relative_to(SOURCE_ROOT).as_posix()))
            paths.append(path)
        host = directory / 'host'; host.mkdir()
        (host / 'BrassHost.exe').write_bytes(b'fake-host')
        config = evaluation_config(read(SOURCE_ROOT/'configs/step5/recovery-smoke.json'),setups=2,
            include_legacy=False,lineups=['recent-neural','mixed-neural'])
        config.update(host=str(host),threads=2)
        config['retention']=dict(config['retention'],enabled=False)
        return paths, config

    def test_batched_evaluation_resume_and_holdout_enforcement(self):
        parent = SOURCE_ROOT / 'artifacts/step5'; parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            directory = Path(temporary)
            paths, config = self.assets(directory)
            output = directory / 'evaluation'
            with patch('brass_ai.recovery.evaluation.ActorPool.collect', autospec=True, side_effect=fixture_collect) as collect:
                report = evaluate(SOURCE_ROOT, output, paths[0], paths[1:], config)
            self.assertEqual(collect.call_count, 2)
            for entry in report['results'].values():
                self.assertEqual(entry['games'], 8)
                self.assertEqual(entry['seat_counts'], {str(s): 2 for s in range(4)})
            self.assertFalse(report['promotion_eligible'])
            with patch('brass_ai.recovery.evaluation.ActorPool.collect', autospec=True, side_effect=fixture_collect), \
                 patch(__name__+'.write_shard',side_effect=AssertionError('Replay should be cached')):
                resumed = evaluate(SOURCE_ROOT, output, paths[0], paths[1:], config)
            self.assertEqual(report['results'], resumed['results'])
            with self.assertRaisesRegex(ValueError, 'specification changed'):
                evaluate(SOURCE_ROOT, output, paths[0], paths[1:], dict(config, seed=config['seed']+1))
            with self.assertRaisesRegex(ValueError, 'Evaluation-only'):
                read_shards(list(output.glob('*.jsonl.gz')), partition='train')

    def test_missing_games_never_publish_complete_evaluation(self):
        parent = SOURCE_ROOT / 'artifacts/step5'; parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            directory = Path(temporary)
            paths, config = self.assets(directory)
            output = directory / 'evaluation'
            def missing(*args, **kwargs):
                return fixture_collect(*args, **kwargs)[:-1]
            with patch('brass_ai.recovery.evaluation.ActorPool.collect', autospec=True, side_effect=missing), self.assertRaisesRegex(ValueError, 'Missing'):
                evaluate(SOURCE_ROOT, output, paths[0], paths[1:], config)
            self.assertFalse((output / 'report.json').exists())
            self.assertTrue(list(output.glob('*.jsonl.gz')))  # Completed partial work remains available to resume.

    def test_search_workers_own_private_evaluator_channels(self):
        from types import SimpleNamespace
        from brass_ai.recovery.execution import make_policy
        from brass_ai.recovery.config import EXECUTION_V21
        from brass_ai.search.common import SearchConfig
        owners=[]
        class SearchStub:
            def __init__(self, *args, **kwargs):
                self.evaluator=kwargs['evaluator'];self.game=SimpleNamespace(host=object());self.simulation_games=[]
                owners.append(self)
        def forest(policy,workers,role,factory):
            self.assertEqual((workers,role),(4,'lab'))
            policy.replicas=[factory(i) for i in range(workers)]
            return policy
        endpoints=[object() for _ in range(5)]
        spec=dict(kind='masked_ismcts',model_id='a'*64,search=SearchConfig().to_dict(),cpu_role='lab')
        with patch('brass_ai.recovery.execution.MaskedSearchPolicy',SearchStub), \
             patch('brass_ai.recovery.parallel.parallelize',side_effect=forest), \
             patch('brass_ai.recovery.teacher_schedule.work_search',side_effect=lambda p,w:p) as wrap, \
             patch('brass_ai.recovery.execution.instrument_host'):
            policy=make_policy(spec,SOURCE_ROOT,endpoints,EXECUTION_V21,None)
        self.assertEqual(len(owners),5)
        self.assertEqual([p.evaluator.endpoint for p in owners],endpoints)
        self.assertEqual(len(policy.replicas),4)
        self.assertEqual(wrap.call_count,1)  # One whole-turn budget, not one per replica.


if __name__ == '__main__':
    unittest.main()
