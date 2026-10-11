"""Scoped cross-run IL evaluation checks; native four-game smoke is opt-in."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fixtures import game, result
from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.step5.league import seeds_for, write_json

spec = importlib.util.spec_from_file_location('il_head_to_head', SOURCE_ROOT / 'scripts/il_head_to_head.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class HeadToHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def spec(self):
        return dict(origins=[dict(model_id='joint'), dict(model_id='original6')],
                    seeds=seeds_for(1947000, 2, 'validation')[0], config=dict(partition='validation'))

    def episodes(self, spec, scores=None):
        jobs = helper.build_jobs(SOURCE_ROOT / 'artifacts/fixture', spec)
        games, records = [], []
        for job in jobs:
            episode = game()
            ids = [p['policy_id'] for p in job['policies']]
            rotation = job['rotation']
            episode.update(game_id=f"game-{job['index']}", split='validation',
                setup_group=digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=job['seed'])),
                opponents=ids[-rotation:]+ids[:-rotation] if rotation else ids,
                result=result(scores or [(80, 10, 5)] * 4))
            games.append(episode)
            records.append(dict(job=job['index'], game_id=episode['game_id']))
        return jobs, games, records

    def test_defaults_seeded_rotations_and_exact_opponents(self):
        args = helper.parser().parse_args([])
        self.assertEqual(args.reference_run, 'artifacts/step5/v4-batch6')
        self.assertEqual(args.candidate_run, 'artifacts/step5/v4-batch7-joint')
        self.assertEqual(args.setups, 40)
        spec = self.spec()
        jobs = helper.build_jobs(Path('fixture'), spec)
        self.assertEqual(jobs, helper.build_jobs(Path('fixture'), spec))
        self.assertEqual(len(jobs), 8)
        for seed in spec['seeds']:
            self.assertEqual([j['rotation'] for j in jobs if j['seed'] == seed], [0, 1, 2, 3])
        for job in jobs:
            self.assertEqual([p['model_id'] for p in job['policies']], ['joint'] + ['original6'] * 3)
            self.assertTrue(all(p['kind'] == 'neural' and p['exploration'] == 0 for p in job['policies']))
            self.assertEqual(job['provenance']['purpose'], 'evaluation')

    def test_official_ties_tiebreakers_and_shuffled_attribution(self):
        spec = self.spec()
        jobs, games, records = self.episodes(spec)
        rows = helper.checked_outcomes(games[::-1], records[::-1], jobs, spec)
        report = helper.summarize_outcomes(rows)
        self.assertEqual(report['win_share'], .25)
        self.assertEqual(report['seat_counts'], {str(s): 2 for s in range(4)})
        self.assertEqual(report['win_share_setup_bootstrap_95'], [.25, .25])
        jobs, games, records = self.episodes(spec, [(80, 10, 5), (80, 10, 5), (80, 9, 100), (80, 9, 99)])
        rows = helper.checked_outcomes(games, records, jobs, spec)
        self.assertEqual([r['win'] for r in rows[:4]], [.5, .5, 0, 0])
        self.assertEqual([r['rank'] for r in rows[:4]], [1, 1, 3, 4])

    def test_wrong_opponent_missing_game_and_rotation_rejected(self):
        for failure in ('batch7', 'two-candidates', 'missing', 'seat'):
            with self.subTest(failure=failure):
                spec = self.spec(); jobs, games, records = self.episodes(spec)
                if failure == 'batch7': games[0]['opponents'][1] = 'original7'
                if failure == 'two-candidates': games[0]['opponents'][1] = games[0]['opponents'][0]
                if failure == 'missing': records.pop()
                if failure == 'seat': records[0]['job'], records[1]['job'] = records[1]['job'], records[0]['job']
                with self.assertRaises(ValueError): helper.checked_outcomes(games, records, jobs, spec)

    def test_completed_origins_keep_identity_and_reject_incomplete(self):
        model = SimpleNamespace(config=ModelConfig.preset('small'))
        expected = [dict(model_id='joint', bc_run='joint'), dict(model_id='six', bc_run='six')]
        with patch.object(helper, 'completed_imitation', side_effect=[(model, o) for o in expected]) as loader:
            self.assertEqual(helper.frozen_origins(SOURCE_ROOT, 'joint', 'six'), expected)
            self.assertEqual([call.args[1] for call in loader.call_args_list], ['joint', 'six'])
        with patch.object(helper, 'completed_imitation', side_effect=ValueError('incomplete')):
            with self.assertRaisesRegex(ValueError, 'incomplete'): helper.frozen_origins(SOURCE_ROOT, 'joint', 'six')
        with patch.object(helper, 'completed_imitation', return_value=(model, expected[0])):
            with self.assertRaisesRegex(ValueError, 'identical'): helper.frozen_origins(SOURCE_ROOT, 'joint', 'six')

    def test_output_protection_and_frozen_cache(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            base = Path(temp)
            for output in (base, base / 'joint', base / 'joint/eval'):
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    helper.checked_output(SOURCE_ROOT, output, base / 'joint', base / 'six')
            args = helper.parser().parse_args(['--root', str(SOURCE_ROOT), '--device', 'cpu', '--setups', '2',
                '--candidate-run', str(base / 'joint'), '--reference-run', str(base / 'six'), '--output', str(base / 'eval')])
            def fake_collect(root, directory, spec, **kwargs):
                jobs, games, records = self.episodes(spec)
                rows = helper.checked_outcomes(games, records, jobs, spec)
                return dict(spec=spec, failed_games=0, results=dict(**helper.summarize_outcomes(rows), outcomes=rows))
            with patch('brass_ai.step5.assets.require_preflight'), \
                 patch.object(helper, 'frozen_origins', return_value=self.spec()['origins']), \
                 patch.object(helper, 'collect', side_effect=fake_collect) as collect, \
                 contextlib.redirect_stdout(io.StringIO()):
                summary = helper.evaluate(args)
                self.assertEqual(helper.evaluate(args), summary)
                collect.assert_called_once()
                self.assertFalse(summary['gameplay_promoted'])
                self.assertFalse(summary['lower_95_above_symmetric'])
                self.assertEqual(summary['reference']['model_id'], 'original6')
                self.assertTrue((base / 'eval/report.json').is_file())
                args.seed += 1
                with self.assertRaisesRegex(ValueError, 'specification changed'): helper.evaluate(args)

    @unittest.skipUnless(os.environ.get('BRASS_IL_HEAD_TO_HEAD_NATIVE') == '1', 'Opt-in four-game native smoke')
    def test_native_four_seats_with_frozen_fixture_models(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            base = Path(temp); origins = []
            for index, name in enumerate(('joint-fixture', 'six-fixture')):
                torch.manual_seed(19 + index)
                path = base / name / 'model.pt'
                identity = save_checkpoint(path, FourSeatNet(ModelConfig.preset('small')))
                origins.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(), model_id=identity, sha256=sha(path)))
            config = read(SOURCE_ROOT / 'configs/step5/recovery-smoke.json')
            spec = dict(format=helper.FORMAT, purpose='evaluation', origins=origins,
                        seeds=seeds_for(1947000, 1, 'validation')[0], config=dict(config, partition='validation'))
            report = helper.collect(SOURCE_ROOT, base / 'evaluation', spec, device='cpu')
            self.assertEqual(report['results']['games'], 4)
            self.assertEqual(report['results']['seat_counts'], {str(s): 1 for s in range(4)})
            self.assertEqual(report['failed_games'], 0)
            self.assertFalse(report['gameplay_promoted'])
            for origin in origins:
                self.assertEqual(sha(SOURCE_ROOT / origin['path']), origin['sha256'])
            # Identical interrupted/completed collection inputs reuse authoritative shards.
            repeated = helper.collect(SOURCE_ROOT, base / 'evaluation', spec, device='cpu')
            self.assertEqual(repeated['cached_games'], 4)
            self.assertEqual(repeated['results'], report['results'])


if __name__ == '__main__':
    unittest.main()
