"""Focused checks for the standalone joint-IL helper; no live games or PPO runs."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, digest
from brass_ai.learning.replay import split_for
from brass_ai.recovery import demonstrations as demos
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.replay import write_shard, cache_receipt
from brass_ai.step5.league import seeds_for, write_json, implementation_id

spec = importlib.util.spec_from_file_location('il_replay', SOURCE_ROOT / 'scripts/il_replay.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class JointReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def manifest(self, directory, number):
        teacher = next(iter(read(SOURCE_ROOT / 'training/opponents/pool.json')['training']))
        entries = []
        for i, partition in enumerate(('train', 'validation', 'test')):
            seed = seeds_for(160000 + number * 1000 + i * 100, 1, partition)[0][0]
            episode = game()
            episode.update(game_id=f'batch-{number}-{i}',
                setup_group=digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed)), opponents=[teacher]*4)
            episode['split'] = split_for(episode['setup_group'])
            for row in episode['records']:
                row['behavior'].update(kind='external', policy_id=teacher)
            path = directory / f'batch-{number}-{i}.jsonl.gz'
            write_shard(path, [episode], dict(purpose='training', seed=seed))
            entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(), sha256=sha(path),
                game_id=episode['game_id'], setup_group=episode['setup_group'], split=partition,
                semantic_sha256=digest(dict(group=episode['setup_group'], opponents=episode['opponents'],
                    decisions=[(r['seat'], r['choices'][r['chosen']]['id']) for r in episode['records']])),
                teachers={teacher: len(episode['records'])}, label_kind='qualified-v3-choice/1.0.0',
                context='neutral-predecision/1.0.0', transitions_verified=True, features=cache_receipt(path)))
        path = directory / f'manifest-{number}.json'
        write_json(path, dict(format='bc-dataset/1.0.0', compatibility=compatibility(),
            implementation=implementation_id(), label_kind='qualified-v3-choice/1.0.0',
            context='neutral-predecision/1.0.0', teachers=[teacher], calibration=None, entries=entries,
            partitions=dict(train=1, validation=1, test=1), excluded=[]))
        return path

    def receipt_run(self, directory, number, manifest):
        run = directory / f'v4-batch{number}'
        signature = dict(manifest_path=manifest.relative_to(SOURCE_ROOT).as_posix(),
                         manifest_sha256=sha(manifest), epochs=1)
        write_json(run / 'state.json', dict(signature=signature, epoch=1, stopped=False))
        write_json(run / 'report.json', dict(format='bc-training-report/1.0.0', signature=signature, epochs=1))
        return run

    def fixtures(self, directory, numbers=(7, 6)):
        return [self.receipt_run(directory, n, self.manifest(directory, n)) for n in numbers]

    def test_all_six_batches_preserve_entries_splits_and_loader_compatibility(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            directory = Path(temp)
            runs = self.fixtures(directory, (7, 6, 5, 4, 3, 2))
            merged = helper.merge_manifests(SOURCE_ROOT, runs)
            self.assertEqual(merged['partitions'], dict(train=6, validation=6, test=6))
            self.assertIsNone(merged['rehearsal']['latest_fraction'])
            self.assertEqual([s['name'] for s in merged['rehearsal']['sources']], [f'v4-batch{n}' for n in range(7, 1, -1)])
            originals = [entry for n in range(7, 1, -1) for entry in read(directory / f'manifest-{n}.json')['entries']]
            self.assertEqual(merged['entries'], originals)
            path = directory / 'joint.json'; write_json(path, merged)
            training, _ = demos.load_dataset(SOURCE_ROOT, path, 'train', workers=1, compact=True)
            validation, _ = demos.load_dataset(SOURCE_ROOT, path, 'validation', workers=1, compact=True)
            self.assertEqual(len(training), 24)
            self.assertEqual(len(validation), 24)
            self.assertTrue({x[2][0] for x in training}.isdisjoint({x[2][0] for x in validation}))
            self.assertEqual(len({x[2][0] for x in training}), 6)

    def test_overlap_is_deduplicated_and_latest_owns_shared_games(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            directory = Path(temp); latest = self.manifest(directory, 7)
            older = directory / 'older.json'; write_json(older, read(latest))
            runs = [self.receipt_run(directory, 7, latest), self.receipt_run(directory, 6, older)]
            merged = helper.merge_manifests(SOURCE_ROOT, runs)
            self.assertEqual(len(merged['entries']), 3)
            self.assertEqual(merged['rehearsal']['sources'][1]['duplicate_games'], 3)
            self.assertEqual(merged['rehearsal']['sources'][1]['selected_game_ids'], [])

    def test_tampered_receipts_conflicting_ids_and_partition_leakage_fail(self):
        for failure in ('manifest', 'split', 'game_id', 'calibration', 'audit'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
                directory = Path(temp); runs = self.fixtures(directory)
                path = directory / 'manifest-6.json'; value = read(path)
                if failure == 'split': value['entries'][0]['split'] = 'validation'
                if failure == 'game_id': value['entries'][0]['game_id'] = 'batch-7-0'
                if failure == 'calibration': value['calibration'] = dict(format='different')
                if failure == 'audit': value['entries'][0]['transitions_verified'] = False
                if failure == 'manifest': value['extra'] = True
                write_json(path, value)
                if failure != 'manifest': self.receipt_run(directory, 6, path)
                with self.assertRaises(ValueError): helper.merge_manifests(SOURCE_ROOT, runs)

    def test_prepare_is_idempotent_and_never_overwrites_sources_or_changed_recipe(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            directory = Path(temp); runs = self.fixtures(directory)
            before = {str(p): sha(p) for p in directory.rglob('*.json')}
            with contextlib.redirect_stdout(io.StringIO()):
                a, _ = helper.prepare(SOURCE_ROOT, runs[0], runs[1:], directory / 'experiment', directory / 'child')
                first = sha(a)
                b, _ = helper.prepare(SOURCE_ROOT, runs[0], runs[1:], directory / 'experiment', directory / 'child')
                self.assertEqual(first, sha(b))
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    helper.prepare(SOURCE_ROOT, runs[0], runs[1:], directory / 'experiment2', runs[0])
                with self.assertRaisesRegex(ValueError, 'recipe differs'):
                    helper.prepare(SOURCE_ROOT, runs[1], runs[:1], directory / 'experiment', directory / 'child')
            self.assertEqual(before, {p: sha(p) for p in before})

    def test_cli_keeps_existing_trainer_and_baseline_selection(self):
        args = helper.parser().parse_args(['train'])
        self.assertEqual(len(args.history_runs), 5)
        command = helper.training_command(args, Path('/data/not_backed_up/yxu209/ai-training/joint.json'))
        self.assertIn('--from-imitation-run', command)
        self.assertNotIn('--initial', command)
        self.assertIn('artifacts/step5/v4-batch7', command)
        rows = [dict(run='7', overall=dict(loss=1.0)), dict(run='6', overall=dict(loss=1.1)),
                dict(run='joint', overall=dict(loss=1.2))]
        self.assertEqual(helper.validation_choice(rows, 'joint')['selected_imitation_run'], '7')
        rows[-1]['overall']['loss'] = .9
        self.assertEqual(helper.validation_choice(rows, 'joint')['selected_imitation_run'], 'joint')
        self.assertFalse(helper.validation_choice(rows, 'joint')['gameplay_promoted'])
        rows[-1]['overall']['loss'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            helper.validation_choice(rows, 'joint')

    def test_aggregate_uses_policy_and_value_denominators_separately(self):
        reports = [dict(decisions=10, meaningful_decisions=2, policy_loss=3., value_loss=.2, meaningful_top1=.5),
                   dict(decisions=20, meaningful_decisions=8, policy_loss=1., value_loss=.4, meaningful_top1=.75)]
        result = helper.aggregate(reports, dict(actor_nonforced=True, value_weight=.5))
        self.assertAlmostEqual(result['policy_loss'], 1.4)
        self.assertAlmostEqual(result['value_loss'], 1/3)
        self.assertAlmostEqual(result['loss'], 1.4 + 1/6)
        self.assertAlmostEqual(result['meaningful_top1'], .7)

    def test_train_action_always_runs_original_model_validation(self):
        with patch.object(helper, 'prepare', return_value=(SOURCE_ROOT / 'fixture.json', {})), \
             patch.object(helper, 'launch') as launch, patch.object(helper, 'validate') as validate:
            helper.main(['train', '--root', str(SOURCE_ROOT), '--device', 'cpu'])
        launch.assert_called_once()
        self.assertIn('imitate', launch.call_args.args[0])
        self.assertNotIn('train', launch.call_args.args[0])
        validate.assert_called_once()

    def test_evaluation_keeps_common_setups_and_each_runs_own_initial(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            args = helper.parser().parse_args(['evaluate', '--root', str(SOURCE_ROOT), '--device', 'cpu',
                '--experiment', temp])
            calls = []
            def fake_launch(command, root):
                calls.append(command)
                destination = Path(command[command.index('--output') + 1])
                panels = command[command.index('--lineups') + 1:]
                write_json(destination / 'report.json', dict(results={p: dict(games=160, win_share=.25,
                    win_share_setup_bootstrap_95=[.2,.3], mean_rank=2.5, mean_vp=80) for p in panels}))
            with patch.object(helper, 'launch', side_effect=fake_launch), contextlib.redirect_stdout(io.StringIO()):
                result = helper.evaluate(args)
            self.assertEqual(len(calls), 3)
            self.assertEqual({c[c.index('--seed')+1] for c in calls}, {'1947000'})
            self.assertEqual({c[c.index('--setups')+1] for c in calls}, {'40'})
            for command in calls:
                candidate = Path(command[command.index('--candidate')+1])
                reference = Path(command[command.index('--references')+1])
                self.assertEqual(reference, candidate.parent / 'initial.pt')
                self.assertIn('--no-search', command)
            self.assertEqual(calls[-1][calls[-1].index('--lineups')+1:], ['recent-neural', 'mixed-v3'])
            self.assertFalse(result['gameplay_promoted'])

    def test_smoke_joint_bc_continuation_validation_and_existing_handoff(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            directory = Path(temp); cfg = dict(UPDATE_V21, effective_batch_size=4, microbatch_size=2, value_weight=.5)
            manifest6, manifest7 = self.manifest(directory, 6), self.manifest(directory, 7)
            run6, run7, child = [directory / name for name in ('v4-batch6', 'v4-batch7', 'joint')]
            with contextlib.redirect_stdout(io.StringIO()):
                demos.train_bc(SOURCE_ROOT, run6, manifest6, None, cfg, epochs=1, min_epochs=1, patience=1)
                demos.train_bc(SOURCE_ROOT, run7, manifest7, None, cfg, epochs=1, min_epochs=1, patience=1, from_imitation_run=run6)
                original_hashes = [sha(p / 'best.pt') for p in (run6, run7)]
                merged_path, merged = helper.prepare(SOURCE_ROOT, run7, [run6], directory / 'experiment', child)
                result = demos.train_bc(SOURCE_ROOT, child, merged_path, None, cfg,
                    epochs=1, min_epochs=1, patience=1, from_imitation_run=run7)
                _, origin = demos.completed_imitation(SOURCE_ROOT, child)
                self.assertEqual(origin['sha256'], result['candidate_sha256'])
                config = directory / 'config.json'; write_json(config, dict(threads=2, update=cfg, audit_workers=1))
                args = helper.parser().parse_args(['validate', '--root', str(SOURCE_ROOT), '--device', 'cpu',
                    '--latest-run', str(run7), '--history-runs', str(run6), '--baseline-runs', str(run6),
                    '--run', str(child), '--experiment', str(directory / 'experiment'), '--config', str(config)])
                comparison = helper.validate(args, merged_path, merged)
            self.assertEqual(original_hashes, [sha(p / 'best.pt') for p in (run6, run7)])
            self.assertEqual(len(comparison['models']), 3)
            self.assertTrue(all(set(r['per_source']) == {'v4-batch6', 'v4-batch7'} for r in comparison['models']))
            self.assertFalse(comparison['gameplay_promoted'])
            # The legacy handoff remains intact; a report does not rewrite best.pt.
            _, after = demos.completed_imitation(SOURCE_ROOT, child)
            self.assertEqual(origin, after)


if __name__ == '__main__': unittest.main()
