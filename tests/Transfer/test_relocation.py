"""Relocation with absent source roots, real checkpoints and signed evidence."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
import zipfile

import test_step5_transfer as fixtures
from brass_ai.learning.paths import (RELOCATION_FILE, bound_reference, make_relocation,
    read_relocation, resolve_project_path, source_roots)

transfer = fixtures.transfer
ROOT = fixtures.ROOT


class RelocationTests(unittest.TestCase):
    def setUp(self):
        fixtures.TransferTests.setUp(self)
        self.put('training/opponents/pool.json', dict(training={}, evaluation={}))
    put = fixtures.TransferTests.put
    entry = fixtures.TransferTests.entry
    export = fixtures.TransferTests.export
    archive_bytes = fixtures.TransferTests.archive_bytes

    def remove(self, path):
        path = Path(path).resolve()
        self.assertTrue(path.is_relative_to(self.area) and path != self.area)
        shutil.rmtree(path)
        self.assertFalse(path.exists())

    def absolute_references(self):
        """Mix absolute and relative fields as existing schemas permit."""
        def absolute(name):
            return str(self.root / name)
        for entry in (self.state['history'][-1], *self.state['population']):
            entry['path'] = absolute(entry['path'])
        cfg = self.state['signature']['config']
        cfg['host'] = absolute(cfg['host'])
        cfg['rehearsal_manifest'] = absolute(cfg['rehearsal_manifest'])
        dataset = transfer.read(self.root / 'training/datasets/demo.json')
        dataset['entries'][0]['path'] = absolute(dataset['entries'][0]['path'])
        self.put('training/datasets/demo.json', dataset)
        origin = self.state['signature']['imitation_source']
        for field in ('bc_run', 'path', 'manifest'):
            origin[field] = absolute(origin[field])
        origin['manifest_sha256'] = transfer.sha(self.root / 'training/datasets/demo.json')
        self.put('artifacts/step5/selfplay/evaluations/pending/spec.json',
                 dict(entries=[self.state['history'][-1]]))
        self.put('artifacts/step5/selfplay/evaluations/pending/replay.jsonl.gz', b'pending evidence')
        self.put('artifacts/step5/selfplay/pins.json', dict(paths=[absolute('artifacts/step5/selfplay/initial.pt')]))
        self.put('artifacts/step5/selfplay/state.json', self.state)

    def test_absolute_relative_dependencies_and_repeated_move_without_sources(self):
        self.absolute_references()
        self.export()
        manifest = transfer.inspect_bundle(self.bundle)
        original = self.root
        archived = {r['path']: (original / r['path']).read_bytes() for r in manifest['files']
                    if (original / r['path']).exists()}
        self.remove(original)
        second = self.area / 'different-project'
        result = transfer.restore(second, self.bundle, relocate=True)
        self.assertTrue(result['verified'] and result['relocated'])
        self.assertNotIn(str(original), result['resume'])
        for name, content in archived.items():
            self.assertEqual((second / name).read_bytes(), content, name)
        self.assertTrue(transfer.restore(second, self.bundle, verify_only=True)['verified'])
        self.assertEqual(source_roots(second), [str(second), str(original)])
        from brass_ai.recovery.retention import references
        pins = references(second, second / 'artifacts/step5/selfplay')
        self.assertIn(second / 'artifacts/step5/selfplay/generation-0002.pt', pins)
        # A new Python process / later session reads the sidecar without env aliases.
        command = [sys.executable, '-c',
                   'import sys; from brass_ai.learning.paths import resolve_project_path; '
                   'print(resolve_project_path(sys.argv[1],sys.argv[2]))',
                   str(second), str(original / 'training/datasets/demo.json')]
        output = subprocess.check_output(command, cwd=ROOT, text=True)
        self.assertEqual(output.strip(), str(second / 'training/datasets/demo.json'))
        moved_bundle = self.area / 'second.zip'
        transfer.export(second, str(original / 'artifacts/step5/selfplay'), moved_bundle,
                        dependencies='bundled', archive_format='zip', part_mb=0)
        receipt = (second / RELOCATION_FILE).read_bytes()
        self.assertTrue(transfer.restore(second, moved_bundle, verify_only=True)['verified'])
        self.assertEqual((second / RELOCATION_FILE).read_bytes(), receipt)
        self.remove(second)
        third = self.area / 'third-project'
        transfer.restore(third, moved_bundle, relocate=True)
        self.assertEqual(source_roots(third), [str(third), str(second), str(original)])
        for origin in (original, second, third):
            self.assertEqual(resolve_project_path(third, str(origin / 'artifacts/step5/selfplay/state.json')),
                             third / 'artifacts/step5/selfplay/state.json')
        for name, content in archived.items():
            self.assertEqual((third / name).read_bytes(), content, name)

    def test_real_resume_preserves_adam_rng_weights_binding_and_progress(self):
        import torch
        from brass_ai.learning.checkpoint import save_checkpoint, resume_optimizer
        from brass_ai.learning.contracts import digest
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet, utility_config
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.recovery.resume import load_resume
        torch.set_num_threads(2)
        self.absolute_references()
        cfg = dict(transfer.read(ROOT / 'configs/step5/lab.json'), **self.state['signature']['config'])
        self.state['signature']['config'] = cfg
        model = UtilityNet(utility_config(ModelConfig.preset('small'), reward_spec(cfg)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        sum(p.square().mean() for p in model.parameters()).backward()
        optimizer.step()
        checkpoint = self.run / 'generation-0002.pt'
        identity = save_checkpoint(checkpoint, model, optimizer, step=2,
            training=dict(mode='step5-population-ppo', fresh_run=str(self.run), signature=digest(self.state['signature'])))
        self.state['history'][-1].update(sha256=transfer.sha(checkpoint), model_id=identity)
        self.put('artifacts/step5/selfplay/evaluations/pending/spec.json', dict(entries=[self.state['history'][-1]]))
        self.put('artifacts/step5/selfplay/state.json', self.state)
        state_bytes = (self.run / 'state.json').read_bytes()
        checkpoint_bytes = checkpoint.read_bytes()
        self.export()
        self.remove(self.root)
        destination = self.area / 'resumed-project'
        transfer.restore(destination, self.bundle, relocate=True)
        run = destination / 'artifacts/step5/selfplay'
        state = transfer.read(run / 'state.json')
        restored, raw, changed = load_resume(destination, run, state, state['signature'])
        resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=9e-2)
        resume_optimizer(raw, resumed_optimizer)
        self.assertFalse(changed)
        self.assertEqual(state, self.state)
        self.assertEqual((run / 'state.json').read_bytes(), state_bytes)
        self.assertEqual((run / checkpoint.name).read_bytes(), checkpoint_bytes)
        self.assertTrue(torch.equal(raw['rng'], torch.get_rng_state()))
        # Restoring RNG predicts the same next random draw, not only same bytes.
        expected = torch.rand(8)
        resume_optimizer(raw, resumed_optimizer)
        self.assertTrue(torch.equal(expected, torch.rand(8)))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
        before, after = optimizer.state_dict(), resumed_optimizer.state_dict()
        self.assertEqual(before['param_groups'], after['param_groups'])
        for key, values in before['state'].items():
            for name, value in values.items():
                torch.testing.assert_close(after['state'][key][name], value, rtol=0, atol=0)
        bad = copy.deepcopy(state)
        bad['signature']['config']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'bound to this run signature'):
            load_resume(destination, run, bad, bad['signature'])
        # A valid state hash cannot excuse modified checkpoint bytes.
        (run / checkpoint.name).write_bytes(checkpoint_bytes + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
            load_resume(destination, run, state, state['signature'])

    def test_legacy_zip_and_zstd_split_archives_need_no_reexport(self):
        self.absolute_references()
        archives = []
        for recipe, fmt, split in ((1, 'zip', False), (2, 'zip', True), (2, 'zstd', True), (3, 'zstd', False)):
            with self.subTest(recipe=recipe, fmt=fmt, split=split):
                self.state['signature']['implementation'] = transfer.source_identity(self.root, recipe=recipe)
                self.put('artifacts/step5/selfplay/state.json', self.state)
                name = self.area / f'archive-{recipe}-{fmt}'
                with patch.object(transfer, 'SOURCE_RECIPE', recipe):
                    transfer.export(self.root, self.run, name, dependencies='bundled', archive_format=fmt, part_mb=0)
                # Reconstruct historical metadata: no relocation roots, recipe 1
                # omitted, original format used project/ paths rather than objects/.
                payloads = self.archive_bytes(name)
                manifest = json.loads(payloads['transfer.json'])
                manifest.pop('path_roots')
                if recipe == 1:
                    manifest.pop('source_recipe')
                    for record in manifest['files']:
                        payloads['project/' + record['path']] = payloads[transfer.payload_name(manifest, record)]
                        record.pop('external')
                    payloads = {k:v for k,v in payloads.items() if k.startswith('project/')}
                    manifest['format'] = transfer.OLD_FORMAT
                manifest['id'] = transfer.digest({k:v for k,v in manifest.items() if k != 'id'})
                payloads['transfer.json'] = transfer.encode(manifest)
                legacy = self.area / (name.name + '-legacy')
                with transfer.create_archive(legacy, fmt, 3, 1) as out:
                    for key, content in payloads.items():
                        transfer.add_member(out, key, content, len(content))
                if split:
                    legacy, count = transfer.split_bundle(legacy, self.area / (legacy.name + '-parts'), 1000)
                    self.assertGreater(count, 1)
                archives.append(legacy)
        self.remove(self.root)
        for i, archive in enumerate(archives):
            destination = self.area / f'restored-{i}'
            result = transfer.restore(destination, archive, relocate=True)
            self.assertTrue(result['verified'])
            self.assertEqual(result['requires_code_update'], i != 3)

    def test_completed_imitation_ancestry_and_spawned_rehearsal_readers(self):
        import torch
        from fixtures import game
        from brass_ai.learning.checkpoint import save_checkpoint
        from brass_ai.learning.contracts import compatibility, digest
        from brass_ai.learning.model import FourSeatNet, ModelConfig
        from brass_ai.recovery.demonstrations import completed_imitation, load_dataset
        from brass_ai.recovery.replay import write_shard
        from brass_ai.step5.league import implementation_id, seeds_for
        torch.set_num_threads(2)
        self.put('training/opponents/pool.json', dict(training={}, evaluation={}))
        identity = dict(bank=dict(parents=dict(parent='historical-qualified-heuristic')))
        evidence = self.put('artifacts/demos/calibration.json', dict(identity=identity, training_eligible=True))
        teacher = 'lab-v3/' + digest(dict(child=identity['bank']['parents']['parent'], identity=identity))
        entries = []
        for index in range(2):
            episode = game()
            seed = seeds_for(95000 + index * 100, 1, 'train')[0][0]
            episode.update(game_id=f'demo-{index}', setup_group=digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed)),
                           split='train', opponents=[teacher] * 4)
            for row in episode['records']:
                row['behavior'].update(kind='external', policy_id=teacher)
            path = self.root / f'artifacts/demos/data-{index}.jsonl.gz'
            write_shard(path, [episode], dict(purpose='training', seed=seed))
            entries.append(dict(path=str(path) if index == 0 else path.relative_to(self.root).as_posix(),
                sha256=transfer.sha(path), game_id=episode['game_id'], setup_group=episode['setup_group'],
                split='train', transitions_verified=True))
        manifest = self.put('training/datasets/demo.json', dict(format='bc-dataset/1.0.0', compatibility=compatibility(),
            label_kind='qualified-v3-choice/1.0.0', context='neutral-predecision/1.0.0', teachers=[teacher], entries=entries,
            calibration=dict(format='archived-budget/1.0.0', identity=identity, budget=dict(calibration=str(evidence)),
                             sha256=transfer.sha(evidence))))
        previous = None
        for name in ('parent-il', 'il'):
            run = self.root / 'artifacts/step5' / name
            model = FourSeatNet(ModelConfig.preset('small'))
            initialization = 'fresh-imitation-continuation' if previous else 'random'
            signature = dict(implementation=implementation_id(), initialization=initialization,
                manifest_path=str(manifest), manifest_sha256=transfer.sha(manifest), architecture=model.config.to_dict(), epochs=1)
            if previous:
                recorded = dict(previous)
                for field in ('bc_run', 'path', 'manifest'):
                    recorded[field] = str(self.root / recorded[field])
                signature['parent_imitation'] = recorded
            checkpoint = run / 'epoch-0001.pt'
            save_checkpoint(checkpoint, model, step=1, training=dict(mode='step5-v3-bc', initialization=initialization,
                parent=previous['model_id'] if previous else None, fresh_run=str(run), signature=digest(signature),
                manifest_sha256=transfer.sha(manifest)))
            selected = dict(path=str(checkpoint) if previous else checkpoint.name, sha256=transfer.sha(checkpoint))
            self.put(run.relative_to(self.root).as_posix() + '/state.json', dict(signature=signature, epoch=1, stopped=False, best=selected))
            self.put(run.relative_to(self.root).as_posix() + '/report.json', dict(format='bc-training-report/1.0.0', signature=signature, epochs=1, candidate_sha256=selected['sha256']))
            _, previous = completed_imitation(self.root, run)
        self.state['signature']['imitation_source'] = previous
        for field in ('bc_run', 'path', 'manifest'):
            previous[field] = str(self.root / previous[field])
        self.state['signature']['config']['rehearsal_manifest'] = str(manifest)
        self.put('artifacts/step5/selfplay/state.json', self.state)
        before, _ = load_dataset(self.root, manifest, 'train', workers=1, compact=True)
        self.export()
        original = self.root
        self.remove(original)
        destination = self.area / 'imitation-destination'
        transfer.restore(destination, self.bundle, relocate=True)
        _, origin = completed_imitation(destination, previous['bc_run'])
        self.assertEqual(bound_reference(destination, origin, previous, ('bc_run', 'path', 'manifest')), previous)
        after, _ = load_dataset(destination, str(manifest), 'train', workers=2, compact=True)
        self.assertEqual(len(after), len(before))
        for a, b in zip(before, after):
            self.assertEqual(a[:3], b[:3])
            import numpy as np
            for field in ('state', 'actions', 'stable_to_relative'):
                np.testing.assert_array_equal(getattr(a[3], field), getattr(b[3], field))
        (destination / 'artifacts/demos/calibration.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            load_dataset(destination, str(manifest), 'train')

    def test_evaluation_loads_real_checkpoints_and_keeps_existing_spec_binding(self):
        import torch
        from brass_ai.learning.checkpoint import save_checkpoint
        from brass_ai.learning.model import FourSeatNet, ModelConfig
        from brass_ai.recovery.evaluation import evaluate
        from brass_ai.step5.evaluation import evaluation_config
        from brass_ai.step5.league import implementation_id, seeds_for
        torch.set_num_threads(2)
        cfg = evaluation_config(
            transfer.read(ROOT / 'configs/step5/evaluation.json'), setups=1, include_legacy=False, use_search=False, lineups=['recent-neural'])
        cfg['host'] = str(self.root / cfg['host'])
        pool = dict(training={}, evaluation={})
        self.put('training/opponents/pool.json', pool)
        entries = []
        for name in ('candidate', 'reference'):
            checkpoint = self.run / (name + '.pt')
            model_id = save_checkpoint(checkpoint, FourSeatNet(ModelConfig.preset('small')),
                training=dict(mode='step5-population-ppo', fresh_run=str(self.run)))
            entries.append(dict(path=str(checkpoint), sha256=transfer.sha(checkpoint), model_id=model_id))
        spec = dict(format='step5-evaluation/2.0.0', purpose='evaluation', entries=entries, config=cfg,
                    seeds=seeds_for(cfg['seed'], cfg['setups'], cfg['partition'])[0], lineups=['recent-neural'],
                    implementation=implementation_id(), legacy_pool=pool, teachers={}, held={})
        directory = 'artifacts/step5/selfplay/evaluations/screen'
        self.put(directory + '/spec.json', spec)
        # Pending evaluations include all referenced model dependencies.
        self.export()
        original = self.root
        self.remove(original)
        destination = self.area / 'evaluation-destination'
        transfer.restore(destination, self.bundle, relocate=True)
        spec_bytes = (destination / directory / 'spec.json').read_bytes()
        # Exercise normal checkpoint/spec loading, then the existing pruned
        # summary branch; no fabricated collection or relaxed spec checks.
        report = destination / directory / 'report.json'
        report.write_bytes(transfer.encode(dict(spec=spec, raw_records_pruned=True)))
        result = evaluate(destination, directory, entries[0]['path'], [entries[1]['path']], cfg)
        self.assertTrue(result['cached_summary'])
        self.assertEqual(result['spec'], spec)
        self.assertEqual((destination / directory / 'spec.json').read_bytes(), spec_bytes)
        # The destination's normal evaluation config names the same host
        # relatively (or via its new absolute root). Keep the signed old spec.
        for host in ('artifacts/r1-host/guarded', str(destination / 'artifacts/r1-host/guarded')):
            with self.subTest(host=host):
                result = evaluate(destination, directory, entries[0]['path'], [entries[1]['path']], dict(cfg, host=host))
                self.assertTrue(result['cached_summary'])
                self.assertEqual(result['spec'], spec)
        self.assertEqual((destination / directory / 'spec.json').read_bytes(), spec_bytes)
        bad = copy.deepcopy(cfg)
        bad['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'specification changed'):
            evaluate(destination, directory, entries[0]['path'], [entries[1]['path']], bad)

    def test_invalid_archive_mappings_and_conflicting_sidecar_fail_before_writes(self):
        self.export()
        members = self.archive_bytes()
        manifest = json.loads(members['transfer.json'])
        manifest['path_roots'] = [str(self.root), str(self.root / 'nested')]
        manifest['id'] = transfer.digest({k:v for k,v in manifest.items() if k != 'id'})
        invalid = self.area / 'invalid.zip'
        with zipfile.ZipFile(invalid, 'w') as archive:
            for key, content in members.items():
                archive.writestr(key, transfer.encode(manifest) if key == 'transfer.json' else content)
        destination = self.area / 'invalid-destination'
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            transfer.restore(destination, invalid, relocate=True)
        self.assertFalse(destination.exists())
        destination.mkdir()
        sidecar = destination / RELOCATION_FILE
        sidecar.write_bytes(transfer.encode(make_relocation(destination, [str(self.root)], 'b' * 64)))
        before = sidecar.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Conflicting destination relocation metadata'):
            transfer.restore(destination, self.bundle, relocate=True)
        self.assertEqual(sidecar.read_bytes(), before)
        self.assertEqual(list(destination.iterdir()), [sidecar])

    def test_corruption_and_conflicts_leave_destination_untouched(self):
        self.export()
        self.remove(self.root)
        destination = self.area / 'conflicting'
        with self.assertRaisesRegex(ValueError, '--relocate'):
            transfer.restore(destination, self.bundle)
        self.assertFalse(destination.exists())
        destination.mkdir()
        script = destination / 'scripts/step5.py'
        script.parent.mkdir()
        script.write_bytes(b'user work')
        with self.assertRaisesRegex(ValueError, 'Conflicting destination'):
            transfer.restore(destination, self.bundle, relocate=True)
        self.assertEqual(script.read_bytes(), b'user work')
        self.assertFalse((destination / 'artifacts').exists())
        self.assertFalse((destination / RELOCATION_FILE).exists())
        corrupt = self.area / 'corrupt'
        corrupt.write_bytes(self.bundle.read_bytes()[:-7])
        missing = self.area / 'never-created'
        with self.assertRaises(Exception):
            transfer.restore(missing, corrupt, relocate=True)
        self.assertFalse(missing.exists())

    def test_component_mapping_metadata_validation_and_external_paths(self):
        aliases = ['/data/not_backed_up/yxu209/ai-training', 'D:/previous/project']
        sidecar = self.root / RELOCATION_FILE
        valid = make_relocation(self.root, aliases, 'a' * 64)
        sidecar.write_bytes(transfer.encode(valid))
        for ref in (aliases[0] + '/artifacts/a.pt', r'd:\PREVIOUS\project\artifacts\a.pt', 'artifacts/a.pt'):
            self.assertEqual(resolve_project_path(self.root, ref), self.root / 'artifacts/a.pt')
        self.assertEqual(resolve_project_path(self.root, 'a.pt', base=self.run), self.run / 'a.pt')
        for ref in (aliases[0] + '-other/a.pt', '/elsewhere/a.pt', '../a.pt', aliases[0] + '/../a.pt', 'C:relative.pt', r'\\server\share\a.pt'):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                resolve_project_path(self.root, ref)
        for roots in ([aliases[0], aliases[0]], [aliases[0], aliases[0] + '/nested'], ['/']):
            with self.assertRaises(ValueError):
                make_relocation(self.root, roots, 'b' * 64)
        for field, value in (('destination_root', str(self.area / 'foreign')), ('source_roots', ['/']),
                             ('source_roots', {aliases[0]: 'invalid-shape'}), ('archive_id', 'invalid')):
            invalid = dict(valid, **{field:value})
            invalid['id'] = transfer.digest({k:v for k,v in invalid.items() if k != 'id'})
            sidecar.write_bytes(transfer.encode(invalid))
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_project_path(self.root, 'artifacts/a.pt')
        sidecar.write_bytes(transfer.encode(dict(valid, id='corrupt')))
        with self.assertRaisesRegex(ValueError, 'digest'):
            read_relocation(self.root)
        sidecar.write_bytes(transfer.encode(valid))
        actual = dict(path='artifacts/a.pt', sha256='a' * 64)
        saved = dict(actual, path=aliases[0] + '/artifacts/a.pt')
        self.assertIs(bound_reference(self.root, actual, saved, ('path',)), saved)
        with self.assertRaisesRegex(ValueError, 'sha256'):
            bound_reference(self.root, actual, dict(saved, sha256='b' * 64), ('path',))

    def test_symlink_or_junction_escape_rejected_before_restore(self):
        self.export()
        destination = self.area / 'linked-project'
        destination.mkdir()
        outside = self.area / 'outside'
        outside.mkdir()
        link = destination / 'artifacts'
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            if os.name != 'nt':
                raise
            import _winapi
            try:
                _winapi.CreateJunction(str(outside), str(link))
            except PermissionError:
                self.skipTest('Windows sandbox denies symlink/junction creation; rerun this check without sandbox')
        with self.assertRaisesRegex(ValueError, 'Linked'):
            transfer.restore(destination, self.bundle, relocate=True)
        self.assertEqual(list(outside.iterdir()), [])

    def test_external_dependency_is_rejected_during_discovery(self):
        self.state['signature']['config']['rehearsal_manifest'] = str(self.area / 'outside.json')
        self.put('artifacts/step5/selfplay/state.json', self.state)
        with self.assertRaisesRegex(ValueError, 'outside project.*Supply this dependency'):
            self.export()
        self.assertFalse(self.bundle.exists())

    def test_atomic_publication_cannot_overwrite_a_racing_file(self):
        temporary = self.area / 'temporary'
        target = self.area / 'target'
        temporary.write_bytes(b'archive bytes')
        target.write_bytes(b'concurrent user bytes')
        with self.assertRaises(FileExistsError):
            transfer.publish(temporary, target)
        self.assertEqual(target.read_bytes(), b'concurrent user bytes')
        self.assertEqual(temporary.read_bytes(), b'archive bytes')


if __name__ == '__main__':
    unittest.main()
