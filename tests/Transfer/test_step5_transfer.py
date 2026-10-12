"""Small archive/continuation checks; no game collection or learning campaign."""
import importlib.util
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('step5_transfer', ROOT / 'scripts/step5-transfer.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='transfer-test-', dir=ROOT / 'artifacts')
        self.addCleanup(self.tmp.cleanup)
        self.area = Path(self.tmp.name).resolve()
        self.root = self.area / 'project'
        self.root.mkdir()
        self.run = self.root / 'artifacts/step5/selfplay'
        self.run.mkdir(parents=True)
        self.bundle = self.area / 'migration.tar.zst'
        for tree in transfer.CODE_TREES:
            (self.root / tree).mkdir(parents=True, exist_ok=True)
        for name in ('learning/runner.py', 'learning/batching.py', 'searchlab/spiel.py', 'beliefs.py', 'runtime_contracts.py', 'host.py'):
            self.put('src/brass_ai/' + name, b'# fixture\n')
        for name in ('build-training-host.py', 'build-search-host.py'):
            self.put('scripts/' + name, b'# fixture builder\n')
        self.put('src/BrassTrainingHost/FastHost.cs', b'// fixture\n')
        for kind in ('ordinary', 'search'):
            exe = self.put(f'training/runtime/throughput/{kind}/BrassHost.exe', b'authored-worker-' + kind.encode())
            self.put(f'training/runtime/throughput/{kind}/runtime.json',
                     dict(files=[dict(name='BrassHost.exe', sha256=transfer.sha(exe))]))
        self.put('configs/references/masked-ismcts.lock.json', {})
        self.put('scripts/step5.py', b'# fixture entry point\n')
        self.put('requirements-step5.txt', b'# fixture\n')
        self.put('artifacts/r1-host/guarded/engine.dll', b'private-runtime')
        h = transfer.sha(self.root / 'artifacts/r1-host/guarded/engine.dll')
        self.put('artifacts/r1-host/guarded/runtime.json', dict(files=[dict(name='engine.dll', sha256=h)]))
        self.put('training/runtime/distribution.json', dict(assets=[], private_runtime_files={
            'artifacts/r1-host/guarded/engine.dll': h}))
        self.put('training/runtime/parity.json', dict(fixture=True))
        self.put('artifacts/step5/preflight.json', dict(passed=True, old_host=True))
        self.put('training/datasets/demo.json', dict(format='bc-dataset/1.0.0', entries=[dict(
            path='artifacts/demos/train.jsonl.gz', sha256=transfer.hashlib.sha256(b'demo').hexdigest(), split='train')]))
        self.put('artifacts/demos/train.jsonl.gz', b'demo')
        self.put('artifacts/demos/not-needed.jsonl.gz', b'not included')
        bc = 'artifacts/step5/il'
        self.put(bc + '/state.json', dict(signature={}))
        self.put(bc + '/report.json', dict(completed=True))
        self.put(bc + '/best.pt', b'imitation weights')
        self.origin = dict(bc_run=bc, path=bc + '/best.pt', sha256=transfer.sha(self.root / bc / 'best.pt'),
            manifest='training/datasets/demo.json', manifest_sha256=transfer.sha(self.root / 'training/datasets/demo.json'))
        current = self.entry('generation-0002.pt', b'checkpoint-with-Adam-and-RNG', 2)
        anchor = self.entry('initial.pt', b'initial', 0)
        pool = self.entry('population/historical.pt', b'old opponent', 1)
        self.state = dict(format='step5-recovery-league/2.2.0', iteration=2, games_done=640,
            history=[dict(pool, path='artifacts/step5/selfplay/generation-0001.pt'), current],
            resume=[current], population=[pool], evaluation_anchor=anchor, champion=pool,
            next_seed=123, next_search_games=800, context_counter=83, search_seconds=16., ordinary_seconds=25.,
            legacy_boost=dict(start_games=0, until_games=30000), evaluations=[],
            signature=dict(implementation=transfer.source_identity(self.root), imitation_source=self.origin, config=dict(
                host='artifacts/r1-host/guarded', model_size='small', rehearsal_manifest='training/datasets/demo.json',
                rehearsal_weight=.1, rehearsal_until=10000, imitation=dict(manifest=None))))
        self.put('artifacts/step5/selfplay/state.json', self.state)
        self.put('artifacts/step5/selfplay/iteration-0001/report.json', dict(complete=True))
        self.put('artifacts/step5/selfplay/iteration-0001/games.jsonl.gz', b'last ordinary games')

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else transfer.encode(value))
        return path

    def entry(self, name, content, generation):
        p = self.put('artifacts/step5/selfplay/' + name, content)
        return dict(path=p.relative_to(self.root).as_posix(), sha256=transfer.sha(p), generation=generation, model_id=name)

    def export(self):
        return transfer.export(self.root, self.run, self.bundle, dependencies='bundled', part_mb=0)

    def archive_bytes(self, bundle=None):
        # Only small synthetic fixtures use this helper; production reads stream.
        with transfer.open_bundle(bundle or self.bundle) as members:
            return {name: stream.read() for name, size, stream in members}

    def move_source(self):
        # Simulate an empty second computer at the same absolute project path.
        destination = self.area / 'source-backup'
        self.assertTrue(self.root.resolve().is_relative_to(self.area))
        self.assertTrue(destination.resolve().is_relative_to(self.area))
        self.root.rename(destination)
        return destination

    def test_bundle_contains_resume_dependencies_and_no_old_host_approval(self):
        result = self.export()
        manifest = transfer.inspect_bundle(self.bundle)
        names = {r['path'] for r in manifest['files']}
        for name in ('state.json', 'generation-0002.pt', 'initial.pt', 'population/historical.pt'):
            self.assertIn('artifacts/step5/selfplay/' + name, names)
        self.assertIn('artifacts/step5/il/best.pt', names)
        self.assertIn('artifacts/demos/train.jsonl.gz', names)
        self.assertIn('artifacts/r1-host/guarded/engine.dll', names)
        self.assertNotIn('artifacts/demos/not-needed.jsonl.gz', names)
        self.assertNotIn('artifacts/step5/preflight.json', names)
        self.assertFalse(any('.active-run.json' in name for name in names))
        self.assertEqual(transfer.read(self.run / 'state.json'), self.state)
        self.assertEqual(result['iteration'], 2)

    def test_finished_rehearsal_keeps_manifest_and_il_but_omits_large_dataset(self):
        self.state['games_done'] = 10000
        self.put('artifacts/step5/selfplay/state.json', self.state)
        self.export()
        manifest = transfer.inspect_bundle(self.bundle)
        names = {r['path'] for r in manifest['files']}
        self.assertFalse(manifest['rehearsal_included'])
        self.assertIn('training/datasets/demo.json', names)
        self.assertIn('artifacts/step5/il/best.pt', names)
        self.assertNotIn('artifacts/demos/train.jsonl.gz', names)

    def test_restore_identical_bytes_and_command_then_repeat_verify(self):
        self.export()
        backup = self.move_source()
        result = transfer.restore(self.root, self.bundle)
        self.assertEqual(transfer.read(self.run / 'state.json'), self.state)
        self.assertEqual((self.run / 'generation-0002.pt').read_bytes(), (backup / 'artifacts/step5/selfplay/generation-0002.pt').read_bytes())
        self.assertIn('--imitation-run', result['resume'])
        self.assertIn('--config', result['resume'])
        self.assertNotIn('--initial', result['resume'])
        self.assertNotIn('--resume-reviewed-source', result['resume'])
        self.assertTrue(transfer.restore(self.root, self.bundle, verify_only=True)['verified'])
        self.assertTrue(transfer.restore(self.root, self.bundle)['verified'])
        self.assertFalse((self.run / '.active-run.json').exists())

    def test_active_pending_corrupt_and_source_mismatch_rejected(self):
        lock = self.put('artifacts/step5/selfplay/.active-run.json', dict(pid=os.getpid(), host=socket.gethostname()))
        with self.assertRaisesRegex(ValueError, 'active'):
            self.export()
        lock.unlink()
        pending = self.put('artifacts/step5/selfplay/iteration-0002/partial.jsonl.gz', b'pending')
        with self.assertRaisesRegex(ValueError, 'Uncommitted'):
            self.export()
        pending.unlink()
        source = self.root / 'src/brass_ai/beliefs.py'
        original = source.read_bytes(); source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Source differs'):
            self.export()
        source.write_bytes(original)
        (self.run / 'generation-0002.pt').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.export()
        self.assertFalse(self.bundle.exists())

    def test_restore_conflicts_and_wrong_root_fail_before_writes(self):
        self.export()
        self.move_source()
        conflict = self.put('scripts/step5.py', b'different code')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            transfer.restore(self.root, self.bundle)
        self.assertEqual(conflict.read_bytes(), b'different code')
        self.assertFalse((self.run / 'state.json').exists())
        with self.assertRaisesRegex(ValueError, 'original absolute'):
            transfer.restore(self.area / 'wrong-root', self.bundle)
        self.assertFalse((self.area / 'wrong-root').exists())

    def test_corrupt_archive_is_rejected_before_destination_creation(self):
        self.export()
        bad = self.area / 'bad.zip'
        source = self.archive_bytes()
        with zipfile.ZipFile(bad, 'w') as out:
            manifest = transfer.json.loads(source['transfer.json'])
            record = next(r for r in manifest['files'] if r['path'].endswith('generation-0002.pt'))
            payload = transfer.payload_name(manifest, record)
            for name, data in source.items():
                if name == payload:
                    data = b'x' * len(data)
                out.writestr(name, data)
        self.move_source()
        with self.assertRaisesRegex(ValueError, 'Corrupted'):
            transfer.restore(self.root, bad)
        self.assertFalse(self.root.exists())

    def test_unsafe_names_and_dry_run(self):
        for name in ('../outside', '/absolute', 'C:/drive', 'src/../escape', '.git/config', 'src\\escape'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                transfer.relative(name)
        result = transfer.export(self.root, self.run, self.bundle, dry_run=True)
        self.assertTrue(result['dry_run'])
        self.assertEqual(result['dependencies'], 'external')
        self.assertEqual(result['part_mb'], 1900)
        self.assertEqual(result['archive_format'], 'zstd')
        self.assertEqual(result['zstd_level'], 3)
        self.assertGreaterEqual(result['compression_threads'], 1)
        self.assertFalse(self.bundle.exists())
        self.assertFalse((self.run / '.active-run.json').exists())

    def test_real_checkpoint_resumes_optimizer_rng_and_counters_after_transfer(self):
        import torch
        from brass_ai.learning.checkpoint import save_checkpoint, resume_optimizer
        from brass_ai.learning.contracts import digest
        from brass_ai.learning.model import FourSeatNet, ModelConfig
        from brass_ai.recovery.resume import load_resume
        torch.set_num_threads(2)
        model = FourSeatNet(ModelConfig.preset('small'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        sum(p.square().mean() for p in model.parameters()).backward(); optimizer.step()
        path = self.run / 'generation-0002.pt'
        identity = save_checkpoint(path, model, optimizer, step=2, training=dict(mode='step5-population-ppo',
            fresh_run='artifacts/step5/selfplay', signature=digest(self.state['signature'])))
        entry = dict(path='artifacts/step5/selfplay/generation-0002.pt', sha256=transfer.sha(path), model_id=identity, generation=2)
        self.state['history'][-1] = entry; self.state['resume'] = [entry]
        self.put('artifacts/step5/selfplay/state.json', self.state)
        self.export(); self.move_source(); transfer.restore(self.root, self.bundle)
        state = transfer.read(self.run / 'state.json')
        restored, raw, changed = load_resume(self.root, self.run, state, state['signature'])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-5)
        resume_optimizer(raw, restored_optimizer)
        self.assertFalse(changed)
        self.assertEqual(state, self.state)
        self.assertTrue(torch.equal(torch.get_rng_state(), raw['rng']))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]))
        before, after = optimizer.state_dict(), restored_optimizer.state_dict()
        self.assertEqual(before['param_groups'], after['param_groups'])
        for key, values in before['state'].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, after['state'][key][name]))

    def test_transport_does_not_change_the_training_implementation(self):
        from brass_ai.step5.league import implementation_id
        self.assertEqual(transfer.source_identity(ROOT), implementation_id())

    def test_source_recipe_covers_throughput_inputs_and_normalizes_newlines(self):
        paths = ['src/brass_ai/host.py', 'src/BrassTrainingHost/FastHost.cs',
                 'scripts/build-training-host.py', 'scripts/build-search-host.py',
                 'training/runtime/throughput/ordinary/runtime.json',
                 'training/runtime/throughput/search/runtime.json']
        before = transfer.source_identity(self.root)
        self.assertNotEqual(before, transfer.source_identity(self.root, recipe=1))
        for name in paths:
            with self.subTest(path=name):
                file = self.root / name
                original = file.read_bytes()
                file.write_bytes(original + b'\n')
                self.assertNotEqual(transfer.source_identity(self.root), before)
                with self.assertRaisesRegex(ValueError, 'Source differs'):
                    transfer.export(self.root, self.run, self.bundle, dry_run=True)
                file.write_bytes(original.replace(b'\n', b'\r\n'))
                self.assertEqual(transfer.source_identity(self.root), before)
                file.write_bytes(original)

    def test_compact_omits_consumed_games_caches_trash_and_unused_IL_epochs(self):
        unused = ['artifacts/step5/selfplay/generation-0000.pt',
            'artifacts/step5/selfplay/population/unused.pt',
            'artifacts/step5/selfplay/iteration-0001/games.jsonl.gz.features.npz',
            'artifacts/step5/selfplay/.retention-trash/old/model.pt',
            'artifacts/step5/selfplay/.retention/cleanup-old.json',
            'artifacts/step5/il/epoch-0001.pt']
        for name in unused:
            self.put(name, b'not-needed' * 10000)
        self.export()
        compact = transfer.inspect_bundle(self.bundle)
        names = {r['path'] for r in compact['files']}
        for name in unused + ['artifacts/step5/selfplay/iteration-0001/games.jsonl.gz']:
            self.assertNotIn(name, names)
        full = transfer.export(self.root, self.run, self.area / 'full.tar.zst', mode='full', part_mb=0)
        self.assertGreater(full['bytes'], sum(r['bytes'] for r in compact['files']) + 500000)
        self.assertTrue((self.run / 'iteration-0001/games.jsonl.gz').exists())

    def test_compact_keeps_pending_evaluation_cache_pins_and_selected_gate(self):
        pin = 'artifacts/step5/selfplay/iteration-0001/games.jsonl.gz'
        self.put('artifacts/step5/selfplay/pins.json', dict(paths=[pin]))
        self.put('artifacts/step5/selfplay/evaluations/pending/spec.json', dict(candidate=self.state['history'][-1]))
        self.put('artifacts/step5/selfplay/evaluations/pending/game.jsonl.gz', b'pending gate game')
        report = self.put('artifacts/step5/selfplay/evaluations/accepted/report.json', dict(accepted=True))
        self.put('artifacts/step5/selfplay/evaluations/accepted/spec.json', dict(model='winner'))
        self.state['champion_gate'] = dict(accepted_gate=dict(report=report.relative_to(self.root).as_posix(), report_sha256=transfer.sha(report)))
        self.put('artifacts/step5/selfplay/state.json', self.state)
        self.export()
        names = {r['path'] for r in transfer.inspect_bundle(self.bundle)['files']}
        for name in (pin, 'artifacts/step5/selfplay/evaluations/pending/game.jsonl.gz', 'artifacts/step5/selfplay/evaluations/accepted/report.json'):
            self.assertIn(name, names)

    def test_compact_keeps_existing_history_candidate_outside_active_pool(self):
        candidate = self.entry('generation-0000.pt', b'eligible historical candidate', 0)
        self.state['history'].insert(0, candidate)
        self.put('artifacts/step5/selfplay/state.json', self.state)
        self.export()
        names = {r['path'] for r in transfer.inspect_bundle(self.bundle)['files']}
        self.assertIn(candidate['path'], names)

    def test_default_external_reuses_clone_and_ssh_runtime_without_packing_them(self):
        result = transfer.export(self.root, self.run, self.bundle)
        self.assertEqual(result['dependencies'], 'external')
        manifest = transfer.inspect_bundle(result['bundle'])
        external = {r['path'] for r in manifest['files'] if r['external']}
        self.assertIn('scripts/step5.py', external)
        self.assertIn('artifacts/r1-host/guarded/engine.dll', external)
        self.assertIn('artifacts/r1-host/guarded/runtime.json', external)
        for kind in ('ordinary', 'search'):
            self.assertIn(f'training/runtime/throughput/{kind}/runtime.json', external)
            self.assertIn(f'training/runtime/throughput/{kind}/BrassHost.exe', external)
        self.assertEqual(manifest['source_recipe'], 2)
        self.assertFalse(any(r['path'].startswith('artifacts/training-host/') for r in manifest['files']))
        dll = next(r for r in manifest['files'] if r['path'].endswith('engine.dll'))
        self.assertNotIn(transfer.payload_name(manifest, dll), self.archive_bytes(result['bundle']))
        backup = self.move_source()
        with self.assertRaisesRegex(ValueError, 'Missing external'):
            transfer.restore(self.root, result['bundle'])
        self.assertFalse(self.root.exists())
        for name in external:
            self.put(name, (backup / name).read_bytes())
        dll_name = 'artifacts/r1-host/guarded/engine.dll'
        self.put(dll_name, b'wrong runtime from SSH')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            transfer.restore(self.root, result['bundle'])
        self.assertFalse((self.run / 'state.json').exists())
        self.put(dll_name, (backup / dll_name).read_bytes())
        self.assertTrue(transfer.restore(self.root, result['bundle'])['verified'])
        self.assertEqual(transfer.read(self.run / 'state.json'), self.state)

    def test_identical_runtime_and_model_bytes_are_stored_once(self):
        self.put('artifacts/step5/selfplay/best.pt', b'old opponent')
        self.put('training/runtime/duplicate.dll', b'private-runtime')
        result = self.export()
        manifest = transfer.inspect_bundle(self.bundle)
        included = [r for r in manifest['files'] if not r['external']]
        self.assertEqual(len(self.archive_bytes()) - 1, len({r['sha256'] for r in included}))
        self.assertLess(result['unique_payload_bytes'], result['bytes'])
        self.move_source(); transfer.restore(self.root, self.bundle)
        self.assertEqual((self.run / 'best.pt').read_bytes(), b'old opponent')
        (self.run / 'best.pt').write_bytes(b'updated independently')
        self.assertEqual((self.run / 'population/historical.pt').read_bytes(), b'old opponent')

    def test_repo_mode_omits_clean_tracked_files_but_includes_private_runtime(self):
        tracked = {'scripts/step5.py', 'src/brass_ai/beliefs.py', 'training/runtime/parity.json'}
        with patch.object(transfer, 'git_baseline', return_value=('test-commit', tracked)):
            result = transfer.export(self.root, self.run, self.bundle, dependencies='repo', part_mb=0)
        manifest = transfer.inspect_bundle(self.bundle)
        external = {r['path'] for r in manifest['files'] if r['external']}
        self.assertEqual(external, tracked)
        self.assertEqual(result['git_commit'], 'test-commit')
        private = next(r for r in manifest['files'] if r['path'].endswith('engine.dll'))
        self.assertFalse(private['external'])
        backup = self.move_source()
        with self.assertRaisesRegex(ValueError, 'Missing external'):
            transfer.restore(self.root, self.bundle)
        self.assertFalse(self.root.exists())
        for name in tracked:
            self.put(name, (backup / name).read_bytes())
        self.assertTrue(transfer.restore(self.root, self.bundle)['verified'])

    def test_parts_restore_without_joining_archive_and_detect_missing_or_changed_part(self):
        pin = 'artifacts/step5/selfplay/extra.pt'
        self.put(pin, os.urandom(2_500_000))
        self.put('artifacts/step5/selfplay/pins.json', dict(paths=[pin]))
        result = transfer.export(self.root, self.run, self.bundle, dependencies='bundled', part_mb=1)
        self.assertGreaterEqual(result['parts'], 3)
        self.assertFalse(self.bundle.exists())
        index = Path(result['bundle'])
        parts = transfer.read(index)['parts']
        self.assertTrue(all(p['bytes'] <= 1_000_000 for p in parts))
        self.move_source()
        self.assertTrue(transfer.restore(self.root, index)['verified'])
        self.assertTrue(transfer.restore(self.root, self.bundle, verify_only=True)['verified'])
        self.assertFalse(self.bundle.exists())
        victim = index.parent / parts[1]['name']
        original = victim.read_bytes(); victim.write_bytes(b'x' * len(original))
        with self.assertRaisesRegex(ValueError, 'Corrupted transfer part'):
            transfer.inspect_bundle(index)
        victim.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing/truncated transfer part'):
            transfer.inspect_bundle(index)

    def test_previous_version_archives_still_restore(self):
        self.export()
        old = self.area / 'old.zip'
        src = self.archive_bytes()
        with zipfile.ZipFile(old, 'w') as dst:
            manifest = transfer.json.loads(src['transfer.json'])
            for record in manifest['files']:
                self.assertFalse(record.pop('external'))
                dst.writestr('project/' + record['path'], src[transfer.payload_name(manifest, record)])
            manifest['format'] = transfer.OLD_FORMAT
            manifest['id'] = transfer.digest({k: v for k, v in manifest.items() if k != 'id'})
            dst.writestr('transfer.json', transfer.encode(manifest))
        self.move_source()
        self.assertTrue(transfer.restore(self.root, old)['verified'])

    def test_archive_without_recipe_retains_original_python_only_identity(self):
        self.state['signature']['implementation'] = transfer.source_identity(self.root, recipe=1)
        self.put('artifacts/step5/selfplay/state.json', self.state)
        with patch.object(transfer, 'SOURCE_RECIPE', 1):
            self.export()
        old = self.area / 'before-throughput.zip'
        src = self.archive_bytes()
        with zipfile.ZipFile(old, 'w') as dst:
            manifest = transfer.json.loads(src['transfer.json'])
            manifest.pop('source_recipe')  # Actual old packages had no recipe field.
            manifest['id'] = transfer.digest({k: v for k, v in manifest.items() if k != 'id'})
            for name, data in src.items():
                dst.writestr(name, transfer.encode(manifest) if name == 'transfer.json' else data)
        self.move_source()
        self.assertTrue(transfer.restore(self.root, old)['verified'])
        self.assertEqual(transfer.read(self.run / 'state.json'), self.state)

    def test_default_zstd_multithreaded_archive_is_standard_tar_and_restores_large_payload(self):
        pin = 'artifacts/step5/selfplay/extra.pt'
        content = os.urandom(20_000_000)  # Larger than one zstd worker job.
        self.put(pin, content)
        self.put('artifacts/step5/selfplay/pins.json', dict(paths=[pin]))
        with patch.object(transfer, 'available_cpus', return_value=2):
            result = self.export()
        self.assertEqual(result['archive_format'], 'zstd')
        self.assertEqual(result['compression_threads'], 2)
        with self.bundle.open('rb') as stream:
            self.assertEqual(stream.read(4), transfer.ZSTD_MAGIC)
        # Independently read with the standard tar parser and the zstd backend.
        with transfer.zstd_backend().ZstdFile(self.bundle, 'rb') as decoded, tarfile.open(fileobj=decoded, mode='r|') as archive:
            self.assertEqual(archive.next().name, 'transfer.json')
        self.move_source()
        self.assertTrue(transfer.restore(self.root, self.bundle)['verified'])
        self.assertEqual((self.root / pin).read_bytes(), content)
        self.assertFalse(list(self.area.glob('*.tar')))

    def test_zip_export_and_split_restore_work_without_zstd(self):
        pin = 'artifacts/step5/selfplay/extra.pt'
        self.put(pin, os.urandom(1_200_000))
        self.put('artifacts/step5/selfplay/pins.json', dict(paths=[pin]))
        with patch.object(transfer, 'zstd_backend', side_effect=ValueError('missing zstd')):
            single = self.area / 'legacy.zip'
            split = self.area / 'legacy-parts.zip'
            a = transfer.export(self.root, self.run, single, archive_format='zip', dependencies='bundled', part_mb=0)
            b = transfer.export(self.root, self.run, split, archive_format='zip', dependencies='bundled', part_mb=1)
            self.assertEqual(a['archive_format'], 'zip')
            self.assertTrue(zipfile.is_zipfile(single))
            self.assertGreaterEqual(b['parts'], 2)
            self.move_source()
            self.assertTrue(transfer.restore(self.root, b['bundle'])['verified'])
            self.assertTrue(transfer.restore(self.root, single, verify_only=True)['verified'])

    def test_truncated_corrupt_or_appended_zstd_is_rejected_before_writes(self):
        self.export()
        data = self.bundle.read_bytes()
        zstd = transfer.zstd_backend()
        invalid = [data[:-1], data[:-10], data[:len(data) // 2],
                   data[:-1] + bytes([data[-1] ^ 1]), data + b'trailing garbage',
                   data + zstd.compress(b'unexpected second tar')]
        self.move_source()
        for index, content in enumerate(invalid):
            with self.subTest(case=index):
                bad = self.area / f'bad-{index}.tar.zst'
                bad.write_bytes(content)
                with self.assertRaises((ValueError, EOFError, tarfile.TarError, zstd.ZstdError)):
                    transfer.restore(self.root, bad)
                self.assertFalse(self.root.exists())

    def test_zstd_payload_hash_tampering_is_rejected_before_writes(self):
        self.export()
        zstd = transfer.zstd_backend()
        raw = zstd.decompress(self.bundle.read_bytes())
        original = b'checkpoint-with-Adam-and-RNG'
        self.assertIn(original, raw)
        bad = self.area / 'bad-content.tar.zst'
        bad.write_bytes(zstd.compress(raw.replace(original, b'x' * len(original))))
        self.move_source()
        with self.assertRaisesRegex(ValueError, 'Corrupted transfer file'):
            transfer.restore(self.root, bad)
        self.assertFalse(self.root.exists())

    def test_tar_links_unsafe_paths_duplicates_and_extra_members_are_rejected(self):
        self.export()
        source = self.archive_bytes()
        self.move_source()
        for name, kind in [('escape', tarfile.SYMTYPE), ('escape', tarfile.LNKTYPE),
                           ('../escape', tarfile.REGTYPE), ('transfer.json', tarfile.REGTYPE),
                           ('objects/unexpected', tarfile.REGTYPE)]:
            with self.subTest(name=name, kind=kind):
                raw = io.BytesIO()
                with tarfile.open(fileobj=raw, mode='w') as archive:
                    for member_name, content in source.items():
                        info = tarfile.TarInfo(member_name); info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))
                    info = tarfile.TarInfo(name); info.type = kind; info.linkname = '../outside'
                    archive.addfile(info, io.BytesIO())
                bad = self.area / 'invalid-members.tar.zst'
                bad.write_bytes(transfer.zstd_backend().compress(raw.getvalue()))
                with self.assertRaises(ValueError):
                    transfer.restore(self.root, bad)
                self.assertFalse(self.root.exists())

    def test_missing_zstd_fails_before_writes_but_dry_run_still_works(self):
        with patch.object(transfer, 'zstd_backend', side_effect=ValueError('missing zstd')):
            self.assertTrue(transfer.export(self.root, self.run, self.bundle, dry_run=True)['dry_run'])
            with self.assertRaisesRegex(ValueError, 'missing zstd'):
                self.export()
        self.assertFalse(self.bundle.exists())
        self.assertFalse((self.run / '.active-run.json').exists())
        self.assertFalse(list(self.area.glob('*.partial')))

    def test_export_controls_and_release_sized_parts(self):
        for options in [dict(archive_format='invalid'), dict(threads=-1), dict(zstd_level=0),
                        dict(zstd_level=23), dict(part_mb=2001)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                transfer.export(self.root, self.run, self.bundle, **options)
        result = transfer.export(self.root, self.run, self.bundle, dependencies='bundled',
                                 part_mb=1900, threads=1, zstd_level=1)
        self.assertEqual(result['compression_threads'], 1)
        self.assertEqual(result['zstd_level'], 1)
        self.assertEqual(result['part_mb'], 1900)
        self.move_source()
        self.assertTrue(transfer.restore(self.root, result['bundle'])['verified'])

    def test_cli_default_zstd_and_auto_threads_respect_scheduler(self):
        env = dict(os.environ, SLURM_CPUS_PER_TASK='2')
        command = [sys.executable, str(ROOT / 'scripts/step5-transfer.py'), 'export',
                   '--root', str(self.root), '--run', str(self.run), '--bundle', str(self.bundle),
                   '--part-mb', '0', '--dependencies', 'bundled']
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        self.assertIn('"archive_format": "zstd"', result.stdout)
        with patch.dict(os.environ, SLURM_CPUS_PER_TASK='2'):
            self.assertLessEqual(transfer.available_cpus(), 2)
        self.assertEqual(self.bundle.read_bytes()[:4], transfer.ZSTD_MAGIC)
        self.assertEqual(transfer.json.loads(result.stdout)['archive_format'], 'zstd')
        self.assertTrue(transfer.restore(self.root, self.bundle)['verified'])


if __name__ == '__main__':
    unittest.main()
