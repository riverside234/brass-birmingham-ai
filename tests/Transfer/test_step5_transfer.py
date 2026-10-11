"""Small archive/continuation checks; no game collection or learning campaign."""
import importlib.util
import os
from pathlib import Path
import socket
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
        self.bundle = self.area / 'migration.zip'
        for tree in transfer.CODE_TREES:
            (self.root / tree).mkdir(parents=True, exist_ok=True)
        for name in ('learning/runner.py', 'learning/batching.py', 'searchlab/spiel.py', 'beliefs.py', 'runtime_contracts.py'):
            self.put('src/brass_ai/' + name, b'# fixture\n')
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
        with zipfile.ZipFile(self.bundle) as source, zipfile.ZipFile(bad, 'w') as out:
            manifest = transfer.json.loads(source.read('transfer.json'))
            record = next(r for r in manifest['files'] if r['path'].endswith('generation-0002.pt'))
            payload = transfer.payload_name(manifest, record)
            for name in source.namelist():
                data = source.read(name)
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
        self.assertEqual(result['part_mb'], 100)
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
        self.assertEqual(implementation_id(), '826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16')

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
        full = transfer.export(self.root, self.run, self.area / 'full.zip', mode='full', part_mb=0)
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
        with transfer.open_bundle(result['bundle']) as archive:
            dll = next(r for r in manifest['files'] if r['path'].endswith('engine.dll'))
            self.assertNotIn(transfer.payload_name(manifest, dll), archive.namelist())
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
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertEqual(len(archive.namelist()) - 1, len({r['sha256'] for r in included}))
        self.assertLess(result['unique_payload_bytes'], result['bytes'])
        self.move_source(); transfer.restore(self.root, self.bundle)
        self.assertEqual((self.run / 'best.pt').read_bytes(), b'old opponent')

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

    def test_parts_restore_without_joining_zip_and_detect_missing_or_changed_part(self):
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
        with zipfile.ZipFile(self.bundle) as src, zipfile.ZipFile(old, 'w') as dst:
            manifest = transfer.json.loads(src.read('transfer.json'))
            for record in manifest['files']:
                self.assertFalse(record.pop('external'))
                dst.writestr('project/' + record['path'], src.read(transfer.payload_name(manifest, record)))
            manifest['format'] = transfer.OLD_FORMAT
            manifest['id'] = transfer.digest({k: v for k, v in manifest.items() if k != 'id'})
            dst.writestr('transfer.json', transfer.encode(manifest))
        self.move_source()
        self.assertTrue(transfer.restore(self.root, old)['verified'])


if __name__ == '__main__':
    unittest.main()
