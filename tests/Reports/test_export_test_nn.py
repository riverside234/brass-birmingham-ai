"""Committed selection, lossless checkpoint transport and source preservation."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('export_test_nn', ROOT / 'scripts/export-test-nn.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class CheckpointExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='test-nn-', dir=ROOT / 'artifacts')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = exporter.DEFAULT_RUN
        self.bytes = b'PK\x03\x04\x00\xff\r\ncheckpoint bytes\n' * 4
        self.latest = self.checkpoint('generation-0012.pt', self.bytes, 12)
        self.best = self.checkpoint('population/anchor.pt', b'older anchor\x00\r\n', 9)
        self.state = dict(iteration=12, history=[self.latest],
                          signature=dict(implementation='saved-source', config=dict(exploration=.05)),
                          policy_anchor=dict(active=dict(checkpoint=self.best)))
        self.write_state()

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def checkpoint(self, name, data, generation):
        path = self.put(self.run + '/' + name, data)
        return dict(path=path.relative_to(self.root).as_posix(),
                    sha256=exporter.digest(data), model_id='model-' + str(generation), generation=generation)

    def write_state(self):
        # CRLF bytes are deliberately retained and sealed, not normalized.
        self.state_raw = exporter.encode(self.state).replace(b'\n', b'\r\n')
        self.put(self.run + '/state.json', self.state_raw)

    def export(self, choice='latest', **kwargs):
        return exporter.export(self.root, choice=choice, part_bytes=13, **kwargs)

    def test_latest_uses_committed_receipt_not_newer_files(self):
        self.put(self.run + '/generation-0013.pt', b'uncommitted')
        self.put(self.run + '/generation-0014.pt.tmp', b'incomplete')
        result = self.export()
        bundle = result['latest']
        manifest = exporter.verify_bundle(bundle)
        self.assertEqual(manifest['recorded_checkpoint'], self.latest)
        self.assertEqual((bundle / 'state.json').read_bytes(), self.state_raw)
        self.assertEqual(json.loads((bundle / 'config.json').read_bytes()), dict(exploration=.05))
        self.assertTrue(all(p['bytes'] <= 13 for p in manifest['parts']))
        self.assertFalse((bundle / 'checkpoint.pt').exists())
        restored = exporter.materialize(self.root, 'latest')
        self.assertEqual(restored.read_bytes(), self.bytes)
        self.assertEqual((self.root / self.latest['path']).read_bytes(), self.bytes)
        self.assertEqual((self.root / self.run / 'state.json').read_bytes(), self.state_raw)

    def test_best_uses_qualified_anchor_without_claiming_global_best(self):
        bundle = self.export('best')['best']
        pointer = exporter.read(self.root / 'test_nn/best.json')
        self.assertIn('qualified policy anchor', pointer['selection_reason'])
        self.assertEqual(exporter.verify_bundle(bundle)['recorded_checkpoint'], self.best)
        self.assertEqual(exporter.materialize(self.root, 'best').read_bytes(), b'older anchor\x00\r\n')

    def test_recorded_best_and_champion_have_explicit_priority(self):
        self.state.update(best=self.latest, champion=self.best)
        self.assertEqual(exporter.selections(self.state, 'best')['best']['reason'], 'recorded best')
        del self.state['best']
        self.assertEqual(exporter.selections(self.state, 'best')['best']['reason'], 'accepted champion')

    def test_both_share_bundle_when_the_same_checkpoint_is_selected(self):
        self.state['best'] = self.latest
        self.write_state()
        result = self.export('both')
        self.assertEqual(result['latest'], result['best'])
        self.assertEqual(exporter.materialize(self.root, 'latest'), exporter.materialize(self.root, 'best'))

    def test_rerun_reuses_bundle_and_does_not_delete_prior_exports(self):
        first = self.export()['latest']
        raw = (first / 'manifest.json').read_bytes()
        self.assertEqual(self.export()['latest'], first)
        self.assertEqual((first / 'manifest.json').read_bytes(), raw)
        self.latest = self.checkpoint('generation-0013.pt', b'new version', 13)
        self.state.update(iteration=13, history=self.state['history'] + [self.latest])
        self.write_state()
        second = self.export()['latest']
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())
        self.assertEqual(exporter.materialize(self.root, 'latest').read_bytes(), b'new version')

    def test_changed_source_hash_publishes_no_selection(self):
        self.put(self.latest['path'], b'changed')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.export()
        self.assertFalse((self.root / 'test_nn/latest.json').exists())
        self.assertFalse(list((self.root / 'test_nn').glob('.staging-*')))

    def test_missing_best_fails_without_publishing_latest_in_both_mode(self):
        self.state['policy_anchor']['active'] = None
        self.write_state()
        with self.assertRaisesRegex(ValueError, 'No committed best'):
            self.export('both')
        self.assertFalse((self.root / 'test_nn/latest.json').exists())

    def test_generation_mismatch_and_missing_source_are_rejected(self):
        self.state['iteration'] = 13
        self.write_state()
        with self.assertRaisesRegex(ValueError, 'generation differs'):
            self.export()
        self.state['iteration'] = 12
        self.write_state()
        (self.root / self.latest['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'checkpoint missing'):
            self.export()

    def test_changed_part_and_metadata_are_rejected_before_restore(self):
        bundle = self.export()['latest']
        manifest = exporter.verify_bundle(bundle)
        path = bundle / manifest['parts'][0]['path']
        original = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'part missing/changed'):
            exporter.materialize(self.root, 'latest')
        self.assertFalse((bundle / 'checkpoint.pt').exists())
        path.write_bytes(original)
        (bundle / 'config.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'Metadata hash differs'):
            exporter.materialize(self.root, 'latest')

    def test_existing_wrong_checkpoint_is_preserved(self):
        bundle = self.export()['latest']
        (bundle / 'checkpoint.pt').write_bytes(b'user changed')
        with self.assertRaisesRegex(ValueError, 'Restored checkpoint changed'):
            exporter.materialize(self.root, 'latest')
        self.assertEqual((bundle / 'checkpoint.pt').read_bytes(), b'user changed')

    def test_verify_does_not_materialize_checkpoint(self):
        bundle = self.export()['latest']
        exporter.materialize(self.root, 'latest', restore=False)
        self.assertFalse((bundle / 'checkpoint.pt').exists())

    def test_unrelated_pointer_is_not_overwritten(self):
        self.put('test_nn/latest.json', b'{"mine":true}')
        with self.assertRaisesRegex(ValueError, 'unrelated file'):
            self.export()
        self.assertEqual((self.root / 'test_nn/latest.json').read_bytes(), b'{"mine":true}')

    def test_explicit_run_and_unique_discovery(self):
        different = self.root / 'artifacts/step5/renamed'
        (self.root / self.run).rename(different)
        self.assertEqual(exporter.find_run(self.root), different)
        self.assertEqual(exporter.find_run(self.root, different), different)
        self.put('artifacts/step5/second/state.json', exporter.encode(self.state))
        with self.assertRaisesRegex(ValueError, 'unique run'):
            exporter.find_run(self.root)

    def il_upgrade(self):
        run = self.root / exporter.DEFAULT_IL_RUN
        source = self.put(exporter.DEFAULT_IL_RUN + '/candidate.pt', b'upgraded IL checkpoint')
        entry = dict(path=source.relative_to(self.root).as_posix(),
                     sha256=exporter.receipt(source)['sha256'], model_id='IL-model')
        raw = exporter.encode(dict(contract='completed-il-critic-upgrade/1.0.0',
                                   architecture=dict(architecture='IL'), candidate=entry))
        self.put(exporter.DEFAULT_IL_RUN + '/upgrade.json', raw)
        return run, entry, raw

    def test_il_defaults_to_upgrade_candidate_and_preserves_source_snapshot(self):
        run, entry, raw = self.il_upgrade()
        bundle = self.export('IL', calibration=dict(host='not-used'))['IL']
        manifest = exporter.verify_bundle(bundle)
        self.assertEqual(manifest['recorded_checkpoint'], entry)
        self.assertEqual(manifest['source_run'], exporter.DEFAULT_IL_RUN)
        self.assertEqual(manifest['source_state_path'], exporter.DEFAULT_IL_RUN + '/upgrade.json')
        self.assertEqual((bundle / 'state.json').read_bytes(), raw)
        self.assertEqual(exporter.read(bundle / 'config.json'), dict(architecture=dict(architecture='IL')))
        self.assertNotIn('calibration', manifest)
        self.assertEqual(exporter.materialize(self.root, 'IL').read_bytes(), b'upgraded IL checkpoint')
        self.assertEqual((run / 'upgrade.json').read_bytes(), raw)
        self.assertFalse((self.root / 'test_nn/latest.json').exists())
        self.assertEqual(self.export('IL')['IL'], bundle)

    def test_il_explicit_native_run_uses_recorded_best_not_last_or_alias(self):
        self.run = 'artifacts/step5/custom-il'
        best = self.checkpoint('epoch-0002.pt', b'IL best', 2)
        best = dict(path='epoch-0002.pt', sha256=best['sha256'])
        self.put(self.run + '/best.pt', b'uncommitted alias')
        signature = dict(format='bc-training/2.2.0', architecture=dict(architecture='IL'),
                         update=dict(effective_batch_size=16))
        self.state = dict(signature=signature, epoch=3, best=best,
                          last=dict(path='epoch-0003.pt', sha256='0' * 64))
        self.write_state()
        bundle = self.export('IL', run=self.run)['IL']
        self.assertEqual(exporter.verify_bundle(bundle)['recorded_checkpoint'], best)
        self.assertEqual(exporter.read(bundle / 'config.json'), signature)
        self.assertEqual(exporter.materialize(self.root, 'IL').read_bytes(), b'IL best')

    def test_il_rejects_missing_wrong_run_and_changed_candidate(self):
        with self.assertRaises(FileNotFoundError):
            self.export('IL')
        with self.assertRaisesRegex(ValueError, 'requires an imitation run'):
            self.export('IL', run=self.run)
        run, entry, raw = self.il_upgrade()
        (run / 'candidate.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.export('IL')
        self.assertFalse((self.root / 'test_nn/IL.json').exists())

    def test_il_cli_export_verify_restore_without_utility_validation(self):
        self.il_upgrade()
        for mode in ((), ('--verify',), ('--restore',)):
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/export-test-nn.py'),
                                     '--root', str(self.root), '--select', 'IL', *mode],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('IL:', result.stdout)
            if not mode:
                self.assertIn('utility validation skipped', result.stdout)
                self.assertIn('--restore --select IL', result.stdout)
                pointer = exporter.read(self.root / 'test_nn/IL.json')
                target = self.root / 'test_nn' / pointer['bundle'] / 'checkpoint.pt'
            self.assertEqual(target.exists(), mode == ('--restore',))
        self.assertEqual(target.read_bytes(), b'upgraded IL checkpoint')

    def test_external_or_traversing_checkpoint_paths_are_rejected(self):
        for path in ('../outside.pt', str(self.root.parent / 'outside.pt')):
            with self.subTest(path=path), self.assertRaises(ValueError):
                exporter.checkpoint_path(self.root, self.root / self.run, dict(path=path))

    @unittest.skipUnless(shutil.which('git'), 'Git required for transport test')
    def test_git_checkout_restores_exact_binary_and_json_bytes(self):
        def git(*args, cwd=None):
            result = subprocess.run(['git', *args], cwd=cwd or self.root,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            return result.stdout
        for name in ('.gitignore', '.gitattributes'):
            shutil.copyfile(ROOT / name, self.root / name)
        bundle = self.export()['latest']
        # A reconstructed local file must not accidentally enter Git.
        exporter.materialize(self.root, 'latest')
        git('init', '-q')
        git('config', 'core.autocrlf', 'true')
        git('add', 'test_nn')
        names = git('ls-files').decode().splitlines()
        self.assertTrue(any('.pt.part' in name for name in names))
        self.assertFalse(any(name.endswith('/checkpoint.pt') for name in names))
        git('add', '.gitignore', '.gitattributes')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '-qm', 'checkpoint export fixture')
        clone = self.root / 'clone'
        git('read-tree', 'HEAD')
        git('checkout-index', '--all', '--prefix=clone/')
        restored = exporter.materialize(clone, 'latest')
        self.assertEqual(restored.read_bytes(), self.bytes)
        self.assertEqual((restored.parent / 'state.json').read_bytes(), self.state_raw)
        self.assertEqual(exporter.verify_bundle(restored.parent), exporter.verify_bundle(bundle))


if __name__ == '__main__':
    unittest.main()
