"""Report selection, byte preservation and consistent publication; no training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('export_reports', ROOT / 'scripts/export-training-reports.py')
reports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reports)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='reports-test-', dir=ROOT / 'artifacts')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = 'artifacts/step5/example'
        self.state = dict(format='step5-recovery-league/2.2.0', iteration=60, games_done=19200,
            signature=dict(implementation='recorded-source', config=dict(learning_rate=1e-5)),
            origin=dict(bc_run='artifacts/step5/il'),
            evaluations=[dict(report=self.run + '/evaluations/monitor-0003/report.json')],
            champion_gate=dict(recent=[]))
        self.put(self.run + '/state.json', reports.encode(self.state).replace(b'\n', b'\r\n'))
        self.put(self.run + '/retention-history.json', {'older': 'summary'})
        for iteration in (57, 58, 59, 60):
            self.put(f'{self.run}/iteration-{iteration:04d}/report.json',
                     dict(iteration=iteration, timings=dict(iteration_seconds=1)))

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else reports.encode(value))
        return path

    def export(self, **kwargs):
        return reports.export(self.root, self.run, **kwargs)

    def test_collects_latest_two_all_evaluations_and_il_ancestry_preserving_bytes(self):
        self.put(self.run + '/best.json', {'generation': 0})
        self.put(self.run + '/generation-0060.pt', b'large checkpoint excluded')
        self.put(self.run + '/iteration-0059/games.jsonl.gz', b'replay excluded')
        self.put(self.run + '/iteration-0059/games.jsonl.gz.features.json', {'excluded': True})
        self.put(self.run + '/iteration-0059/teacher/receipt.json', {'qualified': 10})
        self.put(self.run + '/iteration-0059/profile-collection/report.json', {'excluded': True})
        for panel in ('champion-00019200', 'monitor-0003', 'broader-0001'):
            self.put(f'{self.run}/evaluations/{panel}/report.json', {'panel': panel})
            self.put(f'{self.run}/evaluations/{panel}/spec.json', {'seeds': [1, 2]})
        self.put('artifacts/step5/il/state.json', {'signature': {'parent_imitation': {'bc_run': 'artifacts/step5/parent'}}})
        self.put('artifacts/step5/il/report.json', {'epochs': 6})
        self.put('artifacts/step5/parent/state.json', {'signature': {}})
        self.put('artifacts/step5/parent/report.json', {'epochs': 5})
        before = {p: p.read_bytes() for p in (self.root / 'artifacts').rglob('*') if p.is_file()}
        output, manifest = self.export()
        self.assertTrue(output.name.startswith('example-g0060-'))
        self.assertEqual(manifest['latest_iterations'], [58, 59])
        self.assertEqual(manifest['missing'], [])
        self.assertTrue((output / 'iteration-0058/report.json').is_file())
        self.assertTrue((output / 'iteration-0059/teacher/receipt.json').is_file())
        self.assertTrue((output / 'evaluations/broader-0001/spec.json').is_file())
        self.assertTrue((output / 'lineage/artifacts/step5/parent/report.json').is_file())
        self.assertFalse((output / 'iteration-0057').exists())
        self.assertFalse((output / 'iteration-0060').exists())
        self.assertFalse(any(p.suffix in ('.pt', '.gz', '.npz') for p in output.rglob('*')))
        self.assertFalse(any('features' in p.name or 'profile-collection' in p.parts for p in output.rglob('*')))
        for entry in manifest['files']:
            raw = (output / entry['path']).read_bytes()
            self.assertEqual(raw, before[self.root / entry['source']])
            self.assertEqual(reports.checksum(raw), entry['sha256'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(json.loads((output / 'recorded-config.json').read_text()), self.state['signature']['config'])

    def test_missing_reports_are_explicit_not_reconstructed_from_summaries(self):
        (self.root / self.run / 'iteration-0058/report.json').unlink()
        output, manifest = self.export(output='results/snapshot')
        missing = {x['source'] for x in manifest['missing']}
        self.assertIn(self.run + '/iteration-0058/report.json', missing)
        self.assertIn(self.run + '/evaluations/monitor-0003/report.json', missing)
        self.assertIn('artifacts/step5/il/report.json', missing)
        self.assertFalse((output / 'iteration-0058/report.json').exists())

    def test_never_overwrites_existing_output_or_accepts_escaped_paths(self):
        sentinel = self.put('results/existing/keep.json', b'unchanged')
        for output in ('results/existing', 'artifacts/export', '../outside', 'results/../../outside'):
            with self.assertRaises(ValueError):
                self.export(output=output)
        self.assertEqual(sentinel.read_bytes(), b'unchanged')

    def test_detects_changed_source_and_does_not_publish_partial_bundle(self):
        original_encode = reports.encode

        def concurrent_write(value):
            if value.get('format') == reports.FORMAT:
                updated = dict(self.state, iteration=61)
                (self.root / self.run / 'state.json').write_bytes(original_encode(updated))
            return original_encode(value)

        with patch.object(reports, 'encode', side_effect=concurrent_write):
            with self.assertRaisesRegex(ValueError, 'changed during export'):
                self.export(output='results/snapshot')
        self.assertFalse((self.root / 'results/snapshot').exists())
        self.assertEqual(list((self.root / 'results').iterdir()), [])

    def test_more_iterations_external_panels_and_receipt_mismatch_are_recorded(self):
        self.state['evaluations'][0]['report_sha256'] = 'old-receipt'
        self.put(self.run + '/state.json', self.state)
        self.put(self.run + '/evaluations/monitor-0003/report.json', {'pruned': True})
        self.put('artifacts/comparison/report.json', {'candidate': 'generation60'})
        output, manifest = self.export(latest=3, extras=['artifacts/comparison'])
        self.assertTrue((output / 'iteration-0057/report.json').exists())
        self.assertTrue((output / 'related/artifacts/comparison/report.json').exists())
        self.assertEqual(len(manifest['warnings']), 1)
        entry = next(x for x in manifest['files'] if x['path'] == 'evaluations/monitor-0003/report.json')
        self.assertFalse(entry['matches_recorded_sha256'])

    def test_absent_run_is_not_created(self):
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            reports.export(self.root, 'artifacts/missing')
        self.assertFalse((self.root / 'artifacts/missing').exists())

    def modern_run(self, generation):
        self.run = 'artifacts/step5/joint7-v11-1-grouped-lr'
        il_run = 'artifacts/step5/features4-v11-1-batch7-il'
        controller = dict(contract='ppo-categorical-kl-lr/1.1.0', mode='apply',
            kl=.006, median_kl=.005, steps_done=8, steps_planned=8,
            multiplier_before=1.0, multiplier_after=1.0, decision='hold',
            reason='target_band', effective_lr=6e-6,
            health=dict(anchor_distance=.12, anchor_budget=.15,
                anchor_group_budget=.30, anchor_group_min_positions=64,
                anchor_groups={name:dict(positions=512, anchor_distance=.12, status='ok')
                    for name in ('canal_early', 'canal_late', 'rail_early', 'rail_late')}))
        self.state = dict(format='step5-recovery-league/2.2.0', iteration=generation,
            games_done=320 * generation, origin=dict(bc_run=il_run), evaluations=[],
            signature=dict(implementation='v11-1-grouped-source', config=dict(
                selfplay_recipe='latest-model-selfplay/2.0.0', teacher_enabled=False,
                exploration=.08, population=dict(recent=4, anchors=12),
                curriculum=[dict(until=None, selfplay=60, historical=30, legacy=10)],
                ppo_lr_controller=dict(contract=controller['contract'], mode='apply',
                    anchor_kl_budget=.15, anchor_group_kl_budget=.30, anchor_group_min_positions=64))))
        self.put(self.run + '/state.json', self.state)
        self.put(il_run + '/state.json', {'signature': {}})
        self.put(il_run + '/report.json', {'epochs': 10, 'complete': True})
        for iteration in range(max(0, generation - 2), generation):
            self.put(f'{self.run}/iteration-{iteration:04d}/report.json', dict(
                iteration=iteration, actor=dict(lr_controller=controller,
                    sample_usage=dict(actor_distinct_used=100, actor_unused_eligible=5)),
                sampling=dict(learner_trajectories=1024, by_composition=dict(
                    selfplay=dict(tables=192, learner_trajectories=768),
                    historical=dict(tables=96, learner_trajectories=192),
                    legacy=dict(tables=32, learner_trajectories=64))),
                search=dict(enabled=False, collection_count=0, optimizer_steps=0),
                timings=dict(iteration_seconds=100)))
        return controller

    def test_default_cli_exports_new_run_and_keeps_completed_il_name(self):
        controller = self.modern_run(generation=1)
        # An explicitly null gate must also work for a gate-disabled run.
        self.state['champion_gate'] = None
        self.put(self.run + '/state.json', self.state)
        before = {p: p.read_bytes() for p in (self.root / 'artifacts').rglob('*') if p.is_file()}
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts/export-training-reports.py'),
            '--root', str(self.root)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        outputs = list((self.root / 'results').iterdir())
        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self.assertTrue(output.name.startswith('joint7-v11-1-grouped-lr-g0001-'))
        manifest = json.loads((output / 'export-manifest.json').read_text())
        self.assertEqual(manifest['run'], self.run)
        self.assertEqual(manifest['latest_iterations'], [0])
        self.assertEqual(manifest['missing'], [dict(source=self.run + '/retention-history.json',
            reason='Retained iteration trends')])
        self.assertEqual(manifest['warnings'], [])
        report = json.loads((output / 'iteration-0000/report.json').read_text())
        self.assertEqual(report['actor']['lr_controller'], controller)
        self.assertFalse(report['search']['enabled'])
        il_report = output / 'lineage/artifacts/step5/features4-v11-1-batch7-il/report.json'
        self.assertEqual(il_report.read_bytes(), before[self.root / self.state['origin']['bc_run'] / 'report.json'])
        self.assertFalse((output / 'evaluations').exists())
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_full_and_compact_grouped_lr_fields_are_preserved(self):
        controller = self.modern_run(generation=3)
        history = {'iteration-0000/report.json': dict(sha256='retired-full-report',
            summary=dict(summary_version=5, actor=dict(lr_controller=controller),
                search=dict(enabled=False), sampling=dict(learner_trajectories=1024)))}
        self.put(self.run + '/retention-history.json', history)
        self.put(self.run + '/iteration-0003/report.json', {'iteration': 3, 'pending': True})
        output, manifest = self.export()
        self.assertEqual(manifest['latest_iterations'], [1, 2])
        self.assertEqual(manifest['missing'], [])
        self.assertFalse((output / 'iteration-0003').exists())
        self.assertEqual(json.loads((output / 'retention-history.json').read_text()), history)
        for entry in manifest['files']:
            self.assertEqual((output / entry['path']).read_bytes(), (self.root / entry['source']).read_bytes())
        config = json.loads((output / 'recorded-config.json').read_text())
        self.assertEqual(config, self.state['signature']['config'])
        self.assertEqual(config['ppo_lr_controller']['anchor_kl_budget'], .15)
        self.assertEqual(config['ppo_lr_controller']['anchor_group_kl_budget'], .30)
        report = json.loads((output / 'iteration-0002/report.json').read_text())
        self.assertEqual(report['actor']['lr_controller'], controller)
        self.assertEqual(report['actor']['sample_usage']['actor_unused_eligible'], 5)
        self.assertEqual(report['sampling']['by_composition']['historical']['learner_trajectories'], 192)

    def test_windows_publication_retries_transient_handle_error(self):
        stage = self.put('stage/report.json', {'original': True}).parent
        output = self.root / 'published'
        error = PermissionError('temporary Windows handle')
        error.winerror = 32
        original_rename = Path.rename
        def rename(path, destination):
            if mocked_rename.call_count == 1:
                raise error
            return original_rename(path, destination)
        with patch.object(reports, 'os') as system, patch.object(reports.time, 'sleep') as sleep:
            system.name = 'nt'
            with patch.object(Path, 'rename', autospec=True, side_effect=rename) as mocked_rename:
                reports.publish(stage, output)
                self.assertEqual(mocked_rename.call_count, 2)
            sleep.assert_called_once_with(.05)
        self.assertEqual(json.loads((output / 'report.json').read_text()), {'original': True})
        self.assertFalse(stage.exists())

    def test_fixed_il_probe_reports_are_included_without_validation_tensors(self):
        self.modern_run(generation=1)
        self.put(self.run+'/il-kl/manifest.json',dict(format='fixed-il-kl-probe/1.0.0',probe_hash='fixed',
            cache=dict(path=self.run+'/il-kl/probe.npz')))
        self.put(self.run+'/il-kl/initial-report.json',dict(generation=0,measurement=dict(kl_mean=0.)))
        self.put(self.run+'/il-kl/probe.npz',b'dense validation tensors excluded')
        self.put(self.run+'/il-kl/.prepared/shard.json',dict(excluded=True))
        output,manifest=self.export()
        self.assertTrue((output/'il-kl/manifest.json').is_file())
        self.assertTrue((output/'il-kl/initial-report.json').is_file())
        self.assertFalse((output/'il-kl/probe.npz').exists())
        self.assertFalse((output/'il-kl/.prepared').exists())
        self.assertEqual(len([e for e in manifest['files'] if e['path'].startswith('il-kl/')]),2)

    def test_windows_publication_stops_if_output_appears_during_retry(self):
        stage = self.put('stage/report.json', {}).parent
        output = self.root / 'published'
        error = PermissionError('temporary Windows handle')
        error.winerror = 5
        def competing_export(path, destination):
            self.put('published/keep.json', {'sentinel': True})
            raise error
        with patch.object(reports, 'os') as system, patch.object(reports.time, 'sleep'):
            system.name = 'nt'
            with patch.object(Path, 'rename', autospec=True, side_effect=competing_export) as rename:
                with self.assertRaisesRegex(ValueError, 'Output appeared'):
                    reports.publish(stage, output)
                rename.assert_called_once()
        self.assertTrue((stage / 'report.json').is_file())
        self.assertEqual(json.loads((output / 'keep.json').read_text()), {'sentinel': True})

    def test_windows_publication_retry_is_bounded(self):
        stage = self.put('stage/report.json', {}).parent
        output = self.root / 'published'
        error = PermissionError('persistent Windows access error')
        error.winerror = 5
        with patch.object(reports, 'os') as system, patch.object(reports.time, 'sleep') as sleep:
            system.name = 'nt'
            with patch.object(Path, 'rename', autospec=True, side_effect=error) as rename:
                with self.assertRaisesRegex(PermissionError, 'persistent'):
                    reports.publish(stage, output)
                self.assertEqual(rename.call_count, 5)
            self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), .75)
        self.assertFalse(output.exists())
        self.assertTrue((stage / 'report.json').is_file())

    def test_real_iteration60_exports_can_be_bundled_without_models_or_torch(self):
        # Reconstruct only report paths in a temporary fixture, never the real run.
        source = ROOT / 'results/selfplay-joint'
        state = json.loads((source / 'state.json').read_text())
        self.run = 'artifacts/step5/v4-selfplay-joint'
        for filename in ('state.json', 'retention-history.json'):
            self.put(self.run + '/' + filename, (source / filename).read_bytes())
        self.put(self.run + '/iteration-0059/report.json', (source / 'iter59.json').read_bytes())
        self.put(state['evaluations'][-1]['report'], (source / 'monitor.json').read_bytes())
        self.put(state['champion_gate']['recent'][-1]['report'], (source / 'champion.json').read_bytes())
        output, manifest = self.export(output='results/real-evidence-smoke')
        self.assertEqual(manifest['committed_generation'], 60)
        self.assertEqual(len(manifest['files']), 5)
        self.assertEqual((output / 'iteration-0059/report.json').read_bytes(), (source / 'iter59.json').read_bytes())
        self.assertIn(self.run + '/iteration-0058/report.json', {m['source'] for m in manifest['missing']})


if __name__ == '__main__':
    unittest.main()
