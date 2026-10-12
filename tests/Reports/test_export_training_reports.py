"""Report selection, byte preservation and consistent publication; no training."""
import importlib.util
import json
from pathlib import Path
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
