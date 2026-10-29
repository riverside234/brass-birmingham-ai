"""Short result transport checks; no games, checkpoint loads or lab hardware."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import rust_lab_results as reports
spec = importlib.util.spec_from_file_location('lab_runner', ROOT/'scripts/run-rust-lab-qualification.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class Results(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root/'artifacts/rust-backend/example'
        self.evidence.mkdir(parents=True)

    def test_portable_bytes_compression_pruning_and_unique_export(self):
        original = b'{"path":"/lab/private/checkpoint.pt", "passed":false}\n'
        (self.evidence/'qualification.json').write_bytes(original)
        huge = b'compiler trace\n' * 90000
        (self.evidence/'compile.log').write_bytes(huge)
        (self.evidence/'checkpoint.pt').write_bytes(b'never export')
        (self.evidence/'raw.jsonl.gz').write_bytes(b'never export')
        (self.evidence/'target').mkdir(); (self.evidence/'target/build.json').write_text('{}')
        first = reports.export(self.evidence, root=self.root)
        second = reports.export(self.evidence, root=self.root)
        self.assertNotEqual(first, second)
        self.assertEqual((first/'evidence/qualification.json').read_bytes(), original)
        self.assertEqual(gzip.decompress((first/'evidence/compile.log.gz').read_bytes()), huge)
        self.assertFalse((first/'evidence/checkpoint.pt').exists())
        manifest = json.loads((first/'export-manifest.json').read_text())
        self.assertEqual(len(manifest['files']), 2)
        for entry in manifest['files']:
            self.assertFalse(Path(entry['path']).is_absolute())
            self.assertEqual(reports.sha(first/entry['path']), entry['sha256'])
        self.assertEqual(len(manifest['omitted']), 3)

    def test_export_cannot_escape_evidence_root(self):
        with self.assertRaises(ValueError): reports.export(self.root, root=self.root)
        with self.assertRaises(ValueError): reports.export(self.root/'artifacts/rust-backend', root=self.root)

    def test_symlink_is_not_followed(self):
        external = self.root/'outside.json'; external.write_text('{"private":true}')
        link = self.evidence/'link.json'
        try: link.symlink_to(external)
        except OSError: self.skipTest('Platform does not permit unprivileged symlink creation')
        result = reports.export(self.evidence, root=self.root)
        self.assertFalse((result/'evidence/link.json').exists())

    def test_git_will_include_reports_and_compressed_logs(self):
        for name in ('README.md', 'future/evidence/runner.json', 'future/evidence/build.log', 'future/evidence/tables.json.gz'):
            result = subprocess.run(['git', 'check-ignore', '-q', '--no-index', 'results/rust-lab-qualification/'+name], cwd=ROOT)
            self.assertEqual(result.returncode, 1)

    def fixture_run(self, statuses, *, stage='all', preflight_fail=False):
        run = self.root/'artifacts/step5/joint7-v11-2-grouped-lr-critic'
        run.mkdir(parents=True)
        reports.write(run/'state.json', dict(iteration=1326, history=[dict(generation=1326,
            sha256='ada8d72195837cb399c13effb975cce19840e847080a7a0d4e03091235d85cbe')], signature=dict(implementation='original')))
        package = self.root/'distribution/rust-lab-g1326'; package.mkdir(parents=True)
        reports.write(package/'transfer-manifest.json', {})
        calls = []
        def execute(cmd, log):
            calls.append(cmd)
            code, passed = statuses[len(calls)-1]
            option = '--output' if '--output' in cmd else '--out'
            out = Path(cmd[cmd.index(option)+1]); out.mkdir()
            reports.write(out/'qualification.json', dict(passed=passed, lab_qualification='Pending' if not passed else 'Fixture success'))
            log.write_text('console evidence')
            return code
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'PACKAGE', package), \
             patch.object(runner, 'verify_package', side_effect=ValueError('bad transfer') if preflight_fail else None,
                          return_value={'training_reference': {'generation': 1326}}), \
             patch.object(runner, 'execute', side_effect=execute), \
             patch.object(runner, 'export', side_effect=lambda path: reports.export(path, root=self.root)):
            code = runner.main(['--stage', stage, '--run', str(run)])
        exported = list((self.root/reports.RESULTS).iterdir())
        self.assertEqual(len(exported), 1)
        result = json.loads((exported[0]/'evidence/runner.json').read_text())
        return code, result, calls

    def test_full_chain_exports_and_wires_original_receipts(self):
        code, result, calls = self.fixture_run([(0, True)]*5)
        self.assertEqual(code, 0); self.assertTrue(result['passed']); self.assertEqual(len(calls), 5)
        for flag in ('--b02', '--b03'):
            path = Path(calls[4][calls[4].index(flag)+1])
            self.assertTrue(path.is_relative_to(self.root/'artifacts/rust-backend'))
        self.assertIn('--transfer-manifest', calls[2]); self.assertIn('--transfer-manifest', calls[3])

    def test_failure_exports_and_stops_before_next_stage(self):
        code, result, calls = self.fixture_run([(7, False)])
        self.assertEqual(code, 7); self.assertFalse(result['passed']); self.assertEqual(len(calls), 1)

    def test_zero_exit_deferred_receipt_is_not_passed(self):
        code, result, _ = self.fixture_run([(0, False)], stage='b01')
        self.assertEqual(code, 2); self.assertFalse(result['passed'])

    def test_preflight_failure_still_exports_error(self):
        code, result, calls = self.fixture_run([], preflight_fail=True)
        self.assertEqual(code, 1); self.assertIn('bad transfer', result['error']); self.assertFalse(calls)

    def test_generation_or_hash_mismatch_fails(self):
        run = self.root/'run'; run.mkdir()
        reports.write(run/'state.json', dict(iteration=1326, history=[dict(generation=1326, sha256='a'*64)], signature=dict(implementation='original')))
        self.assertEqual(runner.run_identity(run, 1326, 'a'*64)['generation'], 1326)
        for generation, digest in ((1325, 'a'*64), (1326, 'b'*64)):
            with self.assertRaises(ValueError): runner.run_identity(run, generation, digest)


if __name__ == '__main__': unittest.main()
