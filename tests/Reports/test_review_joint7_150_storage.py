"""Generation-150 compact reports preserve complete evidence and verify sidecars."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import review_joint7_150 as report


class CompactAuditTests(unittest.TestCase):
    def data(self):
        return dict(format=report.EXPANDED_FORMAT, source_folder='results/example',
                    manifest_checks=[{'sha256': 'unchanged'}], windows=[{'mean': 1.0}],
                    iterations=[{'iteration': 0, 'actor': {'loss': .15, 'missing': None}},
                                {'iteration': 1, 'actor': {'loss': .12, 'labels': ['a', 'b']}}],
                    missing=['old-report'])

    def save(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_roundtrip_deterministic_output_and_legacy_loading(self):
        data = self.data()
        before = copy.deepcopy(data)
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            path = Path(folder) / 'metrics.json'
            summary = report.write_metrics(path, data)
            self.assertEqual(data, before)
            self.assertNotIn('iterations', summary)
            self.assertEqual(summary['iterations_ref']['count'], 2)
            detail = path.parent / summary['iterations_ref']['path']
            compressed = detail.read_bytes()
            self.assertEqual(hashlib.sha256(compressed).hexdigest(), summary['iterations_ref']['sha256'])
            self.assertEqual(json.loads(gzip.decompress(compressed)), data['iterations'])
            self.assertEqual(compressed[4:8], b'\0' * 4)
            self.assertEqual(compressed[9], 255)
            self.assertEqual(report.read_metrics(path), before)
            contents = {p.name: p.read_bytes() for p in path.parent.iterdir()}
            report.write_metrics(path, report.read_metrics(path))
            self.assertEqual({p.name: p.read_bytes() for p in path.parent.iterdir()}, contents)
            self.save(path, before)
            self.assertEqual(report.read_metrics(path), before)

    def test_bad_checksum_count_and_path_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            path = Path(folder) / 'metrics.json'
            summary = report.write_metrics(path, self.data())
            for key, value in [('count', 3), ('count', True), ('sha256', 'bad'),
                               ('path', '../iterations.json.gz')]:
                bad = copy.deepcopy(summary)
                bad['iterations_ref'][key] = value
                self.save(path, bad)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    report.read_metrics(path)
            self.save(path, dict(summary, iterations=[]))
            with self.assertRaises(ValueError):
                report.read_metrics(path)
            self.save(path, summary)
            detail = path.parent / summary['iterations_ref']['path']
            detail.write_bytes(detail.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                report.read_metrics(path)

    def test_failed_summary_publication_keeps_previous_report_readable(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            path = Path(folder) / 'metrics.json'
            original = self.data()
            report.write_metrics(path, original)
            changed = self.data()
            changed['iterations'].append({'iteration': 2})
            replace = Path.replace

            def fail_summary(source, target):
                if target == path:
                    raise OSError('interrupted summary publication')
                return replace(source, target)

            with patch.object(Path, 'replace', fail_summary), self.assertRaises(OSError):
                report.write_metrics(path, changed)
            self.assertEqual(report.read_metrics(path), original)

    def test_committed_evidence_contains_all_original_iteration_records(self):
        path = ROOT / 'docs/joint7-150-evidence/metrics.json'
        data = report.read_metrics(path)
        self.assertEqual(len(data['iterations']), 150)
        self.assertEqual([row['iteration'] for row in data['iterations']], list(range(150)))
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                               allow_nan=False).encode('utf-8')
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), '278d0c35f62b714c0003d576e2ade24744ee8c71ea83ffb412a17332485e1aa7')


if __name__ == '__main__':
    unittest.main()
