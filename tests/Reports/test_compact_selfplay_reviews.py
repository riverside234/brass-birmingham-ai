"""Exercise the registry compaction CLI without modifying real approvals."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import compact_selfplay_reviews as cli
from brass_ai.recovery.source_reviews import expand_source_reviews


class CompactRegistryCliTests(unittest.TestCase):
    def original(self):
        entry = dict(from_implementation='a' * 64, to_implementation='b' * 64,
                     learning_compatibility='c' * 64, reason='Repeated review details. ' * 40,
                     config_changes={'learning_rate': {'from': .1, 'to': .2}})
        return dict(format='selfplay-resume-sources/1.0.0',
                    transitions=[copy.deepcopy(entry) for _ in range(10)])

    def run_cli(self, path):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/compact_selfplay_reviews.py'),
                               str(path)], cwd=path.parent, capture_output=True, text=True, timeout=30)

    def test_command_roundtrip_and_no_rewrite_when_already_compact(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            path = Path(folder) / 'reviews.json'
            original = self.original()
            path.write_text(json.dumps(original), encoding='utf-8')
            size = path.stat().st_size
            first = self.run_cli(path)
            self.assertEqual(first.returncode, 0, first.stderr)
            compact = path.read_bytes()
            self.assertLess(len(compact), size)
            self.assertEqual(expand_source_reviews(json.loads(compact)), original)
            modified = path.stat().st_mtime_ns
            second = self.run_cli(path)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(path.read_bytes(), compact)
            self.assertEqual(path.stat().st_mtime_ns, modified)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_invalid_receipt_is_not_rewritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            path = Path(folder) / 'reviews.json'
            path.write_bytes(b'{"format":"selfplay-resume-sources/2.0.0","transitions":[{}]}')
            before = path.read_bytes()
            result = self.run_cli(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing source review identity', result.stderr)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_overlapping_compactions_keep_independent_temporaries(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            path = Path(folder) / 'reviews.json'
            original = self.original()
            path.write_text(json.dumps(original), encoding='utf-8')
            replace = Path.replace
            temporaries = []

            def interleaved_publish(source, target):
                if target == path:
                    temporaries.append(source)
                    if len(temporaries) == 1:
                        cli.main()
                return replace(source, target)

            with patch.object(sys, 'argv', ['compact_selfplay_reviews.py', str(path)]), \
                    patch.object(Path, 'replace', interleaved_publish), contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(expand_source_reviews(json.loads(path.read_bytes())), original)
            self.assertEqual(len(set(temporaries)), 2)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_publication_keeps_original_registry_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            path = Path(folder) / 'reviews.json'
            path.write_text(json.dumps(self.original()), encoding='utf-8')
            before = path.read_bytes()
            with patch.object(sys, 'argv', ['compact_selfplay_reviews.py', str(path)]), \
                    patch.object(Path, 'replace', side_effect=OSError('interrupted publication')):
                with self.assertRaises(OSError):
                    cli.main()
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
