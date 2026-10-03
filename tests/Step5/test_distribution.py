"""Regression checks for Git newline conversion of byte-pinned R3 evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_qualification_reports import check_reports, matching_bytes
from brass_ai.bank import admission, read


class QualificationDistributionTests(unittest.TestCase):
    def fixture(self, root, *, stored_lf=False, pinned_lf=False):
        name = 'training/opponents/evaluations/fixture/report.json'
        identity = 'bank-test/balanced-v3'
        report = dict(bank='bank-test', profiles={'balanced-v3': dict(policy_id=identity,
                      training_eligible=True, evaluation_eligible=False, niche_qualified=True)})
        lf = (json.dumps(report, indent=2) + '\n').encode()
        pinned = lf if pinned_lf else lf.replace(b'\n', b'\r\n')
        path = root / name
        path.parent.mkdir(parents=True)
        path.write_bytes(lf if stored_lf else lf.replace(b'\n', b'\r\n'))
        fingerprint = hashlib.sha256(pinned).hexdigest()
        (root / 'training/opponents/pool.json').write_text(json.dumps(dict(training={identity:
            dict(qualification=name, sha256=fingerprint, profile='balanced-v3')}, evaluation={})))
        (root / 'training/opponents/current.json').write_text(json.dumps(dict(bank='bank-test',
            qualification=name, qualification_sha256=fingerprint)))
        return path, pinned, identity

    def test_linux_converted_report_is_accepted_without_rewriting_pins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, pinned, identity = self.fixture(root, stored_lf=True)
            original = path.read_bytes()
            admission(root, identity)
            checked = check_reports(root)
            self.assertEqual(checked['repaired'], [])
            self.assertEqual(len(checked['line_endings_equivalent']), 1)
            self.assertEqual(path.read_bytes(), original)
            result = check_reports(root, repair=True)
            self.assertEqual(len(result['repaired']), 1)
            self.assertEqual(path.read_bytes(), pinned)
            admission(root, identity)
            self.assertEqual(check_reports(root, repair=True)['repaired'], [])

    def test_windows_checkout_of_lf_pinned_evidence_can_be_restored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, pinned, _ = self.fixture(root, pinned_lf=True)
            check_reports(root, repair=True)
            self.assertEqual(path.read_bytes(), pinned)

    def test_changed_report_content_is_rejected_and_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _, _ = self.fixture(root, stored_lf=True)
            altered = path.read_bytes().replace(b'"niche_qualified": true', b'"niche_qualified": false')
            path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, 'refusing repair'):
                check_reports(root, repair=True)
            self.assertEqual(path.read_bytes(), altered)

    def test_conflicting_fingerprints_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _, _ = self.fixture(root, stored_lf=True)
            original = path.read_bytes()
            current = root / 'training/opponents/current.json'
            value = read(current); value['qualification_sha256'] = '0' * 64
            current.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                check_reports(root, repair=True)
            self.assertEqual(path.read_bytes(), original)

    def test_real_pool_reports_and_git_filters_preserve_pinned_bytes(self):
        self.assertTrue(check_reports(ROOT)['passed'])
        pool = read(ROOT / 'training/opponents/pool.json')
        for name in {r['qualification'] for purpose in ('training', 'evaluation') for r in pool[purpose].values()}:
            data = (ROOT / name).read_bytes()
            expected = subprocess.run(['git', 'hash-object', '--stdin'], input=data, cwd=ROOT,
                                      capture_output=True, check=True).stdout
            for autocrlf in ('true', 'false', 'input'):
                actual = subprocess.run(['git', '-c', 'core.autocrlf=' + autocrlf,
                                         'hash-object', '--path=' + name, '--stdin'], input=data,
                                        cwd=ROOT, capture_output=True, check=True).stdout
                self.assertEqual(actual, expected, autocrlf)

    def test_newline_repair_does_not_strip_bom_or_other_whitespace(self):
        original = b'{"result": 1}\r\n'
        expected = hashlib.sha256(original).hexdigest()
        for changed in (b'\xef\xbb\xbf' + original, b'{"result":  1}\n'):
            with self.assertRaisesRegex(ValueError, 'refusing repair'):
                matching_bytes(changed, expected)


if __name__ == '__main__':
    unittest.main()
