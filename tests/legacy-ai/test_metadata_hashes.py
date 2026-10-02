"""LF/CRLF admission regressions using the selected v4's frozen evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from brass_ai.bank import admission, json_sha_matches, load_bank, read, sha


ROOT = Path(__file__).resolve().parents[2]
V4 = 'bank-3dc65253a610191d66a5'
RECEIPT = 'training/opponents/evaluations/user-directed-v4-1.0.0/selection.json'


def form(data, crlf):
    lf = data.replace(b'\r\n', b'\n')
    return lf.replace(b'\n', b'\r\n') if crlf else lf


class MetadataHashTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkout(self, manifest_crlf=False, receipt_crlf=False):
        bank = Path('training/opponents/bank') / V4
        shutil.copytree(ROOT / bank, self.root / bank)
        self.manifest = self.root / bank / 'manifest.json'
        payload = read(self.manifest)['payload']
        # Contract and frozen source bytes remain exact, as required by Git attrs.
        for relative in payload['source_files']:
            if relative.startswith(('configs/contracts/', 'configs/rules/')):
                destination = self.root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
        self.manifest.write_bytes(form(self.manifest.read_bytes(), manifest_crlf))
        self.receipt = self.root / RECEIPT
        self.receipt.parent.mkdir(parents=True)
        original = (ROOT / RECEIPT).read_bytes()
        self.receipt.write_bytes(form(original, receipt_crlf))
        report = read(self.receipt)
        self.pool = dict(training={}, evaluation={})
        for name in payload['profiles']:
            purpose = 'evaluation' if name.startswith('pottery-flex') else 'training'
            self.pool[purpose][V4 + '/' + name] = dict(
                admission='user-directed-masked-v4/1.0.0',
                manifest_sha256=report['manifest_sha256'], qualification=RECEIPT,
                sha256=hashlib.sha256(form(original, True)).hexdigest(), profile=name)
        (self.root / 'training/opponents/pool.json').write_bytes(
            (json.dumps(self.pool, indent=2) + '\n').encode())
        self.teacher = next(iter(self.pool['training']))

    def test_all_lf_crlf_manifest_receipt_combinations_admit_six_styles(self):
        for manifest_crlf, receipt_crlf in ((False, False), (False, True),
                                             (True, False), (True, True)):
            with self.subTest(manifest_crlf=manifest_crlf, receipt_crlf=receipt_crlf):
                # Each subcase uses an independent checkout, without editing pins.
                with tempfile.TemporaryDirectory() as temp:
                    self.root = Path(temp)
                    self.checkout(manifest_crlf, receipt_crlf)
                    for purpose, profiles in self.pool.items():
                        for policy in profiles:
                            admission(self.root, policy, purpose)
                    self.assertEqual(5, len(self.pool['training']))
                    self.assertEqual(1, len(self.pool['evaluation']))

    def test_exact_reported_linux_failure_is_reproduced_by_old_check(self):
        self.checkout()
        self.assertEqual('681b6d45', sha(self.manifest)[:8])
        self.assertEqual('20b75311', self.pool['training'][self.teacher]['manifest_sha256'][:8])
        with mock.patch('brass_ai.bank.json_sha_matches', side_effect=lambda p, h: sha(p) == h):
            with self.assertRaisesRegex(ValueError, 'Selected v4 bank differs'):
                admission(self.root, self.teacher)
        admission(self.root, self.teacher)

    def test_receipt_edits_are_rejected_even_with_lf_endings(self):
        self.checkout()
        original = self.receipt.read_bytes()
        self.receipt.write_bytes(original.replace(b'"productive_progress": true',
                                                 b'"productive_progress": false'))
        with self.assertRaisesRegex(ValueError, 'V4 selection receipt changed'):
            admission(self.root, self.teacher)

    def test_manifest_content_edits_still_fail_bank_identity(self):
        self.checkout()
        content = read(self.manifest)
        content['payload']['sampling_seed'] += 1
        self.manifest.write_bytes((json.dumps(content, indent=2) + '\n').encode())
        with self.assertRaisesRegex(ValueError, 'Frozen bank manifest changed'):
            load_bank(self.root, V4)

    def test_equivalent_json_reformatting_is_not_a_newline_conversion(self):
        self.checkout()
        self.receipt.write_bytes((json.dumps(read(self.receipt), sort_keys=True) + '\n').encode())
        with self.assertRaisesRegex(ValueError, 'V4 selection receipt changed'):
            admission(self.root, self.teacher)

    def test_mixed_newline_pins_and_embedded_manifest_pins(self):
        self.checkout()
        # A pin made on Linux can coexist with the receipt's historical CRLF pin.
        entry = self.pool['training'][self.teacher]
        entry['manifest_sha256'] = sha(self.manifest)
        (self.root / 'training/opponents/pool.json').write_bytes(json.dumps(self.pool).encode())
        admission(self.root, self.teacher)

    def test_binary_and_string_content_is_not_normalized(self):
        binary = self.root / 'worker.exe'
        binary.write_bytes(b'MZ\r\n')
        expected = sha(binary)
        self.assertFalse(json_sha_matches(binary, expected))
        binary.write_bytes(b'MZ\n')
        self.assertNotEqual(expected, sha(binary))
        path = self.root / 'string.json'
        original = b'{\r\n  "text": "keep\\r\\ninside"\r\n}\r\n'
        expected = hashlib.sha256(original).hexdigest()
        path.write_bytes(form(original, False))
        self.assertTrue(json_sha_matches(path, expected))
        path.write_bytes(form(original, False).replace(b'\\r\\n', b'\\n'))
        self.assertFalse(json_sha_matches(path, expected))

    def test_demo_generation_lf_checkout_reaches_collection(self):
        self.checkout()
        from brass_ai.recovery.demonstrations import generate
        config = read(ROOT / 'configs/step5/lab-24x4.json')
        with mock.patch('brass_ai.recovery.demonstrations.InferenceService'), \
             mock.patch('brass_ai.recovery.demonstrations.ActorPool') as actors:
            actors.return_value.__enter__.return_value.collect.return_value = []
            result = generate(self.root, 'artifacts/test-demonstrations', 1,
                              config['execution'], budget=config['legacy_budget'])
            jobs = actors.return_value.__enter__.return_value.collect.call_args.args[0]
            self.assertEqual(1, len(jobs))
            self.assertEqual(4, len(jobs[0]['policies']))
            for policy in jobs[0]['policies']:
                self.assertIn(policy['parent'], self.pool['training'])
                self.assertEqual(1., policy['budget']['seconds'])
                self.assertEqual(4, policy['budget']['search_workers'])
            self.assertEqual(1, result['games'])


if __name__ == '__main__':
    unittest.main()
