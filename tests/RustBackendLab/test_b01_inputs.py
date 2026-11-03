"""Exercise the actual nested B01 archive, not the surrounding development tree."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.rust_backend.lab_contract import SOURCE, extract_new, native_fingerprint

spec = importlib.util.spec_from_file_location('b01_inputs_qualifier', SOURCE_ROOT/'scripts/qualify-rust-lab.py')
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


class NativeInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        extract_new(SOURCE_ROOT/'distribution/rust-lab-b01/b01-search-v1-inputs.zip', self.folder/'inputs')
        extract_new(self.folder/'inputs/source.zip', self.folder/'checkout')
        self.native = self.folder/'checkout/native'

    def test_original_archive_reproduces_missing_external_input(self):
        self.assertEqual(native_fingerprint(self.native), SOURCE)
        with self.assertRaisesRegex(ValueError, 'Missing Rust compile input: tests/GumbelISMCTS/fixtures/'):
            qualifier.check_compile_inputs(self.native)

    def test_staged_archive_closes_all_includes_and_keeps_core_identical(self):
        receipt = qualifier.prepare_test_inputs(self.native)
        self.assertEqual(native_fingerprint(self.native), SOURCE)
        self.assertEqual(receipt['fixtures'], qualifier.TEST_FIXTURES)
        external = {r['input'] for r in receipt['includes'] if not r['input'].startswith('native/')}
        self.assertEqual(external, set(qualifier.TEST_FIXTURES))
        self.assertEqual(qualifier.prepare_test_inputs(self.native), receipt)
        for row in receipt['includes']:
            self.assertTrue((self.native.parent/row['input']).is_file())

    def fixture_root(self, mutate=None):
        root = self.folder/'source'
        for name in qualifier.TEST_FIXTURES:
            dest = root/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = (SOURCE_ROOT/name).read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            dest.write_bytes(data if mutate is None or name != mutate else b'{}\n')
        return root

    def test_windows_line_endings_are_canonicalized(self):
        receipt = qualifier.prepare_test_inputs(self.native, self.fixture_root())
        for name, expected in receipt['fixtures'].items():
            self.assertEqual(qualifier.sha(self.native.parent/name), expected)

    def test_changed_missing_or_conflicting_fixture_fails_before_build(self):
        name = next(iter(qualifier.TEST_FIXTURES))
        source = self.fixture_root(mutate=name)
        with self.assertRaisesRegex(ValueError, 'Changed B01 test fixture'):
            qualifier.prepare_test_inputs(self.native, source)
        self.assertFalse((self.native.parent/'tests').exists())
        (source/name).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing B01 test fixture'):
            qualifier.prepare_test_inputs(self.native, source)
        target = self.native.parent/name
        target.parent.mkdir(parents=True)
        target.write_bytes(b'conflicting data')
        with self.assertRaisesRegex(ValueError, 'Conflicting detached test fixture'):
            qualifier.prepare_test_inputs(self.native)
        self.assertEqual(target.read_bytes(), b'conflicting data')

    def test_changed_native_cannot_be_blessed_by_fixture_preparation(self):
        (self.native/'brass-ai-core/tests/sessions.rs').write_text('// changed\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'sealed B01 native core'):
            qualifier.prepare_test_inputs(self.native)


if __name__ == '__main__':
    unittest.main()
