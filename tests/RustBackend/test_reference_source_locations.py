"""Relocated exporter provenance retains exact frozen-source verification."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from brass_ai.rust_backend.baseline import verify_reference

OLD = 'scripts/export-test-nn.py'
NEW = 'scripts/training/checkpoints/export-test-nn.py'


class ReferenceSourceLocations(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.frozen = b'# exact freeze-time exporter\n'

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def pin(self, name):
        return dict(path=name, sha256=hashlib.sha256((self.root/name).read_bytes()).hexdigest())

    def fixture(self, exporter=OLD, archived=None):
        self.put(exporter, self.frozen)
        self.put('src/oracle.py', b'# executed oracle remains pinned\n')
        self.put('model.pt', b'immutable checkpoint')
        self.put('bundle.json', b'{}\n')
        with zipfile.ZipFile(self.root/'reference-source.zip', 'w') as archive:
            archive.writestr(exporter, self.frozen if archived is None else archived)
        return dict(reference_source=dict(files=[self.pin(exporter), self.pin('src/oracle.py')],
                    archive=self.pin('reference-source.zip')),
                    model=dict(**self.pin('model.pt'), bundle_files=[self.pin('bundle.json')]))

    def relocate(self):
        (self.root/OLD).unlink()
        self.put(NEW, b'# current relocated exporter\n')

    def test_original_matching_exporter_still_needs_no_drift_receipt(self):
        self.assertEqual(verify_reference(self.fixture(), self.root), [])

    def test_old_inventory_uses_archived_bytes_after_exporter_moves(self):
        baseline = self.fixture()
        self.relocate()
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        drift = verify_reference(baseline, self.root)
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]['path'], OLD)
        self.assertEqual(drift[0]['frozen_sha256'], hashlib.sha256(self.frozen).hexdigest())
        self.assertEqual(drift[0]['current_sha256'], self.pin(NEW)['sha256'])
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_new_inventory_keeps_the_same_archive_verification(self):
        baseline = self.fixture(NEW)
        self.put(NEW, b'# later exporter revision\n')
        self.assertEqual(verify_reference(baseline, self.root)[0]['path'], NEW)

    def test_modified_archive_is_rejected(self):
        baseline = self.fixture()
        self.relocate()
        with (self.root/'reference-source.zip').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'Frozen source archive changed'):
            verify_reference(baseline, self.root)

    def test_matching_archive_digest_does_not_hide_wrong_member_bytes(self):
        baseline = self.fixture(archived=b'# wrong frozen exporter\n')
        self.relocate()
        with self.assertRaisesRegex(ValueError, 'Frozen bundle-verifier bytes changed'):
            verify_reference(baseline, self.root)

    def test_oracle_source_drift_is_still_rejected(self):
        baseline = self.fixture()
        self.put('src/oracle.py', b'# changed oracle\n')
        with self.assertRaisesRegex(ValueError, 'Frozen reference source changed'):
            verify_reference(baseline, self.root)

    def test_missing_oracle_has_no_exporter_fallback(self):
        baseline = self.fixture()
        (self.root/'src/oracle.py').unlink()
        with self.assertRaises(FileNotFoundError):
            verify_reference(baseline, self.root)

    def test_missing_current_exporter_is_still_rejected(self):
        baseline = self.fixture()
        (self.root/OLD).unlink()
        with self.assertRaises(FileNotFoundError):
            verify_reference(baseline, self.root)


if __name__ == '__main__':
    unittest.main()
