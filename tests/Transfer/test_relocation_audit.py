"""Independent regressions from the cross-directory transfer audit.

Only disposable projects are mutated. Archive edits represent malformed inputs,
not a way to repair signed training evidence.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import zipfile

import test_step5_transfer as fixtures
from brass_ai.learning.paths import RELOCATION_FILE, make_relocation, resolve_project_path

transfer = fixtures.transfer
ROOT = fixtures.ROOT


class RelocationAuditTests(unittest.TestCase):
    setUp = fixtures.TransferTests.setUp
    put = fixtures.TransferTests.put
    entry = fixtures.TransferTests.entry
    export = fixtures.TransferTests.export
    archive_bytes = fixtures.TransferTests.archive_bytes

    def malformed(self, change, name='malformed.zip'):
        members = self.archive_bytes()
        manifest = json.loads(members['transfer.json'])
        change(manifest, members)
        manifest['id'] = transfer.digest({k:v for k,v in manifest.items() if k != 'id'})
        members['transfer.json'] = transfer.encode(manifest)
        path = self.area / name
        with zipfile.ZipFile(path, 'x') as archive:
            for key, content in members.items():
                archive.writestr(key, content)
        return path

    def test_archive_cannot_claim_a_run_lease_as_restored_payload(self):
        self.export()
        def inject(manifest, members):
            content = b'not a transferable process lease'
            record = dict(path=manifest['run'] + '/.active-run.json', bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), external=False)
            manifest['files'].append(record)
            members[transfer.payload_name(manifest, record)] = content
        bundle = self.malformed(inject)
        destination = self.area / 'new project with spaces'
        with self.assertRaisesRegex(ValueError, 'lease|control|reserved'):
            transfer.restore(destination, bundle, relocate=True)
        self.assertFalse(destination.exists())

    def test_full_export_omits_nested_run_guard_files(self):
        self.put('artifacts/step5/selfplay/evaluations/.run-locks/guard.lock', b'local OS lock')
        transfer.export(self.root, self.run, self.bundle, mode='full', dependencies='bundled', part_mb=0)
        names = {r['path'] for r in transfer.inspect_bundle(self.bundle)['files']}
        self.assertFalse(any('.run-locks' in name for name in names))

    def test_legacy_full_archive_retains_inert_guard_bytes(self):
        self.export()
        name = 'artifacts/step5/selfplay/evaluations/.run-locks/old-guard.lock'
        content = b'\0'
        def legacy(manifest, members):
            manifest.pop('relocation_version')
            record = dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), external=False)
            manifest['files'].append(record)
            members[transfer.payload_name(manifest, record)] = content
        bundle = self.malformed(legacy)
        destination = self.area / 'legacy full'
        transfer.restore(destination, bundle, relocate=True)
        self.assertEqual((destination / name).read_bytes(), content)
        self.assertTrue(transfer.restore(destination, bundle, verify_only=True)['verified'])

    def test_pending_evaluation_carries_its_separate_absolute_runtime(self):
        directory = 'artifacts/evaluation host with spaces'
        dll = self.put(directory + '/engine.dll', b'evaluation rules dependency')
        self.put(directory + '/runtime.json', dict(files=[dict(name=dll.name, sha256=transfer.sha(dll))]))
        self.put('artifacts/step5/selfplay/evaluations/pending/spec.json',
                 dict(entries=[self.state['history'][-1]], config=dict(host=str(self.root / directory))))
        self.export()
        original = self.root
        backup = self.area / 'unavailable-source'
        original.rename(backup)
        destination = self.area / 'new project with spaces'
        transfer.restore(destination, self.bundle, relocate=True)
        spec = transfer.read(destination / 'artifacts/step5/selfplay/evaluations/pending/spec.json')
        host = resolve_project_path(destination, spec['config']['host'])
        metadata = transfer.read(host / 'runtime.json')
        self.assertEqual(transfer.sha(host / metadata['files'][0]['name']), metadata['files'][0]['sha256'])
        self.assertFalse(original.exists())

    def test_full_archive_keeps_pending_dependencies_outside_run(self):
        name = 'artifacts/separate evaluator/runtime.json'
        host = self.put(name, dict(files=[])).parent
        model = self.put('artifacts/separate evaluator/reference.pt', b'outside-run reference')
        self.put('artifacts/step5/selfplay/evaluations/pending/spec.json',
                 dict(config=dict(host=str(host)), entries=[dict(path=str(model), sha256=transfer.sha(model))]))
        transfer.export(self.root, self.run, self.bundle, mode='full', dependencies='bundled', part_mb=0)
        names = {r['path'] for r in transfer.inspect_bundle(self.bundle)['files']}
        self.assertIn(name, names)
        self.assertIn(model.relative_to(self.root).as_posix(), names)

    def test_previous_relocation_archive_requests_reviewed_code_update(self):
        self.export()
        previous = self.malformed(lambda manifest, _: manifest.pop('relocation_version'))
        destination = self.area / 'previous-release'
        result = transfer.restore(destination, previous, relocate=True)
        self.assertTrue(result['requires_code_update'])
        self.assertTrue(transfer.restore(destination, previous, verify_only=True)['verified'])

    def test_reserved_and_nonportable_archive_paths(self):
        for name in ('.GIT/config', '.Aws/credentials', 'artifacts/NUL.pt', 'artifacts/name?.pt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                transfer.relative(name)

    def test_duplicate_and_truncated_sidecars_never_report_verified(self):
        self.export()
        destination = self.area / 'metadata'
        destination.mkdir()
        record = make_relocation(destination, [str(self.root)], 'a' * 64)
        valid = transfer.encode(record)
        for malformed in (valid[:-5], b'{"format":"duplicate",' + valid[1:], b'[]'):
            (destination / RELOCATION_FILE).write_bytes(malformed)
            with self.subTest(malformed=malformed[:40]), self.assertRaises(ValueError):
                transfer.restore(destination, self.bundle, relocate=True)
            self.assertEqual(list(destination.iterdir()), [destination / RELOCATION_FILE])

    def test_oversized_mapping_is_rejected_before_creating_destination(self):
        self.state['signature'].pop('imitation_source')
        self.put('artifacts/step5/selfplay/state.json', self.state)
        self.export()
        def oversized(manifest, _):
            # Individually valid POSIX root components, within the count limit,
            # but together too large for the runtime's 64 KiB sidecar reader.
            deep = '/'.join(['segment' * 30] * 10)
            manifest['path_roots'] += [f'/historical-{i}/{deep}/project' for i in range(35)]
        bundle = self.malformed(oversized)
        destination = self.area / 'oversized'
        with self.assertRaisesRegex(ValueError, 'versized relocation'):
            transfer.restore(destination, bundle, relocate=True)
        self.assertFalse(destination.exists())

    def test_original_relocation_source_requires_the_reviewed_transition(self):
        from brass_ai.learning.contracts import compatibility
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        prior = 'c9c9bb914477feea94b61b1698db93fd5e767d0f2f88b0c715f0cfea12074140'
        signature = dict(implementation=prior, compatibility=compatibility(), legacy_pool={}, teachers={},
                         config=transfer.read(ROOT / 'configs/step5/lab.json'))
        state = dict(format='step5-recovery-league/2.2.0', iteration=2, signature=signature)
        current = dict(signature, implementation=implementation_id())
        with self.assertRaisesRegex(ValueError, 'resume-reviewed-source'):
            check_signature(state, current)
        self.assertTrue(check_signature(state, current, reviewed=True))
        changed = copy.deepcopy(current)
        changed['config']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'unchanged config'):
            check_signature(state, changed, reviewed=True)
        with self.assertRaisesRegex(ValueError, 'not been reviewed'):
            check_signature(state, dict(current, implementation='0' * 64), reviewed=True)

    def test_missing_evaluation_runtime_fails_before_publishing_an_archive(self):
        self.put('artifacts/step5/selfplay/evaluations/pending/spec.json',
                 dict(config=dict(host='artifacts/missing evaluation host')))
        with self.assertRaisesRegex(ValueError, 'Missing dependency.*runtime.json'):
            self.export()
        self.assertFalse(self.bundle.exists())

    def test_host_source_references_are_dependencies_too(self):
        source = self.put('artifacts/custom host sources/Host.cs', b'// pinned host source\n')
        name = 'artifacts/r1-host/guarded/runtime.json'
        metadata = transfer.read(self.root / name)
        metadata['sources'] = [dict(path=str(source), sha256=transfer.sha(source))]
        self.put(name, metadata)
        # The distribution independently pins its host receipt.
        distribution = transfer.read(self.root / 'training/runtime/distribution.json')
        distribution['private_runtime_files'][name] = transfer.sha(self.root / name)
        self.put('training/runtime/distribution.json', distribution)
        self.export()
        names = {r['path'] for r in transfer.inspect_bundle(self.bundle)['files']}
        self.assertIn(source.relative_to(self.root).as_posix(), names)

    def test_file_directory_collision_fails_before_any_destination_writes(self):
        self.export()
        def collide(manifest, members):
            content = b'blocks a required directory'
            record = dict(path='artifacts/step5', bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), external=False)
            manifest['files'].append(record)
            members[transfer.payload_name(manifest, record)] = content
        bundle = self.malformed(collide)
        destination = self.area / 'conflicting-layout'
        with self.assertRaisesRegex(ValueError, 'ancestor|directory|collid'):
            transfer.restore(destination, bundle, relocate=True)
        self.assertFalse(destination.exists())

    def test_destination_parent_file_fails_before_any_writes(self):
        self.export()
        destination = self.area / 'destination'
        destination.mkdir()
        (destination / 'src').write_bytes(b'user file')
        with self.assertRaisesRegex(ValueError, 'directory|Conflicting destination'):
            transfer.restore(destination, self.bundle, relocate=True)
        self.assertEqual(list(destination.iterdir()), [destination / 'src'])
        self.assertEqual((destination / 'src').read_bytes(), b'user file')

    def junction(self, path, target):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != 'nt':
                raise
            import _winapi
            try:
                _winapi.CreateJunction(str(target), str(path))
            except PermissionError:
                self.skipTest('Windows sandbox denies junction creation; rerun outside sandbox')
        self.addCleanup(lambda: path.rmdir() if path.exists() else None)

    def training_runtime(self):
        base = 'artifacts/r1-host/guarded'
        baseline = self.put(base + '/BrassHost.exe', b'baseline host')
        original = transfer.read(self.root / base / 'runtime.json')
        original['files'].append(dict(name=baseline.name, sha256=transfer.sha(baseline)))
        self.put(base + '/runtime.json', original)
        source = 'training/runtime/throughput/ordinary'
        files = [dict(f) for f in original['files'] if f['name'] != 'BrassHost.exe']
        files.append(dict(name='BrassHost.exe', sha256=transfer.sha(self.root / source / 'BrassHost.exe')))
        self.put(source + '/runtime.json', dict(optimization='training-host/1.0.0', kind='ordinary',
            builder_sha256=transfer.sha(self.root / 'scripts/build-training-host.py'), authored_sources={},
            baseline_host_sha256=transfer.sha(baseline), files=files))

    def test_runtime_install_refuses_linked_output_and_cached_output(self):
        from brass_ai.recovery.host_runtime import training_host, _ready
        self.training_runtime()
        # First prove this is a valid installer fixture, including the cache.
        installed = training_host(self.root)
        self.assertEqual(training_host(self.root), installed)
        self.assertTrue(installed.is_relative_to(self.root))
        outside = self.area / 'outside'
        outside.mkdir()
        shutil.rmtree(installed)
        self.junction(installed, outside)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            training_host(self.root)
        _ready.clear()
        with self.assertRaisesRegex(ValueError, 'Linked'):
            training_host(self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_preflight_search_install_checks_links_before_copying(self):
        from brass_ai.step5.assets import runtime_files
        self.put('training/opponents/current.json', dict(bank='fixture-bank'))
        self.put('artifacts/legacy-ai/bank/fixture-bank/runtime/runtime.json', dict(files=[]))
        executable = self.put('training/runtime/search/BrassHost.exe', b'fixture search host')
        self.put('training/runtime/search/runtime.json', dict(extension='brass-search-chance/1.0.0',
            files=[dict(name=executable.name, sha256=transfer.sha(executable))]))
        outside = self.area / 'outside search output'
        outside.mkdir()
        self.junction(self.root / 'artifacts/r4-host', outside)
        with patch('brass_ai.step5.assets.load_bank', return_value={}), \
             patch('brass_ai.step5.assets.control_banks', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Linked'):
                runtime_files(self.root)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
