"""Compiler-free lab installation keeps source, binary and private DLL checks."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brass_ai.search import legacy_runtime as shared

ROOT=Path(__file__).resolve().parents[2]
BANK='bank-build-fixture'


class SharedPrebuiltTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts',prefix='shared-prebuilt-test-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.source=self.root/'src/BrassMaskedSearch/Test.cs'
        self.source.parent.mkdir(parents=True);self.source.write_bytes(b'// fixed strategy\n')
        for name in shared.GAME_POLICY:
            p=self.root/'src/BrassLocalAI/Policy'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'// shared\n')
        profiles=self.root/'configs/legacy-ai';profiles.mkdir(parents=True)
        shared.write_json(profiles/'balanced.json',dict(mode='masked_ismcts',player_count=4))
        self.deps=self.root/'artifacts/legacy-ai/bank'/BANK/'runtime';self.deps.mkdir(parents=True)
        (self.deps/'engine.dll').write_bytes(b'private pinned runtime')
        self.dependencies=[dict(name='engine.dll',sha256=shared.sha(self.deps/'engine.dll'))]
        self.compiler_bytes=b'pinned compiler fixture'
        self.payload=dict(profiles={'balanced':{}},compiler_sha256=hashlib.sha256(self.compiler_bytes).hexdigest(),
            runtime_files=self.dependencies+[dict(name='BrassLegacyAI.exe',sha256='0'*64)],rules='rules',wire='wire')
        mock=patch.object(shared,'load_bank',return_value=dict(payload=self.payload));mock.start();self.addCleanup(mock.stop)
        self.ref=shared.reference(self.root,BANK)
        self.bundle=shared.bundle_directory(self.root,BANK);self.bundle.mkdir(parents=True)
        self.binary=b'MZ portable managed executable fixture'
        (self.bundle/'BrassLegacyAI.exe').write_bytes(self.binary)
        self.metadata=dict(format=shared.PREBUILT,reference=self.ref,compiler_sha256=self.payload['compiler_sha256'],
            dependencies=self.dependencies,executable=dict(name='BrassLegacyAI.exe',sha256=shared.sha(self.bundle/'BrassLegacyAI.exe')))
        shared.write_json(self.bundle/'prebuilt.json',self.metadata)

    def install(self):
        with patch.object(shared.subprocess,'run',side_effect=AssertionError('compiler must not run')):
            return shared.build(self.root,BANK)

    def test_install_without_compiler_and_reuse_without_compiler(self):
        before=shared.sha(self.deps/'engine.dll');self.assertEqual(self.install(),self.ref)
        runtime=shared.runtime_directory(self.root,self.ref)
        self.assertEqual((runtime/'BrassLegacyAI.exe').read_bytes(),self.binary)
        self.assertEqual(shared.sha(self.deps/'engine.dll'),before)
        self.assertEqual(shared.read(runtime.parent/'search.json')['origin'],'prebuilt')
        self.assertEqual(self.install(),self.ref)

    def test_git_newlines_rebind_only_validated_local_source_bytes(self):
        self.source.write_bytes(b'// fixed strategy\r\n')
        self.assertEqual(shared.reference(self.root,BANK),self.ref)
        self.install();runtime=shared.runtime_directory(self.root,self.ref)
        pins=shared.read(runtime/'runtime.json')['sources']
        pin=next(p for p in pins if p['path']=='src/BrassMaskedSearch/Test.cs')
        self.assertEqual(pin['sha256'],shared.sha(self.source))
        self.assertEqual(shared.sha(runtime/'BrassLegacyAI.exe'),self.metadata['executable']['sha256'])

    def test_changed_source_rejects_prebuilt_before_writing_runtime(self):
        self.source.write_bytes(b'// different strategy\n')
        with self.assertRaisesRegex(ValueError,'stale'):self.install()
        ref=shared.reference(self.root,BANK)
        self.assertFalse(shared.directory(self.root,ref['id']).exists())

    def test_corrupt_prebuilt_cannot_publish_a_runtime(self):
        (self.bundle/'BrassLegacyAI.exe').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):self.install()
        self.assertFalse(shared.directory(self.root,self.ref['id']).exists())

    def test_missing_private_dll_has_actionable_error(self):
        (self.deps/'engine.dll').unlink()
        with self.assertRaisesRegex(ValueError,'use --managed.*missing'):self.install()
        self.assertFalse(shared.directory(self.root,self.ref['id']).exists())

    def test_wrong_private_dll_is_not_substituted(self):
        (self.deps/'engine.dll').write_bytes(b'other game version')
        with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):self.install()

    def test_bad_prebuilt_dependency_or_compiler_receipt_is_rejected(self):
        for field,value in [('dependencies',[]),('compiler_sha256','f'*64)]:
            with self.subTest(field=field):
                metadata=copy.deepcopy(self.metadata);metadata[field]=value
                shared.write_json(self.bundle/'prebuilt.json',metadata)
                with self.assertRaisesRegex(ValueError,'contract changed'):self.install()

    def test_absent_prebuilt_and_compiler_gives_setup_instructions(self):
        (self.bundle/'prebuilt.json').unlink()
        with self.assertRaisesRegex(ValueError,'Pull training/runtime/shared-search'):self.install()

    def test_explicit_compile_does_not_silently_take_prebuilt(self):
        with self.assertRaisesRegex(ValueError,'Pinned C# compiler missing'):
            shared.build(self.root,BANK,use_prebuilt=False)
        wrong=self.root/'wrong-compiler.exe';wrong.write_bytes(b'unpinned')
        with self.assertRaisesRegex(ValueError,'Use the pinned compiler'):
            shared.build(self.root,BANK,compiler=wrong)

    def test_publish_contains_only_authored_binary_and_receipt(self):
        self.install();self.assertEqual(shared.publish(self.root,self.ref),self.bundle)
        self.assertEqual({p.name for p in self.bundle.iterdir()},{'prebuilt.json','BrassLegacyAI.exe'})
        self.assertEqual(shared.read(self.bundle/'prebuilt.json'),self.metadata)

    def test_reuse_detects_corrupt_installed_binary(self):
        self.install();runtime=shared.runtime_directory(self.root,self.ref)
        (runtime/'BrassLegacyAI.exe').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'Existing shared runtime changed'):self.install()


if __name__=='__main__':unittest.main()
