"""Tiny frozen-runtime export/repair regressions; no games or worker launches."""
import copy
import unittest
from unittest.mock import patch

import test_step5_transfer as fixtures
from step5_transfer_frozen import SUPPLEMENT

transfer=fixtures.transfer


class FrozenRuntimeTransfer(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.TransferTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.root,self.run=self.case.root,self.case.run
        self.parent='bank-fixture/balanced'
        profile=self.case.put('configs/legacy-ai/balanced.json',dict(style='balanced'))
        recipe=dict(contract='shared-heuristic-search/1.0.0',bank='bank-fixture',
            sources={'src/old-policy.cs':'a'*64},profiles={'balanced':transfer.sha(profile)},adapter='b'*64)
        self.ref=dict(id='search-'+transfer.digest(recipe)[:24],recipe=recipe)
        self.prefix='artifacts/legacy-ai/search/'+self.ref['id'];self.base=self.root/self.prefix
        exe=self.case.put(self.prefix+'/runtime/BrassLegacyAI.exe',b'original authored frozen executable')
        dependency=self.root/'artifacts/r1-host/guarded/engine.dll'
        self.case.put(self.prefix+'/runtime/engine.dll',dependency.read_bytes())
        manifest=self.case.put(self.prefix+'/runtime/runtime.json',dict(files=[
            dict(name='BrassLegacyAI.exe',sha256=transfer.sha(exe)),dict(name='engine.dll',sha256=transfer.sha(dependency))],
            sources=[dict(path='src/old-policy.cs',sha256='a'*64)]))
        self.case.put(self.prefix+'/search.json',dict(reference=self.ref,runtime_sha256=transfer.sha(manifest)))
        self.catalog='training/runtime/shared-search/bank-fixture/frozen/'+self.ref['id']
        files={}
        for name,path in [('search.json',self.base/'search.json'),('runtime.json',manifest),('BrassLegacyAI.exe',exe)]:
            p=self.case.put(self.catalog+'/'+name,path.read_bytes());files[name]=dict(sha256=transfer.sha(p),bytes=p.stat().st_size)
        descriptor=dict(contract=SUPPLEMENT,reference=self.ref,files=files);descriptor['id']=transfer.digest(descriptor)
        self.case.put(self.catalog+'/supplement.json',descriptor)
        self.case.state['ratings']=dict(config=dict(enabled=True),agents={'v4':dict(policy=dict(
            parent=self.parent,search_runtime=self.ref,search_free=True))})
        self.case.put('artifacts/step5/selfplay/state.json',self.case.state)
        self.state=(self.run/'state.json').read_bytes()

    def missing(self):
        backup=self.case.area/'retained-original-runtime'
        self.assertTrue(self.base.resolve().is_relative_to(self.case.area));self.base.rename(backup)

    def test_export_retains_typed_reference_receipts_and_all_files_without_old_sources(self):
        names={self.prefix+'/search.json',self.prefix+'/runtime/runtime.json',self.prefix+'/runtime/BrassLegacyAI.exe',
               self.prefix+'/runtime/engine.dll'}
        with patch.object(transfer,'git_baseline',return_value=('fixture-commit',names)):
            transfer.export(self.root,self.run,self.case.bundle,archive_format='zip',part_mb=0)
        records={r['path']:r for r in transfer.inspect_bundle(self.case.bundle)['files']}
        for name in names:self.assertIn(name,records);self.assertFalse(records[name]['external'])
        self.assertNotIn('src/old-policy.cs',records)

    def test_repair_missing_runtime_preserves_state_checkpoint_and_frozen_identity_and_retries(self):
        self.missing();checkpoint=self.run/'generation-0002.pt';before=checkpoint.read_bytes()
        preview=transfer.repair_runtimes(self.root,self.run,dry_run=True)
        self.assertEqual(preview['files_to_install'],4);self.assertFalse(self.base.exists())
        result=transfer.repair_runtimes(self.root,self.run)
        self.assertTrue(result['state_unchanged']);self.assertFalse(result['rebuilt'])
        from brass_ai.recovery.ratings import validate_frozen_reference
        validate_frozen_reference(self.root,self.case.state['ratings'])
        self.assertEqual((self.run/'state.json').read_bytes(),self.state);self.assertEqual(checkpoint.read_bytes(),before)
        self.assertEqual(transfer.read(self.base/'search.json')['reference'],self.ref)
        self.assertEqual(transfer.repair_runtimes(self.root,self.run)['files_to_install'],0)

    def test_partial_publication_recovers_without_replacing_existing_files(self):
        self.missing();publish=transfer.publish
        def stop(temporary,path):
            if path==self.base/'search.json':raise RuntimeError('Authored receipt interruption')
            return publish(temporary,path)
        with patch.object(transfer,'publish',stop),self.assertRaisesRegex(RuntimeError,'Authored receipt interruption'):
            transfer.repair_runtimes(self.root,self.run)
        self.assertFalse((self.base/'search.json').exists());self.assertTrue((self.base/'runtime/runtime.json').is_file())
        result=transfer.repair_runtimes(self.root,self.run)
        self.assertEqual(result['files_to_install'],1);self.assertEqual((self.run/'state.json').read_bytes(),self.state)

    def test_conflicting_existing_executable_blocks_all_writes(self):
        self.missing();bad=self.case.put(self.prefix+'/runtime/BrassLegacyAI.exe',b'other policy')
        with self.assertRaisesRegex(ValueError,'Conflicting frozen runtime file'):
            transfer.repair_runtimes(self.root,self.run)
        self.assertEqual(bad.read_bytes(),b'other policy');self.assertFalse((self.base/'runtime/engine.dll').exists())
        self.assertFalse((self.base/'search.json').exists());self.assertEqual((self.run/'state.json').read_bytes(),self.state)

    def test_corrupt_supplement_or_missing_private_dll_cannot_publish_a_receipt(self):
        self.missing();source=self.root/self.catalog/'BrassLegacyAI.exe';data=source.read_bytes();source.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'supplement file changed'):transfer.repair_runtimes(self.root,self.run)
        self.assertFalse(self.base.exists());source.write_bytes(data)
        (self.root/'artifacts/r1-host/guarded/engine.dll').write_bytes(b'wrong dependency')
        with self.assertRaisesRegex(ValueError,'Missing exact frozen runtime input'):transfer.repair_runtimes(self.root,self.run)
        self.assertFalse(self.base.exists())

    def test_export_refuses_missing_frozen_runtime_instead_of_producing_incomplete_archive(self):
        self.missing()
        with self.assertRaisesRegex(ValueError,'Missing frozen runtime dependency'):
            transfer.export(self.root,self.run,self.case.bundle,archive_format='zip',part_mb=0)
        self.assertFalse(self.case.bundle.exists());self.assertEqual((self.run/'state.json').read_bytes(),self.state)

    def test_profile_or_supplement_identity_changes_are_rejected(self):
        self.missing();profile=self.root/'configs/legacy-ai/balanced.json';data=profile.read_bytes();profile.write_bytes(b'new profile')
        with self.assertRaisesRegex(ValueError,'profile changed'):transfer.repair_runtimes(self.root,self.run)
        profile.write_bytes(data)
        path=self.root/self.catalog/'supplement.json';binding=copy.deepcopy(transfer.read(path));binding['reference']['recipe']['adapter']='c'*64
        binding['id']=transfer.digest({k:v for k,v in binding.items() if k!='id'});path.write_bytes(transfer.encode(binding))
        with self.assertRaisesRegex(ValueError,'supplement binding differs'):transfer.repair_runtimes(self.root,self.run)
        self.assertFalse(self.base.exists())

    def test_pending_evaluation_reference_is_retained_even_when_ratings_are_disabled(self):
        self.case.state['ratings']['config']['enabled']=False
        self.case.put('artifacts/step5/selfplay/state.json',self.case.state)
        self.case.put('artifacts/step5/selfplay/evaluations/pending/spec.json',dict(policy=dict(
            parent=self.parent,search_runtime=self.ref)))
        transfer.export(self.root,self.run,self.case.bundle,archive_format='zip',dependencies='bundled',part_mb=0)
        self.assertIn(self.prefix+'/search.json',{r['path'] for r in transfer.inspect_bundle(self.case.bundle)['files']})


if __name__=='__main__':unittest.main()
