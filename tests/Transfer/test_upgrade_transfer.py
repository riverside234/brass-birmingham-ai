"""Critic-upgraded IL archives: authored bytes/metadata, no games or training."""
import copy
import unittest

from test_native_transfer import NativeTransfer,ROOT,transfer


class UpgradedIL(unittest.TestCase):
    def setUp(self):
        self.case=NativeTransfer();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.root,self.run=self.case.root,self.case.run
        self.folder='artifacts/step5/il-critic-upgrade'
        parent=dict(self.case.origin,model_id='parent-model',bc_signature='parent-signature')
        candidate=self.case.put(self.folder+'/candidate.pt',b'authored critic-upgraded IL weights')
        self.receipt=dict(contract='completed-il-critic-upgrade/1.0.0',parent=parent,
            compatibility=dict(fixture='unchanged learning contract'),
            candidate=dict(path=candidate.relative_to(self.root).as_posix(),sha256=transfer.sha(candidate),model_id='upgraded-model'))
        self.origin=dict(bc_run=self.folder,**self.receipt['candidate'],manifest=parent['manifest'],
            manifest_sha256=parent['manifest_sha256'],bc_signature=transfer.digest(self.receipt))
        self.case.state['signature'].update(imitation_source=self.origin,compatibility=self.receipt['compatibility'])
        self.publish()

    def publish(self):
        self.case.put(self.folder+'/upgrade.json',self.receipt)
        self.case.put('artifacts/step5/selfplay/state.json',self.case.state)

    def repin(self):
        self.origin['bc_signature']=transfer.digest(self.receipt);self.publish()

    def test_export_and_relocated_restore_include_both_il_formats_without_replay(self):
        original=(self.run/'state.json').read_bytes()
        self.assertFalse((self.root/self.folder/'state.json').exists())
        self.assertFalse((self.root/self.folder/'report.json').exists())
        preview=transfer.export(self.root,self.run,self.case.bundle,dry_run=True,archive_format='zip')
        self.assertTrue(preview['dry_run']);self.assertFalse(self.case.bundle.exists())
        self.case.export();manifest=transfer.inspect_bundle(self.case.bundle)
        names={r['path'] for r in manifest['files']}
        needed=[self.folder+'/upgrade.json',self.folder+'/candidate.pt',self.case.origin['manifest'],
            'artifacts/step5/il/state.json','artifacts/step5/il/report.json',self.case.origin['path'],
            'artifacts/step5/selfplay/il-kl/probe.npz']
        self.assertTrue(set(needed)<=names)
        self.assertNotIn(self.folder+'/state.json',names)
        self.assertNotIn('artifacts/demos/train.jsonl.gz',names)
        self.assertFalse(any('.prepared' in name or '/teacher-labels/' in name or '/search-values/' in name for name in names))
        destination=self.case.area/'new-machine'
        restored=transfer.restore(destination,self.case.bundle,relocate=True)
        self.assertTrue(restored['verified']);self.assertIn(str(destination/self.folder),restored['resume'])
        self.assertEqual((destination/'artifacts/step5/selfplay/state.json').read_bytes(),original)
        self.assertEqual(transfer.read(destination/self.folder/'upgrade.json'),self.receipt)
        for name in needed:self.assertEqual((destination/name).read_bytes(),(self.root/name).read_bytes())
        self.assertTrue(transfer.restore(destination,self.case.bundle,verify_only=True)['verified'])

    def test_receipt_candidate_dataset_and_compatibility_changes_fail_before_archive(self):
        mutations=[('receipt',lambda:self.receipt.update(implementation='unbound edit'),'receipt/source'),
            ('candidate',lambda:self.origin.update(model_id='foreign'),'candidate identity'),
            ('dataset',lambda:self.origin.update(manifest='training/datasets/foreign.json'),'dataset lineage'),
            ('compatibility',lambda:self.case.state['signature'].update(compatibility={}),'receipt/source')]
        original_receipt=copy.deepcopy(self.receipt);original_origin=copy.deepcopy(self.origin)
        for label,mutate,message in mutations:
            with self.subTest(label=label):
                self.receipt=copy.deepcopy(original_receipt);self.origin.clear();self.origin.update(original_origin)
                self.case.state['signature']['compatibility']=self.receipt['compatibility'];mutate()
                if label not in ('receipt','compatibility'):self.origin['bc_signature']=transfer.digest(self.receipt)
                self.publish()
                with self.assertRaisesRegex(ValueError,message):self.case.export()
                self.assertFalse(self.case.bundle.exists())

    def test_candidate_bytes_and_owned_path_are_still_required(self):
        candidate=self.root/self.receipt['candidate']['path'];candidate.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'Dependency hash differs'):self.case.export()
        other=self.case.put('artifacts/step5/il/foreign.pt',b'outside upgrade run')
        self.receipt['candidate'].update(path=other.relative_to(self.root).as_posix(),sha256=transfer.sha(other))
        self.origin.update(self.receipt['candidate']);self.repin()
        with self.assertRaisesRegex(ValueError,'candidate identity/ownership'):self.case.export()

    def test_missing_parent_checkpoint_and_regular_bc_completion_metadata_fail_closed(self):
        parent=self.root/self.case.origin['path'];data=parent.read_bytes();parent.unlink()
        with self.assertRaisesRegex(ValueError,'Missing dependency'):self.case.export()
        parent.write_bytes(data)
        (self.root/'artifacts/step5/il/state.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing dependency.*state.json'):self.case.export()

    def test_upgrade_cycles_and_nested_upgrades_are_rejected(self):
        self.receipt['parent']['bc_run']=self.folder;self.repin()
        with self.assertRaisesRegex(ValueError,'Cyclic/invalid IL lineage'):self.case.export()
        self.receipt['parent']['bc_run']=self.case.origin['bc_run'];self.repin()
        self.case.put('artifacts/step5/il/upgrade.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'cannot be upgraded again'):self.case.export()

    def test_real_g1284_upgrade_receipt_matches_its_pinned_training_origin(self):
        from brass_ai.learning.source_identity import implementation_id
        state=transfer.read(ROOT/'results/joint7-v11-2-g1290-trained-restart-g1284-20261108T193427364056Z/state.json')
        receipt=transfer.read(ROOT/'results/joint7-v11-2-grouped-lr-critic-g0185-20261020T090519451283Z/lineage/artifacts/step5/features4-v11-2-batch7-il/upgrade.json')
        origin=state['signature']['imitation_source']
        self.assertEqual(origin['bc_signature'],transfer.digest(receipt))
        self.assertEqual(receipt['compatibility'],state['signature']['compatibility'])
        self.assertEqual({k:origin[k] for k in ('path','sha256','model_id')},receipt['candidate'])
        review=transfer.read(ROOT/'configs/learning/b07-replay-location-review.json')
        self.assertEqual(implementation_id(ROOT),review['execution_implementation'])


if __name__=='__main__':unittest.main()
