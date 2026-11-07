"""Teacher-wire retry retains prior timing honestly and pins repaired runtime."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.rust_backend.lab_contract import pin
from brass_ai.rust_backend.lab_replay_compat import qualified_implementation as implementation_id
import rust_lab_b07_retry as R


class Retry(unittest.TestCase):
    def temporary(self):
        t=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(t.cleanup);return Path(t.name)

    def ordinary(self):
        root=self.temporary();(root/'evidence').mkdir()
        snapshot=dict(generation=1250,checkpoint={'sha256':'checkpoint'},config_sha256='config')
        q=dict(milestone='B07',implementation='original',failed_phase='canary-1',passed=False,
            generation=1250,checkpoint_sha256='checkpoint',config_sha256='config')
        checks=dict(passed=True,matched={'equal':True},performance={'useful_accepted_work_gain':False})
        evidence={}
        for name,value in [('qualification',q),('ordinary',checks)]:
            p=root/'evidence'/(name+'.json');p.write_text(json.dumps(value))
            evidence[name]=dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        reference=dict(ordinary_baseline=dict(evidence=evidence,from_implementation='original',
            to_implementation=implementation_id(),result='exact-lab-export'))
        return root,reference,snapshot

    def test_baseline_preserves_negative_performance_without_qualifying_current_source(self):
        root,ref,snapshot=self.ordinary()
        with patch.object(R,'DIRECTORY',root):result=R.prior_ordinary(ref,snapshot)
        self.assertTrue(result['passed']);self.assertFalse(result['performance']['useful_accepted_work_gain'])
        self.assertEqual(result['historical_baseline']['original_implementation'],'original')
        self.assertFalse(result['historical_baseline']['current_qualification'])

    def test_changed_reused_checkpoint_config_or_source_is_rejected(self):
        root,ref,snapshot=self.ordinary()
        with patch.object(R,'DIRECTORY',root):
            for bad in [dict(snapshot,generation=1251),dict(snapshot,config_sha256='other'),
                        dict(snapshot,checkpoint={'sha256':'other'})]:
                with self.assertRaisesRegex(ValueError,'checkpoint/config'):R.prior_ordinary(ref,bad)
            ref['ordinary_baseline']['to_implementation']='other'
            with self.assertRaisesRegex(ValueError,'source review'):R.prior_ordinary(ref,snapshot)

    def test_changed_or_failed_phase_evidence_is_rejected(self):
        root,ref,snapshot=self.ordinary();p=root/'evidence/ordinary.json'
        checks=json.loads(p.read_text());checks['passed']=False;p.write_text(json.dumps(checks))
        with patch.object(R,'DIRECTORY',root):
            with self.assertRaisesRegex(ValueError,'evidence changed'):R.prior_ordinary(ref,snapshot)
            ref['ordinary_baseline']['evidence']['ordinary']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'successful phase'):R.prior_ordinary(ref,snapshot)

    def teacher(self):
        root=self.temporary();files=[]
        for name in ('binary','tests','build_log','bootstrap','dependency'):
            p=root/name;p.write_text(name);files.append(pin(p))
        original=dict(core_source='unchanged-core',bootstrap={'binary':files[3],'dependencies':[files[4]]})
        value=dict(contract='brass-lab-b07-teacher-build/1.0.0',producer=R.source_fingerprint(),
            implementation=implementation_id(),core_source=original['core_source'],bootstrap=original['bootstrap'],
            wire_regressions_passed=True,binary=files[0],tests=files[1],build_log=files[2])
        return original,value

    def test_ordinary_rebuild_requires_current_source_exact_bootstrap_and_wire(self):
        original,value=self.teacher()
        original.update(b01={'sha256':'original-b01'})
        ordinary=dict(value,contract='brass-lab-b07-ordinary-build/1.0.0',shared_core=R.SOURCE,
            collector=value['binary'],b01=original['b01'],search_enabled=False,search_method='search_v1',
            v205_enabled=False,collection_wire='compact-arrays-forced/1')
        self.assertEqual(R.verify_ordinary(ordinary,original),ordinary)
        for key,bad in [('producer','old'),('implementation','old'),('collection_wire','old'),
                        ('shared_core','changed'),('search_enabled',True),('wire_regressions_passed',False)]:
            with self.assertRaises(ValueError):R.verify_ordinary(dict(ordinary,**{key:bad}),original)
        path=Path(ordinary['collector']['path']);path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'runtime/evidence'):R.verify_ordinary(ordinary,original)

    def test_teacher_requires_new_producer_and_exact_build_test_bootstrap_bytes(self):
        original,value=self.teacher();self.assertEqual(R.verify_teacher(value,original),value)
        for key,changed in [('producer','old'),('core_source','changed'),('implementation','old'),('wire_regressions_passed',False)]:
            with self.subTest(key=key),self.assertRaises(ValueError):R.verify_teacher(dict(value,**{key:changed}),original)
        for item in [value['binary'],value['tests'],value['build_log'],*value['bootstrap']['dependencies']]:
            p=Path(item['path']);before=p.read_bytes();p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'runtime/evidence changed'):R.verify_teacher(value,original)
            p.write_bytes(before)


if __name__=='__main__':unittest.main()
