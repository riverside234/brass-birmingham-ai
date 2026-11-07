"""Authored replay/cache/retry regressions; no games, benchmark or training run."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.recovery import value_buffer as V,ppg_replay
from brass_ai.rust_backend import lab_admission as A,lab_backend as B,lab_replay_compat as R
from brass_ai.rust_backend.lab_pool import Publisher
from brass_ai.step5.league import implementation_id
from test_b03 import family_fixture


class Paths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name);self.generation=1250
        f,a,self.config=family_fixture(generation=self.generation)
        self.path=Publisher(self.run/'iteration-1250/native-ordinary',f['producer'],self.config,'ordinary').path(a['job'])
        self.record=A.publish(self.path,f,a,validation_receipt=True)
        self.policy=a['job']['provenance']['learner_behavior']['policy_id']
        self.kinds={f['parent']['game']['setup_group']:'selfplay'}
        self.buf=SimpleNamespace(root=ROOT,run=self.run,generation=self.generation,reward_id=a['reward_id'])

    def source(self,record=None):
        return V.DiskBuffer.source(self.buf,[self.record if record is None else record],self.policy,self.kinds,'a'*64)

    def test_real_native_layout_retains_receipts_and_features_without_rereading(self):
        before=copy.deepcopy(self.record);packed=self.path.read_bytes()
        with patch.object(A,'read',side_effect=AssertionError('No repeated full admission')):
            source=self.source()
            self.assertEqual(ppg_replay.Buffer.source(self.buf,[self.record],self.policy,self.kinds,'a'*64),source)
        saved=source['records'][0]
        self.assertEqual(saved['path'],self.path.relative_to(self.run).as_posix())
        self.assertEqual(saved['native_admission'],before['native_admission'])
        self.assertEqual(saved['features'],before['features'])
        self.assertEqual(self.record,before);self.assertEqual(self.path.read_bytes(),packed)
        saved['native_admission']['admission']['generation']=-1
        self.assertEqual(self.record,before)

    def test_reference_root_layout_remains_valid_and_other_nested_layouts_fail(self):
        reference={k:v for k,v in self.record.items() if k!='native_admission'}
        reference['path']=str(self.run/'iteration-1250/reference.jsonl.gz')
        self.assertNotIn('native_admission',self.source(reference)['records'][0])
        for path in (self.run/'iteration-1249/native-ordinary/replay'/self.path.name,
                     self.run/'iteration-1251/native-ordinary/replay'/self.path.name,
                     self.run/'iteration-1250/arbitrary/replay'/self.path.name,
                     self.run/'iteration-1250/native-ordinary/parts'/self.path.name):
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'current ordinary'):
                self.source(dict(self.record,path=str(path)))
        with self.assertRaisesRegex(ValueError,'native admission'):
            self.source({k:v for k,v in self.record.items() if k!='native_admission'})

    def test_stale_diagnostic_or_wrong_model_reward_job_and_cache_binding_fail(self):
        for key,value in [('generation',1249),('scope','diagnostic'),('model_id','b'*64),('reward_id','b'*64)]:
            r=copy.deepcopy(self.record);binding=r['native_admission']['admission'];binding[key]=value
            binding['sha256']=digest({k:v for k,v in binding.items() if k!='sha256'})
            with self.subTest(key=key),self.assertRaises(ValueError):self.source(r)
        for key,value in [('job',-1),('sha256','b'*64),('features',None),
                          ('path',str(self.path.with_name('other.jsonl.gz')))]:
            with self.subTest(key=key),self.assertRaises(ValueError):self.source(dict(self.record,**{key:value}))

    def test_linked_parent_cannot_expand_the_accepted_layout(self):
        from brass_ai.recovery import retention
        original=retention.link
        with patch.object(retention,'link',side_effect=lambda p:Path(p)==self.path.parent or original(p)):
            with self.assertRaisesRegex(ValueError,'Linked'):self.source()

    def test_cold_rebuild_reads_nested_family_with_original_features_and_detects_tampering(self):
        from brass_ai.recovery.ingress import load_collection
        from brass_ai.recovery.utility import reward_spec
        source=self.source()
        _,expected,_=load_collection([self.record],self.policy,self.kinds,
            insolvency=self.config.get('insolvency'),reward=reward_spec(self.config))
        self.assertTrue(expected)
        buf=object.__new__(V.DiskBuffer)
        buf.__dict__.update(vars(self.buf),config=self.config,stratified=False,
            receipt={'source':source,'holdout_fraction':self.config['value_phase']['holdout_fraction']})
        buf.generation+=1
        # Holdout selection is tested independently; this checks all admitted
        # admitted rows through the real cold reader and feature cache.
        with patch('brass_ai.recovery.value_phase.heldout',return_value=False):
            rebuilt=list(buf._rebuild_rows())
        expected=[d for d in expected if d[0].get('_critic_role','learner')=='learner']
        self.assertEqual(len(rebuilt),len(expected))
        for actual,want in zip(rebuilt,expected):
            self.assertEqual(actual[2],want[2])
            self.assertTrue((actual[3].state==want[3].state).all())
            self.assertTrue((actual[3].actions==want[3].actions).all())
            self.assertEqual(actual[0]['_reward_return'],want[0]['_reward_return'])
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'replay changed'):list(buf._rebuild_rows())


class ReviewedRetry(unittest.TestCase):
    def state(self):
        execution=R.execution();qualified=execution['qualified_implementation']
        config={'teacher_phase':{'mode':'off'},'saved':'recipe'}
        signature={'config':config,'implementation':qualified,'bindings':'untouched'}
        return {'signature':signature,'native_backend':{'mode':'rust','implementation':qualified}},execution

    def test_current_code_is_explicitly_reviewed_and_retry_keeps_frozen_signature(self):
        state,execution=self.state();before=copy.deepcopy(state)
        self.assertIsNotNone(execution['replay_location_review'])
        self.assertEqual(execution['implementation'],implementation_id())
        self.assertNotEqual(execution['implementation'],execution['qualified_implementation'])
        incoming=dict(state['signature'],implementation=implementation_id())
        result=R.training_signature(state,incoming)
        self.assertEqual(result,state['signature']);self.assertEqual(state,before)
        self.assertEqual(incoming['implementation'],implementation_id())
        self.assertEqual(digest(result),digest(before['signature']))  # same frozen job IDs/cache scope

    def test_unreviewed_code_recipe_teacher_and_backend_cannot_reuse_old_signature(self):
        state,execution=self.state();incoming=dict(state['signature'],implementation=implementation_id())
        with patch.object(R,'implementation_id',return_value='f'*64):
            self.assertIsNone(R.execution()['replay_location_review'])
            with self.assertRaisesRegex(ValueError,'exact replay-location'):R.training_signature(state,incoming)
        variants=[dict(incoming,bindings='changed'),dict(incoming,config={'teacher_phase':{'mode':'train'}})]
        for signature in variants:
            with self.assertRaises(ValueError):R.training_signature(state,signature)
        bad=copy.deepcopy(state);bad['native_backend']['implementation']='f'*64
        with self.assertRaises(ValueError):R.training_signature(bad,incoming)
        plain={'signature':state['signature']}
        self.assertIs(R.training_signature(plain,incoming),incoming)

    def test_new_execution_signature_is_not_rewound_and_old_rollback_target_stays_valid(self):
        from brass_ai.rust_backend import lab_continuation as C
        state,execution=self.state()
        with patch('brass_ai.recovery.config.validate_config'):
            self.assertEqual(C.target_signature(state)['implementation'],execution['qualified_implementation'])
            state['signature']['implementation']=execution['implementation']
            state['native_backend']['implementation']=execution['implementation']
            self.assertEqual(C.target_signature(state)['implementation'],execution['implementation'])
            self.assertIs(R.training_signature(state,state['signature']),state['signature'])

    def test_backend_keeps_old_receipt_and_checks_new_checkpoint_execution_provenance(self):
        state,execution=self.state();value=state['native_backend']
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            rollback=Path(tmp)/'rollback.json';rollback.write_text('{}')
            value.update(contract=B.CONTRACT,config_sha256=digest(state['signature']['config']),
                build={},qualification={},rollback={'path':str(rollback),'sha256':sha(rollback)})
            q={'passed':True,'implementation':execution['qualified_implementation']}
            raw={'training':{'native_backend':copy.deepcopy(value)}}
            with patch.object(B,'check_build',return_value={}),patch.object(B,'checked_pin',return_value=q):
                self.assertEqual(B.validate_state(state,raw),value)
                raw['training']['native_backend_execution']=execution
                self.assertEqual(B.validate_state(state,raw),value)
                for previous in json.loads(R.REVIEW.read_text()).get('predecessor_executions',[]):
                    raw['training']['native_backend_execution']=previous
                    self.assertEqual(B.validate_state(state,raw),value)
                raw['training']['native_backend_execution']=dict(execution,implementation='f'*64)
                with self.assertRaisesRegex(ValueError,'execution source'):B.validate_state(state,raw)
                with patch.object(R,'implementation_id',return_value='f'*64):
                    with self.assertRaisesRegex(ValueError,'source/config'):B.validate_state(state)


if __name__=='__main__':unittest.main()
