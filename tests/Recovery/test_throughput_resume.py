"""Throughput-only source/layout transitions keep committed learning state."""
import copy
import unittest
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.step5.league import implementation_id

OLD='f739475371ce3d0a7dfe1c505780513a130474de3c17b8bb4501da0b67b017b3'


def signatures(name='lab.json'):
    cfg=read(SOURCE_ROOT/'configs/step5'/name);cfg['model_size']='small'
    new=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
    old=copy.deepcopy(new);old['implementation']=OLD
    execution=old['config']['execution']
    for key in ('shared_slot_bytes','inference_routing','replica_backlog'):execution.pop(key)
    if name.startswith('lab'):
        execution['processes']*=2;execution['hosts_per_process']=1
    return old,new


class ThroughputResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_layout_upgrade_and_source_only_preserve_other_settings(self):
        for name in ('lab.json','lab-24x4.json','lab-16x2.json','lab-small.json','lab-reference.json','recovery-smoke.json','smoke.json'):
            with self.subTest(preset=name):
                old,new=signatures(name)
                state=dict(format='step5-recovery-league/2.2.0',signature=old,iteration=70)
                self.assertTrue(check_signature(state,new,True))
                self.assertTrue(check_signature(state,dict(old,implementation=implementation_id()),True))
                changed=copy.deepcopy(new);changed['config']['learning_rate']*=2
                with self.assertRaises(ValueError):check_signature(state,changed,True)
                changed=copy.deepcopy(new);changed['config']['execution']['cpu_budget']+=4
                with self.assertRaises(ValueError):check_signature(state,changed,True)
                with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,new)

    def test_weights_adam_rng_and_committed_state_survive_layout_upgrade(self):
        from test_teacher_work_resume import TeacherWorkResumeTests
        fixture=TeacherWorkResumeTests();fixture.signatures=signatures
        self.addCleanup(fixture.doCleanups)
        run,state,new,original,optimizer=fixture.fixture()
        path=run/'generation-0070.pt';before=path.read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
        self.assertTrue(changed)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items()))
        restored=torch.optim.AdamW(model.parameters(),lr=new['config']['learning_rate']);resume_optimizer(raw,restored)
        self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
        self.assertEqual(restored.state_dict()['param_groups'],optimizer.state_dict()['param_groups'])
        for key,values in optimizer.state_dict()['state'].items():
            for name,value in values.items():torch.testing.assert_close(value,restored.state_dict()['state'][key][name],rtol=0,atol=0)
        updated=commit_transition(run,state,new)
        for key,value in state.items():
            if key!='signature':self.assertEqual(updated[key],value,key)
        self.assertEqual(path.read_bytes(),before)
        _,_,changed=load_resume(SOURCE_ROOT,run,updated,new);self.assertFalse(changed)

if __name__=='__main__':unittest.main()
