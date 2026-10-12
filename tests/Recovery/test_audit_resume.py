"""Exact committed Step 5 continuation through the comprehensive bug audit."""
import copy
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.recovery.resume import check_signature, load_resume, commit_transition
from brass_ai.step5.league import implementation_id

PREVIOUS='aeef0917b3645bdbdfe8dcb9d2bfd34ab05593c8b5fb9433dbc1e988b4ef131f'


class AuditResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def signatures(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        cfg['model_size']='small'
        old=dict(config=cfg,compatibility=compatibility(),implementation=PREVIOUS,
                 legacy_pool={},teachers={})
        return old,dict(old,implementation=implementation_id())

    def test_committed_weights_optimizer_rng_population_and_progress_survive(self):
        from test_teacher_work_resume import TeacherWorkResumeTests
        fixture=TeacherWorkResumeTests()
        fixture.signatures=self.signatures
        self.addCleanup(fixture.doCleanups)
        run,state,new,original,optimizer=fixture.fixture()
        checkpoint=run/'generation-0070.pt'
        before=checkpoint.read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
        self.assertTrue(changed)
        restored=torch.optim.AdamW(model.parameters(),lr=new['config']['learning_rate'])
        resume_optimizer(raw,restored)
        self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
        for name,tensor in original.state_dict().items():
            torch.testing.assert_close(tensor,model.state_dict()[name],rtol=0,atol=0)
        expected,actual=optimizer.state_dict(),restored.state_dict()
        self.assertEqual(expected['param_groups'],actual['param_groups'])
        for key,values in expected['state'].items():
            for name,value in values.items():
                torch.testing.assert_close(value,actual['state'][key][name],rtol=0,atol=0)
        updated=commit_transition(run,state,new)
        for name in ('iteration','games_done','next_search_games','search_seed_cursor',
                     'ordinary_seconds','search_seconds','population','champion','resume','legacy_boost'):
            self.assertEqual(updated[name],state[name])
        self.assertFalse(model.utility_calibration['ready'])
        self.assertEqual(checkpoint.read_bytes(),before)
        self.assertFalse(load_resume(SOURCE_ROOT,run,updated,new)[2])

    def test_review_does_not_admit_changed_config_contracts_or_unknown_source(self):
        old,new=self.signatures()
        state=dict(format='step5-recovery-league/2.2.0',iteration=70,signature=old)
        self.assertTrue(check_signature(state,new,True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):
            check_signature(state,new)
        for field in ('config','compatibility','implementation'):
            changed=copy.deepcopy(new)
            if field=='implementation':changed[field]='0'*64
            else:changed[field]['unknown-change']=True
            with self.subTest(field=field),self.assertRaises(ValueError):
                check_signature(state,changed,True)


if __name__=='__main__':
    unittest.main()
