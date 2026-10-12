"""Clock migration keeps committed learning state and refuses stale teaching."""
import copy
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.recovery.config import (TEACHER_WORK_BUDGET,LEGACY_OPPONENT_WORK_BUDGET,
    TIME_CONFIG_MIGRATION,time_budget_config)
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.step5.league import implementation_id

OLD='9a60a40aa1e4d8193243681d68de2f4661b82d2cf213b2d86f9e78bd3a6e5541'


def signatures(name='lab.json'):
    cfg=read(SOURCE_ROOT/'configs/step5'/name);cfg['model_size']='small'
    new=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
    old=copy.deepcopy(new);old['implementation']=OLD
    old['config'].update(teacher_work_budget=copy.deepcopy(TEACHER_WORK_BUDGET),
        legacy_opponent_work_budget=copy.deepcopy(LEGACY_OPPONENT_WORK_BUDGET),search_wall_fraction=.9)
    if 'smoke' in name:old['config']['search']['seconds']=4.
    return old,new


class TimeResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        from test_teacher_work_resume import TeacherWorkResumeTests
        fixture=TeacherWorkResumeTests();fixture.signatures=signatures
        self.addCleanup(fixture.doCleanups)
        return fixture.fixture()

    def test_active_presets_exact_clock_migration_and_no_other_changes(self):
        for path in (SOURCE_ROOT/'configs/step5').glob('*.json'):
            if path.name.startswith('previous-') or read(path)['format']!='step5-training/2.2.0':continue
            with self.subTest(preset=path.name):
                old,new=signatures(path.name)
                self.assertEqual(time_budget_config(old['config']),new['config'])
                state=dict(format='step5-recovery-league/2.2.0',signature=old,iteration=70)
                self.assertTrue(check_signature(state,new,True))
                with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,new)
                for key in ('learning_rate','search_every_games','seed'):
                    changed=copy.deepcopy(new);changed['config'][key]+=1
                    with self.assertRaises(ValueError):check_signature(state,changed,True)
                changed=copy.deepcopy(new);changed['implementation']='0'*64
                with self.assertRaisesRegex(ValueError,'not been reviewed'):check_signature(state,changed,True)

    def test_weights_adam_rng_counters_and_search_debt_survive(self):
        run,state,new,original,optimizer=self.fixture()
        checkpoint=run/'generation-0070.pt';before=checkpoint.read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
        self.assertTrue(changed);self.assertFalse(model.utility_calibration['ready'])
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items()))
        restored=torch.optim.AdamW(model.parameters(),lr=new['config']['learning_rate']);resume_optimizer(raw,restored)
        self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
        self.assertEqual(restored.state_dict()['param_groups'],optimizer.state_dict()['param_groups'])
        for key,values in optimizer.state_dict()['state'].items():
            for name,value in values.items():torch.testing.assert_close(value,restored.state_dict()['state'][key][name],rtol=0,atol=0)
        updated=commit_transition(run,state,new)
        for key,value in state.items():
            if key!='signature':self.assertEqual(updated[key],value,key)
        self.assertEqual(updated['source_transitions'][-1]['config_migration'],TIME_CONFIG_MIGRATION)
        self.assertEqual(checkpoint.read_bytes(),before)
        _,_,changed=load_resume(SOURCE_ROOT,run,updated,new);self.assertFalse(changed)

    def test_partial_iteration_and_old_teacher_targets_cannot_be_reused(self):
        run,state,new,_,_=self.fixture();pending=run/'iteration-0070';pending.mkdir()
        partial=pending/'search-000000.jsonl.gz';partial.write_bytes(b'old count-primary labels')
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):
            load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
        self.assertEqual(partial.read_bytes(),b'old count-primary labels')
        from brass_ai.learning.replay import validate_teacher
        from test_teacher_readiness_review import TeacherReadinessReviewTests
        from brass_ai.recovery.updates import distill_update
        model,opt,episode,cfg,quality=TeacherReadinessReviewTests().fixture()
        for row in episode['records']:
            label=row['teacher'];label['source_fingerprint']=OLD
            label['fingerprint']=digest(dict(model=label['model_id'],config=label['search'],
                contract='masked-ismcts/'+label['source'].rsplit('/',1)[1],source=OLD))
            validate_teacher(label,len(row['choices'])) # Archived replay format stays valid.
        with self.assertRaisesRegex(ValueError,'fresh current-model/current-source'):
            distill_update(model,opt,[episode],cfg,steps=1,quality=quality)


if __name__=='__main__':unittest.main()
