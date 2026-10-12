"""Training work budgets preserve exact reviewed checkpoint continuation."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import (TEACHER_WORK_BUDGET,LEGACY_OPPONENT_WORK_BUDGET,
    teacher_work_budget_config,legacy_opponent_work_budget_config,validate_config)
from brass_ai.recovery.curriculum import activate_boost
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import implementation_id,write_json


OLD='f9f94171ead6383c012d402aadf8781f51074a1f4cb5e6575cfb1487deb4e5f0'


class TeacherWorkResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def signatures(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json');cfg['model_size']='small'
        cfg.pop('teacher_work_budget',None)
        cfg.pop('legacy_opponent_work_budget',None)
        old=dict(config=cfg,compatibility=compatibility(),implementation=OLD,legacy_pool={},teachers={})
        new=copy.deepcopy(old);new['implementation']=implementation_id()
        new['config']['teacher_work_budget']=copy.deepcopy(TEACHER_WORK_BUDGET)
        new['config']['legacy_opponent_work_budget']=copy.deepcopy(LEGACY_OPPONENT_WORK_BUDGET)
        return old,new

    def fixture(self):
        temporary=tempfile.TemporaryDirectory(prefix='teacher-work-resume-',dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(temporary.cleanup);run=Path(temporary.name).resolve()
        old,new=self.signatures()
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(old['config'])))
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility),scope='old fixture')
        optimizer=torch.optim.AdamW(model.parameters(),lr=old['config']['learning_rate'])
        sum(p.square().mean() for p in model.parameters()).backward();optimizer.step()
        path=run/'generation-0070.pt'
        identity=save_checkpoint(path,model,optimizer,step=70,training=dict(mode='step5-population-ppo',
            fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(old)))
        entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=70)
        state=dict(format='step5-recovery-league/2.2.0',signature=old,iteration=70,games_done=22400,
            next_search_games=22560,search_seed_cursor=18970000,ordinary_seconds=12345.,search_seconds=54321.,
            history=[entry],population=[entry],evaluation_anchor=entry,champion=entry,resume=[entry])
        activate_boost(state,old['config']);write_json(run/'state.json',state)
        return run,state,new,model,optimizer

    def test_all_active_lab_presets_adopt_only_training_work_budget(self):
        for path in (SOURCE_ROOT/'configs/step5').glob('lab*.json'):
            with self.subTest(preset=path.name):
                cfg=read(path);validate_config(cfg)
                self.assertEqual(cfg['teacher_work_budget'],TEACHER_WORK_BUDGET)
                self.assertEqual(teacher_work_budget_config(cfg['teacher_work_budget']),TEACHER_WORK_BUDGET)
                self.assertEqual(cfg['legacy_opponent_work_budget'],LEGACY_OPPONENT_WORK_BUDGET)
                self.assertEqual(legacy_opponent_work_budget_config(cfg['legacy_opponent_work_budget']),LEGACY_OPPONENT_WORK_BUDGET)
                self.assertEqual(cfg['search']['seconds'],12.)
                self.assertEqual(cfg['execution']['search_workers_per_turn'],4)
                legacy=copy.deepcopy(cfg);legacy.pop('teacher_work_budget')
                legacy.pop('legacy_opponent_work_budget')
                self.assertEqual(validate_config(legacy),legacy)
        for name in ('smoke.json','recovery-smoke.json'):
            cfg=read(SOURCE_ROOT/'configs/step5'/name);validate_config(cfg)

    def test_work_budget_rejects_malformed_and_unbounded_counts(self):
        self.assertIsNone(teacher_work_budget_config(None))
        for change in ('missing-phase','unknown-phase','zero','negative','bool','fraction','too-large','unknown-mode','unknown-contract'):
            cfg=copy.deepcopy(TEACHER_WORK_BUDGET);phases=cfg['simulations_per_phase']
            if change=='missing-phase':phases.pop('rail_late')
            elif change=='unknown-phase':phases['future']=8
            elif change=='zero':phases['rail_late']=0
            elif change=='negative':phases['rail_late']=-1
            elif change=='bool':phases['rail_late']=True
            elif change=='fraction':phases['rail_late']=1.5
            elif change=='too-large':phases['rail_late']=100001
            elif change=='unknown-mode':cfg['mode']='time-multiplier'
            else:cfg['contract']='unknown'
            with self.subTest(change=change),self.assertRaises(ValueError):teacher_work_budget_config(cfg)
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        cfg['teacher_work_budget']=copy.deepcopy(TEACHER_WORK_BUDGET)
        cfg['teacher_work_budget']['simulations_per_phase']['rail_late']=1
        with self.assertRaisesRegex(ValueError,'cannot support workers'):validate_config(cfg)

    def test_only_exact_reviewed_budget_addition_is_accepted(self):
        old,new=self.signatures();state=dict(format='step5-recovery-league/2.2.0',signature=old,iteration=70)
        self.assertTrue(check_signature(state,new,True))
        self.assertTrue(check_signature(state,dict(old,implementation=implementation_id()),True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,new)
        for field in ('count','legacy_count','learning_rate','compatibility','source'):
            other=copy.deepcopy(new)
            if field=='count':other['config']['teacher_work_budget']['simulations_per_phase']['rail_late']+=1
            elif field=='legacy_count':other['config']['legacy_opponent_work_budget']['simulations_per_turn']+=1
            elif field=='learning_rate':other['config']['learning_rate']*=2
            elif field=='compatibility':other['compatibility']['changed']=True
            else:other['implementation']='0'*64
            with self.subTest(field=field),self.assertRaises(ValueError):check_signature(state,other,True)

    def test_previous_capped_work_revision_upgrades_without_reset(self):
        _,new=self.signatures();old=copy.deepcopy(new)
        old['implementation']='c7bffaf7bbfd2d7495da1af9bd9c8a63cba55c9be2a699b29827065bf1bf4f38'
        for key in ('teacher_work_budget','legacy_opponent_work_budget'):
            old['config'][key]['contract']=old['config'][key]['contract'].replace('/1.1.0','/1.0.0')
            old['config'][key].pop('watchdog_seconds')
        state=dict(format='step5-recovery-league/2.2.0',signature=old,iteration=70)
        self.assertTrue(check_signature(state,new,True))
        changed=copy.deepcopy(new);changed['config']['legacy_opponent_work_budget']['watchdog_seconds']=600
        with self.assertRaises(ValueError):check_signature(state,changed,True)

    def test_resume_preserves_weights_adam_rng_counters_pool_and_window(self):
        run,state,new,original,optimizer=self.fixture();saved=run/'generation-0070.pt';before=saved.read_bytes()
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
        self.assertEqual(updated['source_transitions'][-1]['config_changes'],
            {'teacher_work_budget':dict(before=None,after=TEACHER_WORK_BUDGET),
             'legacy_opponent_work_budget':dict(before=None,after=LEGACY_OPPONENT_WORK_BUDGET)})
        self.assertEqual(read(run/updated['source_transitions'][-1]['backup']),state)
        self.assertEqual(saved.read_bytes(),before)
        again,_,changed=load_resume(SOURCE_ROOT,run,updated,new)
        self.assertFalse(changed);self.assertFalse(again.utility_calibration['ready'])

    def test_old_partial_collection_cannot_be_relabelled_as_work_budget_data(self):
        run,state,new,_,_=self.fixture();pending=run/'iteration-0070';pending.mkdir()
        partial=pending/'search-000000.jsonl.gz';partial.write_bytes(b'old fixed-time partial data')
        before=(run/'state.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):
            load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
        self.assertEqual((run/'state.json').read_bytes(),before)
        self.assertEqual(partial.read_bytes(),b'old fixed-time partial data')


if __name__=='__main__':unittest.main()
