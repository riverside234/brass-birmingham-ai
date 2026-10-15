"""Ordinary learner exploration, exact behavior and reviewed same-run changes."""
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch

import test_latest_model_recipe as fixtures
from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.curriculum import activate_boost,boost_window
from brass_ai.recovery.execution import neural_spec,_collection_recipe,ActorPool
from brass_ai.recovery.history import compact
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.ordinary import plan_collection,sampling_report
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset,probabilities,policy_measure,ppo_update
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import implementation_id,write_json
from brass_ai.step5.evaluation import evaluation_config
from brass_ai.step5.policy import exploration_distribution,INDUSTRIES

PREVIOUS='f71f3b85f1b68a4d128d57dbc85ff0f61526385804029f71078e869ce594911f'


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def config(self):return read(SOURCE_ROOT/'configs/step5/joint7-latest.json')

    def examples(self,exploration=.08):
        helper=fixtures.LearningTests();self.addCleanup(helper.doCleanups)
        return helper.examples(exploration)

    def test_generation_changes_only_ordinary_learner_exploration(self):
        cfg=self.config();old=copy.deepcopy(cfg);old['exploration']=.05
        new=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(new,dict(old,exploration=.08));self.assertEqual(cfg['exploration'],.08)
        self.assertEqual(evaluation_config(old),evaluation_config(cfg))
        self.assertEqual((cfg['update']['entropy_weight'],cfg['update']['value_weight']),(.005,.5))
        validate_config(cfg)
        for name in ('joint7-g150-gate72','lab','evaluation'):
            archived=read(SOURCE_ROOT/f'configs/step5/{name}.json')
            if 'exploration' in archived:self.assertEqual(archived['exploration'],.05)

    def test_family_industry_context_mixture_and_legal_mask(self):
        choices=[dict(kind='build',industry='Pottery')]*2+[dict(kind='build',industry='CottonMill')]+\
                [dict(kind='develop',industry='CoalMine')]*2+[dict(kind='loan')]
        intent=[float(x=='Pottery') for x in INDUSTRIES]+[0.,0.]
        q=exploration_distribution(choices,intent)
        np.testing.assert_allclose(q,[.2,.2,.2,.1,.1,.2]);self.assertAlmostEqual(q.sum(),1.)
        logits=torch.tensor([[4.,1.,0.,-2.,-1.,1.,float('-inf'),float('-inf')]])
        padded=torch.tensor([list(q)+[0.,0.]],dtype=torch.float32)
        p=probabilities(logits,padded,.08,1.)
        torch.testing.assert_close(p,.92*logits.softmax(-1)+.08*padded)
        torch.testing.assert_close(p.sum(-1),torch.ones(1));self.assertTrue(bool((p[:,6:]==0).all()))
        self.assertTrue(bool((p[:,:6]>0).all()))
        neutral=exploration_distribution(choices,[0.]*8)
        self.assertNotEqual(neutral.tolist(),[1/6]*6)
        self.assertAlmostEqual(sum(neutral[:2]),.25);self.assertAlmostEqual(sum(q[:2]),.4)

    def plan(self,directory,exploration):
        anchor_path=directory/'anchor.pt';anchor_path.write_bytes(b'planner-only frozen model')
        anchor=dict(path=anchor_path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(anchor_path),model_id='older',generation=0)
        cfg=self.config();cfg.update(games_per_iteration=20,exploration=exploration)
        state=dict(games_done=0,next_seed=cfg['seed'],context_counter=0,history=[anchor],population=[anchor],evaluation_anchor=anchor)
        activate_boost(state,cfg);service=Mock();service.models={'older':None,'current':None}
        profiles=['v4-a','v4-b','v4-c'];legacy={p:dict(kind='legacy',policy_id=p) for p in profiles}
        return plan_collection(SOURCE_ROOT,directory,state,cfg,dict(config=cfg),service,'current',profiles,legacy,legacy,boost_window(state,cfg)),cfg

    def test_four_learner_seats_and_frozen_opponents_have_distinct_behavior(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            plan,cfg=self.plan(Path(tmp),.08)
            self.assertEqual(plan['composition'],dict(selfplay=3,historical=5,legacy=12))
            learner=neural_spec('current',.08,'mixed')
            for job in plan['jobs']:
                kind=job['provenance']['kind'];self.assertEqual(job['policies'].count(learner),4 if kind=='selfplay' else 1)
                self.assertEqual(job['provenance']['learner_behavior']['exploration'],.08)
                self.assertEqual(job['provenance']['learner_behavior']['temperature'],1.)
                if kind=='historical':
                    for opponent in job['policies'][1:]:
                        self.assertEqual(opponent['exploration'],.05);self.assertEqual(opponent['focus'],'neutral')
                if kind=='legacy':self.assertTrue(all(p['kind']=='legacy' and 'exploration' not in p for p in job['policies'][1:]))
            self.assertNotEqual(learner['policy_id'],neural_spec('current',.05,'mixed')['policy_id'])

    def test_real_collected_probabilities_reconstruct_before_ppo(self):
        model,policy,games,kinds,data=self.examples();cfg=self.config()
        behavior=Behavior(policy.evaluator.model_id,.08);self.assertEqual(behavior.policy_id,policy.policy_id)
        ds=Dataset(data,'cpu');indices=list(range(len(data)))
        kl=policy_measure(model,ds,indices,behavior,cfg['update'],Timings(),verify=True)
        self.assertLess(abs(kl),1e-7)
        metrics=sampling_report(games,data,behavior.policy_id,kinds,cfg)
        self.assertEqual(metrics['behavior']['exploration'],.08);self.assertEqual(metrics['behavior']['temperature'],1.)
        report=compact(dict(actor={},sampling=metrics))
        self.assertEqual(report['sampling']['behavior'],metrics['behavior'])
        result=ppo_update(model,torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']),games,behavior,cfg['update'],learner_data=data,sample_groups=kinds)
        self.assertEqual(result['sample_usage']['actor_distinct_used'],5)

    def test_old_policy_rows_cannot_be_relabelled_or_admitted(self):
        model,old,games,_,data=self.examples(.05);behavior=Behavior(old.evaluator.model_id,.08)
        adam=torch.optim.AdamW(model.parameters(),lr=1e-5);before=copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError,'External or stale actions'):
            ppo_update(model,adam,games,behavior,self.config()['update'],learner_data=data)
        # Forging only the policy identity must not bypass probability verification.
        for row,_,_,_ in data:row['behavior']['policy_id']=behavior.policy_id
        with self.assertRaisesRegex(ValueError,'behavior probabilities'):
            ppo_update(model,adam,games,behavior,self.config()['update'],learner_data=data)
        self.assertTrue(_same_payload(before,model.state_dict()));self.assertEqual(adam.state,{})

    def test_cached_collection_identity_changes_and_stale_cache_fails(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);old,cfg=self.plan(directory,.05);new,_=self.plan(directory,.08)
            old_job=old['jobs'][0];job=new['jobs'][0]
            a=_collection_recipe([old_job],cfg['execution']);b=_collection_recipe([job],cfg['execution'])
            self.assertNotEqual(digest(a),digest(b))
            # A stale shard renamed to the new prefix is rejected before actor startup.
            path=directory/f'games-{digest(b)[:16]}-part-000000.jsonl.gz'
            with gzip.open(path,'wt',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(provenance=dict(collection=digest(a),recipe=a)))+'\n')
            service=Mock();pool=ActorPool(SOURCE_ROOT,cfg['host'],service,cfg['execution']);self.addCleanup(pool.close)
            with patch.object(pool,'start',side_effect=AssertionError('Actor unexpectedly started')):
                with self.assertRaisesRegex(ValueError,'Cached collection differs'):pool._collect([job])


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        from review_fixture import install
        install(self,[dict(from_implementation=source,to_implementation=implementation_id(),
            config_changes=dict(exploration=dict(**{'from':.05,'to':.08}))) for source in (PREVIOUS,implementation_id())])
        temporary=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(temporary.cleanup)
        run=Path(temporary.name);cfg=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');cfg.update(model_size='small',exploration=.05)
        signature=dict(config=cfg,compatibility=compatibility(),implementation=PREVIOUS,legacy_pool={},teachers={})
        signature['imitation_source']=cfg['imitation_checkpoint']
        torch.manual_seed(408);model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(cfg)))
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility),scope='authored fixture')
        adam=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        anchor_path=run/'initial.pt';anchor_id=save_checkpoint(anchor_path,model,adam,step=0)
        anchor=dict(path=anchor_path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(anchor_path),model_id=anchor_id,generation=0)
        sum(p.square().mean() for p in model.parameters()).backward();adam.step();adam.zero_grad()
        path=run/'generation-0075.pt';identity=save_checkpoint(path,model,adam,step=75,
            training=dict(mode='step5-population-ppo',fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(signature)))
        entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=75)
        state=dict(format='step5-recovery-league/2.2.0',signature=signature,iteration=75,games_done=0,history=[anchor,entry],population=[anchor,entry],resume=[entry],evaluation_anchor=anchor,
            next_seed=18512000,context_counter=456,next_search_games=24160,evaluations=[],ordinary_seconds=123.,search_seconds=456.,origin=dict(initialization='fresh-heuristic-imitation',**cfg['imitation_checkpoint']))
        activate_boost(state,cfg);state['games_done']=24000;write_json(run/'state.json',state)
        target=copy.deepcopy(signature);target['implementation']=implementation_id();target['config']['exploration']=.08
        return run,state,target,model,adam,path

    def test_only_exact_explicit_exploration_change_is_reviewed(self):
        _,state,target,_,_,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,target)
        self.assertTrue(check_signature(state,target,True))
        same=copy.deepcopy(state);same['signature']['implementation']=implementation_id()
        self.assertTrue(check_signature(same,target,True))
        for key,value in (('exploration',.09),('learning_rate',2e-5),('entropy_weight',.01)):
            bad=copy.deepcopy(target)
            if key=='entropy_weight':bad['config']['update'][key]=value
            else:bad['config'][key]=value
            with self.assertRaises(ValueError):check_signature(state,bad,True)

    def test_weights_adam_rng_progress_population_and_lineage_survive(self):
        run,state,target,model,adam,path=self.fixture();before=sha(path)
        restored,raw,changed=load_resume(SOURCE_ROOT,run,state,target,reviewed=True)
        self.assertTrue(changed);self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()))
        next_adam=torch.optim.AdamW(restored.parameters(),lr=target['config']['learning_rate'])
        resume_optimizer(raw,next_adam)
        self.assertTrue(_same_payload(adam.state_dict(),next_adam.state_dict()));self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
        updated=commit_transition(run,state,target)
        for key,value in state.items():
            if key not in ('signature','source_transitions'):self.assertEqual(updated[key],value,key)
        self.assertEqual(updated['source_transitions'][-1]['config_changes'],dict(exploration=dict(before=.05,after=.08)))
        self.assertEqual(sha(path),before)
        again,_,changed=load_resume(SOURCE_ROOT,run,updated,target)
        self.assertFalse(changed);self.assertTrue(_same_payload(restored.state_dict(),again.state_dict()))

    def test_pending_old_collection_is_refused_without_mutation(self):
        run,state,target,_,_,path=self.fixture();before=(run/'state.json').read_bytes();checkpoint=sha(path)
        pending=run/'iteration-0075';pending.mkdir();(pending/'old-005.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):
            load_resume(SOURCE_ROOT,run,state,target,reviewed=True)
        self.assertEqual((run/'state.json').read_bytes(),before);self.assertEqual(sha(path),checkpoint)


if __name__=='__main__':unittest.main()
