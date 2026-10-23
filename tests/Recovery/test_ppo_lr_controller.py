"""Decision sequences, exact restart, PPO scope, and the fixed fresh recipe."""
import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.batching import collate,model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,read_json,digest,compatibility
from brass_ai.learning.features import Encoded
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import lr_controller as controller
from brass_ai.recovery.learning_rate import configure,step,KEY as WARMUP_KEY
from brass_ai.recovery.resume import apply_optimizer_config,check_signature
from brass_ai.recovery.config import (validate_config,latest_model_config,ARCHIVED_CURRICULUM,
    ARCHIVED_LATEST_MODEL_RECIPE,LEGACY_BOOST,table_kind,table_rotation)
from brass_ai.recovery.curriculum import training_table,boost_window
from brass_ai.recovery.ordinary import plan_collection,sampling_report
from brass_ai.recovery.population import retain
from brass_ai.recovery.history import compact
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.updates import ppo_update,supervised_epoch,probabilities,Dataset
from brass_ai.recovery.teacher_collection import collect_due,due_collections
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id
from test_mlp_v11 import inputs


def config():
    # These regressions cover the explicit 1.1 ordinary-probe controller.
    value=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json')
    from brass_ai.recovery.exploration import BLOCKS
    for key in BLOCKS:value.pop(key,None)
    value['exploration']=.08
    value['ppo_lr_controller']['contract']=controller.CONTRACT
    value.pop('il_kl_limit',None)
    for key in ('table_plan','opponent_pool','rating','v4_search_enabled','v4_share_gate','advantage','value_phase',
                'critic_data','seat_descriptors','panel_gates','policy_anchor','rating_coverage'):value.pop(key,None)
    value.update(selfplay_recipe='latest-model-selfplay/2.0.0',
        curriculum=[dict(until=None,selfplay=60,historical=30,legacy=10)],population=dict(recent=4,anchors=12))
    return value


def measurements(**kw):
    return dict(dict(kl=.002,steps_done=10,steps_planned=10,early_stop=False,warmup_done=True,
        data_signature='a'*64,anchor_distance=.02,selfplay_vp_per_seat=100.,zero_vp_fraction=.01,
        anchor_groups={p:dict(positions=64,anchor_distance=.02) for p in ('canal_early','canal_late','rail_early','rail_late')},
        anchor_unassigned_positions=0),**kw)


def ready(mode='apply'):
    state=controller.initial_state(dict(controller.DEFAULT_CONTROLLER,mode=mode))
    state['health_window']=[100.]*5
    return state


class Decisions(unittest.TestCase):
    def test_strict_config_and_same_ppo_target(self):
        validate_config(config())
        for key,value in [('mode','automatic'),('extra',1),('target_kl',True),('raise_factor',1.),
                          ('cut_divisor',float('nan')),('min_multiplier',0.),('max_multiplier',.2),
                          ('window_iterations',True),('cooldown_iterations',2),('anchor_kl_budget',-1.)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                controller.validate_config(dict(controller.DEFAULT_CONTROLLER,**{key:value}))
        cfg=config();cfg['ppo_lr_controller']['target_kl']=.02
        with self.assertRaisesRegex(ValueError,'must match'):validate_config(cfg)
        cfg=config();cfg['update']['kl_probe_rows']=1024
        with self.assertRaisesRegex(ValueError,'2048'):validate_config(cfg)

    def test_median_three_and_minimum_three_between_raises(self):
        state=ready()
        for i,kl in enumerate((.002,.013,.002,.002,.002,.002)):
            state,report=controller.decide(state,measurements(kl=kl))
            self.assertEqual(report['decision'],'up' if i in (2,5) else 'hold')
        self.assertEqual(state['multiplier'],1.25**2)

    def test_warmup_and_empty_or_nonfinite_updates(self):
        state=ready()
        for kw in (dict(steps_done=0),dict(kl=float('nan')),dict(kl=float('inf')),dict(kl=None)):
            other,report=controller.decide(state,measurements(**kw))
            self.assertEqual(state,other);self.assertEqual(report['reason'],'empty_or_nonfinite_update')
            # Reporting must remain writable even when measurements are NaN.
            digest(report)
        for _ in range(4):
            state,report=controller.decide(state,measurements(kl=.5,early_stop=True,warmup_done=False))
            self.assertEqual(report['reason'],'warmup');self.assertEqual(state['multiplier'],1.)
            self.assertEqual(state['window'],[])

    def test_cuts_are_immediate_and_two_of_three_even_with_a_gap(self):
        for seq in ([measurements(early_stop=True,steps_done=4)],
                    [measurements(kl=.015001)],
                    [measurements(early_stop=True),measurements(),measurements(early_stop=True)]):
            state=ready()
            for m in seq:state,report=controller.decide(state,m)
            self.assertEqual(report['decision'],'down');self.assertAlmostEqual(state['multiplier'],.8)
            self.assertEqual(state['window'],[]);self.assertEqual(state['cooldown'],3)
        # Exactly half and exactly the upper boundary do not trigger these cuts.
        _,report=controller.decide(ready(),measurements(early_stop=True,steps_done=5,kl=.015))
        self.assertEqual(report['decision'],'hold')

    def test_cooldown_and_both_clamps(self):
        state,report=controller.decide(ready(),measurements(kl=.02))
        for expected in (2,1,0):
            state,report=controller.decide(state,measurements())
            self.assertEqual(report['decision'],'cooldown');self.assertEqual(state['cooldown'],expected)
        state,report=controller.decide(state,measurements());self.assertEqual(report['decision'],'up')
        for _ in range(40):state,_=controller.decide(state,measurements())
        self.assertEqual(state['multiplier'],3.)
        for _ in range(20):state,report=controller.decide(state,measurements(kl=.02))
        self.assertEqual(state['multiplier'],.5);self.assertTrue(report['at_min'])

    def test_health_only_blocks_increases_and_five_iteration_vp_reference(self):
        for kw,reason in ((dict(anchor_distance=.1501),'anchor_budget_exceeded'),
                          (dict(selfplay_vp_per_seat=89.9),'selfplay_vp_declined'),
                          (dict(zero_vp_fraction=.05),'too_many_zero_vp_seats'),
                          (dict(anchor_distance=None),'anchor_distance_unavailable')):
            state=ready()
            for _ in range(3):state,report=controller.decide(state,measurements(**kw))
            self.assertEqual(report['decision'],'hold');self.assertIn(reason,report['health']['reasons'])
            state,report=controller.decide(state,measurements(kl=.02,**kw))
            self.assertEqual(report['decision'],'down')
        state=controller.initial_state(controller.DEFAULT_CONTROLLER)
        for _ in range(5):state,report=controller.decide(state,measurements())
        self.assertEqual(state['multiplier'],1.)
        state,report=controller.decide(state,measurements(selfplay_vp_per_seat=90.))
        self.assertEqual(report['decision'],'up');self.assertEqual(report['health']['selfplay_vp_five_iterations_ago'],100.)
        state=ready()
        for _ in range(3):state,report=controller.decide(state,measurements(anchor_distance=.15))
        self.assertEqual(report['decision'],'up');self.assertEqual(report['health']['anchor_budget'],.15)

    def test_collection_change_resets_only_streak_not_multiplier_or_cooldown(self):
        state=ready();state['multiplier']=2.;state['cooldown']=2;state['data_signature']='a'*64
        state['window']=[dict(kl=.002,early_stop=False)]*2
        state,report=controller.decide(state,measurements(data_signature='b'*64))
        self.assertTrue(report['window_reset']);self.assertEqual(len(state['window']),1)
        self.assertEqual(state['multiplier'],2.);self.assertEqual(state['cooldown'],1)

    def test_off_state_and_rng_are_unchanged(self):
        rng=torch.get_rng_state().clone();state=ready('off')
        other,report=controller.decide(state,measurements(kl=.2))
        self.assertEqual(other,state);self.assertEqual(report['reason'],'off')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


class OptimizerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def optimizer(self,model,mode='apply',warmup=0):
        optimizer=torch.optim.AdamW(model.parameters(),lr=6e-6,weight_decay=1e-4)
        configure(optimizer,6e-6,warmup,.1,'selfplay')
        controller.configure(optimizer,dict(controller.DEFAULT_CONTROLLER,mode=mode))
        optimizer.param_groups[0][controller.KEY]=ready(mode)
        return optimizer

    def test_adam_checkpoint_streak_restart_has_same_decision_and_next_weights(self):
        model=FourSeatNet(ModelConfig.preset('small'));optimizer=self.optimizer(model,warmup=100)
        optimizer.param_groups[0][WARMUP_KEY]['completed_steps']=100
        state=optimizer.param_groups[0][controller.KEY]
        for _ in range(2):state,_=controller.decide(state,measurements())
        optimizer.param_groups[0][controller.KEY]=state
        def update(net,opt):
            opt.zero_grad(set_to_none=True);sum(p.square().mean() for p in net.parameters()).backward();step(opt,ppo=True)
        update(model,optimizer)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
            path=Path(folder)/'checkpoint.pt';save_checkpoint(path,model,optimizer,step=7)
            loaded,raw=load_checkpoint(path);restored=torch.optim.AdamW(loaded.parameters(),lr=1.)
            resume_optimizer(raw,restored);apply_optimizer_config(restored,config())
            self.assertEqual(restored.param_groups[0][controller.KEY],state)
            self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
            for opt in (optimizer,restored):
                opt.param_groups[0][controller.KEY],report=controller.decide(opt.param_groups[0][controller.KEY],measurements())
                self.assertEqual(report['decision'],'up')
            update(model,optimizer);update(loaded,restored)
            self.assertTrue(_same_payload(model.state_dict(),loaded.state_dict()))
            self.assertTrue(_same_payload(optimizer.state_dict(),restored.state_dict()))
            self.assertEqual(restored.param_groups[0]['lr'],7.5e-6)

    def test_shadow_and_nonppo_steps_never_use_multiplier(self):
        for mode in ('apply','shadow','off'):
            p=torch.nn.Parameter(torch.ones(1));opt=self.optimizer(torch.nn.ParameterList([p]),mode)
            opt.param_groups[0][controller.KEY]['multiplier']=3.
            for ppo,expected in ((True,18e-6 if mode=='apply' else 6e-6),(False,6e-6)):
                p.grad=torch.ones_like(p);step(opt,ppo=ppo)
                self.assertAlmostEqual(opt.param_groups[0]['lr'],expected)
            self.assertEqual(opt.param_groups[0][controller.KEY]['multiplier'],3.)
        opt=self.optimizer(torch.nn.ParameterList([p]),warmup=100)
        opt.param_groups[0][controller.KEY]['multiplier']=3.;p.grad=torch.ones_like(p);step(opt,ppo=True)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],6e-7)

    def data(self,model):
        x,a,mask,mapping=inputs(8,2);mask[:]=True
        behavior=Behavior(model_id(model),.08)
        choices=[dict(kind='pass'),dict(kind='loan')]
        with torch.no_grad():
            p=probabilities(model(x,a,mask,mapping)[0],torch.full((8,2),.5),.08,1.)
        rows=[]
        for i in range(8):
            row=dict(chosen=i%2,seat=i%4,intent=[0.]*8,choices=choices,
                behavior=dict(kind='model',policy_id=behavior.policy_id,probability=float(p[i,i%2])),
                view=dict(context=dict(era='CanalEra',round=i+1,phase='action')))
            encoded=Encoded(x[i].numpy(),a[i].numpy(),mapping[i].numpy(),('pass','loan'))
            rows.append((row,dict(win_share=[1.,0.,0.,0.],placement=[1.,2.,3.,4.],vp=[100.,90.,80.,70.]),('g',i%4),encoded))
        return behavior,rows

    def test_real_ppo_exact_probe_planned_steps_anchor_and_early_stop(self):
        torch.manual_seed(33);model=FourSeatNet(ModelConfig.preset('small'));anchor=copy.deepcopy(model).eval()
        behavior,data=self.data(model);opt=self.optimizer(model);opt.param_groups[0][controller.KEY]['multiplier']=1.25
        cfg=dict(config()['update'],effective_batch_size=2,microbatch_size=2,critic_batch_size=2)
        from brass_ai.recovery.review_metrics import CategoricalProbe
        real=CategoricalProbe.measure_step
        def stop(probe,*args,**kw):
            return dict(real(probe,*args,**kw),categorical_kl=.02)
        with patch.object(CategoricalProbe,'measure_step',stop):
            report=ppo_update(model,opt,[dict(split='train')],behavior,cfg,learner_data=data,anchor_model=anchor)
        self.assertEqual(report['optimizer_steps_planned'],8);self.assertEqual(report['optimizer_steps'],1)
        self.assertTrue(report['kl_early_stop']);self.assertEqual(report['learning_rates'],[7.5e-6])
        self.assertEqual(report['kl_guard']['stop_reason'],'categorical_probe')
        self.assertEqual(report['categorical_probe']['positions'],8)
        self.assertEqual(report['categorical_probe']['anchor_unassigned_positions'],0)
        self.assertEqual({p:r['positions'] for p,r in report['categorical_probe']['anchor_groups'].items()},
                         dict(canal_early=4,canal_late=4,rail_early=0,rail_late=0))
        self.assertAlmostEqual(report['categorical_probe']['categorical_kl'],report['categorical_probe']['anchor_distance'],places=7)
        self.assertLess(report['categorical_probe']['categorical_kl'],.02)
        self.assertTrue(report['controller_warmup_done'])

    def test_real_teacher_steps_and_calibration_do_not_use_ppo_multiplier(self):
        model=FourSeatNet(ModelConfig.preset('small'));_,data=self.data(model);other=copy.deepcopy(model)
        left=self.optimizer(model);right=self.optimizer(other,'off')
        left.param_groups[0][controller.KEY]['multiplier']=3.
        cfg=dict(config()['update'],effective_batch_size=8,microbatch_size=8)
        for row,_,_,_ in data:
            row['teacher']=dict(source='fresh-finalist/1.0.0',probabilities=[.75,.25],coverage=[True,True],confidence=1.,samples=4,horizon='terminal')
        rng=torch.get_rng_state().clone()
        supervised_epoch(model,left,data,cfg,kind='teacher')
        torch.set_rng_state(rng);supervised_epoch(other,right,data,cfg,kind='teacher')
        self.assertTrue(_same_payload(model.state_dict(),other.state_dict()))
        self.assertEqual(left.param_groups[0]['lr'],6e-6)
        self.assertEqual(left.param_groups[0][controller.KEY]['multiplier'],3.)
        # Calibration calls Adam directly at its existing fixed 1e-3 rate.
        from brass_ai.recovery.utility import migrate,calibrate,reward_spec
        reward=reward_spec(config());migrate(model,left,reward);migrate(other,right,reward)
        for row,_,_,_ in data:row.update(_reward_return=[1.,.1,0.,0.],_reward_remaining=[.075]*4)
        rng=torch.get_rng_state().clone()
        warmup=copy.deepcopy(left.param_groups[0][WARMUP_KEY]);control=copy.deepcopy(left.param_groups[0][controller.KEY])
        calibrate(model,left,Dataset(data,'cpu',policy=False),cfg)
        torch.set_rng_state(rng);calibrate(other,right,Dataset(data,'cpu',policy=False),cfg)
        self.assertTrue(_same_payload(model.state_dict(),other.state_dict()))
        self.assertEqual(left.param_groups[0][WARMUP_KEY],warmup);self.assertEqual(left.param_groups[0][controller.KEY],control)


class FixedRecipe(unittest.TestCase):
    def test_defaults_generator_and_archived_recipe_validation(self):
        cfg=config();current=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json')
        self.assertEqual(latest_model_config(cfg,cfg['imitation_checkpoint']),current)
        self.assertFalse(cfg['teacher_enabled']);self.assertEqual(cfg['population'],dict(recent=4,anchors=12))
        self.assertEqual(cfg['architecture'],'brass-four-seat-mlp-resnet/11.2.0')
        for start in (0,19920,20000,99920,100000,500000):
            counts=Counter(training_table(i,cfg,None)[0] for i in range(start,start+320))
            self.assertEqual(counts,dict(selfplay=192,historical=96,legacy=32))
        self.assertIsNone(boost_window(dict(games_done=0),cfg))
        old=copy.deepcopy(cfg);old.update(selfplay_recipe=ARCHIVED_LATEST_MODEL_RECIPE,curriculum=ARCHIVED_CURRICULUM,legacy_boost=LEGACY_BOOST)
        validate_config(old)
        for key,value in [('teacher_enabled',0),('legacy_boost',LEGACY_BOOST),('curriculum',ARCHIVED_CURRICULUM)]:
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):validate_config(bad)

    def test_planner_two_learners_opponents_and_rotation_exact_restart(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'anchor.pt';path.write_bytes(b'planner fixture only')
            anchor=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id='older',generation=0)
            cfg=config();state=dict(games_done=0,next_seed=cfg['seed'],context_counter=0,
                history=[anchor],population=[anchor],evaluation_anchor=anchor)
            profiles=['v4a','v4b','v4c'];opponents={p:dict(policy_id=p,kind='legacy') for p in profiles}
            service=Mock();service.models={'older':None,'current':None}
            def plan(state,cfg):return plan_collection(SOURCE_ROOT,Path(directory),state,cfg,dict(config=config()),service,'current',profiles,opponents,{},None)
            whole=plan(state,cfg);seats=Counter()
            for job in whole['jobs']:
                kind=job['provenance']['kind'];learner=job['policies'][0]
                self.assertEqual(job['policies'].count(learner),4 if kind=='selfplay' else 2)
                self.assertEqual(learner['exploration'],.08)
                self.assertEqual(len(job['focuses']),4 if kind=='selfplay' else 2)
                for seat in job['focuses']:seats[(kind,seat)]+=1
                if kind=='historical':self.assertTrue(all(p['exploration']==.05 for p in job['policies'][2:]))
                if kind=='legacy':self.assertEqual(job['provenance']['legacy_mode'],'searched')
            self.assertEqual(whole['context_counter'],1024)
            for kind in ('selfplay','historical','legacy'):
                self.assertEqual(len({seats[(kind,str(s))] for s in range(4)}),1)
            half=dict(cfg,games_per_iteration=160);first=plan(state,half)
            second=plan(dict(state,games_done=160,next_seed=first['cursor'],context_counter=first['context_counter']),half)
            for actual,expected in zip(first['jobs']+second['jobs'],whole['jobs']):
                self.assertEqual(actual['provenance'],expected['provenance']);self.assertEqual(actual['focuses'],expected['focuses'])

    def test_disabled_teacher_does_not_touch_search_seeds_or_optimizer(self):
        cfg=config();state=dict(next_search_games=640,search_seconds=123.)
        self.assertEqual(due_collections(100000,640,cfg),0)
        with patch('brass_ai.recovery.teacher_collection.estimate_collection',side_effect=AssertionError('cost estimated')):
            report,next_search,cursor,seconds=collect_due(None,None,None,cfg,state,None,None,None,None,None,
                done=100000,cursor=999,ordinary_seconds=10,ordinary_game_seconds=1)
        self.assertFalse(report['enabled']);self.assertEqual((next_search,cursor,seconds),(640,999,123.))
        self.assertEqual(report['optimizer_steps'],0)

    def test_sampling_and_health_count_only_actual_learner_seats(self):
        from fixtures import result
        games=[];data=[];kinds={}
        for kind,count in (('selfplay',4),('historical',2),('legacy',2)):
            opponents=['learner']*count+['frozen']*(4-count)
            g=dict(game_id=kind,setup_group=kind,opponents=opponents,
                result=result([(100,1,1),(100,1,1),(0,1,1),(0,1,1)]))
            games.append(g);kinds[kind]=kind
            for seat in range(count):
                row=dict(behavior=dict(policy_id='learner',kind='model'),choices=[1,2])
                data.append((row,None,(kind,seat),None))
        sampling=sampling_report(games,data,'learner',kinds,config())
        self.assertEqual(sampling['learner_trajectories'],8)
        self.assertEqual(sampling['actor_eligible_decisions'],8)
        health=controller.game_health(games,'learner',kinds)
        self.assertEqual(health,dict(selfplay_vp_per_seat=50.,zero_vp_fraction=.25))
        before=controller.collection_signature(config(),sampling)
        # Random variation in decisions must not wipe a valid KL streak.
        other=copy.deepcopy(sampling);other['by_composition']['selfplay']['learner_decisions']+=1
        self.assertEqual(controller.collection_signature(config(),other),before)
        other['by_composition']['historical']['learner_trajectories']+=2
        self.assertNotEqual(controller.collection_signature(config(),other),before)
        data.append((dict(behavior=dict(policy_id='frozen',kind='model'),choices=[1,2]),None,('legacy',3),None))
        with self.assertRaisesRegex(ValueError,'external'):sampling_report(games,data,'learner',kinds,config())

    def test_retention_never_exceeds_16_and_keeps_initial_and_four_recent(self):
        pool=[]
        for generation in range(100):
            pool=retain(pool+[dict(model_id=str(generation),generation=generation)],config()['population'])
            self.assertLessEqual(len(pool),16);self.assertEqual(pool[0]['generation'],0)
        self.assertEqual([e['generation'] for e in pool[-4:]],[96,97,98,99]);self.assertEqual(len(pool),16)

    def test_config_is_signature_bound_and_compact_history_keeps_controller(self):
        cfg=config();signature=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),teachers={},legacy_pool={})
        state=dict(format='step5-recovery-league/2.2.0',signature=signature,games_done=320,iteration=1)
        changed=copy.deepcopy(signature);changed['config']['ppo_lr_controller']['mode']='shadow'
        with self.assertRaisesRegex(ValueError,'configuration/source changed'):check_signature(state,changed)
        with self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)
        _,report=controller.decide(ready(),measurements())
        summary=compact(dict(actor=dict(lr_controller=report,optimizer_steps_planned=10,optimizer_steps_fraction=.7)))
        self.assertEqual(summary['actor']['lr_controller'],report)
        self.assertEqual(summary['actor']['optimizer_steps_planned'],10)

    def test_completed_reviewed_v11_2_il_is_admitted_without_file_changes(self):
        from latest_model_fixture import imitation_fixture
        from brass_ai.recovery.demonstrations import completed_imitation
        before='e2b6bf879c8b9da5b043c8f4c8aedf76b45bec9095867e4246b7decd1c316144'
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
            with patch('latest_model_fixture.implementation_id',return_value=before):
                run,model,cfg=imitation_fixture(Path(folder))
            files={p:sha(p) for p in run.iterdir() if p.is_file()}
            loaded,origin=completed_imitation(SOURCE_ROOT,run)
            self.assertTrue(_same_payload(model.state_dict(),loaded.state_dict()));self.assertEqual(origin,cfg['imitation_checkpoint'])
            self.assertEqual(files,{p:sha(p) for p in files})


if __name__=='__main__':unittest.main()
