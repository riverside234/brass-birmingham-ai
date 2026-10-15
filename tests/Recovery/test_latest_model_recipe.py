"""Fresh latest-model curriculum, attribution, pinned IL and gate-free lifecycle."""
from collections import Counter
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch

import torch

from fixtures import game
from latest_model_fixture import imitation_fixture
from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import champion,champion_mix,updates
from brass_ai.recovery.config import (CURRICULUM,LEGACY_BOOST,TEACHER_TIME_BUDGET,CHAMPION,
    validate_config,latest_model_config,time_budget_config,champion_mix_config,table_rotation)
from brass_ai.recovery.curriculum import activate_boost,boost_window,training_table,curriculum_report
from brass_ai.recovery.history import compact
from brass_ai.recovery.league import train
from brass_ai.recovery.ordinary import plan_collection,sampling_report
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import write_json
from brass_ai.step5.policy import NeuralPolicy


def recipe():return read(SOURCE_ROOT/'configs/step5/joint7-latest.json')


class RecipeTests(unittest.TestCase):
    def test_joint7_pin_agrees_with_completed_exported_lineage(self):
        origin=recipe()['imitation_checkpoint']
        for export in ('v4-selfplay-joint-g0150-20261013T173011108713Z','v4-selfplay-joint-g0179-20261014T200020968013Z'):
            root=SOURCE_ROOT/'results'/export
            self.assertEqual(origin,read(root/'state.json')['signature']['imitation_source'])
            il=root/'lineage'/origin['bc_run'];state=read(il/'state.json');report=read(il/'report.json')
            self.assertEqual(state['best']['sha256'],origin['sha256']);self.assertEqual(report['candidate_sha256'],origin['sha256'])
            self.assertEqual(state['epoch'],report['epochs']);self.assertEqual(state['epoch'],state['signature']['epochs'])
            self.assertEqual(digest(state['signature']),origin['bc_signature'])

    def test_defaults_generation_and_migration_are_idempotent(self):
        cfg=recipe();before=read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json')
        self.assertFalse(CHAMPION['enabled']);self.assertEqual(cfg,latest_model_config(before,cfg['imitation_checkpoint']))
        self.assertEqual(cfg,champion_mix_config(time_budget_config(cfg)));self.assertEqual(cfg,validate_config(cfg))
        changes={k for k in set(before)|set(cfg) if before.get(k)!=cfg.get(k)}
        self.assertEqual(changes,{'selfplay_recipe','imitation_checkpoint','curriculum','legacy_boost','champion','champion_opponents','teacher_work_budget','exploration','architecture','parameter_limit','execution'})
        self.assertIsNone(cfg['search_wall_fraction']);self.assertEqual(cfg['teacher_work_budget'],TEACHER_TIME_BUDGET)
        self.assertEqual(TEACHER_TIME_BUDGET['seconds_per_phase'],dict(canal_early=9.,canal_late=12.,rail_early=12.,rail_late=12.))
        self.assertEqual(cfg['execution'],dict(before['execution'],teacher_search_workers_per_turn=8))
        old=copy.deepcopy(cfg);old['insolvency']['max_cost']=old['search']['reward']['insolvency']['max_cost']=.025
        migrated=champion_mix_config(old)
        self.assertEqual(migrated['insolvency']['max_cost'],.075);self.assertNotIn('champion_opponents',migrated)

    def test_conflicting_gate_curriculum_and_wall_limit_fail_closed(self):
        for change in ('gate','mix','old-base','old-boost','wall','bad-quota','bad-duration'):
            with self.subTest(change=change):
                cfg=recipe()
                if change=='gate':cfg['champion']['enabled']=True
                if change=='mix':cfg['champion_opponents']=dict(contract='staged-champion-opponents/1.0.0',stage_iterations=5,max_champion_opponents=2)
                if change=='old-base':cfg['curriculum'][-1].update(selfplay=70,legacy=5)
                if change=='old-boost':cfg['legacy_boost']['stages'][1]['games']=10000
                if change=='wall':cfg['search_wall_fraction']=.5
                if change=='bad-quota':cfg['curriculum'][0]['selfplay']=16
                if change=='bad-duration':cfg['legacy_boost']['stages'][1]['games']=80001
                with self.assertRaises(ValueError):validate_config(cfg)
        # Defensive scheduler checks also refuse to enable the gate on this recipe.
        cfg=recipe();cfg['champion']['enabled']=True;cfg['champion_opponents']={}
        self.assertFalse(champion.enabled(cfg));self.assertFalse(champion_mix.enabled(cfg))

    def test_ordinary_counts_boundaries_and_resume_window(self):
        cfg=recipe();state=dict(games_done=0);activate_boost(state,cfg)
        window=copy.deepcopy(state['legacy_boost'])
        self.assertEqual((window['start_games'],window['until_games']),(0,100000))
        cases={0:(48,80,192),20000:(144,80,96),100000:(192,80,48),
               19920:(120,80,120),99920:(180,80,60)}
        for start,expected in cases.items():
            state['games_done']=start;activate_boost(state,cfg)
            state=json.loads(json.dumps(state));self.assertEqual(boost_window(state,cfg),window)
            actual=Counter(training_table(i,cfg,window)[0] for i in range(start,start+320))
            self.assertEqual(tuple(actual[k] for k in ('selfplay','historical','legacy')),expected)
        self.assertEqual([s['ordinary_games_this_iteration'] for s in curriculum_report(19920,20240,cfg)['stages']],[80,240,0])
        self.assertEqual([s['ordinary_games_this_iteration'] for s in curriculum_report(99920,100240,cfg)['stages']],[0,80,240])
        self.assertEqual(curriculum_report(99999,100000,cfg)['next_stage'],2)
        state['legacy_boost']['start_games']=1
        with self.assertRaisesRegex(ValueError,'rebased'):boost_window(state,cfg)
        with self.assertRaisesRegex(ValueError,'fresh run'):activate_boost(dict(games_done=160),cfg)

    def test_existing_ppo_run_cannot_switch_recipe_even_with_review(self):
        previous=read(SOURCE_ROOT/'results/v4-selfplay-joint-g0179-20261014T200020968013Z/state.json')
        signature=dict(previous['signature'],config=recipe())
        with self.assertRaisesRegex(ValueError,'fresh IL run'):check_signature(previous,signature,reviewed=True)

    def test_planner_latest_snapshot_seats_rotation_and_exact_final_counts(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);path=root/'anchor.pt';path.write_bytes(b'planner-only frozen fixture')
            anchor=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id='older',generation=0)
            cfg=recipe();state=dict(games_done=0,next_seed=cfg['seed'],context_counter=0,history=[anchor],population=[anchor],evaluation_anchor=anchor)
            activate_boost(state,cfg);state['games_done']=100000
            service=Mock();service.models={'older':object(),'current':object()}
            profiles=['v4-a','v4-b','v4-c'];legacy={p:dict(kind='legacy',policy_id=p) for p in profiles}
            before=copy.deepcopy(state)
            with patch.object(champion_mix,'opponents',side_effect=AssertionError('gate mixed')):
                plan=plan_collection(SOURCE_ROOT,root,state,cfg,dict(config=cfg),service,'current',profiles,legacy,legacy,boost_window(state,cfg))
            self.assertEqual(state,before);self.assertEqual(plan['composition'],dict(selfplay=192,historical=80,legacy=48))
            self.assertEqual(plan['context_counter'],896);self.assertIsNone(plan['incumbent']);self.assertEqual(plan['mix_receipts'],[])
            for job in plan['jobs']:
                kind=job['provenance']['kind'];policies=job['policies'];rotation=job['rotation']
                learner=policies[0];self.assertEqual(learner['model_id'],'current')
                self.assertEqual(policies.count(learner),4 if kind=='selfplay' else 1)
                self.assertEqual(rotation,table_rotation(100000+job['index']))
                self.assertEqual(len(job['focuses']),4 if kind=='selfplay' else 1)
                if kind=='selfplay':self.assertTrue(all(p==learner for p in policies))
            service.register_checkpoint.assert_not_called()


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def examples(self,exploration=None):
        torch.manual_seed(470);model=FourSeatNet(ModelConfig.preset('small'))
        learner=NeuralPolicy(model,focus='mixed',exploration=recipe()['exploration'] if exploration is None else exploration)
        opponent=NeuralPolicy(model,focus='neutral',exploration=.05)
        self.addCleanup(learner.close);self.addCleanup(opponent.close)
        games=[];kinds={}
        for index,kind in enumerate(('selfplay','historical','legacy')):
            g=game();g.update(game_id=kind,setup_group=kind,split='train')
            g['opponents']=[learner.policy_id]*4 if kind=='selfplay' else [learner.policy_id]+[opponent.policy_id if kind=='historical' else 'v4']*3
            for row in g['records']:
                rng=random.Random(index*4+row['seat']);is_learner=kind=='selfplay' or row['seat']==0
                if index==0 and row['seat']==3:row['choices']=row['choices'][:1]
                if kind=='legacy' and not is_learner:
                    row['behavior'].update(kind='external',policy_id='v4');continue
                policy=learner if is_learner else opponent
                chosen,prob,_=policy.select_batch([(row['view'],row['choices'],rng)])[0]
                style,intent=policy.context_features(row['view'],rng)
                row.update(style=style,intent=intent,chosen=chosen,behavior=dict(kind='model',policy_id=policy.policy_id,probability=prob,temperature=1.))
            games.append(g);kinds[kind]=kind
        data=updates.prepared(games,lambda r:r['behavior']['policy_id']==learner.policy_id)
        return model,learner,games,kinds,data

    def test_four_plus_one_plus_one_trajectories_and_real_behavior_verification(self):
        model,policy,games,kinds,data=self.examples()
        sampling=sampling_report(games,data,policy.policy_id,kinds,recipe())
        self.assertEqual([r['learner_trajectories'] for r in sampling['by_composition'].values()],[4,1,1])
        self.assertEqual(sampling['actor_eligible_decisions'],5);self.assertEqual(len(data),6)
        cfg=dict(recipe()['update'],effective_batch_size=2,critic_batch_size=2,microbatch_size=2)
        r=updates.ppo_update(model,torch.optim.AdamW(model.parameters(),lr=1e-5),games,policy,cfg,learner_data=data,sample_groups=kinds)
        usage=r['sample_usage'];self.assertEqual(usage['actor_distinct_used'],5)
        self.assertEqual(usage['actor_presentations'],sum(e['actor_positions'] for e in r['epochs']))
        self.assertEqual(usage['critic_presentations'],sum(e['critic_positions'] for e in r['epochs']))
        report=compact(dict(actor=r,sampling=sampling,curriculum=curriculum_report(0,3,recipe())))
        self.assertEqual(report['actor']['sample_usage'],usage);self.assertEqual(report['sampling'],sampling)

    def test_early_stop_reports_unused_rows_and_rejects_stale_behavior(self):
        model,policy,games,kinds,data=self.examples();cfg=dict(recipe()['update'],effective_batch_size=2,critic_batch_size=2,microbatch_size=2)
        original=updates.policy_measure
        def stop(*args,**kwargs):
            result=original(*args,**kwargs)
            verify=kwargs.get('verify',args[6] if len(args)>6 else False)
            return result if verify else max(.02,result)
        with patch.object(updates,'policy_measure',side_effect=stop):
            r=updates.ppo_update(model,torch.optim.AdamW(model.parameters(),lr=1e-5),games,policy,cfg,learner_data=data,sample_groups=kinds)
        self.assertTrue(r['kl_early_stop']);self.assertEqual(r['optimizer_steps'],1)
        self.assertEqual(r['sample_usage']['actor_unused_eligible'],3)
        model,policy,games,kinds,data=self.examples();data[0][0]['behavior']['probability']*=.5
        with self.assertRaisesRegex(ValueError,'behavior probabilities'):
            updates.ppo_update(model,torch.optim.AdamW(model.parameters()),games,policy,cfg,learner_data=data)

    def test_initialization_is_pinned_il_with_empty_adam_zero_progress_and_no_gate(self):
        class StopAfterInitialization(Exception):pass
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);il,model,cfg=imitation_fixture(directory);run=directory/'new-run'
            with ExitStack() as stack:
                for name in ('initialize','schedule','publish_best'):
                    stack.enter_context(patch.object(champion,name,side_effect=AssertionError('Gate called')))
                stack.enter_context(patch('brass_ai.recovery.demonstrations.fresh_imitation',side_effect=AssertionError('Repeated IL')))
                stack.enter_context(patch('brass_ai.recovery.league.InferenceService',side_effect=StopAfterInitialization))
                with self.assertRaises(StopAfterInitialization):train(SOURCE_ROOT,run,cfg,imitation_run=il,fresh=True)
                before=(run/'state.json').read_bytes()
                with self.assertRaises(StopAfterInitialization):train(SOURCE_ROOT,run,cfg,imitation_run=il)
                self.assertEqual((run/'state.json').read_bytes(),before)
            state=read(run/'state.json');initial,raw=load_checkpoint(run/'initial.pt')
            self.assertEqual(raw['optimizer']['state'],{});self.assertEqual(raw['step'],0)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,initial.state_dict()[k],rtol=0,atol=0)
            self.assertEqual((state['iteration'],state['games_done'],state['context_counter']),(0,0,0))
            self.assertEqual(state['next_seed'],cfg['seed']);self.assertEqual(state['legacy_boost']['start_games'],0)
            self.assertEqual(len(state['population']),1);self.assertEqual(state['origin']['sha256'],cfg['imitation_checkpoint']['sha256'])
            self.assertFalse(any(k in state for k in ('champion','champion_gate','champion_mix')))
            self.assertFalse((run/'best.pt').exists())
            with self.assertRaisesRegex(ValueError,'--fresh'):train(SOURCE_ROOT,run,cfg,imitation_run=il,fresh=True)
            for key in ('champion','champion_gate','champion_mix'):
                bad=copy.deepcopy(state);bad[key]={}
                with self.assertRaisesRegex(ValueError,'forbidden champion'):check_signature(bad,state['signature'])

    def test_missing_or_wrong_il_is_rejected_without_creating_a_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);il,_,cfg=imitation_fixture(directory)
            with self.assertRaisesRegex(ValueError,'requires the pinned'):train(SOURCE_ROOT,directory/'missing',cfg)
            bad=copy.deepcopy(cfg);bad['imitation_checkpoint']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'Bound reference differs'):train(SOURCE_ROOT,directory/'wrong',bad,imitation_run=il)
            (il/'epoch-0001.pt').unlink()
            with self.assertRaises(FileNotFoundError):train(SOURCE_ROOT,directory/'gone',cfg,imitation_run=il)
            self.assertFalse(list(directory.glob('*/initial.pt')))


if __name__=='__main__':unittest.main()
