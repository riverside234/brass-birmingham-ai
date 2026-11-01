"""Trained main-model restart, manual scheduling and NEW ownership/VP heads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.batching import model_id
from brass_ai.recovery import fresh_selfplay as F,trained_startup as T,neutral_training as N
from brass_ai.recovery import gae,exploration as E,panel_gates as panels,rating_coverage,v4_share
from brass_ai.recovery import quality_pool as Q,ratings,local_scoring_auxiliary as aux
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_fresh_neutral_selfplay import recipe,source_fixture
import test_fresh_neutral_selfplay as fresh_tests


def config():
    c=T.configure(N.configure(recipe()));c[aux.KEY]=aux.fresh_config(c[aux.KEY]);return c


class Startup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_explicit_settings_immediate_branching_and_preserved_recipe(self):
        old=N.configure(recipe());before=copy.deepcopy(old);c=validate_config(T.configure(old))
        self.assertEqual(old,before)
        self.assertEqual({k:v for k,v in c.items() if k not in (T.KEY,'rehearsal_weight','exploration_rollout','advantage')},
            {k:v for k,v in old.items() if k not in (T.KEY,'rehearsal_weight','exploration_rollout','advantage')})
        settings=E.effective(c,E.initial(c,0),0)
        self.assertEqual(settings['branching_status'],'active');self.assertEqual(settings['branching'],E.NEUTRAL_BRANCH)
        self.assertEqual(settings['branching']['rate_per_decision'],.01)
        self.assertEqual(settings['branching']['move_distribution'],dict(temperature_1=.6,temperature_2=.3,exploration_q=.1))
        self.assertEqual(settings['branching']['skip_opening_decisions'],3);self.assertEqual(settings['opening'],E.THREE_OPENING)
        fresh_tests.FreshNeutral().test_branch_components_marginal_fallback_and_postfork_three_moves()
        independent=copy.deepcopy(c);independent.pop(aux.KEY)
        self.assertEqual(T.configure(independent),independent)  # No companion artifact/config is a gate prerequisite.
        for change in (lambda x:x[T.KEY].update(authority='inferred'),lambda x:x['advantage'].update(**{'lambda':1.}),
            lambda x:x['learning_rate_warmup'].update(selfplay_steps=10000),lambda x:x['exploration_rollout'].update(branching_after_iterations=20),
            lambda x:x.update(rehearsal_weight=.1),lambda x:x[T.KEY].update(auxiliary_heads='required-trained')):
            bad=copy.deepcopy(c);change(bad)
            with self.assertRaises(ValueError):validate_config(bad)

    def test_new_run_exact_main_weights_fresh_heads_restart_and_manual_planning(self):
        from brass_ai.recovery import league,legacy_search
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.learning_rate import KEY as WARMUP
        from brass_ai.recovery.lr_controller import KEY as LR
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);source,state,il,origin,seed=source_fixture(root)
            before={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
            run=root/'artifacts/trained-start';output='configs/step5/trained-start.json'
            result=F.prepare(root,source,run,output,trained_restart=True)
            self.assertEqual(result['auxiliary_heads'],'fresh');self.assertFalse(result['rehearsal_enabled'])
            self.assertEqual(result['qualification'],'manual_override_not_measured')
            self.assertEqual(F.prepare(root,source,run,output,trained_restart=True),result)
            with self.assertRaisesRegex(ValueError,'another source/config'):F.prepare(root,source,run,output,no_rehearsal=True)
            c=read(root/output);manifest=F.load_manifest(root,run,c)
            write_json(root/'training/opponents/pool.json',dict(training=[f'bot{i}' for i in range(5)]))
            phases=[]
            class Initialized(Exception):pass
            def stop(model,cfg,**kwargs):
                p=TrainingPhases(model,cfg,**kwargs);phases.append(p);raise Initialized()
            with (patch.object(league,'admission'),patch.object(legacy_search,'legacy_spec',side_effect=lambda root,p,*a,**k:dict(kind='legacy',policy_id=p,search_free=True)),
                patch.object(ratings,'check_dependencies'),patch('brass_ai.recovery.demonstrations.completed_imitation',return_value=(il,origin)),
                patch('brass_ai.recovery.actor_value.TrainingPhases',side_effect=stop),
                patch('brass_ai.recovery.domino_auxiliary.Diversity',side_effect=AssertionError('DOMiNO started'))):
                for fresh in (True,False):
                    with self.assertRaises(Initialized):league.train(root,run,c,iterations=1,imitation_run='artifacts/il',fresh=fresh)
            saved=read(run/'state.json');model,raw=load_checkpoint(run/'initial.pt')
            old,_=load_checkpoint(root/seed['path'])
            self.assertTrue(_same_payload(model.state_dict(),old.state_dict()))
            self.assertEqual((raw['step'],saved['iteration'],saved['games_done']),(0,0,0))
            self.assertEqual(raw['optimizer']['state'],{});group=raw['optimizer']['param_groups'][0]
            self.assertEqual(group[WARMUP]['completed_steps'],0);self.assertEqual(group[WARMUP]['steps'],100)
            self.assertEqual(group[LR]['multiplier'],1.)
            companion=aux.Companion(model,c[aux.KEY],raw=raw)
            new=aux.Companion(model,c[aux.KEY],raw=dict(model_id=model_id(model)))
            self.assertTrue(_same_payload(companion.heads.state_dict(),new.heads.state_dict()))
            self.assertEqual(companion.scale,0.);self.assertEqual(companion.age,0)
            self.assertEqual(companion.retained_steps,dict(ppo=0,ppg=0))
            self.assertTrue(all(not o.state for o in companion.optimizers.values()))
            for age in (0,1,2,6,100):
                companion.age=age;self.assertEqual(companion.scale,0.)  # Time alone never trains the new heads.
            receipt=saved['auxiliary_initialization']
            self.assertEqual(receipt['kind'],'fresh');self.assertEqual(receipt['model_id'],raw['model_id'])
            self.assertEqual(receipt['initial_bundle_identity'],raw['training'][aux.KEY]['identity'])
            aux.verify_initialization(saved,raw,c[aux.KEY])
            missing=copy.deepcopy(raw);missing['training'].pop(aux.KEY)
            with self.assertRaisesRegex(ValueError,'initialization/checkpoint'):aux.verify_initialization(saved,missing,c[aux.KEY])
            bad=copy.deepcopy(saved);bad['auxiliary_initialization']['model_id']='0'*64
            with self.assertRaises(ValueError):aux.verify_initialization(bad,raw,c[aux.KEY])
            for phase in phases:
                self.assertEqual(phase.state,gae.initial(c['advantage']));self.assertFalse(phase.optimizer.state)
                self.assertEqual(phase.state['active_lambda'],.98);self.assertEqual(phase.state['healthy_streak'],0)
                self.assertEqual(phase.state['stage_evidence']['entries'],[])
                self.assertEqual(phase.state['table_enabled'],dict(historical=False,legacy=False))
            self.assertNotIn('domino',raw['training']);self.assertNotIn('policy_anchor',saved)
            self.check_panels_and_plan(root,run,c,saved,model)
            self.assertEqual(before,{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError,'progress'):F.prepare(root,source,run,output,trained_restart=True)
            print('Trained-startup initialization smoke: exact g1290 main tensors, fresh heads/Adam, GAE=.98; '
                '160/154/6 tables; zero qualification games; restart verified',flush=True)

    def check_panels_and_plan(self,root,run,c,state,model):
        from brass_ai.recovery.ordinary import plan_collection
        from brass_ai.recovery.plackett_luce import IL
        gates=panels.prepare(root,state,c)
        self.assertEqual(gates,state['ratings']['panel_gates'])
        self.assertEqual(panels.latest_evidence(root,state,c),dict.fromkeys(panels.NAMES))
        self.assertEqual(rating_coverage.prepare(root,state,c),state['ratings']['coverage'])
        self.assertEqual(v4_share.prepare(root,state,c),state['v4_share_gate'])
        self.assertIsNone(state['v4_share_gate']['last_panel'])
        for name,t in gates['retired'].items():
            self.assertTrue(T.manual(t));self.assertFalse(t['measured']);self.assertNotIn('win_share',t)
            panels.verify_evidence(root,state['ratings'],name,t)
            with self.assertRaises(ValueError):panels.validate_trigger(dict(t,win_share=.99))
        bad=copy.deepcopy(state);bad['ratings']['panel_gates']['retired']['il-panel']['authorization']['source_model_id']='0'*64
        with self.assertRaises(ValueError):panels.prepare(root,bad,c)
        self.assertEqual(state['ratings']['agents'][IL]['checkpoint']['model_id'],state['evaluation_anchor']['model_id'])
        spec=ratings.slot_spec(root,run,state['ratings'],dict(state['history'][0],generation=10),c)
        self.assertEqual(len(spec['lineups']),128);self.assertEqual({r['kind'] for r in spec['lineups']},{'rating'})
        self.assertTrue(spec['startup_reference_fallback']);self.assertTrue(spec['coverage']['active'])
        report=ratings.panel_report([],spec['candidate'],None,c['rating']['seed'],gates=gates)
        for r in report['panels'].values():
            self.assertEqual(r['games'],0);self.assertIsNone(r['learner_win_share']);self.assertIsNone(r['win_share_setup_bootstrap_95'])
        bots={f'bot{i}':dict(kind='legacy',policy_id=f'bot{i}',search_free=True) for i in range(5)}
        cfg=copy.deepcopy(c);cfg['local_scoring_auxiliary']['mode']='off'  # Planner fixture does not launch native recorder.
        service=Mock();service.models={m['model_id']:None for m in state['quality_pool']['members']}
        settings=E.effective(c,state['exploration'],0)
        def plan(s):return plan_collection(root,run,s,cfg,dict(config=c),service,model_id(model),list(bots),bots,bots,None,exploration=settings)
        a=plan(state);b=plan(json.loads(json.dumps(state)))
        self.assertEqual(a,b);self.assertEqual(a['composition'],dict(selfplay=160,historical=154,legacy=6))
        self.assertEqual(a['quality_planning']['il_share'],0.)
        self.assertEqual(state['quality_pool']['next_planning']['il_share'],0.)
        self.assertEqual(len(a['quality_planning']['members']),len(state['quality_pool']['members']))
        self.assertTrue(T.manual(a['quality_planning']['il_gate']))
        counts={m['model_id']:0 for m in state['quality_pool']['members']}
        for table in a['quality_tables'].values():
            for o in table['opponents']:
                self.assertEqual(o['source'],'scored');counts[o['model_id']]+=1
        self.assertEqual(sum(counts.values()),308);self.assertGreater(counts[state['evaluation_anchor']['model_id']],0)
        for job in a['jobs']:
            self.assertEqual(set(job['focuses'].values()),{'neutral'})
        self.assertTrue(all(m['S']==5. and m['C']==10. for m in state['quality_pool']['members']))

    def test_gae_98_fresh_evidence_twenty_observations_and_guards(self):
        from test_ppg_window import tick
        c=config()['advantage'];initial=gae.initial(c)
        self.assertEqual(initial['active_lambda'],.98);self.assertEqual(initial['value_iterations'],0)
        self.assertEqual(initial['table_evidence']['entries'],[])
        for mode in ('healthy','no_work','rating_block','economic','kl'):
            s=copy.deepcopy(initial)
            for g in range(1,21):
                s,r=tick(s,c,steps=0 if mode=='no_work' else (6 if g in (1,8,15) else 0),
                    healthy=not(mode=='economic' and g==10),trip=mode=='kl' and g==10,rating=mode!='rating_block')
                if g<20:self.assertEqual(s['active_lambda'],.98)
            self.assertEqual(s['active_lambda'],.95 if mode=='healthy' else .98)
            self.assertFalse(any(s['table_enabled'].values()))
            gae.validate_state(json.loads(json.dumps(s)),c,20)

    def test_optimizer_step_warmup_restarts_exactly_and_never_imports_Adam(self):
        from brass_ai.recovery import learning_rate as lr
        from brass_ai.recovery.resume import apply_optimizer_config
        c=config();p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=c['learning_rate'],weight_decay=c['weight_decay'])
        apply_optimizer_config(opt,c);self.assertFalse(opt.state)
        for _ in range(37):p.grad=torch.ones_like(p);lr.step(opt,ppo=True)
        saved=copy.deepcopy(opt.state_dict());q=torch.nn.Parameter(p.detach().clone())
        other=torch.optim.AdamW([q]);other.load_state_dict(saved);apply_optimizer_config(other,c)
        self.assertEqual(lr.report(other)['completed_steps'],37)
        self.assertEqual(lr.report(other)['next_lr'],lr.report(opt)['next_lr'])
        self.assertAlmostEqual(lr.rate(saved['param_groups'][0][lr.KEY],1),c['learning_rate']*.1)
        for _ in range(63):
            p.grad=torch.ones_like(p);q.grad=torch.ones_like(q);lr.step(opt,ppo=True);lr.step(other,ppo=True)
        self.assertTrue(torch.equal(p,q));self.assertEqual(lr.report(other)['completed_steps'],100)
        self.assertAlmostEqual(lr.report(other)['effective_lr'],c['learning_rate'])

    def test_isolated_ppo_ppg_smoke_with_startup_gae_and_fresh_heads(self):
        # Reuse the bounded authored-data smoke, replacing only its recipe.
        from brass_ai.recovery import neutral_training
        from test_local_scoring import annotate
        from test_critic_descriptors import partition
        original=neutral_training.configure
        required=config();phase_type=fresh_tests.TrainingPhases;phases=[]
        def configure(c):
            c=original(c)
            for k in (aux.KEY,'games_per_iteration','rating','panel_gates','rating_coverage','v4_share_gate'):
                c[k]=copy.deepcopy(required[k])
            return T.configure(c)
        def phase(model,c,**kwargs):
            p=phase_type(model,c,**kwargs)
            p.local_scoring=aux.Companion(model,c[aux.KEY],raw=dict(model_id=model_id(model)));phases.append(p)
            return p
        def labeled_partition(*args,**kwargs):
            learner,snapshot,rest=partition(*args,**kwargs);annotate(learner);annotate(snapshot);return learner,snapshot,rest
        with (patch.object(neutral_training,'configure',side_effect=configure),patch.object(fresh_tests,'TrainingPhases',side_effect=phase),
              patch('test_critic_descriptors.partition',side_effect=labeled_partition)):
            fresh_tests.FreshNeutral().test_isolated_neutral_ppo_ppg_smoke()
        self.assertEqual(phases[0].state['active_lambda'],.98);self.assertEqual(phases[0].local_scoring.scale,0.)
        a=phases[0].local_scoring
        self.assertEqual(a.fitting_steps,phases[0].value_report['optimizer_steps'])
        self.assertGreater(a.report()['warmup']['distinct_fitting_games'],0)
        print('Auxiliary smoke:',a.report()['warmup'],flush=True)


if __name__=='__main__':unittest.main()
