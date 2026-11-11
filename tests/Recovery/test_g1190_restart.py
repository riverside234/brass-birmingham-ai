"""The g1190 seed restart: restored placement, the restart stage schedule, Stage A early stop, PPG floor and the built run."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.learning import post_transfer_reward as P
from brass_ai.learning.reward import specification
from brass_ai.recovery import branch_restart as B,search_value_schedule as V,training_revision_g1491 as R
from brass_ai.recovery import lr_controller as L,quality_pool
from brass_ai.recovery.collector_runtime import recipe
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.teacher_phase import plan_jobs
from brass_ai.recovery.utility import reward_spec
from brass_ai.rust_backend import lab_shadow_config as C,lab_teacher_activation as A
from brass_ai.step5.league import write_json

EXPORT=ROOT/'results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z'
LATER=ROOT/'results/joint7-v11-2-g1290-restart-g1290-g1500-20261110T232721030490Z'
SOURCE='artifacts/step5/joint7-v11-2-g1290-restart-g1290/'
PHASES=('canal_early','canal_late','rail_early','rail_late')


def committed():
    state=read(EXPORT/'state.json')
    return state,recipe(state['signature']['config'])


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state,cls.before=committed()
        cls.g1490=B.rollback_recipe(cls.state['signature']['config'],cls.state,share_target=.75)
        cls.after=B.seed_recipe(cls.state['signature']['config'],cls.state,share_target=.75)

    def test_it_is_the_g1490_recipe_with_placement_and_the_restart_schedule(self):
        validate_config(self.after)
        self.assertEqual({k for k in self.after if self.after[k]!=self.g1490[k]},{'reward','search','teacher_phase'})
        self.assertEqual(self.after['reward'],dict(win=1.,placement=.1,vp=.001,vp_cap=180,vp_timing=P.VP_TIMING))
        self.assertEqual(self.after['insolvency'],P.ZERO_COST)
        spec=specification(P.RESTORED,P.ZERO_COST)
        self.assertEqual(reward_spec(self.after),spec)
        self.assertEqual((spec['version'],spec['placement'],spec['insolvency']['event_cost']),('brass-combined-return/3.0.0',.1,0.))
        old,new=self.g1490['teacher_phase'],self.after['teacher_phase']
        self.assertEqual({k for k in new if new[k]!=old[k]},{'search','fast_search','search_value'})
        self.assertEqual((new['search']['reward'],new['fast_search']['reward']),(spec,spec))
        self.assertEqual(new['search_value']['schedule'],V.restart_schedule())
        self.assertEqual({k for k in new['search_value'] if new['search_value'][k]!=old['search_value'][k]},{'schedule','head'})
        self.assertEqual(new['search_value']['head'],dict(old['search_value']['head'],hidden=256))
        self.assertTrue(P.is_revised(self.after) and P.restored(self.after) and not P.restored(self.g1490))

    def test_it_projects_to_the_reviewed_shadow_recipe(self):
        self.assertTrue(C.is_shadow(self.after))
        self.assertFalse(A.is_on(self.after))
        self.assertEqual(P.previous(self.after)['reward'],P.BEFORE)

    def test_other_schedules_and_rewards_are_not_reviewed(self):
        for key,value in (('stage_a_iterations',40),('stage_b_iterations',20),('stage_c_iterations',20),
                          ('stage_a_fit','fixed'),('ppg_beta_floor_after_stage_a',.25)):
            bad=copy.deepcopy(self.after);bad['teacher_phase']['search_value']['schedule'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(self.after);bad['teacher_phase']['search_value']['head']['hidden']=32
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(self.g1490);bad['teacher_phase']['search_value']['head']['hidden']=256
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(self.after);bad['reward']['placement']=.2
        with self.assertRaises(ValueError):validate_config(bad)
        # The restored reward is only reached from the post-transfer one.
        with self.assertRaisesRegex(ValueError,'starts from the post-transfer'):P.restore_placement(P.previous(self.after))

    def test_jobs_wait_for_readiness_then_search_with_the_restored_reward(self):
        state=dict(self.state,signature=dict(self.state['signature'],config=self.after))
        jobs,_=plan_jobs(self.after,state,model_id='m'*64,export_hash='a'*64,epoch=1491,ready=True)
        self.assertEqual([j['split'] for j in jobs].count('train'),32)
        self.assertTrue(all(j['search']['reward']==reward_spec(self.after) for j in jobs))
        self.assertTrue(all(j['teacher_revision']['short_labels'] is False for j in jobs))
        self.assertEqual(plan_jobs(self.after,state,model_id='m'*64,export_hash='a'*64,epoch=1491,ready=False)[0],[])


class Charged(unittest.TestCase):
    """The second restart: the first one with the insolvency charge kept and 256 teacher rows."""

    @classmethod
    def setUpClass(cls):
        cls.state,cls.before=committed()
        cls.first=B.seed_recipe(cls.state['signature']['config'],cls.state,share_target=.75)
        cls.after=B.seed_recipe(cls.state['signature']['config'],cls.state,share_target=.75,revision=B.SEED_CHARGED)

    def signature(self,config):
        return dict(self.state['signature'],config=config)

    def test_it_is_the_first_restart_with_the_charge_and_256_rows(self):
        validate_config(self.after)
        self.assertEqual({k for k in self.after if self.after[k]!=self.first[k]},{'insolvency','search','teacher_phase'})
        self.assertEqual(self.after['reward'],dict(win=1.,placement=.1,vp=.001,vp_cap=180,vp_timing=P.VP_TIMING))
        self.assertEqual(self.after['insolvency'],dict(event_cost=.0025,max_cost=.025))
        spec=specification(P.RESTORED,P.OLD_COST)
        self.assertEqual(reward_spec(self.after),spec)
        self.assertEqual((spec['version'],spec['placement'],spec['insolvency']),('brass-combined-return/3.0.0',.1,P.OLD_COST))
        old,new=self.first['teacher_phase'],self.after['teacher_phase']
        self.assertEqual({k for k in new if new[k]!=old[k]},{'search','fast_search',R.KEY,'term'})
        self.assertEqual((new['search']['reward'],new['fast_search']['reward']),(spec,spec))
        self.assertEqual({k for k in new['term'] if new['term'][k]!=old['term'][k]},{'rows_per_step'})
        self.assertEqual((old['term']['rows_per_step'],new['term']['rows_per_step']),(72,256))
        self.assertEqual({k:new[k] for k in ('tables','buffer','search_value')},{k:old[k] for k in ('tables','buffer','search_value')})
        self.assertEqual(new['search_value']['schedule'],V.restart_schedule())
        self.assertEqual(new[R.KEY]['variant'],R.WIDE_RESTART)
        self.assertEqual(self.after['update']['epochs'],2)
        self.assertTrue(P.is_revised(self.after) and P.charged(self.after) and P.restored(self.after) and not P.charged(self.first))

    def test_it_projects_to_the_reviewed_shadow_recipe(self):
        self.assertTrue(C.is_shadow(self.after))
        self.assertFalse(A.is_on(self.after))
        previous=P.previous(self.after)
        self.assertEqual((previous['reward'],previous['insolvency']),(P.BEFORE,P.OLD_COST))
        from brass_ai.recovery import entropy_schedule
        entropy_schedule.validate(self.after)

    def test_the_charge_and_row_count_are_exact(self):
        for cost in (P.ZERO_COST,dict(event_cost=.005,max_cost=.025),dict(event_cost=.0025,max_cost=.05)):
            bad=copy.deepcopy(self.after);bad['insolvency']=cost
            with self.subTest(cost=cost),self.assertRaises(ValueError):validate_config(bad)
        for rows in (72,2048,255):
            bad=copy.deepcopy(self.after);bad['teacher_phase']['term']['rows_per_step']=rows
            with self.subTest(rows=rows),self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(self.first);bad['teacher_phase']['term']['rows_per_step']=256
        with self.assertRaises(ValueError):validate_config(bad)
        with self.assertRaisesRegex(ValueError,'starts from the restored-placement'):P.keep_charge(self.state['signature']['config'])
        self.assertEqual(P.keep_charge(self.after),self.after)
        with self.assertRaisesRegex(ValueError,'Unknown seed restart revision'):
            B.seed_recipe(self.state['signature']['config'],self.state,share_target=.75,revision='g1190-other')

    def test_jobs_search_with_the_kept_charge(self):
        state=dict(self.state,signature=self.signature(self.after))
        jobs,_=plan_jobs(self.after,state,model_id='m'*64,export_hash='a'*64,epoch=1491,ready=True)
        self.assertEqual(sorted(j['split'] for j in jobs).count('train'),32)
        self.assertEqual(len(jobs),36)
        self.assertTrue(all(j['search']['reward']['insolvency']==P.OLD_COST for j in jobs))
        self.assertTrue(all(j['teacher_revision']['short_labels'] is False for j in jobs))

    def test_calibration_then_teaching_on_and_it_is_never_adopted_inside_a_run(self):
        on=A.configure(self.after,.75,revise_stages=True)
        self.assertTrue(A.is_on(on) and P.charged(on))
        self.assertEqual(on['teacher_phase']['term']['rows_per_step'],256)
        self.assertTrue(C.signature_transition(self.signature(self.after),self.signature(on)))
        self.assertFalse(C.signature_transition(self.state['signature'],self.signature(self.after)))
        # Neither restart can turn into the other, nor change rows, inside a run.
        self.assertFalse(C.signature_transition(self.signature(self.first),self.signature(self.after)))
        self.assertFalse(C.signature_transition(self.signature(self.after),self.signature(self.first)))
        # The row count is the variant's own: any other value fails the recipe's validation.
        slow=copy.deepcopy(on);slow['teacher_phase']['term']['rows_per_step']=72
        with self.assertRaises(ValueError):validate_config(slow)


class Transitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state,cls.before=committed()
        cls.shadow=B.seed_recipe(cls.state['signature']['config'],cls.state,share_target=.75)
        cls.on=A.configure(cls.shadow,.75,revise_stages=True)

    def signature(self,config):
        return dict(self.state['signature'],config=config)

    def test_never_adopted_inside_a_run(self):
        g1490=B.rollback_recipe(self.state['signature']['config'],self.state,share_target=.75)
        self.assertFalse(C.approved(dict(self.state,signature=self.signature(g1490)),self.signature(self.shadow)))
        self.assertFalse(C.signature_transition(self.state['signature'],self.signature(self.shadow)))

    def test_calibration_then_teaching_on_and_the_schedule_is_fixed(self):
        self.assertTrue(A.is_on(self.on))
        self.assertEqual(self.on['teacher_phase']['search_value']['schedule'],V.restart_schedule())
        self.assertTrue(C.signature_transition(self.signature(self.shadow),self.signature(self.on)))
        narrow=copy.deepcopy(self.on);narrow['teacher_phase']['search_value']['head']['hidden']=32
        self.assertFalse(C.signature_transition(self.signature(self.on),self.signature(narrow)))
        for key,value in (('stage_b_iterations',20),('ppg_beta_floor_after_stage_a',None)):
            other=copy.deepcopy(self.on);schedule=other['teacher_phase']['search_value']['schedule']
            if value is None:
                schedule.pop(key);schedule.pop('stage_a_fit');schedule['stage_a_iterations']=50
            else:schedule[key]=value
            with self.subTest(key=key):self.assertFalse(C.signature_transition(self.signature(self.on),self.signature(other)))


class Schedule(unittest.TestCase):
    def test_fifty_head_only_iterations_then_fifty_blend_steps(self):
        options=dict(schedule=V.restart_schedule())
        state=dict(stage='A',w=0.,scale=0.,generation=0,history=[],lam=1.)
        V.initialize(state,options)
        self.assertEqual((state['stage_a_iterations'],state['stage_b_iterations']),(0,0))
        for g in range(1,51):V.advance(state,g,{},stage_a=50,stage_b=50)
        self.assertEqual((state['stage'],state['stage_a_iterations']),('B',50))
        weights=[]
        for g in range(51,101):weights.append(V.advance(state,g,{},stage_a=50,stage_b=50)['w'])
        self.assertEqual(weights[:3],[.02,.04,.06]);self.assertEqual(weights[-1],1.)
        self.assertEqual(len(set(weights)),50)
        self.assertEqual((state['stage'],state['scale']),('C',0.))
        V.initialize(state,options)
        bad=dict(state,stage='B',stage_b_iterations=10,w=.5)
        with self.assertRaisesRegex(ValueError,'blend differ'):V.initialize(bad,options)
        self.assertEqual((V.blend_step(options),V.beta_floor(options,'A'),V.beta_floor(options,'B'),V.beta_floor(options,'C')),(.02,None,.5,.5))

    def test_the_earlier_schedules_are_unchanged(self):
        for a in (10,30):
            options=dict(schedule=dict(V.defaults(),stage_a_iterations=a))
            self.assertEqual((V.stage_b_iterations(options),V.blend_step(options),V.stage_a_holdout(options),V.beta_floor(options,'C')),(20,.05,False,None))

    def test_the_first_stage_b_collection_uses_one_small_step(self):
        state,_=committed()
        config=A.configure(B.seed_recipe(state['signature']['config'],state,share_target=.75),.75,revise_stages=True)
        progress=dict(state,search_value=dict(stage=dict(stage='B',w=0.)))
        jobs,_=plan_jobs(config,progress,model_id='m'*64,export_hash='a'*64,epoch=1491,ready=True)
        self.assertEqual({j['search_value']['w'] for j in jobs},{.02})


class StageA(unittest.TestCase):
    """Stage A head fitting with the Stage C held-out early stop."""

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        from test_b025 import model_fixture
        from test_b045 import value_fixture
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.model=model_fixture()
        options=A.configure(read(ROOT/'results/joint7-v11-2-g1290-trained-restart-g1284-20261108T193427364056Z/recorded-config.json'),
            revise_stages=True)['teacher_phase']['search_value']
        options['schedule']=V.restart_schedule();options['head']['hidden']=256
        self.phase=SearchValuePhase(Path(self.tmp.name)/'values',self.model,options,digest(self.model.config.utility))
        self.row,_,_=value_fixture(self.model,generation=1284)
        self.row.update(reward_id=self.phase.reward_id,terminal_utility=self.row['u'])
        for game in range(10):
            rows=[dict(self.row,generation=1284,game_id=f'game-{game}',index=i) for i in range(64)]
            self.phase.buffer.publish(1284,'train',rows,game_id=rows[0]['game_id'])

    def snapshot(self):
        from brass_ai.recovery.retention import _same_payload
        state=copy.deepcopy((self.model.search_value_head.state_dict(),self.phase.companion.snapshot()))
        return lambda:_same_payload(state,(self.model.search_value_head.state_dict(),self.phase.companion.snapshot()))

    def test_intervals_stop_when_the_heldout_loss_rises(self):
        with mock.patch('brass_ai.recovery.search_value_retention.HeldoutLoss',return_value=iter([1.,.9,.95]).__next__):
            result=self.phase.after_ppg(1284,mode='on')
        update=result['update']
        self.assertEqual((update['steps'],update['heldout_loss_after'],update['early_stop_reason']),(16,.9,'heldout_loss_increased'))
        self.assertEqual([i['accepted'] for i in update['intervals']],[True,False])
        self.assertEqual(self.phase.companion.steps,16)
        self.assertTrue(result['retention_holdout']['ready'])
        self.assertEqual(result['stage']['stage_a_iterations'],1)

    def test_a_first_interval_rise_restores_the_head_and_adam(self):
        same=self.snapshot()
        with mock.patch('brass_ai.recovery.search_value_retention.HeldoutLoss',return_value=iter([1.,1.1]).__next__):
            result=self.phase.after_ppg(1284,mode='on')
        self.assertEqual(result['update']['steps'],0);self.assertTrue(same())
        self.assertFalse(result['updated'])

    def test_no_rise_runs_all_four_intervals(self):
        with mock.patch('brass_ai.recovery.search_value_retention.HeldoutLoss',return_value=lambda:1.):
            result=self.phase.after_ppg(1284,mode='on')
        trunk=self.phase.options['trunk']
        self.assertEqual((result['update']['steps'],len(result['update']['intervals'])),
                         (trunk['intervals']*trunk['interval_steps'],trunk['intervals']))
        self.assertEqual(trunk['intervals']*trunk['interval_steps'],self.phase.options['steps_per_iteration'])
        # Held-out games never train.
        held=set(result['retention_holdout']['games'])
        self.assertEqual(len(held),8)

    def test_the_real_heldout_loss_does_not_rise_after_a_kept_interval(self):
        result=self.phase.after_ppg(1284,mode='on')
        update=result['update']
        self.assertLessEqual(update['heldout_loss_after'],update['heldout_loss_before'])

    def test_the_fresh_head_is_256_wide(self):
        from brass_ai.learning.search_value import width
        self.assertEqual(width(self.model),256)
        self.assertEqual(self.model.search_value_head[-1].in_features,256)

    def test_fixed_steps_without_the_restart_schedule(self):
        from test_b025 import model_fixture
        from brass_ai.learning.search_value import width
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        options=copy.deepcopy(self.phase.options);options['schedule']=dict(V.defaults(),stage_a_iterations=30)
        options['head']['hidden']=32
        model=model_fixture()
        phase=SearchValuePhase(Path(self.tmp.name)/'values',model,options,self.phase.reward_id,
            saved=dict(stage=self.phase.machine.state,buffer=self.phase.buffer.payload()))
        self.assertEqual(width(model),32)
        result=phase.after_ppg(1284,mode='on')
        self.assertEqual(result['update']['steps'],options['steps_per_iteration']);self.assertNotIn('intervals',result['update'])


class Floor(unittest.TestCase):
    def phases(self,beta,floor):
        from brass_ai.recovery.actor_value import TrainingPhases
        state,_=committed();config=state['signature']['config']
        phases=TrainingPhases.__new__(TrainingPhases)
        phases.options=dict(value_phase=copy.deepcopy(config['value_phase']));phases.beta=beta;phases.beta_floor=floor
        return phases

    def test_the_floor_rises_to_one_half_past_stage_a(self):
        phases=self.phases(.25,.5)
        options=phases.floor_options()
        self.assertEqual((options['beta_clone_min'],phases.options['value_phase']['beta_clone_min']),(.5,.25))
        plain=self.phases(.25,None);self.assertIs(plain.floor_options(),plain.options['value_phase'])
        from brass_ai.recovery import ppg_g210
        report=dict(scheduled=True,kl_guard_tripped=False,optimizer_steps=10,probe_kl=0.)
        beta,_,feedback=ppg_g210.beta_after_phase(.5,[],report,options)
        self.assertEqual((beta,feedback['minimum'],feedback['reason']),(.5,.5,'at_minimum'))
        beta,_,_=ppg_g210.beta_after_phase(.5,[],report,plain.floor_options())
        self.assertEqual(beta,.25)

    def test_the_search_value_iteration_sets_the_floor_for_its_stage(self):
        from brass_ai.recovery.search_value_iteration import SearchValueIteration
        it=SearchValueIteration.__new__(SearchValueIteration)
        state,_=committed()
        it.config=B.seed_recipe(state['signature']['config'],state,share_target=.75)
        it.mode='on';it.generation=1491;it._publication=None
        phases=type('Phases',(),{})()
        for stage,floor in (('A',None),('B',.5),('C',.5)):
            it.phase=type('Phase',(),dict(machine=type('Machine',(),dict(state=dict(stage=stage)))()))()
            it.prepare_ppg(phases,1492)
            self.assertEqual(phases.beta_floor,floor)


class SeedUtility(unittest.TestCase):
    """Restart keeps the learned utility correction and its full-game coordinate."""

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_charged_restart_keeps_nonzero_utility_head_predictions_exactly(self):
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
        state,before=committed()
        config=dict(B.seed_recipe(before,state,share_target=.05,revision=B.SEED_CHARGED),model_size='small')
        old=specification(P.BEFORE,P.OLD_COST);new=reward_spec(config)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'version'})
        torch.manual_seed(1190)
        seed=UtilityNet(utility_config(ModelConfig.preset('small'),old))
        with torch.no_grad():
            for parameter in seed.utility_head.parameters():parameter.uniform_(-.1,.1)
        context=torch.randn(4,seed.config.hidden)
        mapping=torch.tensor([[0,1,2,3],[3,0,1,2],[2,3,0,1],[1,2,3,0]])
        heads=(torch.softmax(torch.randn(4,4),-1),torch.tensor([[1.,2.,3.,4.]]*4),
               torch.tensor([[220.,170.,80.,0.]]*4))
        remaining=torch.tensor([[.025,.01,.0025,0.]]*4);embeddings=torch.randn(4,4,32)
        def predict(model):
            with torch.inference_mode():return model.predict_utility(context,mapping,heads,remaining,embeddings)
        expected=predict(seed)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            path=Path(tmp)/'seed1190.pt';save_checkpoint(path,seed,None,step=1190)
            restarted,_=B.seed_model(path,config)
            for name,tensor in seed.state_dict().items():self.assertTrue(torch.equal(tensor,restarted.state_dict()[name]),name)
            torch.testing.assert_close(predict(restarted),expected,rtol=0,atol=0)
            checkpoint=Path(tmp)/'generation-1491.pt';save_checkpoint(checkpoint,restarted,None,step=1491)
            loaded,raw=load_checkpoint(checkpoint)
            self.assertEqual(raw['step'],1491)
            torch.testing.assert_close(predict(loaded),expected,rtol=0,atol=0)

    def test_charged_restart_targets_match_g1190_in_both_eras_and_at_the_cost_cap(self):
        from brass_ai.learning.reward import replay_returns,terminal_utility,CostLedger
        from test_insolvency_recovery import row
        from test_era_vp_reward import EraVP
        old=specification(P.BEFORE,P.OLD_COST);new=specification(P.RESTORED,P.OLD_COST)
        records=[]
        for era in ('CanalEra','RailEra'):
            for round_ in range(1,8):
                for record in (row(0,round=round_,era=era),row(0,round=round_,era=era,phase='debt',kind='liquidate',debts={0:5})):
                    for player in record['view']['players']:player['vp']=0 if era=='CanalEra' else 50-10*player['seat']
                    records.append(record)
        result=EraVP().result()
        before,old_summary=replay_returns(records,result,old)
        after,new_summary=replay_returns(records,result,new)
        self.assertEqual(old_summary['cost'],new_summary['cost'])
        self.assertAlmostEqual(new_summary['cost'][0],.025)
        self.assertEqual(old_summary['utility'],new_summary['utility'])
        np.testing.assert_array_equal(terminal_utility(result,old),terminal_utility(result,new))
        for a,b in zip(before,after):
            self.assertEqual(a['remaining'],b['remaining'])
            np.testing.assert_array_equal(a['utility'],b['anchored_utility'])
            np.testing.assert_allclose(np.asarray(b['utility'])+b['vp_paid'],a['utility'],rtol=0,atol=1e-15)
        ledger=CostLedger(new);ledger.charge(0);self.assertEqual(ledger.spent[0],.0025)
        for _ in range(20):ledger.charge(0)
        self.assertEqual(ledger.spent[0],.025)


class Build(unittest.TestCase):
    """A small real run tree built from the exported g1491 state with a g1190-style seed, then loaded back."""

    def setUp(self):
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.recovery.local_scoring_auxiliary import Companion,KEY as COMPANION,bundle_id
        temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(temp.cleanup)
        base=Path(temp.name);self.source=base/'source';self.run=base/'restart'
        new=self.source.relative_to(ROOT).as_posix()+'/'
        state=read(EXPORT/'state.json')
        state={k:B.rewrite(v,SOURCE,new) if k!='signature' else v for k,v in state.items()}
        config=state['signature']['config']
        torch.manual_seed(7)
        old_spec=specification(P.BEFORE,P.OLD_COST)
        model=UtilityNet(utility_config(ModelConfig.preset(config['model_size']),old_spec))
        model.utility_calibration=dict(ready=True,reward_id=digest(old_spec))
        identity=model_id(model)
        slot=next(s for s in state['ratings']['slots'] if s['generation']==1400)
        keep={slot[k]['path'] for k in ('records','report','spec','panels','bootstrap')}
        archive=state['ratings']['archive']
        for where,receipt in B.receipts(state,new):
            if where.startswith(('/il_kl_limit',)+B.RECORD_PREFIXES) or receipt['path'] in keep:continue
            path=ROOT/receipt['path'];path.parent.mkdir(parents=True,exist_ok=True)
            if receipt is archive:path.write_bytes((EXPORT/receipt['path'][len(new):]).read_bytes())
            else:path.write_bytes(('stand-in '+receipt['path']).encode());receipt['sha256']=sha(path)
        entry=next(e for e in state['history'] if e['generation']==1190);entry['model_id']=identity
        agent=state['ratings']['agents']['generation-001190']['checkpoint'];agent['model_id']=identity
        companion=Companion(model,config[COMPANION],raw=dict(model_id=identity)).payload(0)
        companion.pop('identity')
        companion.update(generation=1190,age=1190,retained_steps=dict(ppo=30000,ppg=30000),
            warmup=dict(companion['warmup'],generation=1190,qualifying_iterations=900,fitting_steps=30000,released_at=2,encoder_stage=5))
        companion['identity']=bundle_id(companion)
        old='artifacts/step5/joint7-v11-2-g1290-trained-restart'
        path=ROOT/agent['path']
        save_checkpoint(path,model,None,step=1190,training=dict(mode='step5-population-ppo',recipe='ppo/old',parent=dict(generation=1189),
            initialization='random-lineage',fresh_run=old,signature='0'*64,quality_pool_digest='1'*64,
            exploration=dict(state['exploration'],generation=1190),policy_anchor=dict(active=None),
            startup_initialization=state['startup_initialization'],**{COMPANION:companion}))
        agent['sha256']=sha(path)
        write_json(self.source/'state.json',state)
        snapshot=base/'export';snapshot.mkdir()
        write_json(snapshot/'state.json',state)
        (snapshot/'iteration-1490').mkdir()
        (snapshot/'iteration-1490/report.json').write_bytes((EXPORT/'iteration-1490/report.json').read_bytes())
        write_json(snapshot/'export-manifest.json',dict(run=new[:-1],committed_generation=1491,files=[
            dict(path='state.json',source=new+'state.json',sha256=sha(snapshot/'state.json')),
            dict(path='iteration-1490/report.json',source=new+'iteration-1490/report.json',sha256=sha(snapshot/'iteration-1490/report.json'))]))
        later=base/'later';(later/'ratings').mkdir(parents=True)
        name='ratings/history-5cb5bb67ff13b24695f1e072b180f22bac015c6197b2e11843a981bb93161ac0.json'
        (later/name).write_bytes((LATER/name).read_bytes())
        write_json(later/'state.json',dict(ratings=dict(archive=dict(path=new+name,sha256=sha(later/name)))))
        write_json(later/'export-manifest.json',dict(run=new[:-1],committed_generation=1500,files=[
            dict(path='state.json',source=new+'state.json',sha256=sha(later/'state.json')),
            dict(path=name,source=new+name,sha256=sha(later/name))]))
        self.snapshot,self.later,self.state,self.model,self.weights=snapshot,later,state,model,agent['path']
        probe=type('Probe',(),dict(metadata=dict(probe_hash=state['il_kl_limit']['probe']['probe_hash']),
            probabilities=lambda self,model,update:np.full(8,.125,dtype=np.float32)))()
        self.receipt=dict(path=(self.run/'il-kl/manifest.json').relative_to(ROOT).as_posix(),sha256='b'*64,
            probe_hash=state['il_kl_limit']['probe']['probe_hash'])
        patcher=mock.patch('brass_ai.recovery.il_kl_probe.import_probe',return_value=(probe,self.receipt))
        patcher.start();self.addCleanup(patcher.stop)

    def build(self,revision=B.SEED,**kw):
        options=dict(share_target=.75,weights=self.weights,rollback=dict(later=self.later,revision=revision,learner_generation=1190))
        options.update(kw)
        return B.prepare(ROOT,self.source,self.snapshot,self.run,**options)

    def test_the_new_run_starts_from_g1491_with_the_g1190_learner(self):
        self.check(B.SEED)

    def test_the_second_restart_keeps_the_charge_and_samples_256_rows(self):
        self.check(B.SEED_CHARGED)

    def check(self,revision):
        charged=revision==B.SEED_CHARGED
        before={p:sha(p) for p in self.source.rglob('*') if p.is_file()}
        summary=self.build(revision)
        self.assertEqual(before,{p:sha(p) for p in self.source.rglob('*') if p.is_file()})
        seed=model_id(self.model)
        state=read(self.run/'state.json');config=state['signature']['config']
        spec=reward_spec(config)
        self.assertEqual((summary['learner']['generation'],summary['learner']['seed_model_id']),(1190,seed))
        self.assertNotEqual(summary['learner']['model_id'],seed)
        self.assertEqual(summary['reward_migration']['to_reward'],spec)
        self.assertEqual((summary['compacted_rating_slots'],summary['ppo_lr_multiplier']),([1400],1.25))
        self.assertEqual({k:summary['schedule'][k] for k in ('utility_refit_iterations','shadow_iterations','teacher_on_from_iteration',
            'stage_a_iterations','stage_b_iterations','ppg_beta_floor_after_stage_a')},
            dict(utility_refit_iterations=1,shadow_iterations=1,teacher_on_from_iteration=3,stage_a_iterations=50,
                stage_b_iterations=50,ppg_beta_floor_after_stage_a=.5))
        self.assertEqual(state['iteration'],1491)
        self.assertEqual((state['history'][-1]['generation'],state['history'][-1]['model_id']),(1491,summary['learner']['model_id']))
        self.assertNotIn('search_value',state)
        self.assertTrue(P.restored(config))
        self.assertEqual((config['insolvency'],P.charged(config)),(P.OLD_COST if charged else P.ZERO_COST,charged))
        self.assertEqual(config['teacher_phase']['term']['rows_per_step'],256 if charged else 72)
        self.assertEqual(spec['insolvency'],P.OLD_COST if charged else P.ZERO_COST)
        # The committed entropy clock holds the applied charge, so the run adopts it without a reward step.
        self.assertEqual(state['exploration']['entropy_schedule']['insolvency'],config['insolvency'])
        self.assertEqual(state['native_backend']['shadow']['tables'],dict(train=32,held_out=4))
        self.assertEqual([s['generation'] for s in state['ratings']['slots']],list(range(1410,1500,10)))
        quality_pool.validate_state(state['quality_pool'],config['opponent_pool'],1491)
        from brass_ai.recovery.resume import load_resume,apply_optimizer_config
        model,raw,changed=load_resume(ROOT,self.run,state,state['signature'])
        self.assertFalse(changed)
        self.assertEqual((model.config.utility,model.utility_calibration),(spec,None))
        self.assertFalse(hasattr(model,'search_value_head'))
        # Same tensors as the seed; only the utility specification moved.
        for (name,a),(_,b) in zip(sorted(model.state_dict().items()),sorted(self.model.state_dict().items())):
            self.assertTrue(torch.equal(a,b),name)
        self.assertNotIn('policy_anchor',raw['training'])
        from brass_ai.learning.search_value import ordinary_parameters
        from brass_ai.learning.checkpoint import resume_optimizer
        optimizer=torch.optim.AdamW(ordinary_parameters(model),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        resume_optimizer(raw,optimizer,allow_cuda_topology_change=True);apply_optimizer_config(optimizer,config)
        self.assertEqual(optimizer.param_groups[0][L.KEY]['multiplier'],1.25)
        from brass_ai.recovery import exploration
        exploration.restore(config,state,raw['training']['exploration'])
        from brass_ai.recovery.actor_value import TrainingPhases
        TrainingPhases(model,config,raw=raw,generation=1491,root=ROOT,run=self.run)
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        companion=Companion(model,config['local_scoring_auxiliary'],raw=raw,generation=1491)
        self.assertEqual((companion.generation,companion.warmup['generation'],companion.retained_steps['ppg']),(1491,1491,30000))
        from brass_ai.recovery.teacher_phase import TeacherPhase
        teacher=TeacherPhase(ROOT,self.run,config,state,model,raw)
        # Readiness waits for one value phase under the restart reward.
        self.assertFalse(teacher.ready);self.assertEqual(teacher.weight,{})
        machine=teacher.search_value.phase.machine
        self.assertEqual((machine.state['stage'],machine.stage_a,machine.stage_b),('A',50,50))
        self.assertIsNone(teacher.search_value.phase.companion)
        model.utility_calibration=dict(ready=True,reward_id=digest(spec))
        jobs=teacher.begin(state,model,export_hash=state['history'][-1]['sha256'],epoch=1491)
        self.assertEqual(len(jobs),36)
        with mock.patch('brass_ai.rust_backend.lab_backend.checked_pin',return_value=dict(passed=True,
                implementation=state['native_backend']['implementation'])),\
                mock.patch('brass_ai.rust_backend.lab_backend.pin',side_effect=lambda path:state['native_backend']['rollback']):
            __import__('brass_ai.rust_backend.lab_backend',fromlist=['validate_state']).validate_state(state,raw,check_runtime=False)
        # The head adopted at the first teaching iteration is 256 wide, fits the parameter limit
        # and loads back at its own width.
        from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
        from brass_ai.learning.search_value import marker,width,WIDE_CONTRACT
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        SearchValuePhase(self.run/'values',model,config['teacher_phase']['search_value'],digest(spec),trainable=True)
        self.assertEqual(width(model),256)
        self.assertLessEqual(model.validate_parameter_count(),model.config.parameter_limit)
        self.assertEqual(marker(model),dict(contract=WIDE_CONTRACT,hidden=256))
        wide=self.run/'wide.pt';save_checkpoint(wide,model,None,step=1492,training={})
        loaded,again=load_checkpoint(wide)
        self.assertEqual((width(loaded),again['training']['search_value_head']),(256,marker(model)))
        self.assertEqual(model_id(loaded),model_id(model))
        spec_=importlib.util.spec_from_file_location('continue_g1491',ROOT/'scripts/continue-rust-g1491.py')
        m=importlib.util.module_from_spec(spec_);spec_.loader.exec_module(m)
        self.assertEqual(m.calibrated(state),(False,2))
        with self.assertRaisesRegex(ValueError,'already has training progress'):self.build(revision)

    def test_only_the_g1190_snapshot_seeds_the_learner(self):
        with self.assertRaisesRegex(ValueError,'retained snapshot of g1190'):
            self.build(weights=self.state['ratings']['agents']['generation-001180']['checkpoint']['path'])

    def test_a_seed_with_a_search_head_or_another_reward_is_refused(self):
        from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
        path=ROOT/self.weights
        model,raw=load_checkpoint(path)
        from dataclasses import replace
        model.config=replace(model.config,utility=specification(P.AFTER,P.ZERO_COST))
        save_checkpoint(path,model,None,step=1190,training=raw['training'])
        agent=self.state['ratings']['agents']['generation-001190']['checkpoint']
        agent['sha256']=sha(path)
        for e in self.state['history']:
            if e['generation']==1190:e['model_id']=model_id(model)
        agent['model_id']=model_id(model)
        write_json(self.source/'state.json',self.state);write_json(self.snapshot/'state.json',self.state)
        manifest=read(self.snapshot/'export-manifest.json');manifest['files'][0]['sha256']=sha(self.snapshot/'state.json')
        write_json(self.snapshot/'export-manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'Unreviewed seed reward migration'):self.build()


if __name__=='__main__':unittest.main()
