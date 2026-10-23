"""Actor-only GAE, leakage prevention, value transactions and exact continuation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_mlp_v11 import inputs
from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.features import Encoded
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import gae,value_phase,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.learning_rate import configure,KEY as WARMUP_KEY
from brass_ai.recovery.lr_controller import configure as configure_lr,KEY as LR_KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import migrate,reward_spec,calibrate


def config():
    # Historical transaction/resume fixtures retain their original hash split
    # and base LR. The refinement has its own explicit migration regressions.
    result=copy.deepcopy(read(SOURCE_ROOT/'configs/step5/joint7-latest.json'))
    from brass_ai.recovery.exploration import BLOCKS
    for key in (*BLOCKS,'panel_gates','policy_anchor','rating_coverage'):result.pop(key,None)
    result['exploration']=.08
    result['value_phase']=copy.deepcopy(value_phase.LEGACY_DEFAULT)
    result['advantage']['gate']=copy.deepcopy(gae.LEGACY_GATE)
    return result


def fixture(*,preset='small',games=4,decisions=3):
    torch.manual_seed(313);cfg=config();cfg['model_size']=preset
    cfg['update'].update(epochs=1,effective_batch_size=16,critic_batch_size=16,microbatch_size=8)
    cfg['value_phase'].update(batch_size=32,clone_rows=64,clone_batch_size=8,max_epochs=2,buffer_iterations=1)
    model=FourSeatNet(ModelConfig.preset(preset))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    migrate(model,optimizer,reward_spec(cfg));configure(optimizer,cfg['learning_rate'],100,.1,'selfplay')
    configure_lr(optimizer,cfg['ppo_lr_controller'])
    return model,optimizer,cfg,examples(model,games=games,decisions=decisions)


def examples(model,*,games=4,decisions=3,prefix='fixture-'):
    # Explicit authored fixture: one naturally hashed holdout game, then train
    # games. These IDs and tensors are not claimed to be native game outcomes.
    hold=next(prefix+str(i) for i in range(10000) if value_phase.heldout(prefix+str(i),.05))
    train=[prefix+str(i) for i in range(10000) if not value_phase.heldout(prefix+str(i),.05)][:games-1]
    names=[hold,*train];count=games*4*decisions
    states,actions,_,mapping=inputs(count,4);behavior=Behavior(model_id(model),.08);data=[]
    reward=model.config.utility;row_index=0
    for g,name in enumerate(names):
        win=[float(s==g%4) for s in range(4)];rank=[float(1+(s-g)%4) for s in range(4)]
        vp=[100.-10*(r-1)+g for r in rank];target=dict(win_share=win,placement=rank,vp=vp)
        placements=[(4-r)/3 for r in rank]
        terminal=[reward['win']*w+reward['placement']*p+reward['vp']*v for w,p,v in zip(win,placements,vp)]
        for step in range(decisions):
            for seat in range(4):
                n=1 if step==1 and seat%2==0 else 2+(seat%3)
                cost=.005 if step==0 else 0.
                choices=[dict(kind='loan'),dict(kind='pass'),dict(kind='build',industry='IronWorks'),dict(kind='build',industry='CoalMine')][:n]
                row=dict(chosen=(seat+step)%n,seat=seat,intent=[0.]*8,choices=choices,teacher=None,
                    behavior=dict(kind='model',policy_id=behavior.policy_id,probability=1.),
                    view=dict(context=dict(era='CanalEra' if step<2 else 'RailEra',round=2 if step==0 else 7,phase='action')),
                    _reward_id=digest(reward),_reward_return=[v-cost for v in terminal],
                    _reward_remaining=[reward['insolvency']['max_cost']]*4,_insolvency_cost_to_go=cost)
                if step==0:
                    row['_reward_trajectory']=dict(win=win[seat],placement=placements[seat],vp=vp[seat],cost=cost,utility=terminal[seat]-cost)
                    row['_insolvency_trajectory']=dict(events=1,shortfall=1,cost=cost,by_era={'CanalEra':1})
                encoded=Encoded(states[row_index].numpy(),actions[row_index,:n].numpy(),mapping[row_index].numpy(),tuple(str(j) for j in range(n)))
                data.append((row,target,(name,seat),encoded));row_index+=1
    ds=updates.Dataset(data,'cpu');model.eval()
    with torch.no_grad():
        for part in ds.microbatches(range(count),32,131072):
            ix,batch,q=ds.batch(part,.08);p=updates.probabilities(model(*batch)[0],q,.08,1.)
            for j,i in enumerate(part):data[i][0]['behavior']['probability']=float(p[j,data[i][0]['chosen']])
    return behavior,data


def ppo(model,optimizer,cfg,behavior,data,phases=None):
    reward=reward_spec(cfg)
    return updates.ppo_update(model,optimizer,[dict(split='train')],behavior,cfg['update'],
        learner_data=data,insolvency=reward['insolvency'],reward=reward,phases=phases)


def finish(phases,report,generation):
    return phases.finish(generation=generation,update=report,anchor_gate=dict(ok=True),
        rating=dict(ok=True,slope=1.,points=[[1,0],[2,1],[3,2]]))


class Advantages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_hand_chain_forced_costs_interleaved_seats_and_terminal(self):
        identities=[('g',0),('g',1),('g',0),('g',0),('g',1),('other',0)]
        data=[(dict(view=dict(context=dict(era='CanalEra',round=1,phase='action'))),None,k,None) for k in identities]
        target=torch.tensor([.975,.4,.99,1.,.5,.2]);v=torch.tensor([.2,.1,99.,.3,.15,.4]);mc=target-v
        selected=[0,1,3,4,5];rng=torch.get_rng_state().clone()
        actual,report=gae.compute(data,selected,mc,target,.95,[.98,.95,.9])
        self.assertAlmostEqual(float(actual[0]),.075+.95*.7,places=6)
        self.assertAlmostEqual(float(actual[1]),-.05+.95*.35,places=6)
        self.assertAlmostEqual(float(actual[5]),-.2,places=6)
        self.assertEqual(float(actual[2]),float(mc[2]))
        exact,other=gae.compute(data,selected,mc,target,1.,[.98,.95,.9])
        self.assertTrue(torch.equal(mc,exact));self.assertEqual(other['shadows']['1.0']['same_sign_fraction'],1.)
        self.assertEqual(report['trajectories'],3);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_lambda_one_shadow_has_identical_ppo_weights_rng_and_losses(self):
        model,opt,cfg,(behavior,data)=fixture();other=copy.deepcopy(model)
        cfg['value_phase']['enabled']=False;cfg['advantage']['gate']['mode']='fixed'
        other_opt=torch.optim.AdamW(other.parameters());other_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        phases=TrainingPhases(other,cfg);baseline=TrainingPhases(model,cfg)
        # PPG-off retains the critic holdout. Compare MC vs lambda-one GAE
        # using the same critic split, rather than training on held-out rows.
        baseline.advantages=lambda mc,ids:(mc,{})
        rng=torch.get_rng_state().clone()
        a=ppo(model,opt,cfg,behavior,data,baseline);final_rng=torch.get_rng_state().clone()
        torch.set_rng_state(rng);b=ppo(other,other_opt,cfg,behavior,data,phases)
        self.assertTrue(_same_payload(model.state_dict(),other.state_dict()))
        self.assertTrue(_same_payload(opt.state_dict(),other_opt.state_dict()))
        self.assertTrue(torch.equal(final_rng,torch.get_rng_state()))
        for key in ('epochs','advantage','baseline_explained_variance','categorical_probe'):self.assertEqual(a[key],b[key])
        self.assertEqual(b['gae']['lambda'],1.)

    def test_strict_config_old_recipe_and_new_generator(self):
        cfg=config();validate_config(cfg)
        generated=latest_model_config(cfg,cfg['imitation_checkpoint'])
        # The archived input stays valid, while fresh generation uses the
        # current recipe; do not reconstruct an obsolete migration pipeline.
        validate_config(generated)
        self.assertEqual(generated['advantage'],gae.DEFAULT)
        self.assertEqual(generated['imitation_checkpoint'],cfg['imitation_checkpoint'])
        self.assertEqual(generated['learning_rate'],cfg['learning_rate'])
        self.assertEqual(generated['exploration'],0.)
        self.assertEqual(generated['reward']['vp_cap'],180)
        self.assertEqual(generated['value_phase']['contract'],'ppg-value-phase/8.0.0')
        self.assertTrue(generated['value_phase']['enabled'])
        for key,value in (('lambda',0),('lambda',True),('chain','all')):
            bad=copy.deepcopy(cfg);bad['advantage'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(cfg);bad['value_phase']['learning_rate']*=2
        with self.assertRaisesRegex(ValueError,'base PPO'):validate_config(bad)
        bad=copy.deepcopy(cfg);bad['value_phase']['buffer_iterations']=3
        with self.assertRaises(ValueError):validate_config(bad)
        with self.assertRaisesRegex(ValueError,'combined reward'):gae.validate(gae.DEFAULT,combined=False)
        old=copy.deepcopy(cfg)
        for key in ('advantage','value_phase','critic_data','seat_descriptors'):old.pop(key)
        validate_config(old)

    def test_lower_lambda_runs_real_ppo_without_changing_critic_targets(self):
        model,opt,cfg,(behavior,data)=fixture();cfg['advantage']['gate']['mode']='fixed'
        cfg['advantage']['lambda']=.95;cfg['value_phase']['enabled']=False
        targets=copy.deepcopy([(d[1],d[0]['_reward_return']) for d in data])
        phases=TrainingPhases(model,cfg);r=ppo(model,opt,cfg,behavior,data,phases)
        self.assertEqual(r['gae']['lambda'],.95);self.assertGreater(r['optimizer_steps'],0)
        self.assertEqual([(d[1],d[0]['_reward_return']) for d in data],targets)
        self.assertNotEqual(r['gae']['shadows']['0.95']['std'],r['gae']['shadows']['1.0']['std'])
        self.assertEqual(phases.dataset.utility.tolist(),torch.tensor([v for _,v in targets]).tolist())


class Gates(unittest.TestCase):
    def run_gate(self,state,config,*,ev=.6,healthy=True,rating=True,ran=True):
        return gae.advance(state,config,generation=state['generation']+1,
            value_report=dict(optimizer_steps=int(ran),heldout_after=dict(explained_variance=ev,
                groups={'table:selfplay/role:learner':dict(explained_variance=ev)}),early_stop_reason=None),
            health_ok=healthy,rating=dict(ok=rating))

    def test_minimum_ev_streak_and_twenty_consecutive_healthy_iterations(self):
        cfg=copy.deepcopy(gae.DEFAULT);s=gae.initial(cfg)
        for _ in range(9):s,r=self.run_gate(s,cfg)
        self.assertEqual(s['active_lambda'],1.)
        s,r=self.run_gate(s,cfg);self.assertEqual(s['active_lambda'],.98);self.assertEqual(s['stage_origin'],10)
        for _ in range(19):s,r=self.run_gate(s,cfg)
        self.assertEqual(s['active_lambda'],.98)
        s,r=self.run_gate(s,cfg,healthy=False);self.assertEqual(s['healthy_streak'],0)
        for _ in range(20):s,r=self.run_gate(s,cfg,rating=False)
        self.assertEqual(r['reason'],'rating_evidence_not_ready')
        resumed=copy.deepcopy(s)
        a,ar=self.run_gate(s,cfg);b,br=self.run_gate(resumed,cfg)
        self.assertEqual((a,ar),(b,br));self.assertEqual(a['active_lambda'],.95)
        a,ar=self.run_gate(a,cfg,healthy=False,rating=False);self.assertEqual(a['active_lambda'],.95)

    def test_fallback_counts_completed_value_iterations_not_attempts(self):
        cfg=copy.deepcopy(gae.DEFAULT);s=gae.initial(cfg)
        for _ in range(7):s,r=self.run_gate(s,cfg,ran=False,ev=None)
        for _ in range(29):s,r=self.run_gate(s,cfg,ev=.1)
        self.assertEqual(s['active_lambda'],1.)
        s,r=self.run_gate(s,cfg,ev=.1,healthy=False);self.assertEqual(s['active_lambda'],1.)
        s,r=self.run_gate(s,cfg,ev=.1);self.assertEqual(s['active_lambda'],.98)
        self.assertEqual(r['reason'],'value_iteration_fallback')

    def test_rating_uses_verified_common_fit_and_watches_decline(self):
        from brass_ai.step5.league import write_json
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'fit.json'
            rows={f'generation-{g:06d}':dict(points=p) for g,p in [(10,2.),(20,3.),(30,4.)]}
            write_json(path,dict(ratings=rows))
            state=dict(iteration=30,ratings=dict(latest_report=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path))))
            r=gae.rating_evidence(SOURCE_ROOT,state,gae.GATE);self.assertTrue(r['ok']);self.assertAlmostEqual(r['slope'],.1)
            rows['generation-000030']['points']=0.;write_json(path,dict(ratings=rows))
            with self.assertRaisesRegex(ValueError,'evidence changed'):gae.rating_evidence(SOURCE_ROOT,state,gae.GATE)
            state['ratings']['latest_report']['sha256']=sha(path)
            self.assertFalse(gae.rating_evidence(SOURCE_ROOT,state,gae.GATE)['ok'])

    def test_runtime_warmup_and_ppo_stop_block_switch_and_clone_beta_caps(self):
        # Pure controller boundary: even a ready critic must not bypass warmup
        # or PPO health. Restoring the serialized streak keeps this decision.
        cfg=config();phase=object.__new__(TrainingPhases);phase.config=cfg
        phase.options={k:copy.deepcopy(cfg[k]) for k in ('advantage','value_phase')}
        phase.state=gae.initial(cfg['advantage'],9);phase.state.update(value_iterations=9,ev_streak=4)
        phase.critic_state=None
        phase.beta=1.;phase.trips=[];phase.evidence_migration=None
        phase.value_report=dict(optimizer_steps=1,heldout_after=dict(explained_variance=.6,
            groups={'table:selfplay/role:learner':dict(explained_variance=.6)}),
            kl_guard_tripped=False,early_stop_reason=None)
        update=dict(gae={},categorical_probe=dict(categorical_kl=.003),controller_warmup_done=False,
            optimizer_steps=1,kl_early_stop=False)
        phase.finish(generation=10,update=update,anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(phase.state['active_lambda'],1.)
        update['controller_warmup_done']=True;update['kl_early_stop']=True
        phase.finish(generation=11,update=update,anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(phase.state['active_lambda'],1.)
        update['kl_early_stop']=False
        phase.finish(generation=12,update=update,anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(phase.state['active_lambda'],.98)
        phase.value_report['kl_guard_tripped']=True
        for g in range(13,19):phase.finish(generation=g,update=update,anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(phase.beta,8.);self.assertEqual(phase.trips,[True]*5)


class ValuePhase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_split_is_deterministic_all_seats_and_calibration_excludes_it(self):
        model,opt,cfg,(behavior,data)=fixture();ds=updates.Dataset(data,'cpu');rng=torch.get_rng_state().clone()
        train,held,report=value_phase.split(data,.05)
        self.assertEqual((train,held,report),value_phase.split(data,.05));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual({data[i][0]['seat'] for i in held},{0,1,2,3})
        self.assertTrue({data[i][2][0] for i in train}.isdisjoint({data[i][2][0] for i in held}))
        seen=[];real=ds.value_batch
        def batch(part,*args,**kwargs):seen.extend(part);return real(part,*args,**kwargs)
        with patch.object(ds,'value_batch',side_effect=batch):calibrate(model,opt,ds,cfg['update'],indices=train)
        self.assertTrue(seen);self.assertTrue(set(seen).isdisjoint(held))

    def test_ppo_critic_excludes_holdout_actor_keeps_it(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg);seen=[];real=updates.Dataset.value_batches
        def batches(ds,parts,*args,**kwargs):
            parts=list(parts);seen.extend(i for part in parts for i in part);yield from real(ds,parts,*args,**kwargs)
        with patch.object(updates.Dataset,'value_batches',batches):report=ppo(model,opt,cfg,behavior,data,phases)
        self.assertTrue(set(seen).isdisjoint(phases.heldout_ids))
        self.assertEqual(report['meaningful_decisions'],sum(len(d[0]['choices'])>1 for d in data))
        self.assertTrue(set(phases.actor_ids)&set(phases.heldout_ids))

    def test_raw_clone_zero_and_gradients_and_separate_optimizer(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg)
        report=ppo(model,opt,cfg,behavior,data,phases);before=copy.deepcopy(opt.state_dict())
        reference=value_phase.RawPolicyReference(model,phases.dataset,phases.actor_ids,cfg['update'])
        self.assertLess(reference.measure(model),1e-7)
        used={};real_step=phases.optimizer.step
        def step(*args,**kwargs):
            used.update({name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None})
            return real_step(*args,**kwargs)
        with patch.object(phases.optimizer,'step',side_effect=step):value=phases.fit(model)
        self.assertTrue(_same_payload(before,opt.state_dict()))
        self.assertTrue(any(v>0 for k,v in used.items() if k.startswith('utility_head')))
        self.assertTrue(any(v>0 for k,v in used.items() if 'state_encoder' in k))
        self.assertLessEqual(value['probe_kl'],cfg['value_phase']['kl_cap']+1e-7)
        self.assertGreater(value['attempted_steps'],0)
        self.assertEqual(phases.optimizer.param_groups[0]['lr'],cfg['learning_rate'])
        self.assertNotIn(WARMUP_KEY,phases.optimizer.param_groups[0]);self.assertNotIn(LR_KEY,phases.optimizer.param_groups[0])
        self.assertLessEqual(value['heldout_after']['loss'],value['heldout_before']['loss'])
        self.assertEqual(len(value['heldout_after']['win_calibration']),10)
        # The LR controller's saved PPO signal is never overwritten by PPG.
        signal=copy.deepcopy(report['categorical_probe'])
        final=phases.ppo_probe.measure(model)
        self.assertEqual(signal,report['categorical_probe']);self.assertTrue(torch.isfinite(torch.tensor(final['categorical_kl'])))

    def test_guard_rolls_back_weights_and_value_adam_then_adapts_beta(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg)
        report=ppo(model,opt,cfg,behavior,data,phases)
        initial=copy.deepcopy(model.state_dict());adam=copy.deepcopy(phases.optimizer.state_dict())
        real=value_phase.RawPolicyReference.measure;calls=[]
        def measurement(ref,*args,**kwargs):
            calls.append(1)
            return .003 if len(calls)==1 else real(ref,*args,**kwargs)
        with patch.object(value_phase.RawPolicyReference,'measure',measurement),patch.object(
                value_phase,'metrics',wraps=value_phase.metrics) as metrics:value=phases.fit(model)
        self.assertEqual(value['optimizer_steps'],0);self.assertEqual(value['early_stop_reason'],'policy_kl_cap')
        self.assertTrue(value['train_metrics_reused'])
        self.assertEqual(value['train_before'],value['train_after'])
        self.assertEqual(metrics.call_count,4)  # train before, holdout before/attempt, unknown holdout
        self.assertEqual(value['attempted_value_presentations'],min(len(phases.train_ids),cfg['value_phase']['batch_size']))
        self.assertGreater(value['attempted_clone_presentations'],0)
        self.assertEqual(value['value_presentations'],0)
        self.assertTrue(_same_payload(initial,model.state_dict()));self.assertTrue(_same_payload(adam,phases.optimizer.state_dict()))
        phases.trips=[True,False,True,False];finish(phases,report,1)
        self.assertEqual(phases.beta,2.);self.assertEqual(value['beta_clone_next'],2.)

    def test_heldout_loss_guard_rolls_back_and_empty_split_skips(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg)
        ppo(model,opt,cfg,behavior,data,phases);before=copy.deepcopy(model.state_dict())
        real=value_phase.metrics;calls=[]
        def measure(*args,**kwargs):
            r=real(*args,**kwargs);calls.append(1)
            if len(calls)==3:r['loss']+=10.
            return r
        with patch.object(value_phase,'metrics',side_effect=measure):r=phases.fit(model)
        self.assertEqual(r['early_stop_reason'],'heldout_loss_increased');self.assertEqual(r['optimizer_steps'],0)
        self.assertTrue(_same_payload(before,model.state_dict()))
        phases.heldout_ids=[];r=phases.fit(model);self.assertEqual(r['early_stop_reason'],'insufficient_critic_split')

    def test_kl_cap_retains_safe_prefix_and_its_exact_adam_only(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg)
        ppo(model,opt,cfg,behavior,data,phases)
        ppo_before=copy.deepcopy(opt.state_dict());safe=[]
        real_step=phases.optimizer.step;real_measure=value_phase.RawPolicyReference.measure
        def step(*args,**kwargs):
            result=real_step(*args,**kwargs)
            safe.append((copy.deepcopy(model.state_dict()),copy.deepcopy(phases.optimizer.state_dict())))
            return result
        def measurement(ref,*args,**kwargs):
            # One valid step, then a violating step. Subsequent final probes
            # use the real frozen post-PPO distribution on restored weights.
            if len(safe)==2 and not rejected[0]:rejected[0]=True;return .0021471
            return real_measure(ref,*args,**kwargs)
        rejected=[False]
        with patch.object(phases.optimizer,'step',side_effect=step),patch.object(
                value_phase.RawPolicyReference,'measure',measurement):value=phases.fit(model)
        self.assertEqual(value['attempted_steps'],2);self.assertEqual(value['optimizer_steps'],1)
        self.assertTrue(value['kl_guard_tripped']);self.assertEqual(value['early_stop_reason'],'policy_kl_cap')
        self.assertEqual(value['partial_epochs'],1);self.assertEqual(value['completed_epochs'],0)
        self.assertTrue(value['epoch_reports'][0]['partial']);self.assertTrue(value['epoch_reports'][0]['accepted'])
        self.assertEqual(value['epoch_reports'][0]['accepted_steps'],1)
        self.assertEqual(value['distinct_value_rows_used'],cfg['value_phase']['batch_size'])
        self.assertEqual(value['value_presentations'],cfg['value_phase']['batch_size'])
        self.assertEqual(value['attempted_value_presentations'],len(phases.train_ids))
        self.assertEqual(value['clone_presentations'],cfg['value_phase']['clone_batch_size'])
        self.assertEqual(value['attempted_clone_presentations'],2*cfg['value_phase']['clone_batch_size'])
        self.assertLessEqual(value['probe_kl'],.002)
        self.assertLessEqual(value['heldout_after']['loss'],value['heldout_before']['loss'])
        self.assertTrue(_same_payload(safe[0][0],model.state_dict()))
        self.assertTrue(_same_payload(safe[0][1],phases.optimizer.state_dict()))
        self.assertTrue(_same_payload(ppo_before,opt.state_dict()))

    def test_safe_prefix_still_requires_heldout_improvement(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg)
        ppo(model,opt,cfg,behavior,data,phases)
        initial=copy.deepcopy(model.state_dict());adam=copy.deepcopy(phases.optimizer.state_dict())
        real_measure=value_phase.RawPolicyReference.measure;real_metrics=value_phase.metrics
        kl_calls=[];metric_calls=[]
        def measurement(ref,*args,**kwargs):
            kl_calls.append(1)
            return .003 if len(kl_calls)==2 else real_measure(ref,*args,**kwargs)
        def metrics(*args,**kwargs):
            result=real_metrics(*args,**kwargs);metric_calls.append(1)
            if len(metric_calls)==3:result['loss']+=10.
            return result
        with patch.object(value_phase.RawPolicyReference,'measure',measurement),patch.object(
                value_phase,'metrics',side_effect=metrics):value=phases.fit(model)
        self.assertEqual(value['optimizer_steps'],0);self.assertEqual(value['attempted_steps'],2)
        self.assertEqual(value['epoch_reports'][0]['safe_steps'],1)
        self.assertEqual(value['epoch_reports'][0]['rejection_reason'],'heldout_loss_increased')
        self.assertEqual(value['value_presentations'],0);self.assertEqual(value['partial_epochs'],0)
        self.assertTrue(_same_payload(initial,model.state_dict()))
        self.assertTrue(_same_payload(adam,phases.optimizer.state_dict()))

    def test_first_step_cap_in_later_epoch_keeps_prior_epoch(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg)
        ppo(model,opt,cfg,behavior,data,phases);saved=[]
        real_step=phases.optimizer.step;real_measure=value_phase.RawPolicyReference.measure;calls=[]
        def step(*args,**kwargs):
            result=real_step(*args,**kwargs)
            saved.append((copy.deepcopy(model.state_dict()),copy.deepcopy(phases.optimizer.state_dict())))
            return result
        def measurement(ref,*args,**kwargs):
            calls.append(1)
            return .003 if len(calls)==3 else real_measure(ref,*args,**kwargs)
        with patch.object(phases.optimizer,'step',side_effect=step),patch.object(
                value_phase.RawPolicyReference,'measure',measurement):value=phases.fit(model)
        self.assertEqual(value['attempted_steps'],3);self.assertEqual(value['optimizer_steps'],2)
        self.assertEqual(value['completed_epochs'],1);self.assertEqual(value['partial_epochs'],0)
        self.assertEqual(value['epoch_reports'][1]['accepted_steps'],0)
        self.assertEqual(value['epoch_reports'][1]['rejection_reason'],'no_safe_steps')
        self.assertTrue(_same_payload(saved[1][0],model.state_dict()))
        self.assertTrue(_same_payload(saved[1][1],phases.optimizer.state_dict()))

    def test_checkpoint_restores_both_adams_gate_beta_rng_and_next_update(self):
        model,opt,cfg,(behavior,data)=fixture();phases=TrainingPhases(model,cfg)
        report=ppo(model,opt,cfg,behavior,data,phases);phases.fit(model);finish(phases,report,1)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'generation-0001.pt'
            save_checkpoint(path,model,opt,step=1,training={KEY:phases.payload()});checksum=sha(path)
            from brass_ai.recovery.retention import slim_checkpoint
            entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=checksum,model_id=model_id(model),generation=1)
            slim=slim_checkpoint(SOURCE_ROOT,Path(tmp),entry)
            frozen,frozen_raw=load_checkpoint(SOURCE_ROOT/slim['path'])
            self.assertIsNone(frozen_raw['optimizer']);self.assertNotIn(KEY,frozen_raw['training'])
            self.assertEqual(model_id(frozen),model_id(model))
            restored,raw=load_checkpoint(path);other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,other)
            second=TrainingPhases(restored,cfg,raw=raw,generation=1)
            self.assertTrue(_same_payload(phases.payload(),second.payload()));self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
            behavior,data=examples(model,prefix='resume-next-');rng=torch.get_rng_state().clone();results=[]
            for net,adam,phase in ((model,opt,phases),(restored,other,second)):
                torch.set_rng_state(rng);r=ppo(net,adam,cfg,behavior,data,phase);v=phase.fit(net);finish(phase,r,2)
                results.append((r,v,torch.get_rng_state().clone()))
            self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()))
            self.assertTrue(_same_payload(opt.state_dict(),other.state_dict()))
            self.assertTrue(_same_payload(phases.payload(),second.payload()))
            self.assertEqual(results[0][0]['epochs'],results[1][0]['epochs'])
            self.assertTrue(torch.equal(results[0][2],results[1][2]));self.assertEqual(sha(path),checksum)
            with self.assertRaisesRegex(ValueError,'Missing GAE'):TrainingPhases(restored,cfg,generation=1)
            bad=copy.deepcopy(cfg);bad['advantage']['gate']['mode']='fixed'
            with self.assertRaisesRegex(ValueError,'configuration differs'):TrainingPhases(restored,bad,raw=raw,generation=1)

    def test_compact_history_keeps_nested_diagnostics_and_gate(self):
        from brass_ai.recovery.history import compact
        report=dict(actor=dict(gae=dict(shadows={'0.98':dict(std=.1)},gate=dict(lambda_next=.98)),
            final_categorical_probe=dict(categorical_kl=.01),critic_split=dict(heldout_rows=20)),
            value_phase=dict(heldout_after=dict(explained_variance=.6),optimizer_steps=2))
        out=compact(report)
        for key in report['actor']:self.assertEqual(out['actor'][key],report['actor'][key])
        self.assertEqual(out['value_phase'],report['value_phase']);digest(out)


if __name__=='__main__':unittest.main()
