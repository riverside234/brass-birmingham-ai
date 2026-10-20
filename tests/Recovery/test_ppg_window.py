"""Interval rollback, rolling accepted-work support and reviewed g185 migration."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import gae,gae_health,value_phase,il_kl_limit as limits
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.history import compact
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id
from test_gae_value_phase import fixture,ppo
from test_auto_il_kl import baseline,measurement,counts


def upgraded(cfg):
    return value_phase.interval_options(gae.health_options(value_phase.refined_options(cfg)))


def tick(state,cfg,steps=0,healthy=True,trip=False,ev=.4,rating=True):
    return gae.advance(state,cfg,generation=state['generation']+1,health_ok=healthy,rating=dict(ok=rating),
        value_report=dict(enabled=True,optimizer_steps=steps,kl_guard_tripped=trip,
            heldout_after=dict(groups={'table:selfplay/role:learner':dict(explained_variance=ev)})))


def stage():
    cfg=copy.deepcopy(gae.DEFAULT);state=gae.initial(cfg)
    state['active_lambda']=.98
    return cfg,state


class StageWindow(unittest.TestCase):
    def test_twenty_observations_sparse_accepted_work_and_exact_restart(self):
        cfg,state=stage()
        for g in range(1,20):
            state,r=tick(state,cfg,steps=6 if g in (1,8,15) else 0)
            self.assertEqual(state['healthy_streak'],g)
            self.assertEqual(state['active_lambda'],.98)
        saved=json.loads(json.dumps(state))
        a,report=tick(state,cfg);b,other=tick(saved,cfg)
        self.assertEqual((a,report),(b,other));self.assertEqual(a['active_lambda'],.95)
        self.assertEqual(report['stage_evidence']['accepted_steps'],18)
        self.assertEqual(report['stage_evidence']['accepted_iterations'],3)
        self.assertTrue(report['stage_evidence']['ready'])
        retained=compact(dict(actor=dict(gae=dict(gate=report)),value_phase=dict(optimizer_steps=0)))
        self.assertEqual(retained['actor']['gae']['gate']['stage_evidence'],report['stage_evidence'])

    def test_zero_steps_alone_never_reset_but_insufficient_work_cannot_promote(self):
        for schedule in ({},{1:16,2:16},{1:1,2:1,3:1}):
            cfg,state=stage()
            for g in range(1,21):state,r=tick(state,cfg,steps=schedule.get(g,0))
            self.assertEqual(state['healthy_streak'],20)
            self.assertEqual(state['active_lambda'],.98)
            self.assertEqual(r['reason'],'accepted_value_window_not_ready')
            self.assertFalse(r['stage_evidence']['ready'])

    def test_expiry_and_rating_still_block(self):
        cfg,state=stage()
        for g in range(1,21):state,r=tick(state,cfg,steps=6 if g<=3 else 0,rating=False)
        self.assertTrue(r['stage_evidence']['ready']);self.assertEqual(r['reason'],'rating_evidence_not_ready')
        state,r=tick(state,cfg)
        self.assertEqual(state['active_lambda'],.98)
        self.assertEqual(r['stage_evidence']['accepted_steps'],12)
        self.assertEqual(r['stage_evidence']['accepted_iterations'],2)
        self.assertEqual(len(state['stage_evidence']['entries']),20)

    def test_economic_PPO_PPG_and_invalid_EV_failures_reset_health(self):
        for failure in (dict(healthy=False),dict(trip=True),dict(ev=None),dict(ev=float('nan'))):
            cfg,state=stage()
            for _ in range(8):state,r=tick(state,cfg,steps=16)
            state,r=tick(state,cfg,**failure)
            self.assertEqual(state['healthy_streak'],0);self.assertFalse(r['healthy'])
            for _ in range(19):state,r=tick(state,cfg,steps=1)
            self.assertEqual(state['active_lambda'],.98)
            state,r=tick(state,cfg,steps=0);self.assertEqual(state['active_lambda'],.95)

    def test_state_support_and_config_validation(self):
        cfg,state=stage();state,r=tick(state,cfg,steps=16)
        for key,value in (('window_iterations',19),('healthy_iterations',19),
                          ('minimum_accepted_steps',True),('minimum_value_iterations_in_window',21)):
            bad=copy.deepcopy(cfg);bad['gate'][key]=value
            with self.assertRaises(ValueError):gae.validate(bad)
        for mutation in ('skip','negative','extra'):
            bad=copy.deepcopy(state)
            if mutation=='skip':bad['stage_evidence']['entries']=[]
            if mutation=='negative':bad['stage_evidence']['entries'][0]['accepted_steps']=-1
            if mutation=='extra':bad['stage_evidence']['extra']=True
            with self.assertRaises(ValueError):gae.validate_state(bad,cfg,1)


class Intervals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fit_case(self,*,losses,cap_at=None):
        model,opt,cfg,(behavior,data)=fixture(games=5)
        cfg=upgraded(cfg);cfg['value_phase'].update(batch_size=1,clone_batch_size=2,max_epochs=1)
        phase=TrainingPhases(model,cfg);ppo(model,opt,cfg,behavior,data,phase)
        ppo_before=copy.deepcopy(opt.state_dict())
        origin=(copy.deepcopy(model.state_dict()),copy.deepcopy(phase.optimizer.state_dict()))
        retained={0:origin};attempts=[];references=set();held_calls=[]
        real_step=phase.optimizer.step;real_metrics=value_phase.metrics;real_measure=value_phase.RawPolicyReference.measure
        def step(*args,**kwargs):
            result=real_step(*args,**kwargs);attempts.append(1)
            if len(attempts) in (16,17,32,36):
                retained[len(attempts)]=(copy.deepcopy(model.state_dict()),copy.deepcopy(phase.optimizer.state_dict()))
            return result
        def metrics(*args,**kwargs):
            result=real_metrics(*args,**kwargs)
            if kwargs.get('evidence'):
                result['loss']=losses[len(held_calls)];held_calls.append(1)
            return result
        def measure(ref,*args,**kwargs):
            references.add(id(ref))
            result=real_measure(ref,*args,**kwargs)
            if cap_at==len(attempts) and not tripped[0]:tripped[0]=True;return .003
            return result
        tripped=[False]
        with patch.object(phase.optimizer,'step',side_effect=step),patch.object(value_phase,'metrics',side_effect=metrics),\
             patch.object(value_phase.RawPolicyReference,'measure',measure):
            report=phase.fit(model)
        count=report['optimizer_steps']
        self.assertTrue(_same_payload(retained[count][0],model.state_dict()))
        self.assertTrue(_same_payload(retained[count][1],phase.optimizer.state_dict()))
        self.assertTrue(_same_payload(ppo_before,opt.state_dict()))
        self.assertEqual(len(references),1)  # cumulative post-PPO reference, never reset at boundaries
        self.assertEqual(report['value_presentations'],count)
        self.assertEqual(report['distinct_value_rows_used'],count)
        self.assertEqual(report['clone_presentations'],2*count)
        self.assertLessEqual(report['probe_kl'],.002)
        self.assertEqual(compact(dict(value_phase=report))['value_phase'],report)
        return report

    def test_later_loss_rejection_keeps_first_sixteen_and_exact_adam(self):
        r=self.fit_case(losses=[20,19,21])
        self.assertEqual((r['optimizer_steps'],r['attempted_steps']),(16,32))
        self.assertEqual((r['accepted_intervals'],r['rejected_intervals']),(1,1))
        self.assertEqual((r['completed_epochs'],r['partial_epochs']),(0,1))
        self.assertEqual(r['heldout_after']['loss'],19)

    def test_first_rejection_restores_entire_origin(self):
        r=self.fit_case(losses=[20,21])
        self.assertEqual((r['optimizer_steps'],r['attempted_steps']),(0,16))
        self.assertTrue(r['train_metrics_reused'])

    def test_cap_in_second_interval_retains_safe_prefix_without_resetting_KL(self):
        r=self.fit_case(losses=[20,19,18],cap_at=18)
        self.assertEqual((r['optimizer_steps'],r['attempted_steps']),(17,18))
        self.assertTrue(r['kl_guard_tripped']);self.assertEqual(r['accepted_intervals'],2)

    def test_short_tail_is_checked_and_counted(self):
        r=self.fit_case(losses=[20,19,18,17])
        self.assertEqual((r['optimizer_steps'],r['attempted_steps']),(36,36))
        self.assertEqual([v['steps'] for v in r['interval_reports']],[16,16,4])
        self.assertEqual((r['completed_epochs'],r['partial_epochs']),(1,0))


class ContinuationAndPassRemoval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pass_evidence_cannot_affect_shared_health_limit_or_LR(self):
        original=baseline();expected=None
        for behavior in (None,{},dict(ok=False,passing=dict(ok=False,failures=999)),
                         dict(ok=True,passing=dict(ok=True)),dict(passing=float('nan'))):
            state,r=limits.advance(original,measurement(behavior=behavior),counts(),generation=11,warmup_done=True)
            self.assertTrue(r['health']['ok']);self.assertTrue(r['lr_gate']['ok'])
            self.assertNotIn('behavior',r['health']);self.assertNotIn('behavior',r['measurement'])
            if expected is None:expected=(state,r)
            else:self.assertEqual(expected,(state,r))
        self.assertFalse(limits.health(original['baseline'],limits.rates([counts(passes_with_alternative=1000)]*5))['ok'])

    def test_g185_migration_keeps_adams_rng_counters_and_prior_evidence(self):
        model,opt,cfg,_=fixture();old=value_phase.refined_options(cfg)
        phase=TrainingPhases(model,old)
        p=next(model.parameters());p.grad=torch.ones_like(p)
        phase.optimizer.step();phase.optimizer.zero_grad(set_to_none=True)
        phase.state=gae.initial(old['advantage'],185)
        phase.state.update(active_lambda=.98,stage_origin=81,value_iterations=100,healthy_streak=2)
        phase.critic_state['generation']=185
        saved=phase.payload();raw=dict(training={KEY:saved})
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        new=upgraded(old);resumed=TrainingPhases(model,new,raw=raw,generation=185)
        self.assertTrue(_same_payload(resumed.optimizer.state_dict(),phase.optimizer.state_dict()))
        self.assertTrue(_same_payload(model.state_dict(),weights));self.assertTrue(_same_payload(opt.state_dict(),adam))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(resumed.state,{**phase.state,'stage_evidence':gae_health.initial(185)})
        self.assertEqual(resumed.critic_state,phase.critic_state)
        restarted=TrainingPhases(model,new,raw=dict(training={KEY:resumed.payload()}),generation=185)
        self.assertIsNone(restarted.window_migration)
        self.assertTrue(_same_payload(restarted.payload(),resumed.payload()))
        for value in (17,0,True):
            bad=copy.deepcopy(new);bad['value_phase']['acceptance_steps']=value
            with self.assertRaises(ValueError):value_phase.validate(bad['value_phase'])
        bad=copy.deepcopy(new);bad['value_phase']['learning_rate']*=.5
        with self.assertRaisesRegex(ValueError,'configuration differs'):TrainingPhases(model,bad,raw=raw,generation=185)

    def test_real_g185_config_source_and_only_explicit_upgrades(self):
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0185-20261020T090519451283Z/state.json')
        old=state['signature']['config'];new=upgraded(old)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'advantage','value_phase'})
        self.assertEqual(new,read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-ppg-window.json'))
        signature=dict(state['signature'],config=new,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        for field in ('learning_rate','kl_cap'):
            bad=copy.deepcopy(signature);bad['config']['value_phase'][field]*=1.1
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)


if __name__=='__main__':unittest.main()
