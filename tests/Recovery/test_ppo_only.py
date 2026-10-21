"""PPG-off continuation, real PPO critic work and gate evidence without PPG."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import gae,value_phase,value_buffer,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY,PPO_ONLY_CONTRACT
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json
from test_gae_value_phase import fixture,ppo
from test_critic_descriptors import mixed_fixture
from test_health_ev_window import report as ev_report
from test_ppg_window import upgraded

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0210-20261020T145156876449Z'


def measurement(*,steps=8,ev=.6):
    metrics=dict(groups={})
    metrics['groups'].update({
        'table:selfplay/role:learner':dict(explained_variance=ev),
        'role:learner':dict(explained_variance=ev)})
    return dict(enabled=False,optimizer_steps=0,attempted_steps=0,
        critic_training=dict(source='ppo',optimizer_steps=steps),
        heldout_after=metrics,descriptor_gain=dict(learner_ev_gain=.1),kl_guard_tripped=False)


class PPOOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_disabled_default_and_exact_current_config_transition(self):
        state=read(EXPORT/'state.json');before=state['signature']['config']
        cfg=value_phase.disabled_options(before);validate_config(cfg)
        self.assertEqual([k for k in cfg if cfg[k]!=before[k]],['value_phase'])
        expected=copy.deepcopy(before['value_phase']);expected['enabled']=False
        self.assertEqual(cfg['value_phase'],expected)
        self.assertFalse(value_phase.DEFAULT['enabled'])
        fresh=latest_model_config(before,before['imitation_checkpoint'])
        self.assertTrue(fresh['value_phase']['enabled'])
        self.assertEqual(fresh['value_phase']['buffer_iterations'],8)
        self.assertFalse(fresh['teacher_enabled'])
        self.assertEqual(fresh['advantage']['gate']['healthy_iterations'],20)
        signature=dict(state['signature'],config=cfg,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)

    def test_ppo_only_real_update_preserves_holdout_snapshots_and_no_ppg_mutation(self):
        model,opt,cfg,behavior,data,snap,_=mixed_fixture()
        cfg=value_phase.disabled_options(upgraded(cfg));phase=TrainingPhases(model,cfg)
        phase.snapshot_data=snap
        report=ppo(model,opt,cfg,behavior,data,phase)
        self.assertGreater(report['critic_rows']['optimizer_steps'],0)
        self.assertGreater(report['critic_rows']['snapshot_distinct_used'],0)
        self.assertTrue(phase.heldout_ids)
        train_games={phase.critic_dataset.data[i][2][0] for i in phase.train_ids}
        held_games={phase.critic_dataset.data[i][2][0] for i in phase.heldout_ids}
        self.assertTrue(train_games.isdisjoint(held_games))
        self.assertEqual(report['critic_rows']['optimizer_steps'],report['optimizer_steps'])
        before=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
        ppg=copy.deepcopy(phase.optimizer.state_dict());rng=torch.get_rng_state().clone()
        with patch.object(phase.optimizer,'step',side_effect=AssertionError('PPG step')), \
             patch.object(value_phase,'RawPolicyReference',side_effect=AssertionError('PPG clones')), \
             patch.object(value_buffer.DiskBuffer,'stage',side_effect=AssertionError('PPG cache')):
            value=phase.fit(model)
        self.assertFalse(value['enabled']);self.assertEqual(value['early_stop_reason'],'disabled')
        self.assertEqual(value['optimizer_steps'],0);self.assertEqual(value['attempted_steps'],0)
        self.assertEqual(value['distinct_previous_rows_used'],0)
        self.assertIsNone(phase.buffer);self.assertNotIn('buffer',value)
        self.assertEqual(value['critic_training']['optimizer_steps'],report['critic_rows']['optimizer_steps'])
        self.assertIn('table:legacy/role:learner',value['heldout_after']['groups'])
        self.assertIn('ev_evidence',value['heldout_after']['groups']['table:legacy/role:learner'])
        self.assertIn('learner_ev_gain',value['descriptor_gain'])
        self.assertTrue(_same_payload(before,model.state_dict()))
        self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(_same_payload(ppg,phase.optimizer.state_dict()))
        self.assertIsNone(model.utility_calibration)
        self.assertEqual(value['learning_rate'],0.)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        saved_trips=[True,True,False];phase.trips=saved_trips[:]
        phase.finish(generation=1,update=report,anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(phase.trips,saved_trips)
        self.assertEqual(value['beta_reason'],'ppg_disabled')
        gate=report['gae']['gate']
        self.assertEqual(gate['critic_training']['source'],'ppo')
        self.assertEqual(gate['stage_evidence']['source'],'ppo')
        self.assertEqual(compact(dict(actor=report,value_phase=value))['value_phase'],value)

    def test_no_critic_weight_means_no_qualifying_work(self):
        model,opt,cfg,(behavior,data)=fixture(games=4)
        cfg=value_phase.disabled_options(upgraded(cfg));cfg['update']['value_weight']=0.
        phase=TrainingPhases(model,cfg)
        report=ppo(model,opt,cfg,behavior,data,phase);value=phase.fit(model)
        self.assertGreater(report['optimizer_steps'],0)
        self.assertEqual(value['critic_training']['optimizer_steps'],0)
        phase.finish(generation=1,update=report,anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(phase.state['value_iterations'],0)

    def test_disable_and_restart_preserve_weights_both_adams_and_attained_stages(self):
        model,opt,cfg,_=fixture();cfg=upgraded(cfg);phase=TrainingPhases(model,cfg)
        p=next(model.parameters());p.grad=torch.ones_like(p);phase.optimizer.step()
        phase.optimizer.zero_grad(set_to_none=True)
        p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
        phase.state=gae.initial(cfg['advantage'],210)
        phase.state.update(active_lambda=.95,value_iterations=175,stage_origin=206,
            table_enabled=dict(historical=True,legacy=True),
            table_origins=dict(historical=130,legacy=150))
        phase.critic_state.update(generation=210,snapshots_active=True,snapshot_weight=.75,
            learner_ev_baseline=.2,activation_generation=15)
        phase.critic_state['weight_recovery']=critic_data.recovery_initial(0)  # Pre-R2 checkpoint.
        phase.critic_state['weight_recovery']['healthy_streak']=3
        phase.trips=[True,False,False];saved=phase.payload()
        # The disabled route must not require or rebuild an old disk buffer.
        saved['options']['value_phase']['buffer_iterations']=2
        saved['buffer']=dict(generation=210,unavailable_derived_cache=True)
        cfg['value_phase']['buffer_iterations']=2
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
        rng=torch.get_rng_state().clone();new=value_phase.disabled_options(cfg)
        with patch.object(value_buffer,'DiskBuffer',side_effect=AssertionError('disabled disk cache')):
            resumed=TrainingPhases(model,new,raw=dict(training={KEY:saved}),generation=210)
        self.assertEqual(resumed.contract,PPO_ONLY_CONTRACT)
        self.assertEqual(resumed.state['active_lambda'],.95);self.assertEqual(resumed.state['stage_origin'],206)
        self.assertEqual(resumed.state['table_origins'],phase.state['table_origins'])
        self.assertEqual(resumed.state['table_enabled'],phase.state['table_enabled'])
        self.assertEqual(resumed.state['stage_evidence']['origin'],210)
        self.assertEqual(resumed.state['table_evidence']['entries'],[])
        self.assertEqual(resumed.state['value_iterations'],0)
        self.assertEqual(resumed.critic_state['snapshot_weight'],.75)
        self.assertIsNone(resumed.critic_state['learner_ev_baseline'])
        self.assertEqual(resumed.critic_state['weight_recovery']['legacy_ev_baseline'],.2)
        self.assertEqual(resumed.critic_state['activation_generation'],15)
        self.assertEqual(resumed.critic_state['weight_recovery']['healthy_streak'],0)
        self.assertTrue(_same_payload(saved['optimizer'],resumed.optimizer.state_dict()))
        self.assertTrue(_same_payload(weights,model.state_dict()))
        self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        restarted=TrainingPhases(model,new,raw=dict(training={KEY:resumed.payload()}),generation=210)
        self.assertTrue(_same_payload(restarted.payload(),resumed.payload()))
        self.assertIsNone(restarted.disable_migration)
        bad=copy.deepcopy(new);bad['value_phase']['kl_cap']*=.5
        with self.assertRaisesRegex(ValueError,'configuration differs'):
            TrainingPhases(model,bad,raw=dict(training={KEY:saved}),generation=210)
        corrupt=copy.deepcopy(saved);corrupt['critic_data']['weight_recovery']['healthy_streak']=-1
        with self.assertRaisesRegex(ValueError,'Invalid snapshot weight recovery'):
            TrainingPhases(model,new,raw=dict(training={KEY:corrupt}),generation=210)
        corrupt=resumed.payload();corrupt['gae']['value_iterations']=1
        with self.assertRaisesRegex(ValueError,'predates the mode transition'):
            TrainingPhases(model,new,raw=dict(training={KEY:corrupt}),generation=210)

    def test_gae_transitions_need_real_ppo_work_health_twenty_iterations_and_rating(self):
        cfg=copy.deepcopy(gae.DEFAULT);state=gae.initial(cfg)
        def tick(steps=8,healthy=True,rating=True):
            nonlocal state
            state,r=gae.advance(state,cfg,generation=state['generation']+1,value_report=measurement(steps=steps),
                health_ok=healthy,rating=dict(ok=rating))
            return r
        for _ in range(12):tick(steps=0)
        self.assertEqual(state['active_lambda'],1.)
        for _ in range(10):tick()
        self.assertEqual(state['active_lambda'],.98)
        for _ in range(19):tick()
        self.assertEqual(state['active_lambda'],.98)
        r=tick(rating=False);self.assertEqual(r['reason'],'rating_evidence_not_ready')
        tick(healthy=False);self.assertEqual(state['healthy_streak'],0)
        for _ in range(19):tick()
        self.assertEqual(state['active_lambda'],.98)
        prior=copy.deepcopy(state)
        r=tick()
        replay,other=gae.advance(prior,cfg,generation=prior['generation']+1,value_report=measurement(),
            health_ok=True,rating=dict(ok=True))
        self.assertEqual((state,r),(replay,other));self.assertEqual(state['active_lambda'],.95)
        self.assertEqual(r['stage_evidence']['source'],'ppo')

    def test_mixed_table_ev_still_requires_games_and_updates(self):
        cfg=copy.deepcopy(gae.DEFAULT);state=gae.initial(cfg)
        for generation in range(1,11):
            value=measurement(steps=8)
            value['heldout_after']=ev_report(generation)['heldout_after']
            state,r=gae.advance(state,cfg,generation=generation,value_report=value,
                health_ok=True,rating=dict(ok=True))
        self.assertTrue(state['table_enabled']['historical'])
        self.assertTrue(state['table_enabled']['legacy'])
        self.assertEqual(r['table_gate']['legacy']['evidence']['source'],'ppo')
        # Sparse or entirely absent accepted critic work is not a bypass.
        empty=gae.initial(cfg)
        for generation in range(1,11):
            value=measurement(steps=0);value['heldout_after']=ev_report(generation)['heldout_after']
            empty,r=gae.advance(empty,cfg,generation=generation,value_report=value,
                health_ok=True,rating=dict(ok=True))
        self.assertFalse(empty['table_enabled']['legacy'])
        self.assertIn('insufficient_value_updates',r['table_gate']['legacy']['evidence']['reasons'])

    def test_snapshot_adoption_and_weight_recovery_without_ppg(self):
        cfg=copy.deepcopy(critic_data.DEFAULT);state=critic_data.initial(cfg)
        value=measurement()
        state,r=critic_data.advance(state,cfg,generation=1,report=value,healthy=True,fingerprint_now=None)
        self.assertTrue(state['snapshots_active'])
        state['snapshot_weight']=.5
        for generation in range(2,12):
            state,r=critic_data.advance(state,cfg,generation=generation,report=value,healthy=True,
                fingerprint_now=None,ppo_snapshot_rows=20)
            if generation==6:self.assertEqual(state['snapshot_weight'],.75)
        self.assertEqual(state['snapshot_weight'],1.)
        self.assertEqual(r['value_snapshot_distinct_used'],0)
        self.assertEqual(r['ppo_snapshot_distinct_used'],20)
        self.assertEqual(r['critic_training']['source'],'ppo')

    def test_ppo_only_warmup_empty_nan_stop_and_economic_guards_still_apply(self):
        model,opt,cfg,_=fixture();cfg=value_phase.disabled_options(upgraded(cfg))
        for failure in (None,'warmup','empty','nan','stop','kl','health','missing_ev'):
            with self.subTest(failure=failure):
                phase=TrainingPhases(model,cfg);phase.state=gae.initial(cfg['advantage'],9)
                phase.state.update(value_iterations=9,ev_streak=4)
                phase.critic_state['generation']=9;phase.value_report=measurement()
                from brass_ai.recovery.snapshot_r2 import initial
                phase.critic_state['weight_recovery']=initial(9,phase.critic_state['snapshot_weight'])
                update=dict(optimizer_steps=1,controller_warmup_done=True,kl_early_stop=False,
                    categorical_probe=dict(categorical_kl=.001),gae={})
                if failure=='warmup':update['controller_warmup_done']=False
                if failure=='empty':update['optimizer_steps']=0;phase.value_report['critic_training']['optimizer_steps']=0
                if failure=='nan':update['categorical_probe']['categorical_kl']=float('nan')
                if failure=='kl':update['categorical_probe']['categorical_kl']=.02
                if failure=='stop':update['kl_early_stop']=True
                if failure=='missing_ev':phase.value_report['heldout_after']={}
                phase.finish(generation=10,update=update,anchor_gate=dict(ok=failure!='health'),rating=dict(ok=True))
                self.assertEqual(phase.state['active_lambda'],.98 if failure is None else 1.)

    def test_disabled_disk_cache_retirement_is_postcommit_and_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);cache=run/'.ppg-value-cache';cache.mkdir()
            write_json(cache/'owner.json',value_buffer.OWNER)
            (cache/'generation-000210.npy').write_bytes(b'old derived data')
            write_json(run/'state.json',dict(iteration=210,signature=dict(config=dict(value_phase=dict(enabled=True)))))
            with self.assertRaises(ValueError):value_buffer.retire_disabled(SOURCE_ROOT,run,211)
            self.assertTrue((cache/'generation-000210.npy').exists())
            write_json(run/'state.json',dict(iteration=211,signature=dict(config=dict(value_phase=dict(enabled=False)))))
            (cache/'unknown.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Unknown PPG cache artifact'):
                value_buffer.retire_disabled(SOURCE_ROOT,run,211)
            self.assertTrue((cache/'generation-000210.npy').exists())
            (cache/'unknown.txt').unlink()
            report=value_buffer.retire_disabled(SOURCE_ROOT,run,211)
            self.assertEqual(report['deleted_files'],['generation-000210.npy'])
            self.assertTrue((cache/'owner.json').exists())
            self.assertEqual(value_buffer.retire_disabled(SOURCE_ROOT,run,211)['deleted_files'],[])


if __name__=='__main__':unittest.main()
