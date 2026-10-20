"""Snapshot critic weight recovery, evidence isolation and committed continuity."""
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import critic_data,gae,updates,snapshot_r2
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.history import compact
from brass_ai.recovery.retention import _same_payload
from test_critic_descriptors import mixed_fixture,partition
from test_gae_value_phase import fixture,examples,ppo,finish
from test_selfplay55_review import phase_fixture


def active(weight=.5,generation=85):
    cfg=copy.deepcopy(critic_data.DEFAULT);s=critic_data.initial(cfg)
    s.update(generation=generation,snapshots_active=True,snapshot_weight=weight,
        learner_ev_baseline=.6,activation_generation=56 if generation>=56 else 0)
    s['weight_recovery']=critic_data.recovery_initial(generation)
    return cfg,s


def report(ev=.6,*,steps=0,rows=0,trip=False):
    return dict(optimizer_steps=steps,attempted_steps=3,kl_guard_tripped=trip,
        role_usage=dict(snapshot=dict(distinct_used=rows)),
        heldout_after=dict(groups={'role:learner':dict(explained_variance=ev)}))


def advance(s,c,*,ev=.6,healthy=True,ppo_rows=8,value_rows=0,value_steps=0,trip=False):
    return critic_data.advance(s,c,generation=s['generation']+1,healthy=healthy,
        fingerprint_now=None,ppo_snapshot_rows=ppo_rows,
        report=report(ev,steps=value_steps,rows=value_rows,trip=trip))


class Recovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_two_separate_five_iteration_stages_and_cap(self):
        c,s=active();original=copy.deepcopy(s)
        for generation in range(86,99):
            s,r=advance(s,c)
            expected=.5 if generation<90 else .75 if generation<95 else 1.
            self.assertEqual(s['snapshot_weight'],expected)
            self.assertEqual(r['weight_used'],.5 if generation<=90 else .75 if generation<=95 else 1.)
            self.assertEqual(r['decision'],'increase_snapshot_weight' if generation in (90,95) else 'hold')
            self.assertEqual(s['weight_recovery']['healthy_streak'],
                generation-85 if generation<90 else generation-90 if generation<95 else 0)
            if generation in (90,95):self.assertEqual(r['weight_recovery']['evidence_streak'],5)
        self.assertEqual(s['weight_recovery']['last_change_generation'],95)
        self.assertIsNone(r['weight_recovery']['next_weight'])
        self.assertEqual(original['snapshot_weight'],.5)

    def test_health_ev_or_exposure_failure_resets_streak(self):
        c,s=active()
        for _ in range(4):s,_=advance(s,c)
        failures=[dict(healthy=False),dict(ev=None),dict(ev=float('nan')),dict(ev=float('inf')),
            dict(ev=.599),dict(ppo_rows=0),dict(ppo_rows=0,value_steps=0,value_rows=99),dict(trip=True)]
        for kwargs in failures:
            with self.subTest(kwargs=kwargs):
                state,r=advance(s,c,**kwargs)
                self.assertEqual(state['snapshot_weight'],.5)
                self.assertEqual(state['weight_recovery']['healthy_streak'],0)
                self.assertFalse(r['weight_recovery']['eligible'])
                for _ in range(4):state,_=advance(state,c)
                self.assertEqual(state['snapshot_weight'],.5)
                state,_=advance(state,c);self.assertEqual(state['snapshot_weight'],.75)

    def test_accepted_value_exposure_or_ppo_exposure_each_qualifies(self):
        for kwargs in (dict(ppo_rows=0,value_rows=7,value_steps=1),dict(ppo_rows=7,value_steps=0)):
            c,s=active()
            for _ in range(5):s,r=advance(s,c,**kwargs)
            self.assertEqual(s['snapshot_weight'],.75)
            self.assertTrue(r['snapshot_rows_used'])

    def test_reduction_hysteresis_and_recovery_after_relapse(self):
        for weight in (.75,1.):
            c,s=active(weight)
            same,r=advance(s,c,ev=.6-.02)
            self.assertEqual(same['snapshot_weight'],weight)
            self.assertEqual(same['weight_recovery']['healthy_streak'],0)
            no_exposure,_=advance(s,c,ev=.579,ppo_rows=0)
            self.assertEqual(no_exposure['snapshot_weight'],weight)
            s,r=advance(s,c,ev=.579,healthy=False)
            self.assertEqual((s['snapshot_weight'],r['decision']),(.5,'reduce_snapshot_weight'))
            for _ in range(4):s,_=advance(s,c)
            self.assertEqual(s['snapshot_weight'],.5)
            s,_=advance(s,c);self.assertEqual(s['snapshot_weight'],.75)

    def test_unknown_baseline_and_learner_only_cannot_recover(self):
        c,s=active();s['learner_ev_baseline']=None
        for _ in range(10):s,r=advance(s,c)
        self.assertEqual(s['snapshot_weight'],.5)
        self.assertEqual(r['reason'],'snapshot_recovery_ev_unavailable')
        c['rollout']['mode']='learner_only';s['snapshots_active']=False
        s['learner_ev_baseline']=.6
        for _ in range(10):s,r=advance(s,c)
        self.assertEqual(s['weight_recovery']['healthy_streak'],0)
        self.assertEqual(r['reason'],'configured_learner_only')

    def test_restart_mid_both_stages_exact_and_no_randomness(self):
        c,s=active()
        for n in (4,5):
            for _ in range(n):s,_=advance(s,c)
            b=io.BytesIO();torch.save(s,b);b.seek(0);restored=torch.load(b,weights_only=False)
            trng=torch.get_rng_state().clone();nrng=np.random.get_state()
            a,ar=advance(s,c);b,br=advance(restored,c)
            self.assertEqual((a,ar),(b,br));self.assertTrue(torch.equal(trng,torch.get_rng_state()))
            after=np.random.get_state();self.assertEqual(nrng[0],after[0])
            np.testing.assert_array_equal(nrng[1],after[1]);self.assertEqual(nrng[2:],after[2:])

    def test_strict_versioned_state_and_no_backfill(self):
        c,s=active();old=copy.deepcopy(s);old.pop('weight_recovery')
        before=copy.deepcopy(old);new=critic_data.migrate_state(old,c,85)
        self.assertEqual(old,before);self.assertEqual(new,s)
        for changes in (dict(healthy_streak=5),dict(healthy_streak=True),dict(origin=86),
                        dict(last_change_generation=84),dict(contract='unknown'),dict(extra=0),dict(healthy_streak=1)):
            bad=copy.deepcopy(new);bad['weight_recovery'].update(changes)
            with self.assertRaises(ValueError):critic_data.validate_state(bad,c,85)
        with self.assertRaises(ValueError):critic_data.migrate_state(new,c,85)
        for weight in (.7,True):
            bad=copy.deepcopy(new);bad['snapshot_weight']=weight
            with self.assertRaises(ValueError):critic_data.validate_state(bad,c,85)

    def test_full_health_gate_and_lr_budget_independence(self):
        cases=[({},True),(dict(optimizer_steps=0),False),(dict(controller_warmup_done=False),False),
            (dict(kl_early_stop=True),False),(dict(categorical_probe=dict(categorical_kl=.016)),False)]
        for overrides,eligible in cases:
            p,u=phase_fixture(85);_,p.critic_state=active()
            u.update(critic_rows=dict(snapshot_distinct_used=8),**overrides)
            p.value_report=report()
            gate=dict(ok=False,reasons=['anchor_budget_exceeded'],health=dict(ok=True))
            p.finish(generation=86,update=u,anchor_gate=gate,rating=dict(ok=True))
            self.assertEqual(p.critic_state['weight_recovery']['healthy_streak'],int(eligible))
        for gate,trip in ((dict(health=dict(ok=False)),False),(dict(health=dict(ok=True)),True)):
            p,u=phase_fixture(85);_,p.critic_state=active();u['critic_rows']=dict(snapshot_distinct_used=8)
            p.value_report=report(trip=trip)
            p.finish(generation=86,update=u,anchor_gate=gate,rating=dict(ok=True))
            self.assertEqual(p.critic_state['weight_recovery']['healthy_streak'],0)

    def test_compact_history_and_console_keep_decision_and_streak(self):
        c,s=active();s,r=advance(s,c)
        self.assertEqual(compact(dict(value_phase=dict(critic_rollout=r)))['value_phase']['critic_rollout'],r)
        self.assertEqual(critic_data.recovery_console(r),' snapshot_recovery=1/5:hold')
        self.assertEqual(critic_data.recovery_console({}),'')


class CheckpointIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_legacy_upgrade_preserves_every_existing_state_and_new_resume_streak(self):
        model,opt,cfg,(b,data)=fixture();phase=TrainingPhases(model,cfg)
        r=ppo(model,opt,cfg,b,data,phase);phase.fit(model);finish(phase,r,1)
        # Simulate the exact old v4 schema, with populated PPO/value Adam state.
        _,phase.critic_state=active(generation=1)
        raw=dict(training={KEY:phase.payload()})
        raw['training'][KEY]['contract']='actor-value-phases/4.0.0'
        old=raw['training'][KEY]['critic_data'];old.pop('weight_recovery')
        original=copy.deepcopy(raw);weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
        rng=torch.get_rng_state().clone()
        a=TrainingPhases(model,cfg,raw=raw,generation=1);b=TrainingPhases(model,cfg,raw=raw,generation=1)
        self.assertEqual(a.contract,'actor-value-phases/5.0.0')
        self.assertEqual({k:v for k,v in a.critic_state.items() if k!='weight_recovery'},
                         dict(old,learner_ev_baseline=None))
        self.assertEqual(a.critic_state['weight_recovery'],snapshot_r2.initial(1,.5,.6))
        self.assertTrue(_same_payload(a.payload(),b.payload()));self.assertTrue(_same_payload(raw,original))
        self.assertTrue(_same_payload(model.state_dict(),weights));self.assertTrue(_same_payload(opt.state_dict(),adam))
        self.assertTrue(_same_payload(a.optimizer.state_dict(),phase.optimizer.state_dict()))
        self.assertEqual((a.state,a.beta,a.trips),(phase.state,phase.beta,phase.trips))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for generation in range(2,5):
            a.critic_state,_=advance(a.critic_state,cfg['critic_data'])
            a.state,_=gae.advance(a.state,cfg['advantage'],generation=generation,
                value_report=report(),health_ok=True,rating=dict(ok=True))
        resumed=TrainingPhases(model,cfg,raw=dict(training={KEY:a.payload()}),generation=4)
        self.assertTrue(_same_payload(a.payload(),resumed.payload()));self.assertIsNone(resumed.critic_migration)
        bad=copy.deepcopy(a.payload());bad['critic_data'].pop('weight_recovery')
        with self.assertRaisesRegex(ValueError,'critic rollout checkpoint'):
            TrainingPhases(model,cfg,raw=dict(training={KEY:bad}),generation=4)

    def test_combined_old_ev_and_critic_migrations(self):
        model,_,cfg,_=fixture();old_cfg=copy.deepcopy(cfg)
        old_cfg['advantage']['table_gate']=copy.deepcopy(gae.ARCHIVED_TABLE_GATE)
        old=TrainingPhases(model,old_cfg);old.state['generation']=85
        _,old.critic_state=active();payload=old.payload();payload['contract']='actor-value-phases/3.0.0'
        payload['critic_data'].pop('weight_recovery');before=copy.deepcopy(payload)
        upgraded=TrainingPhases(model,cfg,raw=dict(training={KEY:payload}),generation=85)
        self.assertEqual(upgraded.critic_state['weight_recovery'],snapshot_r2.initial(85,.5,.6))
        self.assertEqual(upgraded.critic_state['snapshot_weight'],.5)
        self.assertEqual(upgraded.state['table_evidence']['origin'],85)
        self.assertTrue(_same_payload(payload,before))

    def test_three_quarter_weight_applies_to_ppo_and_cached_value_rows_only(self):
        from brass_ai.recovery import value_buffer
        from test_value_buffer import source,commit
        import tempfile
        from pathlib import Path
        model,opt,cfg,b,data,snap,_=mixed_fixture();cfg['value_phase']['buffer_iterations']=2
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=Path(tmp));p.critic_state['snapshot_weight']=.5
            p.critic_state['weight_recovery']=snapshot_r2.initial(0,.5)
            p.snapshot_data=snap;p.collection_source=dict(source(b),snapshot_rows=True)
            update=ppo(model,opt,cfg,b,data,p)
            p.fit(model);finish(p,update,1);commit(Path(tmp),p,1)
            # Reuse the disk buffer through fit: old rows receive the CURRENT
            # weight, so recovery does not leave cached snapshots at 0.5.
            b,rows=examples(model,games=9,decisions=4,prefix='recovery-second-')
            data,snap,_=partition(rows,'recovery-second-');p.snapshot_data=snap
            p.critic_state['snapshot_weight']=.75;p.collection_source=dict(source(b),snapshot_rows=True)
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,.75)
            update=ppo(model,opt,cfg,b,data,p)
            self.assertEqual(p.critic_dataset.critic_weights.tolist(),[.75 if r=='snapshot' else 1. for r in p.critic_dataset.roles])
            self.assertEqual(update['critic_rows']['snapshot_weight'],.75)
            seen=[]
            def inspect(*args,**kwargs):
                ds=kwargs['value_dataset'];seen.append(ds)
                self.assertEqual(ds.critic_weights.tolist(),[.75 if r=='snapshot' else 1. for r in ds.roles])
                return {}
            with patch('brass_ai.recovery.value_phase.fit',side_effect=inspect):p.fit(model)
            self.assertIsInstance(seen[0],value_buffer.ValueDataset)


if __name__=='__main__':unittest.main()
