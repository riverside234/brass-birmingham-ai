"""Eight-age disk replay, paired guards, uniform epochs and durable scheduling."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_config,ppg_replay,ppg_sampling,ppg_auxiliary,ppg_epochs,value_phase,updates,gae
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.critic_holdout import DEFAULT as HOLDOUT
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_gae_value_phase import fixture,examples,ppo,finish
from test_critic_descriptors import partition
from test_value_buffer import source,commit


def setup(preset='small'):
    model,opt,cfg,_=fixture(preset=preset,games=9,decisions=3)
    cfg=value_phase.interval_options(value_phase.refined_options(gae.health_options(cfg)))
    cfg['value_phase']['holdout']=copy.deepcopy(HOLDOUT)
    cfg['value_phase']['enabled']=False
    cfg['critic_data']['rollout']['mode']='snapshots'
    cfg['value_phase']['learning_rate']=3e-6
    cfg=ppg_config.configure(cfg)
    cfg['value_phase'].update(batch_size=32,clone_rows_per_age=24,clone_batch_size=16,
        probe_rows_per_age=12,current_probe_rows=24,acceptance_steps=2)
    return model,opt,cfg


def collect(model,phase,generation,*,branches=False):
    b,rows=examples(model,games=18,decisions=3,prefix=f'age{generation}-')
    learner,snapshot,_=partition(rows,f'age{generation}-')
    if branches:
        # Valid continuation-shaped rows, with no copied prefix/fork move.
        r,t,(g,s),e=learner[-1]
        first=next(x[0] for x in learner if x[2]==(g,s) and '_reward_trajectory' in x[0])
        trajectory=dict(first['_reward_trajectory'],cost=r['_insolvency_cost_to_go'],utility=r['_reward_return'][s])
        learner.append((dict(r,_branch=True,_parent_game_id=g,_reward_trajectory=trajectory,
            _insolvency_trajectory=dict(events=0,shortfall=0,cost=0.,by_era={})),t,(g+'/branch/test',s),e))
    phase.snapshot_data=snapshot;phase.collection_source=dict(source(b),snapshot_rows=True)
    return b,learner


def stage_only(model,phase,generation):
    b,data=collect(model,phase,generation,branches=True)
    ds=updates.Dataset(data,'cpu');phase.prepare(ds)
    actor=[i for i in range(len(data)) if len(data[i][3].choice_ids)>1]
    phase.after_ppo(model,actor,actor[:4],{},critic_steps=1,critic_rows=len(phase.train_ids))
    cache=phase.buffer.stage(phase.critic_dataset,phase.train_ids,phase.heldout_ids,actor,
        phase.collection_source,model,phase.split['selection'])
    return cache


class Pure(unittest.TestCase):
    def test_schedule_restart_origin_eight_fresh_and_uniform_age_weights(self):
        s=ppg_config.initial(418);phases=[]
        for g in range(419,443):
            self.assertEqual(ppg_config.due(s),ppg_config.due(json.loads(json.dumps(s))))
            run=ppg_config.due(s)
            if run:phases.append(g)
            s=ppg_config.advance(s,g,run)
        self.assertEqual(phases,[426,434,442])
        self.assertEqual([ppg_config.weight(a,ppg_config.DEFAULT) for a in range(8)],[1.]*8)
        with self.assertRaises(ValueError):ppg_config.advance(s,444,False)
        with self.assertRaises(ValueError):ppg_config.advance(ppg_config.initial(418),419,True)

    def test_paired_game_loss_one_SE_and_mean_not_row_weighted(self):
        self.assertFalse(ppg_auxiliary.paired({'a':1.,'b':1.},{'a':.8,'b':1.3})['reject'])
        self.assertTrue(ppg_auxiliary.paired({'a':1.,'b':1.},{'a':1.2,'b':1.2})['reject'])
        self.assertTrue(ppg_auxiliary.paired({'a':1.},{'a':.9})['improved'])
        with self.assertRaises(ValueError):ppg_auxiliary.paired({'a':1.},{'b':1.})

    def test_config_exact_and_unrelated_preserved(self):
        from brass_ai.recovery.config import validate_config
        c=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-entropy-groups.json')
        n=ppg_config.configure(c);validate_config(n)
        self.assertEqual({k:v for k,v in c.items() if k!='value_phase'},{k:v for k,v in n.items() if k!='value_phase'})
        self.assertEqual(n,ppg_config.configure(n))
        for k,v in [('half_life',3),('weighting','recency'),('kl_cap',.003),('buffer_iterations',9),
                    ('phase_interval',True),('phase_interval',4),('validation_iterations',4),('max_epochs',3),
                    ('acceptance_steps',33),('beta_clone_max',64)]:
            bad=copy.deepcopy(n['value_phase']);bad[k]=v
            with self.assertRaises(ValueError):ppg_config.validate(bad)

    def test_stratum_correction_and_disjointness_full_quotas(self):
        # All 12 table/decision strata, heavily uneven populations.
        from types import SimpleNamespace
        rows=[]
        for t in ('selfplay','historical','legacy'):
            for era in ('CanalEra','RailEra'):
                for phase in ('action','merchant'):
                    n=900 if t=='selfplay' else 600
                    for j in range(n):
                        r=dict(_table_kind=t,view=dict(context=dict(era=era,phase=phase)),behavior={})
                        rows.append((r,None,(t+era+phase,j),SimpleNamespace(choice_ids=('a','b'))))
        selected,pop=ppg_sampling.select(rows,range(len(rows)),ppg_config.DEFAULT)
        entries=[dict(e,row=e['index']) for e in selected]
        a,w,sa=ppg_sampling.phase_indices(entries,pop,2048,'clone')
        b,v,sb=ppg_sampling.phase_indices(entries,pop,512,'probe')
        self.assertEqual(len(a),2048);self.assertEqual(len(b),512);self.assertFalse(set(a)&set(b))
        self.assertAlmostEqual(sum(w),1.);self.assertAlmostEqual(sum(v),1.)
        for k,x in sb['strata'].items():
            self.assertGreater(x['selected'],0);self.assertAlmostEqual(x['weight_mass'],pop[k]/sum(pop.values()))

    def test_uniform_epoch_covers_every_age_once_including_unequal_sizes(self):
        train={a:list(range(a*1000,(a+1)*1000-10*a)) for a in range(8)}
        order,total,counts=ppg_auxiliary.epoch_order(train,ppg_config.DEFAULT)
        self.assertEqual(len(order),len(set(order)))
        self.assertEqual(set(order),{i for ids in train.values() for i in ids})
        self.assertEqual(counts,{str(a):len(ids) for a,ids in train.items()})
        self.assertEqual(total,sum(map(len,train.values())))

    def test_epoch_shuffles_and_32_step_intervals_keep_short_final_batches(self):
        cfg=dict(ppg_config.DEFAULT,batch_size=8)
        train={0:list(range(259))};blocks=list(ppg_auxiliary.intervals(train,cfg,2))
        self.assertEqual([(e,len(b),end) for e,_,b,end in blocks],[(0,32,False),(0,1,True),(1,32,False),(1,1,True)])
        for epoch in range(2):
            rows=[i for e,_,b,_ in blocks if e==epoch for batch in b for i in batch]
            self.assertEqual(sorted(rows),train[0]);self.assertEqual(len(rows),len(set(rows)))

    def test_value_rate_warmup_then_decay_uses_planned_progress(self):
        cfg=ppg_config.DEFAULT
        rates=[ppg_config.phase_rate(cfg,s,506) for s in range(1,507)]
        self.assertAlmostEqual(rates[0],3e-7);self.assertAlmostEqual(rates[31],3e-6)
        self.assertAlmostEqual(rates[-1],9e-7)
        self.assertTrue(all(a<b for a,b in zip(rates[:31],rates[1:32])))
        self.assertTrue(all(a>b for a,b in zip(rates[31:-1],rates[32:])))
        self.assertEqual(ppg_config.phase_rate(cfg,80,506),rates[79])
        with self.assertRaises(ValueError):ppg_config.phase_rate(cfg,507,506)
        with self.assertRaises(ValueError):ppg_config.phase_rate(cfg,0,506)


class EpochGate(unittest.TestCase):
    cfg=ppg_config.DEFAULT

    def step(self,state,best,*,kl=False):
        report=dict(scheduled=True,epoch_cap_used=state['cap'],steps_per_epoch=100,
            planned_steps=state['cap']*100,best_budget_step=best,kl_guard_tripped=kl)
        return ppg_epochs.advance(state,report,self.cfg,state['last_phase']+8)

    def test_three_of_four_late_best_increases_and_clears_stage_window(self):
        state=ppg_epochs.initial(418,self.cfg)
        for n,best in enumerate((150,200,120,190)):
            state,r=self.step(state,best)
            self.assertEqual(r['decision'],'up' if n==3 else 'hold')
        self.assertEqual(state['cap'],3);self.assertEqual(state['window'],[])
        self.assertEqual(r['late_best_phases'],3);self.assertEqual(state['window_origin'],4)
        state,r=self.step(state,300)
        self.assertEqual(state['cap'],3);self.assertEqual(len(state['window']),1)

    def test_first_epoch_decreases_floor_and_late_best_can_leave_floor(self):
        state=ppg_epochs.initial(0,self.cfg)
        for best in (100,0,160,1):state,r=self.step(state,best)
        self.assertEqual(state['cap'],1);self.assertEqual(r['decision'],'down')
        for _ in range(4):state,r=self.step(state,0)
        self.assertEqual(state['cap'],1)
        for _ in range(4):state,r=self.step(state,state['cap']*100)
        self.assertEqual(state['cap'],2)

    def test_restart_window_maximum_and_KL_limited_phases(self):
        state=ppg_epochs.initial(418,self.cfg)
        for best in (200,150,100):state,r=self.step(state,best)
        self.assertEqual(self.step(state,190),self.step(json.loads(json.dumps(state)),190))
        state=ppg_epochs.initial(0,self.cfg)
        for kl in (False,False,True,True):state,r=self.step(state,200,kl=kl)
        self.assertEqual(state['cap'],2);self.assertEqual(r['late_best_phases'],2)
        for _ in range(24):state,r=self.step(state,state['cap']*100)
        self.assertEqual(state['cap'],6)
        with self.assertRaises(ValueError):ppg_epochs.advance(state,dict(scheduled=False),self.cfg,state['last_phase']+1)
        bad=copy.deepcopy(state);bad['window'][0]['generation']+=1
        with self.assertRaises(ValueError):ppg_epochs.validate(bad,self.cfg)


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)/'run'

    def test_mmap_targets_descriptors_holdout_branches_and_deletion_after_commit(self):
        model,opt,cfg=setup();p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=self.run)
        for g in range(1,10):
            r=stage_only(model,p,g)
            if g==9:
                self.assertTrue(p.buffer.path(1,'critic.npy').exists())
                with self.assertRaisesRegex(ValueError,'commit'):p.buffer.committed(g)
            with p.buffer.datasets('cpu') as (critic,policy,receipts):
                self.assertIsInstance(critic.data.parts[0].array,np.memmap)
                tr,held,validation,ages,support=ppg_auxiliary.layout(critic,receipts,cfg['value_phase'],.5)
                held_games=set(p.split['selection']['heldout_games'])
                self.assertTrue(all(critic.data[i][2][0] not in held_games for i in tr[0]))
                self.assertTrue(all('/branch/' not in critic.data[i][2][0] and critic.roles[i]=='learner' for i in held))
                self.assertEqual(validation,held)
                self.assertEqual(set(ages[i] for i in validation),set(range(min(g,8))))
                self.assertEqual(len(ppg_auxiliary.by_game(critic,{i:0. for i in validation})),min(g,8)*6)
                for i in (0,len(critic.data.parts[0])-1):
                    d=critic.data[i];actual=p.critic_dataset.data[i]
                    np.testing.assert_array_equal(d[3].state,actual[3].state)
                    self.assertEqual(critic.utility[[i]].tolist(),torch.tensor([actual[0]['_reward_return']]).tolist())
                    np.testing.assert_array_equal(d[0]['_seat_descriptors'],actual[0]['_seat_descriptors'])
                self.assertEqual(len(policy.data.parts[0]),len(receipts[0]['policy_entries']))
                self.assertTrue(all(policy.data[i][2][0] not in held_games for i in range(len(policy.data.parts[0]))))
            write_json(self.run/'state.json',dict(iteration=g));p.buffer.committed(g)
            if g==9:self.assertFalse(p.buffer.path(1,'critic.npy').exists())
        restored=ppg_replay.Buffer(SOURCE_ROOT,self.run,cfg,generation=9,receipts=p.buffer.receipts,origin=0)
        self.assertEqual(len(restored.receipts),8)
        path=restored.path(9,'critic.npy');path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'corrupt'):ppg_replay.Buffer(SOURCE_ROOT,self.run,cfg,generation=9,receipts=restored.receipts)

    def test_full_eight_age_policy_support_raw_reference_and_masks(self):
        model,opt,cfg=setup();p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=self.run)
        for g in range(1,9):
            stage_only(model,p,g)
            if g<8:write_json(self.run/'state.json',dict(iteration=g));p.buffer.committed(g)
        with p.buffer.datasets('cpu') as (critic,policy,receipts):
            rng=torch.get_rng_state().clone();ref=ppg_auxiliary.Reference(model,policy,receipts,cfg['value_phase'],cfg['update'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertFalse(set(ref.clone_ids)&set(ref.probe_ids));m=ref.probes(model)
            self.assertLess(m['overall'],1e-6);self.assertLess(m['age0'],1e-6)
            self.assertEqual(len(ref.clone_ids),8*24);self.assertEqual(len(ref.probe_ids),24+7*12)
            population=[sum(r['policy_population'].values()) for r in receipts]
            for age in range(8):
                expected=population[age]/sum(population)
                for split in ('clone','probe'):
                    self.assertAlmostEqual(sum(w for i,w in zip(getattr(ref,split+'_ids'),getattr(ref,split+'_weights')) if ref.ages[i]==age),expected)
            for i in ref.ids:
                self.assertAlmostEqual(float(ref.rows[i].sum()),1.,places=5)
                self.assertEqual(len(ref.rows[i]),len(policy.rows[i]['choices']))
                self.assertEqual(policy.rows[i]['_critic_role'],'learner')

    def test_migration_preserves_Adam_RNG_progress_and_gates(self):
        model,opt,cfg=setup();old=copy.deepcopy(cfg)
        old['value_phase']=copy.deepcopy(value_phase.DEFAULT)
        old_phase=TrainingPhases(model,old)
        g=418;old_phase.state=gae.initial(old['advantage'],g)
        from brass_ai.recovery import critic_data,snapshot_r2
        old_phase.critic_state=critic_data.initial(old['critic_data']);old_phase.critic_state['generation']=g
        old_phase.critic_state['weight_recovery']=snapshot_r2.initial(g,old_phase.critic_state['snapshot_weight'])
        old_phase.critic_evidence['origin']=g
        old_phase.beta=8.
        param=next(model.parameters());param.grad=torch.ones_like(param);old_phase.optimizer.step();old_phase.optimizer.zero_grad(set_to_none=True)
        raw=dict(training={KEY:old_phase.payload()});before=copy.deepcopy(raw)
        rng=torch.get_rng_state().clone();weights=copy.deepcopy(model.state_dict())
        target=ppg_config.configure(old)
        resumed=TrainingPhases(model,target,raw=raw,generation=g,root=SOURCE_ROOT,run=self.run)
        self.assertTrue(_same_payload(raw,before));self.assertTrue(_same_payload(weights,model.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(_same_payload(old_phase.optimizer.state_dict(),resumed.optimizer.state_dict()))
        self.assertEqual(resumed.state,old_phase.state);self.assertEqual(resumed.critic_state,old_phase.critic_state)
        self.assertEqual(resumed.schedule,ppg_config.initial(418));self.assertFalse(ppg_config.due(resumed.schedule))
        self.assertEqual(resumed.beta,8.);self.assertEqual(resumed.epoch_control,ppg_epochs.initial(418,target['value_phase']))

    def test_frozen_reference_KL_violation_restores_best_weights_and_value_Adam(self):
        model,opt,cfg=setup();p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=self.run)
        for g in range(1,9):
            stage_only(model,p,g)
            if g<8:write_json(self.run/'state.json',dict(iteration=g));p.buffer.committed(g)
        origin=copy.deepcopy(model.state_dict());adam=copy.deepcopy(p.optimizer.state_dict());ppo_adam=copy.deepcopy(opt.state_dict())
        real=ppg_auxiliary.Reference.probes;calls=[]
        def cap(ref,net):
            result=real(ref,net);calls.append(1)
            if len(calls)==1:result['age0']=.003
            return result
        with patch.object(ppg_auxiliary.Reference,'probes',cap),patch.object(ppg_auxiliary,'paired',return_value=
                dict(games=48,mean_change=.1,standard_error=.01,reject=True,improved=False)):
            report=ppg_auxiliary.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                cfg['value_phase'],cfg['update'],beta=1.,snapshot_weight=1.,dropout=None,run_phase=True,ppo_work=p.ppo_work)
        self.assertTrue(report['kl_guard_tripped']);self.assertEqual(report['optimizer_steps'],0)
        self.assertTrue(_same_payload(model.state_dict(),origin));self.assertTrue(_same_payload(p.optimizer.state_dict(),adam))
        self.assertTrue(_same_payload(opt.state_dict(),ppo_adam));self.assertEqual(report['attempted_steps'],1)

    def test_best_interval_and_patience_restore_matching_Adam(self):
        model,opt,cfg=setup();p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=self.run)
        for g in range(1,9):
            stage_only(model,p,g)
            if g<8:write_json(self.run/'state.json',dict(iteration=g));p.buffer.committed(g)
        saved=[];real=p.optimizer.step;comparisons=[]
        def step(*a,**kw):
            result=real(*a,**kw)
            saved.append((copy.deepcopy(model.state_dict()),copy.deepcopy(p.optimizer.state_dict())))
            return result
        def comparison(*a):
            comparisons.append(1)
            self.assertEqual(len(a[0]),48)
            self.assertEqual(a[0].keys(),a[1].keys())
            return dict(games=48,mean_change=-.1 if len(comparisons)==1 else .01,
                standard_error=.02,reject=False,improved=len(comparisons)==1)
        with patch.object(p.optimizer,'step',step),patch.object(ppg_auxiliary,'paired',comparison):
            r=ppg_auxiliary.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                cfg['value_phase'],cfg['update'],beta=1.,snapshot_weight=1.,dropout=None,run_phase=True,ppo_work=p.ppo_work)
        self.assertEqual(r['early_stop_reason'],'validation_patience')
        self.assertEqual(r['optimizer_steps'],2);self.assertEqual(r['attempted_steps'],6)
        self.assertTrue(_same_payload(model.state_dict(),saved[1][0]));self.assertTrue(_same_payload(p.optimizer.state_dict(),saved[1][1]))
        self.assertEqual(r['value_presentations'],64)
        self.assertEqual(r['acceptance_generations'],list(range(8,0,-1)))
        self.assertEqual(r['acceptance_games'],48)
        self.assertEqual(value_phase.critic_work(r),p.ppo_work)

    def fit_retry(self,*,always_trip=False,two_epochs=False,seed_adam=False):
        model,opt,cfg=setup();cfg['value_phase'].update(batch_size=850,acceptance_steps=1)
        p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=self.run)
        for g in range(1,9):
            stage_only(model,p,g)
            if g<8:write_json(self.run/'state.json',dict(iteration=g));p.buffer.committed(g)
        if seed_adam:
            param=next(model.parameters());param.grad=torch.ones_like(param)
            p.optimizer.step();p.optimizer.zero_grad(set_to_none=True)
        origin=copy.deepcopy(model.state_dict());adam=copy.deepcopy(p.optimizer.state_dict())
        probes=ppg_auxiliary.Reference.probes;calls=[]
        def cap(ref,net):
            result=probes(ref,net);calls.append(1)
            if not two_epochs and len(calls)<=(3 if always_trip else 2):result['age0']=.003
            return result
        with patch.object(ppg_auxiliary.Reference,'probes',cap),patch.object(ppg_auxiliary,'paired',return_value=
                dict(games=48,mean_change=-.01,standard_error=.01,reject=False,improved=True)):
            r=ppg_auxiliary.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                cfg['value_phase'],cfg['update'],beta=8.,snapshot_weight=1.,dropout=None,
                run_phase=True,ppo_work=p.ppo_work,epoch_cap=2 if two_epochs else 1)
        return model,p,r,origin,adam

    def test_KL_improving_value_retries_same_interval_with_beta_8_16_32(self):
        model,p,r,origin,adam=self.fit_retry()
        self.assertEqual(r['beta_retries'],2);self.assertEqual(r['beta_clone_final'],32.)
        self.assertEqual(r['attempted_steps'],r['planned_steps']+2)
        self.assertEqual(r['optimizer_steps'],r['planned_steps'])
        self.assertEqual(r['best_budget_step'],r['planned_steps'])
        self.assertLessEqual(r['probe_kl'],.002);self.assertLessEqual(r['age0_probe_kl'],.002)
        self.assertEqual({int(s['step']) for s in p.optimizer.state.values()},{r['optimizer_steps']})
        self.assertTrue(all(x['retry'] for x in r['interval_reports'][:2]))

    def test_beta_bound_stops_unsafe_retries_and_keeps_best_Adam(self):
        model,p,r,origin,adam=self.fit_retry(always_trip=True)
        self.assertEqual(r['beta_retries'],2);self.assertEqual(r['attempted_steps'],3)
        self.assertEqual(r['early_stop_reason'],'policy_kl_cap_beta_max');self.assertEqual(r['optimizer_steps'],0)
        self.assertTrue(_same_payload(model.state_dict(),origin));self.assertTrue(_same_payload(p.optimizer.state_dict(),adam))
        self.assertLessEqual(r['age0_probe_kl'],.002)

    def test_two_epochs_count_short_batch_optimizer_steps_and_distinct_rows(self):
        model,p,r,_,_=self.fit_retry(two_epochs=True)
        self.assertEqual(r['epochs_completed'],2)
        self.assertEqual(r['optimizer_steps'],2*r['steps_per_epoch'])
        self.assertEqual({int(s['step']) for s in p.optimizer.state.values()},{r['optimizer_steps']})
        self.assertEqual(r['distinct_value_rows_used'],r['rows_per_epoch'])
        self.assertEqual(r['value_presentations'],2*r['rows_per_epoch'])
        self.assertEqual(r['best_budget_step'],r['planned_steps'])

    def test_phase_resets_stale_value_moments_and_warms_and_decays(self):
        model,p,r,_,old_adam=self.fit_retry(two_epochs=True,seed_adam=True)
        self.assertTrue(old_adam['state']);self.assertTrue(r['adam_reset']['reset'])
        self.assertEqual(r['adam_reset']['previous_max_step'],1)
        self.assertEqual({int(s['step']) for s in p.optimizer.state.values()},{r['optimizer_steps']})
        self.assertAlmostEqual(r['rate_schedule']['first_lr'],3e-7)
        self.assertAlmostEqual(r['rate_schedule']['final_lr'],9e-7)
        self.assertAlmostEqual(r['last_attempted_lr'],9e-7)
        self.assertAlmostEqual(p.optimizer.param_groups[0]['lr'],r['learning_rate'])

    def test_real_g418_transition_review_and_unrelated_config_rejection(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0418-20261021T181921767029Z/state.json')
        before=copy.deepcopy(state);config=ppg_config.configure(state['signature']['config'])
        signature=dict(state['signature'],implementation=implementation_id(),config=config)
        self.assertTrue(check_signature(state,signature,reviewed=True));self.assertEqual(state,before)
        with self.assertRaises(ValueError):check_signature(state,signature)
        for key,value in [('exploration',.08),('learning_rate',1e-3)]:
            bad=copy.deepcopy(signature);bad['config'][key]=value
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)


if __name__=='__main__':unittest.main()
