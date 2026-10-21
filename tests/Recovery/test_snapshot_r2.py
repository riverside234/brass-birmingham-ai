"""Bias-aware pooled evidence, stage isolation and exact snapshot-only recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery import critic_data, snapshot_r2 as r2, snapshot_r2_hotfix as hotfix
from brass_ai.recovery.actor_value import TrainingPhases, KEY
from brass_ai.recovery.ev_evidence import statistics
from brass_ai.recovery.history import compact
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_gae_value_phase import fixture, ppo, finish
from test_ppg_window import upgraded
from brass_ai.recovery import value_phase, critic_holdout

EXPORT = SOURCE_ROOT / 'results/joint7-v11-2-grouped-lr-critic-g0213-20261020T173708767483Z'


def value_report(generation, *, error=.5, bias=0., healthy=True, trained=True):
    groups = {}
    selection = dict(games={}, heldout_games=[])
    for kind, games in (('selfplay', 8), ('historical', 8), ('legacy', 2)):
        # Balanced target/error correlation, with independent constant bias.
        y = np.tile([-1., -1., 1., 1.], games)
        p = y + np.tile([-error, error, -error, error], games) + bias
        ids = [f'g{generation}-{kind}-{j//4}' for j in range(len(y))]
        groups['table:'+kind+'/role:learner'] = dict(ev_evidence=statistics(y, p, ids))
        selection['games'].update({g: kind for g in ids})
        selection['heldout_games'].extend(sorted(set(ids)))
    return dict(enabled=False, contract=value_phase.CONTRACT, optimizer_steps=0,
        kl_guard_tripped=not healthy, critic_training=dict(source='ppo', optimizer_steps=4 if trained else 0),
        role_usage=dict(snapshot=dict(distinct_used=0)), heldout_after=dict(groups=groups),
        split=dict(**copy.deepcopy(critic_holdout.DEFAULT), fraction=.05, selection=selection,
                   by_table={k:dict(games=n) for k,n in zip(r2.TABLES,(160,144,16))}))


def observe(generation, **kwargs):
    v = value_report(generation, **kwargs)
    return r2.entry(generation, v, healthy=kwargs.get('healthy', True), trained=kwargs.get('trained', True))


def tick(controller, generation, **kwargs):
    return r2.advance(controller, controller['stage_weight'], generation=generation,
        observation=observe(generation, **kwargs), current_regime=dict(source='ppo'),
        active=True, tolerance=.02)


class Statistics(unittest.TestCase):
    def test_bias_is_squared_error_not_centered_variance(self):
        y = np.array([-1., 0., 2., 5.])
        s = statistics(y, y+2., ['a','a','b','b'])
        got = r2.pool([s])
        self.assertEqual(got['learner_mse'], 4.)
        self.assertEqual(got['prediction_bias'], 2.)
        self.assertAlmostEqual(got['r2'], 1-4/np.var(y))
        self.assertNotEqual(got['r2'], 1.)  # The old EV is 1 for constant bias.

    def test_pool_matches_raw_arrays_unequal_counts_means_and_biases(self):
        ys = [np.array([-1., 1.]), np.arange(13.)+4., np.array([7.,8.,9.])]
        ps = [ys[0]+3., ys[1]+np.arange(13.)/10, ys[2]-2.]
        parts = [statistics(y,p,[f'{i}-{j//2}' for j in range(len(y))])
                 for i,(y,p) in enumerate(zip(ys,ps))]
        y=np.concatenate(ys); error=np.concatenate(ps)-y
        a=r2.pool(parts)
        self.assertAlmostEqual(a['learner_mse'],float(np.square(error).mean()),14)
        self.assertAlmostEqual(a['prediction_bias'],float(error.mean()),14)
        self.assertAlmostEqual(a['target_variance'],float(y.var()),14)
        self.assertAlmostEqual(a['r2'],float(1-np.square(error).mean()/y.var()),14)
        self.assertNotAlmostEqual(a['r2'],np.mean([r2.pool([s])['r2'] for s in parts]))
        with self.assertRaisesRegex(ValueError,'Repeated'):r2.pool(parts+[parts[0]])
        with self.assertRaisesRegex(ValueError,'Branch'):
            r2.pool([statistics([0.,1.],[0.,1.],['x/branch/a','x/branch/a'])])
        self.assertIsNone(r2.pool([statistics([1.,1.],[0.,0.],['a','a'])])['r2'])

    def test_entry_excludes_snapshot_and_branches_and_preserves_game_minima(self):
        v=value_report(1)
        expected=r2.entry(1,v,healthy=True,trained=True)
        v['heldout_after']['groups'].update({
            'role:snapshot':dict(ev_evidence=statistics([0.,1.],[99.,99.],['snap','snap'])),
            'source:branch':dict(ev_evidence=statistics([0.,1.],[99.,99.],['x/branch/a']*2))})
        self.assertEqual(r2.entry(1,v,healthy=True,trained=True),expected)
        v['enabled']=True  # Zero accepted PPG does not erase retained PPO exposure.
        self.assertTrue(r2.entry(1,v,healthy=True,trained=True)['qualified'])
        stats=v['heldout_after']['groups']['table:legacy/role:learner']['ev_evidence']
        stats['game_ids']=stats['game_ids'][:1]
        self.assertFalse(r2.entry(1,v,healthy=True,trained=True)['qualified'])
        stats['game_ids']=['not-held-out']
        with self.assertRaisesRegex(ValueError,'outside'):r2.entry(1,v,healthy=True,trained=True)


class Stages(unittest.TestCase):
    def test_report_distinguishes_committed_next_weight_from_recovery_target(self):
        s=r2.initial(0,.5)
        for g in range(1,16):
            s,w,_,_,report=tick(s,g)
            self.assertEqual(report['next_weight'],w)
            self.assertEqual(report['recovery_target_weight'],.75 if w==.5 else 1.)

    def test_existing_point_zero_two_tolerance_includes_bias(self):
        baseline=r2.summarize([observe(g) for g in range(1,11)])
        for degradation,expected in ((.0199,.75),(.0201,.5)):
            s=r2.initial(10,.75)
            s['regime']=dict(source='ppo')
            s['reference']=r2.reference(baseline,10,.5,s['regime'])
            for g in range(11,21):s,w,_,_,_=tick(s,g,bias=degradation**.5)
            self.assertEqual(w,expected)

    def test_minimum_ten_then_five_new_commits_and_empty_new_stage(self):
        s=r2.initial(0,.5,hotfix.OLD_EV)
        for g in range(1,16):
            s,w,decision,_,report=tick(s,g)
            self.assertEqual(w,.5 if g<15 else .75)
            if g<10:self.assertIsNone(s['reference'])
            if g==10:
                self.assertEqual(s['healthy_streak'],0)
                self.assertEqual(s['reference']['evidence']['contributing_generations'],list(range(1,11)))
            if g==15:
                self.assertEqual(decision,'increase_snapshot_weight')
                self.assertEqual(s['entries'],[])
                self.assertEqual(s['reference']['generation'],15)
        for g in range(16,30):
            s,w,decision,_,report=tick(s,g)
            self.assertEqual(w,.75 if g<29 else 1.)
            if g<25:self.assertFalse(report['learner']['supported'])
        self.assertEqual(s['reference']['evidence']['contributing_generations'],list(range(16,30)))
        self.assertEqual(s['reference']['weight'],.75)

    def test_cut_preserves_reference_and_does_not_reuse_old_stage_samples(self):
        s=r2.initial(0,.5)
        for g in range(1,16):s,*_=tick(s,g)
        ref=copy.deepcopy(s['reference'])
        for g in range(16,26):
            s,w,d,_,r=tick(s,g,bias=1.)
            self.assertEqual(w,.75 if g<25 else .5)
        self.assertEqual(d,'reduce_snapshot_weight')
        self.assertEqual(s['reference'],ref)
        self.assertEqual(s['entries'],[])
        for g in range(26,46):s,w,_,_,r=tick(s,g,error=.8)
        self.assertEqual(w,.5)
        self.assertEqual(s['reference'],ref)
        self.assertLess(r['learner']['r2'],ref['evidence']['r2'])
        self.assertEqual(len(s['entries']),20)

    def test_health_no_work_and_bias_deterioration_break_streak(self):
        for failure in (dict(healthy=False),dict(trained=False),dict(bias=2.)):
            s=r2.initial(0,.5)
            for g in range(1,14):s,*_=tick(s,g)
            self.assertEqual(s['healthy_streak'],3)
            s,w,_,_,report=tick(s,14,**failure)
            self.assertEqual(w,.5)
            self.assertEqual(s['healthy_streak'],0)
            self.assertFalse(report['eligible'])

    def test_only_committed_progress_and_regime_scoped_rolling_window(self):
        s=r2.initial(0,.5)
        for g in range(1,13):s,*_=tick(s,g)
        original=copy.deepcopy(s)
        a=tick(s,13); b=tick(json.loads(json.dumps(s)),13)
        self.assertEqual(a,b);self.assertEqual(s,original)
        with self.assertRaisesRegex(ValueError,'progress'):tick(a[0],13)
        with self.assertRaisesRegex(ValueError,'progress'):tick(s,14)
        changed=r2.advance(s,.5,generation=13,observation=observe(13),current_regime=dict(source='ppg'),
                          active=True,tolerance=.02)[0]
        self.assertEqual(len(changed['entries']),1)
        self.assertEqual(changed['healthy_streak'],0)
        self.assertEqual(changed['reference'],s['reference'])

    def test_minimum_qualifying_commits_not_just_overlapping_windows(self):
        s=r2.initial(0,.5)
        for g in range(1,31):
            s,w,_,_,rep=tick(s,g,trained=g%3==0)
            self.assertEqual(w,.5);self.assertFalse(rep['learner']['supported'])
        self.assertEqual(len(s['entries']),20)
        self.assertEqual(rep['learner']['contributing_generations'],[12,15,18,21,24,27,30])

    def test_reference_can_rise_before_increase_but_never_falls(self):
        s=r2.initial(0,.5)
        for g in range(1,11):s,*_=tick(s,g)
        for g in range(11,16):s,*_=tick(s,g,error=.2)
        self.assertGreater(s['reference']['evidence']['r2'],.75)
        ref=copy.deepcopy(s['reference'])
        for g in range(16,26):s,*_=tick(s,g,error=.9)
        self.assertEqual(s['reference'],ref)


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_migration_archives_EV_and_keeps_GAE_optimizers_RNG_exact(self):
        model,opt,cfg,(behavior,data)=fixture()
        cfg=critic_holdout.quota_options(value_phase.disabled_options(upgraded(cfg)))
        phase=TrainingPhases(model,cfg)
        report=ppo(model,opt,cfg,behavior,data,phase);phase.fit(model);finish(phase,report,1)
        saved=phase.payload()
        legacy=critic_data.initial(cfg['critic_data'])
        legacy.update(generation=1,snapshot_weight=.5,snapshots_active=True,learner_ev_baseline=hotfix.OLD_EV)
        legacy['weight_recovery']=critic_data.recovery_initial(1)
        saved['critic_data']=legacy
        before=copy.deepcopy(saved);weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
        rng=torch.get_rng_state().clone()
        new=TrainingPhases(model,cfg,raw=dict(training={KEY:saved}),generation=1)
        self.assertTrue(_same_payload(before,saved))
        self.assertEqual(new.state,saved['gae'])
        self.assertTrue(_same_payload(new.optimizer.state_dict(),saved['optimizer']))
        self.assertTrue(_same_payload(weights,model.state_dict()))
        self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(new.critic_state['learner_ev_baseline'])
        r=new.critic_state['weight_recovery']
        self.assertEqual(r['legacy_ev_baseline'],hotfix.OLD_EV)
        self.assertIsNone(r['reference'])
        restarted=TrainingPhases(model,cfg,raw=dict(training={KEY:new.payload()}),generation=1)
        self.assertTrue(_same_payload(restarted.payload(),new.payload()))
        self.assertIsNone(restarted.snapshot_r2_migration)

    def test_production_metrics_only_ordinary_learner_rows(self):
        from brass_ai.recovery.updates import Dataset
        from test_critic_descriptors import mixed_fixture
        model,_,cfg,_,data,snap,_=mixed_fixture()
        all_rows=copy.deepcopy(data+snap)
        branch=copy.deepcopy(data[:4])
        for i,row in enumerate(branch):
            row=list(row);row[2]=(f'parent/branch/{i}',row[2][1]);branch[i]=tuple(row)
        ds=Dataset(all_rows+branch,'cpu',policy=False)
        metrics=value_phase.metrics(model,ds,list(range(len(ds.data))),cfg['update'],evidence=True)
        parts=[g['ev_evidence'] for k,g in metrics['groups'].items() if k.startswith('table:') and k.endswith('/role:learner')]
        got=r2.pool(parts)
        learner=metrics['groups']['role:learner']
        self.assertEqual(got['rows'],learner['rows'])
        self.assertAlmostEqual(got['learner_mse'],learner['learner_utility_mse'],12)
        self.assertGreater(metrics['rows'],got['rows'])

    def test_report_compaction_keeps_R2_moments_and_reference(self):
        s=r2.initial(0,.5)
        for g in range(1,16):s,w,d,reason,report=tick(s,g)
        value=dict(critic_rollout=dict(weight_recovery=report,decision=d,reason=reason))
        self.assertEqual(compact(dict(value_phase=value))['value_phase'],value)

    def test_economic_PPO_and_PPG_guards_still_gate_the_new_R2_streak(self):
        from brass_ai.recovery import gae
        model,_,cfg,_=fixture()
        cfg=critic_holdout.quota_options(value_phase.disabled_options(upgraded(cfg)))
        state=r2.initial(0,.5)
        for g in range(1,14):state,*_=tick(state,g)
        for failure in (None,'economy','warmup','empty','early_stop','kl','nan','ppg_kl'):
            with self.subTest(failure=failure):
                phase=TrainingPhases(model,cfg)
                phase.state=gae.initial(cfg['advantage'],13)
                phase.critic_state.update(generation=13,snapshots_active=True,snapshot_weight=.5)
                phase.critic_state['weight_recovery']=copy.deepcopy(state)
                phase.value_report=value_report(14)
                update=dict(optimizer_steps=1,controller_warmup_done=True,kl_early_stop=False,
                    categorical_probe=dict(categorical_kl=.001),critic_rows=dict(snapshot_distinct_used=8),gae={})
                if failure=='warmup':update['controller_warmup_done']=False
                if failure=='empty':update['optimizer_steps']=0
                if failure=='early_stop':update['kl_early_stop']=True
                if failure=='kl':update['categorical_probe']['categorical_kl']=.016
                if failure=='nan':update['categorical_probe']['categorical_kl']=float('nan')
                if failure=='ppg_kl':phase.value_report['kl_guard_tripped']=True
                phase.critic_state['weight_recovery']['regime']=r2.regime(phase.value_report,update)
                phase.finish(generation=14,update=update,
                    anchor_gate=dict(ok=False,health=dict(ok=failure!='economy')),rating=dict(ok=True))
                self.assertEqual(phase.critic_state['weight_recovery']['healthy_streak'],4 if failure is None else 0)


class Hotfix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_g213_has_only_three_current_regime_iterations(self):
        report=hotfix.preview(EXPORT)
        self.assertIsNone(report['baseline'])
        e=report['evidence']
        self.assertEqual(e['contributing_generations'],[211,212,213])
        self.assertEqual(e['games'],54);self.assertEqual(e['rows'],14917)
        self.assertAlmostEqual(e['r2'],.3624826418501088,13)
        self.assertAlmostEqual(e['learner_mse'],.15825827904097786,13)
        self.assertEqual(report['missing_qualifying_iterations'],7)

    def test_exact_latest_ten_and_deferred_seed_never_convert_EV(self):
        state=read(EXPORT/'state.json');records=hotfix.reports(EXPORT,state)
        c=state['signature']['config']['critic_data']
        old=critic_data.initial(c);old.update(generation=213,snapshots_active=True,snapshot_weight=.5,learner_ev_baseline=hotfix.OLD_EV)
        old['weight_recovery']=critic_data.recovery_initial(213)
        seeded,report=hotfix.seeded(old,c,213,records)
        self.assertIsNone(seeded['weight_recovery']['reference'])
        expected_regime=seeded['weight_recovery']['regime']
        for g in range(214,221):
            observation=observe(g)
            s=seeded['weight_recovery']
            seeded['weight_recovery'],w,_,_,rep=r2.advance(s,.5,generation=g,observation=observation,
                current_regime=expected_regime,active=True,tolerance=.02)
            self.assertEqual(w,.5)
        ref=seeded['weight_recovery']['reference']
        self.assertEqual(ref['generation'],220)
        self.assertEqual(ref['evidence']['contributing_generations'],list(range(211,221)))
        self.assertNotEqual(ref['evidence']['r2'],hotfix.OLD_EV)
        # Ten-plus eligible archived commits: freeze ONLY the last ten.
        latest=records[213]
        extended={}
        for g in range(202,214):
            record=copy.deepcopy(latest)
            record['value_phase'].update(value_report(g))
            record['value_phase']['critic_rollout']=copy.deepcopy(latest['value_phase']['critic_rollout'])
            extended[g]=record
        new,r=hotfix.seeded(old,c,213,extended)
        self.assertEqual(r['evidence']['contributing_generations'],list(range(204,214)))
        self.assertEqual(new['weight_recovery']['reference']['evidence']['games'],180)
        self.assertEqual(new['weight_recovery']['healthy_streak'],0)

    def test_overlay_is_bound_idempotent_and_cannot_change_other_training_state(self):
        state=read(EXPORT/'state.json');records=hotfix.reports(EXPORT,state)
        model,opt,cfg,_=fixture()
        cfg=critic_holdout.quota_options(value_phase.disabled_options(upgraded(cfg)))
        old=critic_data.initial(cfg['critic_data'])
        old.update(generation=213,snapshots_active=True,snapshot_weight=.5,learner_ev_baseline=hotfix.OLD_EV)
        old['weight_recovery']=critic_data.recovery_initial(213)
        next_state,rep=hotfix.seeded(old,cfg['critic_data'],213,records)
        phase=TrainingPhases(model,cfg)
        raw=dict(step=213,model_id=state['history'][-1]['model_id'],training={KEY:dict(critic_data=old)})
        receipt=dict(contract=hotfix.CONTRACT,generation=213,checkpoint=copy.deepcopy(state['history'][-1]),
            before_digest=digest(old),after_digest=digest(next_state),controller=next_state['weight_recovery'],report=rep)
        state[hotfix.KEY]=receipt
        before=copy.deepcopy(phase.state);adam=copy.deepcopy(opt.state_dict());value_adam=copy.deepcopy(phase.optimizer.state_dict())
        rng=torch.get_rng_state().clone()
        hotfix.restore(state,raw,phase);hotfix.restore(state,raw,phase)
        self.assertEqual(phase.critic_state,next_state)
        self.assertEqual(phase.state,before)
        self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(_same_payload(value_adam,phase.optimizer.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        bad=copy.deepcopy(state);bad[hotfix.KEY]['controller']['reference']=dict(r2=hotfix.OLD_EV)
        with self.assertRaises(ValueError):hotfix.restore(bad,raw,phase)
        bad=copy.deepcopy(raw);bad['model_id']='wrong'
        with self.assertRaisesRegex(ValueError,'bound'):hotfix.restore(state,bad,phase)

    def test_real_disk_hotfix_preserves_checkpoint_and_backups_and_rejects_short_evidence(self):
        from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint
        from brass_ai.recovery import gae
        from brass_ai.step5.league import implementation_id
        for count in (3,10):
            with self.subTest(qualifying=count), tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                root=Path(tmp);directory=root/hotfix.RUN;directory.mkdir(parents=True)
                model,opt,cfg,(behavior,data)=fixture()
                cfg=critic_holdout.quota_options(value_phase.disabled_options(upgraded(cfg)))
                phase=TrainingPhases(model,cfg)
                actual=ppo(model,opt,cfg,behavior,data,phase)
                phase.fit(model);finish(phase,actual,1)
                phase.state=gae.initial(cfg['advantage'],10)
                phase.state.update(active_lambda=.95,stage_origin=10,
                    table_enabled=dict(historical=True,legacy=True),
                    table_origins=dict(historical=10,legacy=10))
                legacy=critic_data.initial(cfg['critic_data'])
                legacy.update(generation=10,snapshots_active=True,snapshot_weight=.5,learner_ev_baseline=hotfix.OLD_EV)
                legacy['weight_recovery']=critic_data.recovery_initial(10)
                phase.critic_state=legacy
                signature=dict(config=cfg,implementation=implementation_id())
                file=directory/'generation-0010.pt'
                mid=save_checkpoint(file,model,opt,step=10,training=dict(fresh_run=hotfix.RUN,
                    signature=digest(signature),**{KEY:phase.payload()}))
                history=[dict(generation=g,model_id=f'model-{g}',path=f'{hotfix.RUN}/generation-{g:04d}.pt',sha256='unused')
                         for g in range(10)]
                history.append(dict(generation=10,model_id=mid,path=file.relative_to(root).as_posix(),sha256=sha(file)))
                state=dict(format='step5-recovery-league/2.2.0',iteration=10,games_done=3200,history=history,signature=signature)
                for g in range(11-count,11):
                    v=value_report(g);v['critic_rollout']=dict(weight_used=.5,weight_next=.5,
                        snapshot_rows_used=True,weight_recovery=dict(health_ok=True))
                    r=dict(iteration=g-1,post_update_model_id=history[g]['model_id'],
                        incoming_model_id=history[g-1]['model_id'],value_phase=v,
                        actor=dict(gae=dict(lambda_by_table=dict(selfplay=.95,historical=.95,legacy=.95))))
                    write_json(directory/f'iteration-{g-1:04d}'/'report.json',r)
                write_json(directory/'state.json',state)
                before=sha(file);original=(directory/'state.json').read_bytes()
                preview=hotfix.run(root)
                self.assertEqual(preview['evidence']['qualifying_iterations'],count)
                self.assertEqual((directory/'state.json').read_bytes(),original)
                if count<10:
                    with self.assertRaisesRegex(ValueError,'Fewer than ten'):
                        hotfix.run(root,apply=True)
                    self.assertEqual((directory/'state.json').read_bytes(),original)
                applied=hotfix.run(root,apply=True,defer_until_ready=count<10)
                self.assertEqual(applied['baseline'],.75 if count==10 else None)
                updated=read(directory/'state.json')
                self.assertEqual({k:v for k,v in updated.items() if k!=hotfix.KEY},state)
                self.assertEqual((directory/'state-before-snapshot-r2-0010.json').read_bytes(),original)
                self.assertEqual(sha(file),before)
                self.assertTrue(hotfix.run(root,apply=True)['already_applied'])
                loaded,raw=load_checkpoint(file)
                resumed=TrainingPhases(loaded,cfg,raw=raw,generation=10,root=root,run=directory)
                gae_before=copy.deepcopy(resumed.state)
                hotfix.restore(updated,raw,resumed)
                self.assertEqual(resumed.state,gae_before)
                self.assertTrue(_same_payload(resumed.optimizer.state_dict(),phase.optimizer.state_dict()))
                self.assertEqual(resumed.critic_state['snapshot_weight'],.5)
                self.assertEqual(len(resumed.critic_state['weight_recovery']['entries']),count)


if __name__ == '__main__':unittest.main()
