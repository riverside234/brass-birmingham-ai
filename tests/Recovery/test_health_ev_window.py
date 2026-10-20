"""Removed sale guard, bounded whole-game EV support and reviewed continuation."""
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
from brass_ai.recovery import ev_evidence as ev,gae,il_kl_limit as limits,il_kl_probe as probes,value_phase,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.history import compact
from brass_ai.recovery.retention import _same_payload
import test_auto_il_kl as auto_tests
from test_auto_il_kl import ToyPolicy,baseline,measurement,counts
from test_gae_value_phase import fixture,ppo,finish


def group(generation,kind,quality=.6,games=2,rows=16):
    target=np.tile([-.4,.4],rows//2)
    prediction=target+np.sqrt(1-quality)*target
    ids=[f'{generation}-{kind}-{i%games}' for i in range(rows)]
    return dict(explained_variance=quality,rows=rows,games=games,
        ev_evidence=ev.statistics(target,prediction,ids))


def report(generation,*,legacy=.6,historical=.6,steps=1,games=2,trip=False,enabled=True):
    groups={'table:selfplay/role:learner':group(generation,'selfplay')}
    for kind,q in (('historical',historical),('legacy',legacy)):
        if q is not None:groups['table:'+kind+'/role:learner']=group(generation,kind,q,games)
    return dict(enabled=enabled,optimizer_steps=steps,kl_guard_tripped=trip,heldout_after=dict(groups=groups))


def config():
    c=copy.deepcopy(gae.DEFAULT);c['gate']['mode']='fixed';c['lambda']=.95
    return c


def tick(s,c,healthy=True,**kwargs):
    g=s['generation']+1
    return gae.advance(s,c,generation=g,value_report=report(g,**kwargs),health_ok=healthy,rating=dict(ok=True))


class SaleRemoval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_sale_mass_can_fall_without_behavior_failure_or_cache_rebuild(self):
        rows=auto_tests.ProbeTests().data();model=ToyPolicy();cfg=copy.deepcopy(limits.DEFAULT)
        update=dict(microbatch_size=3,max_candidate_cells=100)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            root=Path(temporary);source=root/'validation';source.write_bytes(b'fixed dataset')
            path=root/'manifest.json';path.write_text(json.dumps(dict(entries=[dict(split='validation',path='validation',sha256=sha(source))])))
            origin=dict(manifest='manifest.json',manifest_sha256=sha(path),model_id='a'*64,sha256='b'*64)
            with patch('brass_ai.recovery.demonstrations.load_dataset',return_value=(rows,{})):
                probe,receipt=probes.load(root,root/'run',origin,model,cfg,update,1)
            cache=probe.metadata['cache'];before=sha(root/cache['path'])
            # IL sale mass ~.245; now ~.00012. The pass mass also falls.
            with torch.no_grad():model.bias.copy_(torch.tensor([-4.,-4.,5.]))
            current=probe.measure(model,update)
            self.assertNotIn('behavior',current);self.assertGreater(current['kl_mean'],0)
            loaded,again=probes.load(root,root/'run',origin,model,cfg,update,1,receipt)
            self.assertEqual(again,receipt);self.assertEqual(sha(root/cache['path']),before)
            self.assertNotIn('behavior',loaded.measure(model,update))
            with torch.no_grad():model.bias.copy_(torch.tensor([-4.,5.,1.]))
            self.assertNotIn('behavior',probe.measure(model,update))

    def test_archived_sale_flag_cannot_affect_any_shared_health_consumer(self):
        s=baseline();before=copy.deepcopy(s)
        old=dict(ok=False,sales=dict(ok=False,failures=79),passing=dict(ok=True,positions=511,failures=0))
        m=measurement(behavior=old)
        new,r=limits.advance(s,m,counts(),generation=11,warmup_done=True)
        self.assertTrue(r['health']['ok']);self.assertTrue(r['lr_gate']['ok'])
        self.assertEqual(r['decision'],'hold');self.assertEqual(new['limit'],s['limit'])
        self.assertNotIn('behavior',r['measurement']);self.assertEqual(s,before);self.assertEqual(m['behavior'],old)
        # No IL action-probability evidence is required by economic health.
        self.assertTrue(limits.health(s['baseline'],limits.rates([counts()]*5),dict(ok=True))['ok'])
        self.assertTrue(limits.health(s['baseline'],limits.rates([counts(sells=0)]*5),old)['ok'])


class Statistics(unittest.TestCase):
    def test_pooled_ev_matches_concatenated_rows_not_average_ev(self):
        t1=np.array([0.,1.,2.,3.]);p1=t1+[0.,1.,0.,1.]
        t2=np.arange(20.)/10+4;p2=t2+np.linspace(0,2,20)
        a=ev.statistics(t1,p1,['a','a','b','b']);b=ev.statistics(t2,p2,['c']*20)
        pooled=ev.pool([a,b]);t=np.concatenate([t1,t2]);p=np.concatenate([p1,p2])
        self.assertAlmostEqual(pooled['explained_variance'],1-np.var(p-t)/np.var(t),places=14)
        self.assertNotAlmostEqual(pooled['explained_variance'],np.mean([1-np.var(p1-t1)/np.var(t1),1-np.var(p2-t2)/np.var(t2)]))
        self.assertEqual((pooled['rows'],pooled['games']),(24,3))

    def test_constant_targets_missing_nan_and_duplicate_games(self):
        self.assertIsNone(ev.pool([])['explained_variance'])
        flat=ev.statistics([1.,1.],[1.,2.],['g','g'])
        self.assertIsNone(ev.pool([flat])['explained_variance'])
        with self.assertRaisesRegex(ValueError,'Repeated game'):ev.pool([flat,flat])
        with self.assertRaises(ValueError):ev.statistics([1.,float('nan')],[1.,2.],['a','b'])

    def test_only_heldout_learner_groups_supply_moments(self):
        from test_critic_descriptors import mixed_fixture
        model,opt,cfg,b,data,snap,_=mixed_fixture();ds=updates.Dataset(data+snap,'cpu')
        train,held,_=value_phase.split(ds.data,.05)
        metrics=value_phase.metrics(model,ds,held,cfg['update'],evidence=True)
        for kind in ev.TABLES:
            g=metrics['groups']['table:'+kind+'/role:learner'];stats=g['ev_evidence']
            self.assertEqual(stats['rows'],g['rows']);self.assertEqual(len(stats['game_ids']),g['games'])
            self.assertAlmostEqual(ev.pool([stats])['explained_variance'],g['explained_variance'])
            self.assertTrue(all(value_phase.heldout(game,.05) for game in stats['game_ids']))
            self.assertTrue(set(stats['game_ids']).isdisjoint(ds.data[i][2][0] for i in train))
        self.assertNotIn('ev_evidence',metrics['groups']['role:snapshot'])
        other=value_phase.metrics(model,ds,train,cfg['update'])
        self.assertTrue(all('ev_evidence' not in g for g in other['groups'].values()))


class EvidenceWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_sparse_v4_support_and_new_sample_required_to_activate(self):
        c=config();s=gae.initial(c)
        for g in range(1,11):
            s,r=tick(s,c,legacy=.56 if g%2 else None,steps=int(g%3==1))
        self.assertTrue(s['table_enabled']['historical']);self.assertFalse(s['table_enabled']['legacy'])
        e=r['table_gate']['legacy']['evidence']
        self.assertEqual((e['mixed']['games'],e['contributing_iterations']),(10,5))
        self.assertTrue(e['supported']);self.assertFalse(e['fresh']);self.assertEqual(s['table_streaks']['legacy'],10)
        s,r=tick(s,c,legacy=.56,steps=0)
        self.assertTrue(s['table_enabled']['legacy']);self.assertEqual(s['table_origins']['legacy'],11)
        self.assertEqual(r['table_gate']['legacy']['lambda_used'],1.)
        self.assertEqual(r['table_gate']['legacy']['lambda_next'],.95)
        # Held-out-loss rejection with restored weights is valid evidence even
        # when this iteration accepted zero PPG steps. Recent value work is required.
        self.assertEqual(r['table_gate']['legacy']['evidence']['value_iterations'],4)

    def test_many_correlated_rows_cannot_replace_games(self):
        c=config();s=gae.initial(c)
        for g in range(1,11):s,r=tick(s,c,legacy=.6 if g in (1,5,10) else None,games=1)
        self.assertFalse(s['table_enabled']['legacy'])
        e=r['table_gate']['legacy']['evidence'];self.assertEqual(e['mixed']['games'],3)
        self.assertIn('insufficient_games',e['reasons'])

    def test_single_iteration_cannot_supply_all_support(self):
        c=config();s=gae.initial(c)
        for g in range(1,11):s,r=tick(s,c,legacy=.6 if g==10 else None,games=8)
        self.assertIn('insufficient_iterations',r['table_gate']['legacy']['evidence']['reasons'])
        self.assertFalse(s['table_enabled']['legacy'])

    def test_no_recent_value_updates_and_unhealthy_or_kl_trip_cannot_enable(self):
        for kwargs in (dict(steps=0),dict(healthy=False),dict(trip=True),dict(enabled=False)):
            c=config();s=gae.initial(c)
            for _ in range(12):s,r=tick(s,c,**kwargs)
            self.assertFalse(any(s['table_enabled'].values()))
        c=config();s=gae.initial(c)
        for _ in range(9):s,r=tick(s,c)
        s,r=tick(s,c,healthy=False);self.assertEqual(s['table_streaks']['legacy'],0)
        s,r=tick(s,c);self.assertEqual(s['table_streaks']['legacy'],1)

    def test_relative_ev_gate_independence_and_selfplay_only_mode(self):
        c=config();s=gae.initial(c)
        for _ in range(10):s,r=tick(s,c,historical=.54,legacy=.56)
        self.assertEqual(s['table_enabled'],dict(historical=False,legacy=True))
        self.assertIn('ev_below_selfplay_tolerance',r['table_gate']['historical']['evidence']['reasons'])
        c['table_gate']['mode']='selfplay_only';s=gae.initial(c)
        for _ in range(11):s,r=tick(s,c)
        self.assertFalse(any(s['table_enabled'].values()))

    def test_stale_evidence_expires_and_window_is_bounded(self):
        c=config();s=gae.initial(c)
        for _ in range(9):s,r=tick(s,c)
        for _ in range(25):s,r=tick(s,c,legacy=None)
        self.assertFalse(s['table_enabled']['legacy'])
        e=r['table_gate']['legacy']['evidence'];self.assertEqual(e['mixed']['games'],0)
        self.assertIsNone(e['mixed']['explained_variance']);self.assertEqual(len(s['table_evidence']['entries']),20)
        self.assertEqual(s['table_evidence']['entries'][0]['generation'],15)

    def test_compare_only_matching_generations_and_no_snapshot_substitution(self):
        c=config();s=gae.initial(c)
        for g in range(1,11):
            v=report(g,legacy=.6 if g%2 else None)
            if g%2==0:
                v['heldout_after']['groups']['table:selfplay/role:learner']=group(g,'selfplay',.99)
                v['heldout_after']['groups']['table:legacy/role:snapshot']=group(g,'legacy',.99)
            s,r=gae.advance(s,c,generation=g,value_report=v,health_ok=True,rating={})
        e=r['table_gate']['legacy']['evidence']
        self.assertEqual(e['contributing_generations'],[1,3,5,7,9])
        self.assertAlmostEqual(e['selfplay']['explained_variance'],.6)

    def test_resume_mid_window_same_next_decision_and_corruption_rejected(self):
        c=config();s=gae.initial(c)
        for g in range(1,10):s,r=tick(s,c,legacy=.6 if g%2 else None)
        before=copy.deepcopy(s);rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'state.pt';torch.save(s,path);loaded=torch.load(path,weights_only=True)
            self.assertEqual(tick(s,c),tick(loaded,c))
        self.assertEqual(s,before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        bad=copy.deepcopy(s);bad['table_evidence']['entries'][-1]['generation']-=1
        with self.assertRaises(ValueError):gae.validate_state(bad,c,s['generation'])
        bad=copy.deepcopy(s);bad['table_evidence']['entries'][-1]['tables']['legacy']=copy.deepcopy(bad['table_evidence']['entries'][0]['tables']['legacy'])
        with self.assertRaisesRegex(ValueError,'Repeated game'):gae.validate_state(bad,c,s['generation'])

    def test_compact_history_keeps_support_reasons_and_explicit_scope(self):
        c=config();s=gae.initial(c);s,r=tick(s,c,legacy=None)
        v=dict(actor=dict(gae=dict(gate=r)))
        self.assertEqual(compact(v)['actor']['gae']['gate'],r)
        e=r['table_gate']['legacy']['evidence']
        self.assertIn('insufficient_games',e['reasons']);self.assertIn('online EV',e['scope'])

    def test_strict_config_and_no_archived_backfill(self):
        c=config();old=copy.deepcopy(c);old['table_gate']=copy.deepcopy(gae.ARCHIVED_TABLE_GATE)
        s=gae.initial(old,55);s.update(value_iterations=20,ev_streak=0,table_streaks=dict(historical=9,legacy=6))
        s['table_enabled']['historical']=True;s['table_origins']['historical']=50
        updated=gae.migrate_state(s,old,c,55)
        self.assertEqual(updated['table_evidence'],ev.initial(55));self.assertEqual(updated['active_lambda'],s['active_lambda'])
        self.assertEqual(updated['table_enabled'],s['table_enabled']);self.assertEqual(updated['table_origins'],s['table_origins'])
        self.assertEqual(updated['table_streaks'],dict(historical=0,legacy=0))
        for changes in (dict(window_iterations=100),dict(minimum_games=True),dict(minimum_iterations=21),dict(minimum_value_iterations=0),dict(extra=1)):
            bad=copy.deepcopy(c);bad['table_gate'].update(changes)
            with self.assertRaises(ValueError):gae.validate(bad)


class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_upgrade_preserves_weights_adams_rng_beta_critic_and_is_repeatable(self):
        model,opt,new,(behavior,data)=fixture();old=copy.deepcopy(new)
        old['advantage']['table_gate']=copy.deepcopy(gae.ARCHIVED_TABLE_GATE)
        p=TrainingPhases(model,old);r=ppo(model,opt,old,behavior,data,p);p.fit(model);finish(p,r,1)
        p.beta=8.;p.trips=[True,False,True]
        raw=dict(training={KEY:p.payload()});before=copy.deepcopy(raw)
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        a=TrainingPhases(model,new,raw=raw,generation=1);b=TrainingPhases(model,new,raw=raw,generation=1)
        self.assertTrue(_same_payload(a.payload(),b.payload()));self.assertTrue(_same_payload(raw,before))
        self.assertTrue(_same_payload(weights,model.state_dict()));self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(_same_payload(a.optimizer.state_dict(),p.optimizer.state_dict()))
        self.assertEqual((a.beta,a.trips,a.critic_state),(p.beta,p.trips,p.critic_state))
        self.assertEqual(a.state['table_evidence'],ev.initial(1));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        restored=TrainingPhases(model,new,raw=dict(training={KEY:a.payload()}),generation=1)
        self.assertTrue(_same_payload(a.payload(),restored.payload()))
        bad=copy.deepcopy(new);bad['value_phase']['beta_clone']=2.
        with self.assertRaisesRegex(ValueError,'configuration differs'):TrainingPhases(model,bad,raw=raw,generation=1)

    def test_current_run_upgrade_requires_review_and_only_changes_table_gate(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        path=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0055-20261019T053031361382Z/state.json'
        s=read(path);before=copy.deepcopy(s);signature=copy.deepcopy(s['signature'])
        signature.update(implementation=implementation_id(),config=gae.evidence_config(signature['config']))
        with self.assertRaises(ValueError):check_signature(s,signature)
        self.assertTrue(check_signature(s,signature,reviewed=True));self.assertEqual(s,before)
        self.assertEqual({k for k,v in signature['config'].items() if v!=s['signature']['config'][k]},{'advantage'})
        signature['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(s,signature,reviewed=True)

    def test_restart_after_source_commit_migrates_same_old_checkpoint(self):
        from brass_ai.learning.contracts import compatibility,digest
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.resume import load_resume,commit_transition,apply_optimizer_config
        from brass_ai.step5.league import implementation_id,write_json
        model,opt,new,(behavior,data)=fixture();old=copy.deepcopy(new)
        old['advantage']['table_gate']=copy.deepcopy(gae.ARCHIVED_TABLE_GATE)
        p=TrainingPhases(model,old);r=ppo(model,opt,old,behavior,data,p);p.fit(model);finish(p,r,1)
        previous=dict(config=old,compatibility=compatibility(),teachers={},legacy_pool={},
            implementation='c13907c7564371269b2ae86fa690dcb6a0cd7d18c74d7322e9fcc83ac6a61d80')
        current=dict(previous,config=new,implementation=implementation_id())
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);path=run/'generation-0001.pt'
            identity=save_checkpoint(path,model,opt,step=1,training=dict(signature=digest(previous),
                fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),**{KEY:p.payload()}))
            entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=1)
            state=dict(format='step5-recovery-league/2.2.0',signature=previous,iteration=1,games_done=320,
                next_seed=2000,context_counter=960,history=[entry],population=[entry],evaluation_anchor=entry,
                next_search_games=320)
            write_json(run/'state.json',state)
            restored,raw,changed=load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
            self.assertTrue(changed);adam=torch.optim.AdamW(restored.parameters())
            resume_optimizer(raw,adam);apply_optimizer_config(adam,new)
            phase=TrainingPhases(restored,new,raw=raw,generation=1)
            migrated=copy.deepcopy(phase.payload())
            self.assertTrue(_same_payload(opt.state_dict()['state'],adam.state_dict()['state']))
            updated=commit_transition(run,state,current)
            for key in state:
                if key!='signature':self.assertEqual(updated[key],state[key])
            resumed,again,changed=load_resume(SOURCE_ROOT,run,updated,current)
            self.assertFalse(changed)
            again_phase=TrainingPhases(resumed,new,raw=again,generation=1)
            self.assertTrue(_same_payload(migrated,again_phase.payload()))
            self.assertEqual(sha(path),entry['sha256'])


if __name__=='__main__':unittest.main()
