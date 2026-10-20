"""Fixed-probe correctness, adaptive limits, rehearsal boundaries and restart."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,read_json
from brass_ai.learning.features import Encoded
from brass_ai.recovery import il_kl_limit as limits,il_kl_probe as probes,lr_controller as lr
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.rehearsal import schedule,DEFAULT_FLOOR
from brass_ai.recovery.history import compact
from test_mlp_v11 import inputs
from test_ppo_lr_controller import measurements


def counts(**kwargs):
    return dict(dict(tables=25,all_learner_tables=25,seat_games=100,vp_sum=10000.,sells=200,
        passes_with_alternative=10,zero_vp_seats=1,insolvent_seats=10,insolvency_events=15),**kwargs)


def measurement(distance=.1,**kwargs):
    return dict(dict(positions=2048,kl_mean=distance,groups={k:dict(positions=512,kl_mean=distance) for k in limits.GROUPS},
        unassigned_positions=0,behavior=dict(ok=True,passing=dict(ok=True,positions=10,failures=0))),**kwargs)


def tick(state,distance=.1,health=None,warmup=True,**kwargs):
    return limits.advance(state,measurement(distance,**kwargs),counts() if health is None else health,
        generation=state['last_generation']+1,warmup_done=warmup)


def baseline():
    state=limits.initial_state(limits.DEFAULT,dict(probe_hash='a'*64))
    for _ in range(10):state,_=tick(state)
    return state


class Rules(unittest.TestCase):
    def test_defaults_signature_and_legacy_config_are_explicit(self):
        cfg=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json');validate_config(cfg)
        self.assertEqual(cfg['ppo_lr_controller']['contract'],lr.AUTO_CONTRACT)
        self.assertEqual(latest_model_config(cfg,cfg['imitation_checkpoint']),cfg)
        for change in (dict(initial_limit=.2),dict(group_factor=3),dict(rolling_iterations=True),dict(extra=1)):
            bad=copy.deepcopy(cfg);bad['il_kl_limit'].update(change)
            with self.assertRaises(ValueError):validate_config(bad)
        old=copy.deepcopy(cfg);old.pop('il_kl_limit');old['ppo_lr_controller']['contract']=lr.CONTRACT
        from brass_ai.recovery.exploration import BLOCKS
        for key in BLOCKS:old.pop(key,None)
        for key in ('table_plan','opponent_pool','rating','advantage','value_phase','critic_data','seat_descriptors'):old.pop(key,None)
        old.update(selfplay_recipe='latest-model-selfplay/2.0.0',
            curriculum=[dict(until=None,selfplay=60,historical=30,legacy=10)],population=dict(recent=4,anchors=12))
        validate_config(old)
        self.assertNotIn('il_kl_limit',old)
        # Explicit fresh config generation opts into the new health-gated pool.
        self.assertEqual(latest_model_config(old,old['imitation_checkpoint']),cfg)

    def test_rehearsal_floor_boundaries_and_restart_use_committed_progress(self):
        cfg=dict(rehearsal_manifest='completed.json',rehearsal_weight=.1,rehearsal_until=10000,rehearsal_floor=DEFAULT_FLOOR)
        self.assertEqual(schedule(cfg,0,0)['weight'],.1)
        for iteration,expected in ((0,.03),(59,.03),(60,.02),(89,.02),(90,.02),(119,.02),(120,.01),(179,.01),(180,0.)):
            self.assertAlmostEqual(schedule(cfg,320*max(40,iteration),iteration)['weight'],expected)
            self.assertEqual(schedule(cfg,320*max(40,iteration),iteration),
                             schedule(json.loads(json.dumps(cfg)),320*max(40,iteration),iteration))
        self.assertEqual(schedule(dict(cfg,rehearsal_weight=0),20000,65)['weight'],0)
        old=copy.deepcopy(cfg);old.pop('rehearsal_floor')
        self.assertEqual(schedule(old,10000,32)['weight'],0)
        legacy=dict(cfg,rehearsal_floor=dict(DEFAULT_FLOOR,interval_iterations=30))
        self.assertEqual(schedule(legacy,32000,90)['weight'],.01)
        self.assertEqual(schedule(legacy,38400,120)['weight'],0.)

    def test_warmup_baseline_and_first_raise_at_generation_23(self):
        state=limits.initial_state(limits.DEFAULT,{})
        for _ in range(3):state,r=tick(state,distance=.2,warmup=False)
        self.assertEqual(state['baseline_rows'],[])
        for _ in range(10):state,r=tick(state,distance=.2)
        self.assertEqual(state['last_generation'],13);self.assertEqual(state['last_update'],13)
        for _ in range(9):state,r=tick(state,distance=.2);self.assertEqual(r['decision'],'hold')
        state,r=tick(state,distance=.2)
        self.assertEqual(state['last_generation'],23);self.assertEqual(r['decision'],'raise')
        self.assertEqual(r['limit_after'],.1875);self.assertEqual(r['group_limit_after'],.375)
        self.assertTrue(r['blocked'])

    def test_small_distances_never_lower_the_limit(self):
        state=baseline()
        for _ in range(35):
            state,r=tick(state,distance=1e-8)
            self.assertEqual(state['limit'],.15);self.assertEqual(r['group_limit_after'],.30)

    def test_immediate_health_tightening_clears_clock_and_never_raises(self):
        state=baseline();state['kl_last_healthy']=.12
        state,r=tick(state,health=counts(passes_with_alternative=1000))
        self.assertEqual((r['decision'],r['limit_after'],r['group_limit_after']),('tighten',.12,.24))
        self.assertEqual(state['last_update'],11);self.assertEqual(state['healthy_window'],[])
        state['kl_last_healthy']=.9
        state,r=tick(state,health=counts(passes_with_alternative=1000));self.assertEqual(state['limit'],.12)
        state['kl_last_healthy']=.001
        state,r=tick(state,health=counts(passes_with_alternative=1000));self.assertEqual(state['limit'],.05)

    def test_five_iteration_health_rates_and_all_thresholds(self):
        b=limits.baseline_record([counts()]*10,limits.DEFAULT)
        for key,value in (('vp_sum',8499),('passes_with_alternative',71),
                          ('zero_vp_seats',6),('insolvent_seats',21)):
            current=limits.rates([counts(**{key:value})]*5)
            self.assertFalse(limits.health(b,current,dict(passing=dict(ok=True)))['ok'])
        current=limits.rates([counts(vp_sum=8500,sells=140,passes_with_alternative=70,zero_vp_seats=5,insolvent_seats=20)]*5)
        self.assertTrue(limits.health(b,current,dict(passing=dict(ok=True)))['ok'])
        state=baseline()
        for _ in range(4):state,_=tick(state,health=counts(vp_sum=0))
        self.assertEqual(state['limit'],.1)

    def test_raise_cap_ceiling_restart_and_group_limits(self):
        state=baseline()
        for _ in range(100):
            before=state['limit'];state,r=tick(state,distance=.95)
            self.assertLessEqual(state['limit'],1.25*before)
            self.assertLessEqual(state['limit'],1.0)
            self.assertEqual(r['group_limit_after'],2*state['limit'])
        self.assertEqual(state['limit'],1.)
        state=baseline()
        for _ in range(7):state,_=tick(state,distance=.14)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'checkpoint.pt';torch.save(dict(il_kl_limit=state),path)
            restored=torch.load(path,weights_only=True)['il_kl_limit']
            for _ in range(3):
                state,left=tick(state,distance=.14);restored,right=tick(restored,distance=.14)
                self.assertEqual(state,restored);self.assertEqual(left,right)
            self.assertEqual(left['decision'],'raise')

    def test_latch_requires_mean_and_every_group_to_recover(self):
        state=baseline();groups=measurement()['groups'];groups['rail_other']['kl_mean']=.31
        state,r=tick(state,groups=groups);self.assertTrue(r['blocked'])
        groups['rail_other']['kl_mean']=.25
        state,r=tick(state,groups=groups);self.assertTrue(r['blocked'])
        groups['rail_other']['kl_mean']=.23
        state,r=tick(state,distance=.119,groups=groups);self.assertFalse(r['blocked'])
        state,r=tick(state,distance=.301);self.assertTrue(r['alarm'])

    def test_probe_missing_group_blocks_raise_and_lr_cuts_still_run(self):
        state=baseline();groups=measurement()['groups'];groups['rail_other']['positions']=63
        state,r=tick(state,groups=groups)
        self.assertFalse(r['lr_gate']['ok'])
        control=lr.initial_state(lr.AUTO_CONTROLLER)
        updated,decision=lr.decide(control,measurements(kl=.02,calibrated_anchor=r['lr_gate']))
        self.assertEqual(decision['decision'],'down')
        self.assertEqual(updated['multiplier'],1/1.5)
        compacted=compact(dict(actor=dict(lr_controller=decision),il_kl_limit=r,
            rehearsal_schedule=dict(weight=.03),by_composition=dict(selfplay=dict(health_counts=counts()))))
        self.assertEqual(compacted['il_kl_limit'],r)
        self.assertEqual(compacted['health_by_composition']['selfplay'],counts())

    def test_missing_health_counts_remain_unavailable_and_restartable(self):
        state=baseline()
        state,r=limits.advance(state,measurement(),{},generation=11,warmup_done=True)
        self.assertIsNone(state['rolling'][-1]);self.assertFalse(r['lr_gate']['ok'])
        limits.validate_state(state,state['config'],11)
        restored=json.loads(json.dumps(state))
        self.assertEqual(tick(state),tick(restored))

    def test_lr_raise_uses_new_limits_and_preserves_shadow_cooldown(self):
        state=baseline()
        for _ in range(10):state,r=tick(state,distance=.14)
        self.assertAlmostEqual(r['limit_after'],.175)
        self.assertAlmostEqual(r['group_limit_after'],.35)
        control=lr.initial_state(dict(lr.AUTO_CONTROLLER,mode='shadow'))
        for _ in range(3):control,decision=lr.decide(control,measurements(calibrated_anchor=r['lr_gate']))
        self.assertEqual(decision['decision'],'up');self.assertEqual(control['multiplier'],1.25)
        from brass_ai.recovery.learning_rate import configure,step
        p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([p],lr=6e-6)
        configure(optimizer,6e-6,0,.1,'selfplay');optimizer.param_groups[0][lr.KEY]=control
        p.grad=torch.ones_like(p);step(optimizer,ppo=True)
        self.assertEqual(optimizer.param_groups[0]['lr'],6e-6)
        control['cooldown']=2
        control,decision=lr.decide(control,measurements(calibrated_anchor=r['lr_gate']))
        self.assertEqual(decision['decision'],'cooldown')

    def test_health_counts_distinguish_zero_seats_events_and_opponent_tables(self):
        from brass_ai.recovery.diagnostics import health_counts
        records=[dict(seat=s,choices=[dict(kind='sell'),dict(kind='pass')],chosen=s%2) for s in range(4)]
        game=dict(opponents=['learner']*4,result={},records=records)
        events=[dict(seat=0),dict(seat=0),dict(seat=3)]
        with patch('brass_ai.recovery.diagnostics.terminal_targets',return_value=dict(vp=[0,10,0,20])),\
             patch('brass_ai.recovery.insolvency.events',return_value=events):
            actual=health_counts([game],{'learner'})
            self.assertEqual((actual['seat_games'],actual['zero_vp_seats'],actual['insolvent_seats'],actual['insolvency_events']),
                             (4,2,2,3))
            self.assertEqual((actual['sells'],actual['passes_with_alternative']),(2,2))
            game['opponents']=['learner','old','learner','old']
            actual=health_counts([game],{'learner'})
            self.assertEqual((actual['all_learner_tables'],actual['seat_games'],actual['insolvent_seats'],actual['insolvency_events']),
                             (0,2,1,2))

    def test_archived_vp_check_reaches_11520_only_with_stated_baseline_assumption(self):
        archive=read_json(SOURCE_ROOT/'results/selfplay-v2-70/retention-history.json')
        rows=[]
        for item in archive.values():
            report=item['summary']
            if 'iteration' not in report:continue
            table=report['by_composition']['selfplay'];policies=list(table['matches']['policies'].values())
            self.assertEqual(len(policies),1)
            p=policies[0];self.assertEqual(p['seat_games'],4*table['matches']['games'])
            rows.append((report['iteration'],report['games_done'],p['seat_games'],p['vp']))
            # Missing metrics must stay unknown, not be fabricated as zero.
            self.assertNotIn('health_counts',table)
        rows.sort()
        mean=lambda values:sum(n*vp for _,_,n,vp in values)/sum(n for _,_,n,_ in values)
        baseline_vp=mean(rows[3:13])  # Illustrative warmup ends after generation 3.
        first=next(rows[i][1] for i in range(13,len(rows)) if mean(rows[i-4:i+1])<.85*baseline_vp)
        self.assertEqual(first,11520)
        self.assertIsNone(limits.rates([dict(seat_games=512,vp_sum=baseline_vp*512)]))


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bias=torch.nn.Parameter(torch.tensor([1.,0.,2.]))
    def forward(self,state,actions,mask,mapping):
        return (self.bias[:actions.shape[1]].expand(len(state),-1).masked_fill(~mask,-torch.inf),)


class Rows(list):
    def record(self,i,features=True):return self[i]
    def close(self):pass


class ProbeTests(unittest.TestCase):
    def data(self):
        x,a,mask,mapping=inputs(8,3);rows=Rows()
        for i in range(8):
            context=dict(era='CanalEra' if i<4 else 'RailEra',phase='action' if i%2==0 else 'continuation')
            choices=[dict(id=k,kind=k) for k in ('sell','pass','loan')]
            row=dict(view=dict(context=context),choices=choices,chosen=0)
            rows.append((row,{},('game',i%4),Encoded(x[i].numpy(),a[i].numpy(),mapping[i].numpy(),tuple(c['id'] for c in choices))))
        return rows

    def test_seeded_stratification_forced_filter_and_rng(self):
        data=self.data();before=torch.get_rng_state().clone()
        first=probes.select_indices(data,4,112)
        self.assertEqual(first,probes.select_indices(data,4,112));self.assertEqual(len(set(first)),4)
        self.assertEqual({probes.group_for(data[i][0]['view']['context']) for i in first},set(limits.GROUPS))
        data[0][0]['choices']=data[0][0]['choices'][:1]
        self.assertNotIn(0,probes.select_indices(data,8,112))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_kl_identity_floor_mask_and_raw_probability_behavior(self):
        p=torch.tensor([[1e-10,.9,.1,0.]])
        mask=torch.tensor([[True,True,True,False]])
        self.assertEqual(float(probes.stable_kl(p,p,mask,1e-6)),0.)
        q=torch.tensor([[.1,.8,.1,0.]])
        self.assertGreater(float(probes.stable_kl(p,q,mask,1e-6)),0)
        actual=probes.raw_probabilities(torch.tensor([[0.,0.,0.,100.]]),mask)
        self.assertEqual(actual[0,3],0);self.assertAlmostEqual(float(actual.sum()),1)

    def test_real_cache_rebuild_identity_behavior_and_changed_source_rejection(self):
        rows=self.data();model=ToyPolicy();cfg=copy.deepcopy(limits.DEFAULT)
        update=dict(microbatch_size=3,max_candidate_cells=100)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            root=Path(temporary);source=root/'validation.jsonl.gz';source.write_bytes(b'test source')
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(entries=[dict(split='validation',path='validation.jsonl.gz',sha256=sha(source))])))
            origin=dict(manifest='manifest.json',manifest_sha256=sha(manifest),model_id='a'*64,sha256='b'*64)
            before=torch.get_rng_state().clone()
            with patch('brass_ai.recovery.demonstrations.load_dataset',return_value=(rows,{})):
                probe,receipt=probes.load(root,root/'run',origin,model,cfg,update,1)
                other,rebuilt=probes.load(root,root/'rebuilt',origin,model,cfg,update,1)
            self.assertEqual(receipt['probe_hash'],rebuilt['probe_hash'])
            report=probe.measure(model,update)
            self.assertEqual(report['kl_mean'],0.);self.assertNotIn('behavior',report)
            self.assertEqual(len(report['groups']),4)
            for group in report['groups'].values():
                self.assertGreater(group['raw_entropy'],0.)
                self.assertTrue(0.<group['normalized_entropy']<=1.)
            loaded,again=probes.load(root,root/'run',origin,model,cfg,update,1,receipt)
            self.assertEqual(again,receipt);self.assertEqual(loaded.measure(model,update)['kl_mean'],0.)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            with torch.no_grad():model.bias.copy_(torch.tensor([-4.,5.,1.]))
            changed=loaded.measure(model,update)
            self.assertGreater(changed['kl_mean'],0)
            self.assertNotIn('behavior',changed)
            # Even an uncommitted bootstrap cache must verify its semantic hash.
            path=root/'run/il-kl/manifest.json';original=path.read_bytes()
            edited=json.loads(original);edited['rows'][0]['identity']=['different',0]
            path.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError,'semantic hash'):
                probes.load(root,root/'run',origin,model,cfg,update,1)
            path.write_bytes(original)
            source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError,'validation data changed'):
                probes.load(root,root/'run',origin,model,cfg,update,1,receipt)

    def test_retention_follows_probe_manifest_and_pins_its_cache(self):
        from brass_ai.recovery.retention import references
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            root=Path(temporary);run=root/'artifacts/run';directory=run/'il-kl';directory.mkdir(parents=True)
            cache=directory/'probe.npz';cache.write_bytes(b'cache')
            manifest=directory/'manifest.json'
            manifest.write_text(json.dumps(dict(cache=dict(path='artifacts/run/il-kl/probe.npz'))))
            (run/'state.json').write_text(json.dumps(dict(format='step5-recovery-league/2.2.0',
                il_kl_limit=dict(probe=dict(path='artifacts/run/il-kl/manifest.json')))))
            pins=references(root,run)
            self.assertIn(manifest,pins);self.assertIn(cache,pins)


if __name__=='__main__':unittest.main()
