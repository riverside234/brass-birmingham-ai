"""Quality-score arithmetic, independent seats, pool gates and exact planning."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,read_json
from brass_ai.recovery import quality_pool as q
from brass_ai.recovery.quality_config import LEGACY_POOL as OPPONENT_POOL,TABLE_PLAN,maximum_members
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.ordinary import plan_collection


def entry(i):
    return dict(model_id=f'model-{i}',generation=i,path=f'population/{i}.pt',sha256='a'*64)


def pool(size=3,generation=100):
    value=q.initial(entry(0),OPPONENT_POOL,dict(path='probe.npz',sha256='b'*64,probe_hash='p'))
    value['generation']=generation
    for i in range(1,size):
        value['members'].append(dict(entry(i),admitted_iteration=i,q=0.,probe={},admission_health=dict(ok=True),recent_rates=[]))
    value['health_window']=[True]*5
    return value


class Quality(unittest.TestCase):
    def test_defaults_and_legacy_config_validation_are_versioned(self):
        c=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json');validate_config(c)
        self.assertEqual(c,latest_model_config(c,c['imitation_checkpoint']))
        self.assertEqual(c['table_plan'],TABLE_PLAN);self.assertEqual(sum(c['population'].values()),24)
        old=copy.deepcopy(c)
        for k in ('table_plan','opponent_pool','rating','advantage','value_phase','critic_data','seat_descriptors','v4_share_gate'):old.pop(k)
        from brass_ai.recovery.exploration import BLOCKS
        for k in BLOCKS:old.pop(k,None)
        old['exploration']=.08
        old.update(selfplay_recipe='latest-model-selfplay/2.0.0',curriculum=[dict(until=None,selfplay=60,historical=30,legacy=10)],
                   population=dict(recent=4,anchors=12))
        validate_config(old)
        for field in ('table_plan','opponent_pool','rating'):
            broken=copy.deepcopy(c);broken.pop(field)
            with self.assertRaises(ValueError):validate_config(broken)
        broken=copy.deepcopy(c);broken['opponent_pool']['max_members']=25
        with self.assertRaises(ValueError):validate_config(broken)

    def test_hand_computed_win_tie_loss_importance_recentre_and_IL(self):
        p=pool();p['members'][2]['q']=-math.log(3)
        plan=q.distribution(p);self.assertAlmostEqual(plan['members'][0]['p'],.7)
        table=dict(index=0,learner_seats=[0,1],policy_ids=['learner','learner','m1','IL'],opponents=[
            dict(seat=2,model_id='model-1',source='scored',N=2,p=plan['members'][0]['p']),
            dict(seat=3,model_id='model-0',source='IL',N=2,p=None)])
        games=[dict(setup_group='setup',opponents=table['policy_ids'],result={})]
        for ranks,win_sum in (([1,1,2,3],2.),([1,2,2,4],1.5),([3,4,1,2],0.)):
            with patch('brass_ai.recovery.quality_pool.terminal_targets',return_value=dict(placement=ranks)):
                updated,report=q.update_scores(p,games,{'setup':table},plan,generation=101)
            self.assertIsNone(updated['members'][0]['q'])
            self.assertEqual(updated['members'][1]['q'],0.)
            self.assertAlmostEqual(updated['members'][2]['q'],-math.log(3)+.01*win_sum/(2*.7))
            self.assertEqual(report['members'][1]['comparisons'],2)
            self.assertEqual(report['members'][1]['outrank_rate'],win_sum/2)
            self.assertEqual(report['members'][0]['comparisons'],2)
            self.assertEqual(report['members'][0]['outrank_rate'],sum(
                1. if ranks[s]<ranks[3] else .5 if ranks[s]==ranks[3] else 0. for s in (0,1))/2)
        with patch('brass_ai.recovery.quality_pool.terminal_targets',return_value=dict(placement=[1,2,3,4])):
            altered=copy.deepcopy(table);altered['opponents'][0]['p']*=.8
            with self.assertRaisesRegex(ValueError,'planning probability'):
                q.update_scores(p,games,{'setup':altered},plan,generation=101)
        self.assertEqual(p['members'][1]['q'],0.)

    def test_both_independent_opponent_seats_contribute_all_four_pairs(self):
        p=pool(2);d=q.distribution(p)
        t=dict(index=0,learner_seats=[0,1],policy_ids=['l','l','h','h'],opponents=[
            dict(seat=s,model_id='model-1',source='scored',N=1,p=1.) for s in (2,3)])
        with patch('brass_ai.recovery.quality_pool.terminal_targets',return_value=dict(placement=[1,2,2,4])):
            _,r=q.update_scores(p,[dict(setup_group='s',opponents=t['policy_ids'],result={})],{'s':t},d,generation=101)
        self.assertEqual(r['members'][1]['comparisons'],4);self.assertEqual(r['members'][1]['outrank_rate'],.875)

    def test_sampler_floor_IL_share_startup_and_global_rng_unchanged(self):
        p=pool(24);p['members'][1]['q']=-1000;d=q.distribution(p)
        self.assertAlmostEqual(sum(r['p'] for r in d['members']),1)
        self.assertTrue(all(r['p']>=.2/23 for r in d['members']))
        before=random.getstate();rng=random.Random(1234)
        draws=[q.draw(p,d,rng)[1] for _ in range(10000)]
        self.assertAlmostEqual(sum(r['source']=='IL' for r in draws)/len(draws),.2,delta=.02)
        self.assertEqual(random.getstate(),before)
        first=pool(1)
        self.assertTrue(all(q.draw(first,q.distribution(first),rng)[1]['seat_probability']==1 for _ in range(30)))

    def test_expected_sampler_concentrates_then_recovers_after_learner_stops_winning(self):
        p=pool(24);rng=np.random.default_rng(73);initial=1/23
        def iterate(rates):
            d=q.distribution(p);prob=np.array([r['p'] for r in d['members']]);selected=rng.choice(23,460,p=prob)
            wins=rng.random(460)<np.asarray(rates)[selected]
            for i,m in enumerate(p['members'][1:]):m['q']-=.01*float(wins[selected==i].sum())/(23*prob[i])
            top=max(m['q'] for m in p['members'][1:])
            for m in p['members'][1:]:m['q']-=top
        for _ in range(20):iterate([.9,.1]+[.5]*21)
        d=q.distribution(p);self.assertGreater(d['members'][1]['p'],d['members'][0]['p'])
        before=d['members'][0]['p']
        for _ in range(20):iterate([0.]+[.8]*22)
        after=q.distribution(p)['members'][0]['p']
        self.assertGreater(after,before);self.assertGreater(after,initial)

    def test_admission_health_distinctness_new_score_and_capacity_protection(self):
        p=pool(4);p['health_window'][-1]=False
        distances={m['model_id']:.03 for m in p['members']}
        _,r=q.admission(p,entry(100),dict(ok=True),distances,lambda a,b:.1)
        self.assertFalse(r['admitted'])
        p['health_window'][-1]=True;distances['model-0']=.0199
        _,r=q.admission(p,entry(100),dict(ok=True),distances,lambda a,b:.1)
        self.assertEqual(r['reason'],'insufficient_distinctness')
        distances['model-0']=.02
        new,r=q.admission(p,entry(100),dict(ok=True),distances,lambda a,b:.1)
        self.assertTrue(r['admitted']);self.assertEqual(new['members'][-1]['q'],0)
        p=pool(24)
        p['members'][1]['q']=p['members'][2]['q']=-3
        p['members'][3]['q']=-99;p['members'][3]['admitted_iteration']=90
        p['members'][-1]['q']=-100
        distances={m['model_id']:.1 for m in p['members']}
        new,r=q.admission(p,entry(100),dict(ok=True),distances,lambda a,b:.001 if a['model_id']=='model-2' else .1)
        self.assertEqual(r['evicted']['model_id'],'model-2');self.assertEqual(len(new['members']),24)
        for identity in ('model-0','model-3','model-23'):self.assertIn(identity,{m['model_id'] for m in new['members']})
        for m in p['members'][1:]:m['admitted_iteration']=90
        new,r=q.admission(p,entry(100),dict(ok=True),distances,lambda a,b:.1)
        self.assertEqual(r['reason'],'all_members_protected');self.assertEqual(len(new['members']),24)

    def test_probe_KL_distinctness_masks_by_offsets_and_identity(self):
        a=np.array([.2,.8,.1,.2,.7]);b=np.array([.8,.2,.1,.2,.7]);offsets=np.array([0,2,5])
        self.assertEqual(q.mean_kl(a,a,offsets),0.)
        self.assertAlmostEqual(q.mean_kl(a,b,offsets),(.2*math.log(.25)+.8*math.log(4))/2)

    def test_score_commit_roundtrip_digest_and_compact_reporting(self):
        from brass_ai.recovery.history import compact
        p=pool(3);d=q.distribution(p)
        table=dict(index=0,learner_seats=[0,1],policy_ids=['l','l','a','b'],opponents=[
            dict(seat=i+2,model_id=row['model_id'],source='scored',N=d['N'],p=row['p'])
            for i,row in enumerate(d['members'])])
        games=[dict(setup_group='s',opponents=table['policy_ids'],result={})]
        with patch('brass_ai.recovery.quality_pool.terminal_targets',return_value=dict(placement=[1,3,2,4])):
            next_pool,report=q.update_scores(p,games,{'s':table},d,generation=101)
            restored=json.loads(json.dumps(next_pool))
            a=q.update_scores(next_pool,games,{'s':dict(table,opponents=[dict(o,p=row['p'])
                for o,row in zip(table['opponents'],q.distribution(next_pool)['members'])])},q.distribution(next_pool),generation=102)
            b=q.update_scores(restored,games,{'s':dict(table,opponents=[dict(o,p=row['p'])
                for o,row in zip(table['opponents'],q.distribution(restored)['members'])])},q.distribution(restored),generation=102)
        self.assertEqual(a,b)
        token=q.state_digest(restored);restored['members'][1]['path']='derived/weights.pt'
        self.assertEqual(token,q.state_digest(restored))
        restored['members'][1]['q']-=.001;self.assertNotEqual(token,q.state_digest(restored))
        rating=dict(generation=100,ratings={'generation-0':dict(points=0)})
        summary=compact(dict(quality_pool=report,strength_rating=rating))
        self.assertEqual(summary['quality_pool'],report);self.assertEqual(summary['strength_rating'],rating)

    def test_retention_pins_members_probes_and_evicted_but_rated_checkpoints(self):
        from brass_ai.recovery import retention
        from brass_ai.step5.league import write_json
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            def put(name,raw=b'fixture'):
                path=run/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                return dict(path=path.relative_to(root).as_posix(),sha256=sha(path))
            active=[dict(put(f'population/weights-{i}.pt'),model_id=f'm{i}',generation=i,
                         probe=put(f'opponent-pool/probes/{i}.npz')) for i in (0,5)]
            rated=dict(put('population/weights-10.pt'),generation=10,model_id='m10')
            put('population/weights-8.pt');put('opponent-pool/probes/8.npz')
            slot={k:put('ratings/slots/generation-000010/'+name,raw) for k,name,raw in (
                ('records','games.jsonl',b'{"game_id":"evidence"}\n'),('bootstrap','bootstrap.npz',b'bootstrap'),
                ('spec','spec.json',b'{}'),('panels','panels.json',b'{}'),('report','report.json',b'{}'))}
            history=[dict(put(f'generation-{g:04d}.pt'),generation=g) for g in (19,20)]
            write_json(run/'state.json',dict(format='step5-recovery-league/2.2.0',iteration=20,
                signature=dict(config={}),history=history,resume=history,population=[q.reference(m) for m in active],
                quality_pool=dict(members=active),ratings=dict(agents={'generation-000010':dict(checkpoint=rated)},slots=[slot])))
            plan=retention.plan(root,run);protected={e['path'] for e in plan['protected']};removed={e['path'] for e in plan['entries']}
            for path in ('population/weights-0.pt','population/weights-5.pt','population/weights-10.pt',
                         'opponent-pool/probes/5.npz','ratings/slots/generation-000010/games.jsonl',
                         'ratings/slots/generation-000010/bootstrap.npz'):
                self.assertIn(path,protected);self.assertNotIn(path,removed)
            self.assertIn('population/weights-8.pt',removed);self.assertIn('opponent-pool/probes/8.npz',removed)

    def test_320_table_plan_six_pairs_receipts_and_restart(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);path=root/'weights.pt';path.write_bytes(b'frozen fixture')
            p=pool(3,generation=20)
            for m in p['members']:m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path))
            c=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json')
            c['opponent_pool']=copy.deepcopy(OPPONENT_POOL)
            state=dict(iteration=20,games_done=6400,next_seed=c['seed'],context_counter=0,
                quality_pool=p,population=[q.reference(m) for m in p['members']],history=[q.reference(p['members'][0])],evaluation_anchor=q.reference(p['members'][0]))
            service=Mock();service.models={m['model_id']:None for m in p['members']}
            opponents={'v4':dict(kind='legacy',policy_id='v4')}
            args=(SOURCE_ROOT,root,state,c,dict(config=c),service,'current',['v4'],opponents,opponents,None)
            fp=dict(values=[1/8]*16+[1/6]*12)
            for member in p['members']:member['fingerprint']=fp
            fingerprints=dict(learner=fp,rule_bot=fp)
            before=copy.deepcopy(state);planned=plan_collection(*args,fingerprints=fingerprints)
            restarted=plan_collection(*args[:2],json.loads(json.dumps(state)),*args[3:],fingerprints=fingerprints)
            self.assertEqual(planned,restarted);self.assertEqual(state,before)
            self.assertEqual(planned['composition'],dict(selfplay=160,historical=144,legacy=16))
            self.assertEqual(planned['context_counter'],960)
            pairs={kind:Counter() for kind in ('historical','legacy')}
            for job in planned['jobs']:
                physical=[job['policies'][(s-job['rotation'])%4] for s in range(4)]
                table=job['provenance']['critic_table']
                self.assertEqual([s['policy_id'] for s in table['seats']],[p['policy_id'] for p in physical])
                pair=tuple(s for s,policy in enumerate(physical) if policy.get('model_id')=='current')
                kind=job['provenance']['kind']
                if kind=='selfplay':self.assertEqual(len(pair),4)
                else:pairs[kind][pair]+=1;self.assertEqual(len(pair),2)
            self.assertEqual(set(pairs['historical'].values()),{24})
            self.assertEqual(set(pairs['legacy'].values()),{2,3})
            self.assertTrue(any(t['opponents'][0]['model_id']==t['opponents'][1]['model_id'] for t in planned['quality_tables'].values()))


if __name__=='__main__':unittest.main()
