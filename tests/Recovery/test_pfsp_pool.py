"""PFSP arithmetic, plans, protections, migration and next-collection continuity."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import quality_pool as q,pfsp_pool as p
from brass_ai.recovery.quality_config import OPPONENT_POOL,LEGACY_POOL,validate_pool,sampling_config
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.ordinary import plan_collection
from brass_ai.recovery.resume import prepare_opponent_pool,commit_transition,check_signature
from brass_ai.step5.league import write_json,implementation_id
from test_quality_pool import entry,pool as legacy_pool


def pool(size=4,generation=100,games=None):
    value=q.initial(entry(0),OPPONENT_POOL,dict(path='probe.npz',sha256='b'*64,probe_hash='probe'))
    value.update(generation=generation,games_done=generation*320 if games is None else games,health_window=[True]*5)
    for i in range(1,size):value['members'].append(dict(entry(i),admitted_iteration=i,probe={},admission_health=dict(ok=True),**p.prior(OPPONENT_POOL)))
    return p.seal(value)


def table_plan(value,ids=(1,0)):
    plan=q.distribution(value);refs={r['model_id']:r for r in plan['members']};opponents=[]
    for seat,i in enumerate(ids,2):
        identity=f'model-{i}';pr=refs[identity]['p'] if i else None
        opponents.append(dict(seat=seat,model_id=identity,source='scored' if i else 'IL',N=plan['N'],p=pr,
            il_share=plan['il_share'],seat_probability=(1-plan['il_share'])*pr if i else plan['il_share']))
    table=dict(index=0,learner_seats=[0,1],opponents=opponents,policy_ids=['learner','learner']+[f'model-{i}' for i in ids])
    return plan,table


def tick(value,ids=(1,0),ranks=(1,2,2,4)):
    plan,table=table_plan(value,ids)
    with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=list(ranks))):
        return q.update_scores(value,[dict(setup_group='h',opponents=table['policy_ids'],result={})],{'h':table},plan,
            generation=value['generation']+1,games_done=value['games_done']+320)


class PFSP(unittest.TestCase):
    def test_strict_config_and_fresh_defaults_preserve_unrelated_recipe(self):
        c=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic.json');before=copy.deepcopy(c)
        new=validate_config(sampling_config(c));self.assertEqual(c,before)
        self.assertEqual({k for k in c if c[k]!=new[k]},{'opponent_pool','rehearsal_floor'})
        self.assertEqual(new['opponent_pool'],OPPONENT_POOL);self.assertEqual(new['rehearsal_floor']['interval_iterations'],60)
        self.assertEqual(latest_model_config(new,new['imitation_checkpoint'])['opponent_pool'],OPPONENT_POOL)
        self.assertEqual(sampling_config(new),new);validate_pool(LEGACY_POOL)
        for change in (dict(eta=.01),dict(max_members=24),dict(pinned_seat_share=.2),dict(q=0)):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_pool(dict(OPPONENT_POOL,**change))
        for name,key,value in (('outrank_rate','half_life_iterations',0),('outrank_rate','prior_rate',float('nan')),
                ('weighting','function','softmax'),('pinned_seat_share','offset',True),('evict','min_decayed_comparisons',0),
                ('alarm','iterations',True),('pinned_seat_share','max',0)):
            c=copy.deepcopy(OPPONENT_POOL);c[name][key]=value
            with self.subTest(name=name,key=key),self.assertRaises(ValueError):validate_pool(c)

    def test_IL_share_cap_and_empirical_independent_seats(self):
        before=random.getstate()
        for k,expected in ((1,1.),(2,1/3),(17,1/18),(24,1/25)):
            value=pool(k,games=100000);plan=q.distribution(value);self.assertEqual(plan['il_share'],expected)
            rng=random.Random(1717);draws=[q.draw(value,plan,rng)[1] for _ in range(20000)]
            self.assertAlmostEqual(sum(d['source']=='IL' for d in draws)/len(draws),expected,delta=.012)
            if k>1:self.assertTrue(any(a['model_id']==b['model_id'] for a,b in zip(draws[::2],draws[1::2])))
        value=pool(2);value['config']['pinned_seat_share']['max']=.2;p.seal(value)
        self.assertEqual(q.distribution(value)['il_share'],.2)
        only=pool(1);only['config']['pinned_seat_share']['max']=.2;p.seal(only)
        self.assertEqual(q.distribution(only)['il_share'],1.);self.assertEqual(random.getstate(),before)

    def test_hand_computed_three_iteration_decay_ties_missing_member_and_IL(self):
        value=pool();before=copy.deepcopy(value);d=.5**.2;S=C=None
        expectedS=expectedC=None
        for iteration,ranks in enumerate(((1,2,2,4),(3,4,1,2),(1,1,2,3)),1):
            value,report=tick(value,ranks=ranks);wins=(1.5,0.,2.)[iteration-1]
            expectedS=d*(5. if expectedS is None else expectedS)+wins
            expectedC=d*(10. if expectedC is None else expectedC)+2
            m=value['members'][1]
            self.assertEqual((m['S'],m['C']),(expectedS,expectedC))
            unplayed=value['members'][2];self.assertAlmostEqual(unplayed['C'],10*d**iteration)
            self.assertEqual(p.rate(unplayed),.5);self.assertEqual(report['members'][1]['comparisons'],2)
            self.assertEqual(report['il']['comparisons'],2);self.assertEqual(report['il']['seats'],1)
            self.assertNotIn('model-0',[r['model_id'] for r in report['planning']['members']])
        self.assertEqual(before,pool())
        value,report=tick(pool(2),ids=(1,1))
        self.assertEqual(report['members'][1]['comparisons'],4)
        self.assertEqual(report['members'][1]['outrank_rate'],.875)

    def test_probability_floor_normalization_prior_and_all_easy_fallback(self):
        value=pool();value['members'][1].update(S=10.,C=10.);p.seal(value);plan=q.distribution(value)
        self.assertAlmostEqual(sum(r['p'] for r in plan['members']),1.)
        self.assertEqual(plan['members'][0]['p'],.2/3)
        self.assertEqual(plan['members'][1]['w'],.25)
        for m in value['members']:m['S']=m['C']
        p.seal(value);plan=q.distribution(value)
        self.assertTrue(plan['all_zero_weights']);self.assertTrue(all(r['p']==1/3 for r in plan['members']))
        value['members'][0].update(S=0.,C=10.);p.seal(value)
        self.assertEqual(plan,q.distribution(value))
        value=pool();value['config']['weighting']=dict(function='variance',power=1)
        value['members'][1]['S']=2.;p.seal(value);validate_pool(value['config'])
        self.assertAlmostEqual(q.distribution(value)['members'][0]['w'],.16)

    def test_only_history_pairs_and_original_probabilities_can_update(self):
        value=pool();plan,table=table_plan(value)
        games=[dict(setup_group='h',opponents=table['policy_ids'],result={}),
               dict(setup_group='selfplay',opponents=['learner']*4,result='must not read'),
               dict(setup_group='v4',opponents=['learner']*2+['v4']*2,result='must not read')]
        with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=[1,2,3,4])) as targets:
            _,r=q.update_scores(value,games,{'h':table},plan,generation=101,games_done=32320)
            self.assertEqual(targets.call_count,1);self.assertEqual(sum(m['comparisons'] for m in r['members']),4)
            for key,val in (('p',.001),('il_share',.123),('N',99),('source','IL'),('seat_probability',.1)):
                bad=copy.deepcopy(table);bad['opponents'][0][key]=val
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'planning probability'):
                    q.update_scores(value,games,{'h':bad},plan,generation=101,games_done=32320)
            bad=copy.deepcopy(table);bad['opponents'][0]['seat']=0
            with self.assertRaisesRegex(ValueError,'separate snapshot seats'):
                q.update_scores(value,games,{'h':bad},plan,generation=101,games_done=32320)
        with self.assertRaisesRegex(ValueError,'once per'):
            q.update_scores(value,[],{},plan,generation=100,games_done=32320)

    def test_official_income_cash_tiebreaks_and_residual_ties(self):
        value=pool();plan,table=table_plan(value,ids=(1,0))
        for scores,expected in (([(100,5,20),(100,5,10),(100,5,10),(100,4,30)],(.75,1.)),
                ([(100,5,10)]*4,(.5,.5))):
            best=max(scores);winners=scores.count(best)
            result=dict(objective='win-share/1.0.0',seats=[dict(seat=i,vp=s[0],income=s[1],cash=s[2],
                rank=1+sum(t>s for t in scores),utility=1/winners if s==best else 0.) for i,s in enumerate(scores)])
            _,r=q.update_scores(value,[dict(setup_group='h',opponents=table['policy_ids'],result=result)],{'h':table},plan,
                generation=101,games_done=32320)
            self.assertEqual((r['members'][1]['outrank_rate'],r['members'][0]['outrank_rate']),expected)

    def test_admission_prior_and_share_frozen_until_the_next_plan(self):
        value=pool(2);old=q.distribution(value)
        new,event=q.admission(value,entry(100),dict(ok=True),{m['model_id']:.02 for m in value['members']},lambda a,b:.1)
        self.assertTrue(event['admitted']);self.assertEqual(p.rate(new['members'][-1]),.5)
        self.assertEqual(new['members'][-1]['C'],10.);self.assertEqual(old['il_share'],1/3)
        self.assertEqual(q.distribution(new)['il_share'],1/4);self.assertEqual(q.distribution(value),old)
        for state,health,distances,reason in ((dict(value,health_window=[True]*4),dict(ok=True),{},'five_healthy_iterations_required'),
                (value,dict(ok=False),{},'health_unavailable'),(value,dict(ok=True),None,'probe_unavailable'),
                (value,dict(ok=True),{m['model_id']:.019 for m in value['members']},'insufficient_distinctness')):
            _,r=q.admission(state,entry(100),health,distances,lambda a,b:.1);self.assertEqual(r['reason'],reason)

    def test_easiest_redundancy_band_age_counts_newest_and_cap(self):
        value=pool(17)
        for m in value['members']:m.update(S=16.,C=20.)
        value['members'][1]['S']=19.8;value['members'][2]['S']=19.4
        value['members'][3].update(S=20.,C=20.,admitted_iteration=90) # young
        value['members'][4].update(S=14.,C=14.) # insufficient support
        value['members'][-1]['S']=20. # newest
        p.seal(value);distances={m['model_id']:.1 for m in value['members']}
        new,r=q.admission(value,entry(100),dict(ok=True),distances,lambda a,b:.001 if a['model_id']=='model-2' else .1)
        self.assertEqual(r['evicted']['model_id'],'model-2');self.assertEqual(len(new['members']),17)
        self.assertEqual(set(r['eviction']['shortlist']),{'model-1','model-2'})
        for i in (0,3,4,16):self.assertIn(f'model-{i}',{m['model_id'] for m in new['members']})
        # With no near-equal alternatives, only the highest eligible rate can go.
        value['members'][2]['S']=16.;p.seal(value)
        _,r=q.admission(value,entry(100),dict(ok=True),distances,lambda a,b:.1)
        self.assertEqual(r['evicted']['model_id'],'model-1')
        for m in value['members'][1:]:m['C']=10.;m['S']=5.
        p.seal(value);new,r=q.admission(value,entry(100),dict(ok=True),distances,lambda a,b:.1)
        self.assertEqual(r['reason'],'no_eligible_eviction');self.assertEqual(len(new['members']),17)
        self.assertEqual(p.capacity(OPPONENT_POOL,99999),16);self.assertEqual(p.capacity(OPPONENT_POOL,100000),23)
        value['games_done']=100000;p.seal(value)
        new,r=q.admission(value,entry(100),dict(ok=True),distances,lambda a,b:.1)
        self.assertTrue(r['admitted']);self.assertIsNone(r['evicted']);self.assertEqual(len(new['members']),18)

    def test_three_iteration_regression_alarm_is_reporting_only_and_resumes(self):
        value=pool(2);value['members'][1].update(S=1.,C=30.);p.seal(value)
        for _ in range(2):value,r=tick(value,ids=(1,1),ranks=(3,4,1,2));self.assertFalse(r['alarms'])
        restored=json.loads(json.dumps(value));new,r=tick(value,ids=(1,1),ranks=(3,4,1,2))
        self.assertEqual((new,r),tick(restored,ids=(1,1),ranks=(3,4,1,2)))
        self.assertEqual(r['alarms'][0]['model_id'],'model-1');self.assertEqual(r['alarms'][0]['action'],'review_only')
        self.assertEqual(new['config'],value['config']);self.assertEqual(new['health_window'],value['health_window'])
        from brass_ai.recovery.review_metrics import gate_diagnostics
        self.assertIn('pool_regression:model-1',gate_diagnostics({},quality=r)['warnings'])
        new['members'][1].update(S=25.,C=30.);p.seal(new)
        _,r=tick(new);self.assertFalse(r['alarms'])

    def test_state_validation_and_compact_reporting(self):
        from brass_ai.recovery.history import compact
        value,r=tick(pool());r=q.finish_report(r,value,dict(reason='not_due',admitted=False))
        q.validate_state(value,OPPONENT_POOL,101);self.assertEqual(compact(dict(quality_pool=r))['quality_pool'],r)
        self.assertIn('PFSP=IL:',p.console(r));self.assertEqual(r['summary']['k'],4)
        token=q.state_digest(value);value['members'][0]['path']='derived.pt';self.assertEqual(token,q.state_digest(value))
        for key,val in (('S',float('nan')),('C',0),('S',11),('q',0),('alarm_streak',True)):
            bad=copy.deepcopy(value);bad['members'][2][key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):q.validate_state(bad,OPPONENT_POOL)
        bad=copy.deepcopy(value);bad['next_planning']['il_share']=.123
        with self.assertRaisesRegex(ValueError,'stored next plan'):q.validate_state(bad,OPPONENT_POOL)

    def test_atomic_migration_and_restart_before_first_PPO_commit(self):
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0085-20261019T172419550588Z/state.json')
        old=copy.deepcopy(state);config=sampling_config(state['signature']['config'])
        raw=dict(training=dict(quality_pool_digest=q.state_digest(state['quality_pool'])))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);new,receipt=prepare_opponent_pool(SOURCE_ROOT,run,state,raw,config,True)
            self.assertEqual(state,old);self.assertEqual(new['health_window'],old['quality_pool']['health_window'])
            for a,b in zip(old['quality_pool']['members'],new['members']):
                for key in ('model_id','generation','path','sha256','probe','fingerprint','admitted_iteration','admission_health'):
                    if key in a:self.assertEqual(a[key],b[key])
                n=sum(r['comparisons'] for r in a['recent_rates'])
                self.assertEqual(b['C'],10.);self.assertAlmostEqual(p.rate(b),sum(r['outcomes'] for r in a['recent_rates'])/n if n else .5)
            signature=dict(state['signature'],config=config,implementation='test-new-source')
            updated=commit_transition(run,state,signature,opponent_pool=new,pool_migration=receipt)
            again,extra=prepare_opponent_pool(SOURCE_ROOT,run,read(run/'state.json'),raw,config)
            self.assertEqual(new,again);self.assertIsNone(extra)
            for key in old:
                if key not in ('signature','quality_pool','source_transitions'):self.assertEqual(updated[key],old[key])
            bad=copy.deepcopy(updated);bad['quality_pool']['members'][1]['S']+=.01;p.seal(bad['quality_pool'])
            with self.assertRaisesRegex(ValueError,'committed checkpoint'):prepare_opponent_pool(SOURCE_ROOT,run,bad,raw,config)
            # After the first commit the new generation's digest is authoritative.
            updated['quality_pool']['generation']+=1;updated['iteration']+=1
            updated['quality_pool']['games_done']+=320;updated['games_done']+=320;p.seal(updated['quality_pool'])
            raw['training']['quality_pool_digest']=q.state_digest(updated['quality_pool'])
            self.assertEqual(prepare_opponent_pool(SOURCE_ROOT,run,updated,raw,config)[0],updated['quality_pool'])

    def test_migration_requires_boundary_review_and_rejects_oversize(self):
        old=legacy_pool(generation=86)
        with self.assertRaisesRegex(ValueError,'admission boundary'):p.migrate(old,OPPONENT_POOL,86,27520)
        old=legacy_pool(size=18,generation=85)
        with self.assertRaisesRegex(ValueError,'exceeds'):p.migrate(old,OPPONENT_POOL,85,27200)
        old=legacy_pool(generation=85);raw=dict(training=dict(quality_pool_digest=q.state_digest(old)))
        state=dict(quality_pool=old,iteration=85,games_done=27200)
        with self.assertRaisesRegex(ValueError,'reviewed source/config'):
            prepare_opponent_pool(SOURCE_ROOT,SOURCE_ROOT/'artifacts',state,raw,dict(opponent_pool=OPPONENT_POOL))

    def test_320_table_plan_frozen_share_rotation_and_exact_resume(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);path=root/'weights.pt';path.write_bytes(b'planning fixture')
            value=pool(4,generation=85);c=read(SOURCE_ROOT/'configs/step5/joint7-latest.json')
            fp=dict(values=[1/8]*16+[1/6]*12)
            for m in value['members']:m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),fingerprint=fp)
            state=dict(iteration=85,games_done=27200,next_seed=c['seed'],context_counter=81600,
                quality_pool=value,population=[q.reference(m) for m in value['members']],history=[q.reference(value['members'][0])],
                evaluation_anchor=q.reference(value['members'][0]))
            service=Mock();service.models={m['model_id']:None for m in value['members']}
            profiles=[f'v4-{i}' for i in range(5)];bots={s:dict(kind='legacy',policy_id=s,search_free=True) for s in profiles}
            def plan(s):return plan_collection(SOURCE_ROOT,root,s,c,dict(config=c),service,'current',profiles,bots,bots,None,
                fingerprints=dict(learner=fp,rule_bot=fp))
            from brass_ai.recovery.panel_gates import initial as initial_panels
            state['ratings']=dict(panel_gates=initial_panels(state['iteration']))
            before=copy.deepcopy(state);a=plan(state);b=plan(json.loads(json.dumps(state)))
            self.assertEqual(a,b);self.assertEqual(state,before)
            self.assertEqual(a['composition'],dict(selfplay=160,historical=144,legacy=16));self.assertEqual(a['context_counter']-81600,960)
            self.assertEqual(a['quality_planning'],value['next_planning'])
            pairs=Counter(tuple(t['learner_seats']) for t in a['quality_tables'].values());self.assertEqual(set(pairs.values()),{24})
            self.assertTrue(all(o['il_share']==.2 for t in a['quality_tables'].values() for o in t['opponents']))

    def test_exact_review_allows_only_sampler_and_rehearsal_changes(self):
        from brass_ai.recovery.gae import evidence_config
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0085-20261019T172419550588Z/state.json')
        signature=copy.deepcopy(state['signature']);signature['implementation']=implementation_id()
        signature['config']=sampling_config(evidence_config(signature['config']))
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaises(ValueError):check_signature(state,signature)
        for key,value in (('learning_rate',1e-4),('exploration',.2),('games_per_iteration',640)):
            bad=copy.deepcopy(signature);bad['config'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)


if __name__=='__main__':unittest.main()
