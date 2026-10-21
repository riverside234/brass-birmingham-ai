"""90% panel retirement, scored IL seats, frozen R2 reference and resume guards."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch, Mock

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery import panel_gates as g, ratings, pfsp_pool as pfsp, quality_pool as q
from brass_ai.recovery import v4_share, temperature_rating, snapshot_r2 as r2
from brass_ai.recovery.config import validate_config, latest_model_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.plackett_luce import IL, V4
from brass_ai.recovery.quality_config import RATING
from brass_ai.recovery.resume import check_signature, commit_transition
from brass_ai.step5.league import write_json, implementation_id
import test_strength_rating as rating_fixtures
from test_strength_rating import fixture_rows
from test_pfsp_pool import pool
from test_snapshot_r2 import observe, tick

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0270-20261020T233304700657Z'


def exported(): return read(EXPORT/'state.json')


def fixture(root, rates=(.9,.9), generation=10):
    state=rating_fixtures.Slots().state(root,copy.deepcopy(RATING));state['iteration']=generation
    state['history'][-1]['generation']=generation
    state['ratings']['agents'][V4]['policy']['search_free']=True
    c=dict(rating=copy.deepcopy(RATING),host='host',execution={},panel_gates=g.DEFAULT)
    spec=ratings.slot_spec(root,root,state['ratings'],state['history'][-1],c)
    panels=ratings.panel_report(fixture_rows(spec),spec['candidate'],None,RATING['seed'])
    for name,rate in zip(g.NAMES,rates):panels['panels'][name]['learner_win_share']=rate
    directory=root/f'slot-{generation}';directory.mkdir()
    write_json(directory/'spec.json',spec);write_json(directory/'panels.json',panels)
    slot=dict(generation=generation,**{k:ratings.receipt(root,directory/(k+'.json')) for k in ('spec','panels')})
    state['ratings']['slots']=[slot]
    return state,c,spec,panels


def trigger(model_id='model-0', generation=10):
    return dict(generation=generation,candidate=f'generation-{generation:06d}',games=32,setups=8,
        win_share=.9,threshold=.9,reference=digest(dict(model_id=model_id)),
        panels=dict(path='panels.json',sha256='a'*64),spec=dict(path='spec.json',sha256='b'*64))


class PanelGates(unittest.TestCase):
    def test_v4_share_90_boundary_and_threshold_migration_rechecks_latest(self):
        from test_v4_share import panel
        before=exported()['signature']['config'];c=g.configure(before)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp)
            for rate,passes in ((.899999,False),(.9,True),(1.,True)):
                s=dict(iteration=270,games_done=86400,ratings=panel(root,c,rate,generation=270))
                share=v4_share.prepare(SOURCE_ROOT,s,c)
                self.assertEqual(share['activated'] is not None,passes)
            s=dict(iteration=270,games_done=86400,signature=dict(config=before),
                ratings=panel(root,before,.9375,generation=260))
            s['v4_share_gate']=v4_share.prepare(SOURCE_ROOT,s,before)
            self.assertIsNone(s['v4_share_gate']['activated'])
            s['ratings']=panel(root,before,.8,generation=270)
            s['v4_share_gate'],_=v4_share.migrate_state(s,c)
            self.assertIsNone(s['v4_share_gate']['last_panel'])
            self.assertIsNone(v4_share.prepare(SOURCE_ROOT,s,c)['activated'])

    def test_inclusive_threshold_independent_panels_and_full_support(self):
        for rates,expected in (((.899999,.9),(False,True)),((.9,.899999),(True,False)),((1.,1.),(True,True))):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                root=Path(tmp);s,c,_,_=fixture(root,rates)
                before=copy.deepcopy(s);out=g.prepare(root,s,c)
                self.assertEqual(tuple(out['retired'][k] is not None for k in g.NAMES),expected)
                self.assertEqual(s,before)
                s['ratings']['panel_gates']=out
                self.assertEqual(g.prepare(root,json.loads(json.dumps(s)),c),out)
                path=root/s['ratings']['slots'][0]['panels']['path'];path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'receipt'):g.prepare(root,s,c)

    def test_v4_share_after_retired_main_only_backlog_and_restart(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,c,_,_=fixture(root)
            c['v4_share_gate']=copy.deepcopy(v4_share.CURRENT_DEFAULT)
            s.update(games_done=6400,iteration=20)
            s['ratings']['panel_gates']=g.prepare(root,s,c)
            s['v4_share_gate']=v4_share.initial(c,0,0)
            entry=dict(s['history'][-1],generation=20)
            spec=ratings.slot_spec(root,root,s['ratings'],entry,c)
            panels=ratings.panel_report(fixture_rows(spec),spec['candidate'],None,1,
                gates=s['ratings']['panel_gates'])
            directory=root/'slot-20';directory.mkdir()
            for key,value in (('spec',spec),('panels',panels)):write_json(directory/(key+'.json'),value)
            s['ratings']['slots'].append(dict(generation=20,**{
                key:ratings.receipt(root,directory/(key+'.json')) for key in ('spec','panels')}))
            s['ratings']['panel_gates']=g.prepare(root,s,c)
            share=v4_share.prepare(root,s,c)
            self.assertEqual(share['activated'],dict(generation=20,games=6400))
            self.assertEqual(share['last_panel']['generation'],10)
            self.assertEqual(share,v4_share.prepare(root,json.loads(json.dumps(s)),c))
            self.assertEqual(s['v4_share_gate']['activated'],None)
            s['v4_share_gate']=share
            self.assertEqual(share,v4_share.prepare(root,s,c))

    def test_adoption_uses_latest_not_past_peak_and_never_unlatches(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,c,_,_=fixture(root,(1.,1.));old=copy.deepcopy(s['ratings']['slots'][0])
            latched=g.prepare(root,s,c)
            newer,_,_,_=fixture(root,(.4,.4),20)
            s['iteration']=20;s['ratings']['slots']=[old,newer['ratings']['slots'][0]]
            self.assertTrue(all(t is None for t in g.prepare(root,s,c)['retired'].values()))
            s['ratings']['panel_gates']=latched
            out=g.prepare(root,s,c)
            self.assertEqual(out['retired'],latched['retired'])
            self.assertEqual(out['last_generation'],20)

    def test_reject_incomplete_panel_or_reference_changes_even_with_new_hash(self):
        for change in ('support','reference','rotation','nan'):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                root=Path(tmp);s,c,spec,panels=fixture(root)
                if change=='support':panels['panels']['il-panel']['games']=31
                if change=='reference':spec['agents'][IL]['checkpoint']['model_id']='other'
                if change=='rotation':spec['lineups'][-1]['seat']=0
                if change=='nan':panels['panels']['il-panel']['learner_win_share']=None
                for key,value in (('panels',panels),('spec',spec)):
                    path=root/s['ratings']['slots'][0][key]['path'];write_json(path,value)
                    s['ratings']['slots'][0][key]=ratings.receipt(root,path)
                with self.assertRaises(ValueError):g.prepare(root,s,c)

    def test_only_panel_games_removed_main_lineups_and_missing_reports(self):
        for retired in (('v4-panel',),('il-panel',),g.NAMES):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                root=Path(tmp);rates=tuple(.9 if n in retired else .8 for n in g.NAMES)
                s,c,base,_=fixture(root,rates);saved=g.prepare(root,s,c);s['ratings']['panel_gates']=saved
                # Use same candidate with no fitted history to compare identical main setup selection.
                spec=ratings.slot_spec(root,root,s['ratings'],s['history'][-1],c)
                self.assertEqual([r for r in spec['lineups'] if r['kind']=='rating'],
                                 [r for r in base['lineups'] if r['kind']=='rating'])
                self.assertEqual(len(spec['lineups']),192-32*len(retired))
                p=ratings.panel_report(fixture_rows(spec),spec['candidate'],None,1,gates=saved)
                for name in retired:
                    self.assertEqual(p['panels'][name]['games'],0)
                    self.assertIsNone(p['panels'][name]['learner_win_share'])
                    self.assertEqual(p['panels'][name]['status'],'retired')
                with self.assertRaisesRegex(ValueError,'Missing active'):
                    ratings.panel_report([],spec['candidate'],None,1)

    def test_retired_trigger_survives_archive_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,c,_,panels=fixture(root);saved=g.prepare(root,s,c);slot=s['ratings']['slots'][0]
            archive=dict(format=ratings.ARCHIVE_FORMAT,slots=[dict(generation=10,panels=panels,
                original_sha256={k:slot[k]['sha256'] for k in ('panels','spec')})],games=[dict(slot=10)])
            path=root/'archive.json';write_json(path,archive)
            s['ratings'].update(slots=[],panel_gates=saved,archive=ratings.receipt(root,path));s['iteration']=120
            self.assertEqual(g.prepare(root,s,c),saved)
            archive['slots'][0]['panels']['panels']['v4-panel']['learner_win_share']=.5
            write_json(path,archive);s['ratings']['archive']=ratings.receipt(root,path)
            with self.assertRaisesRegex(ValueError,'evidence differs'):g.prepare(root,s,c)

    def test_schedule_partial_retry_restart_and_main_only_next_slot(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);c=dict(RATING,setups=2,panel_setups=1,bootstrap=3)
            s=rating_fixtures.Slots().state(root,c);s['ratings']['agents'][V4]['policy']['search_free']=True
            cfg=dict(rating=c,host='host',execution={},panel_gates=g.DEFAULT,retention=dict(enabled=False))
            def collect(root,directory,spec,*args):
                rows=fixture_rows(spec)
                for r in rows:
                    if r['kind']!='rating':
                        seats=[i for i,a in enumerate(r['agents']) if a==spec['candidate']]
                        r['win_share']=[1/len(seats) if i in seats else 0 for i in range(4)]
                return rows
            with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,old=None:e),\
                 patch.object(ratings,'collect_slot',side_effect=collect) as collector,\
                 patch.object(ratings,'plot_report',side_effect=lambda path,*args:path.write_text('<svg/>')):
                first=ratings.schedule(root,root,s,cfg)
                self.assertEqual(first['protocol']['panel_games'],8)
                self.assertTrue(all(s['ratings']['panel_gates']['retired'].values()))
                checkpoint=copy.deepcopy(s)
                s['iteration']=20;s['history'].append(dict(s['history'][-1],generation=20))
                prior=copy.deepcopy(s)
                # Interrupt after immutable spec/games/reports, before state commit.
                real_write=ratings.write_json
                def fail(path,data):
                    if data['ratings']['latest_report']!=checkpoint['ratings']['latest_report']:raise RuntimeError('interrupted')
                    return real_write(path,data)
                with patch.object(ratings,'write_json',side_effect=fail),self.assertRaisesRegex(RuntimeError,'interrupted'):
                    ratings.schedule(root,root,s,cfg)
                s=prior;second=ratings.schedule(root,root,s,cfg)
                self.assertEqual(second['protocol']['panel_games'],0)
                self.assertEqual(second['games'],16)
                self.assertEqual(collector.call_count,2)
                self.assertIsNone(ratings.schedule(root,root,json.loads(json.dumps(s)),cfg))
                self.assertEqual(second['alerts']['v4_watch'],False)

    def test_config_upgrade_exact_blocks_and_temperature_default(self):
        s=exported();before=copy.deepcopy(s['signature']['config']);c=validate_config(g.configure(before))
        self.assertEqual(before,s['signature']['config'])
        self.assertEqual({k for k in c if c[k]!=before.get(k)},{'panel_gates','v4_share_gate','entropy_control'})
        self.assertEqual(c,g.configure(c));self.assertEqual(c['v4_share_gate'],v4_share.CURRENT_DEFAULT)
        self.assertEqual(latest_model_config(c,c['imitation_checkpoint'])['panel_gates'],g.DEFAULT)
        with patch.object(temperature_rating.ratings,'collect_slot',side_effect=AssertionError('evaluation ran')):
            self.assertIsNone(temperature_rating.schedule(None,None,{},c,device='cpu'))
        for key,value in (('win_share_at_least',.95),('il_history_sampling','uniform')):
            bad=copy.deepcopy(c);bad['panel_gates'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(c);bad['entropy_control']['deployment_temperature_check']['enabled']=0
        with self.assertRaises(ValueError):validate_config(bad)

    def test_generation270_reviewed_transition_preserves_counters_and_pool(self):
        s=exported();before=copy.deepcopy(s);c=validate_config(g.configure(s['signature']['config']))
        signature=dict(copy.deepcopy(s['signature']),implementation=implementation_id(),config=c)
        self.assertTrue(check_signature(s,signature,reviewed=True))
        bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(s,bad,reviewed=True)
        share,receipt=v4_share.migrate_state(s,c)
        for key in ('activated','last_panel','origin_generation','origin_games','allocation_origin'):
            self.assertEqual(share[key],s['v4_share_gate'][key])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=commit_transition(Path(tmp),s,signature)
        for key in s:
            if key not in ('signature','source_transitions','v4_share_gate'):self.assertEqual(out[key],s[key])
        self.assertEqual(s,before)


class ScoredIL(unittest.TestCase):
    def test_real_g270_planner_quotas_probabilities_and_checkpoint_plan_restart(self):
        from brass_ai.recovery.ordinary import plan_collection
        s=exported();c=g.configure(s['signature']['config']);before=copy.deepcopy(s)
        prefix='artifacts/step5/joint7-v11-2-grouped-lr-critic/'
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp)
            # Git's Windows checkout changes JSON newlines. Reconstruct the
            # attested Linux bytes only in this fixture, never relax runtime hashes.
            for key in ('spec','panels'):
                ref=s['ratings']['slots'][-1][key];relative=ref['path'].removeprefix(prefix)
                path=directory/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((EXPORT/relative).read_bytes().replace(b'\r\n',b'\n'))
                self.assertEqual(sha(path),ref['sha256'])
            with patch.object(g,'project_path',side_effect=lambda root,path:directory/path.removeprefix(prefix)):
                gates=g.prepare(SOURCE_ROOT,s,c)
            self.assertTrue(all(gates['retired'].values()))
            s['ratings']['panel_gates']=gates;s['v4_share_gate'],_=v4_share.migrate_state(s,c)
            path=directory/'weights.pt';path.write_bytes(b'only a planner fixture')
            for m in s['quality_pool']['members']:
                m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path))
            service=Mock();service.models={m['model_id']:None for m in s['quality_pool']['members']}
            fp=dict(values=[1/8]*16+[1/6]*12)
            profiles=[f'v4-{i}' for i in range(5)];bots={p:dict(kind='legacy',policy_id=p,search_free=True) for p in profiles}
            def plan(state):return plan_collection(SOURCE_ROOT,directory,state,c,dict(config=c),service,'current',
                profiles,bots,bots,None,fingerprints=dict(learner=fp,rule_bot=fp))
            a=plan(s);b=plan(json.loads(json.dumps(s)))
            self.assertEqual(a,b);self.assertEqual(a['composition'],dict(selfplay=160,historical=154,legacy=6))
            self.assertEqual(a['context_counter']-s['context_counter'],960)
            self.assertEqual(len(a['quality_tables']),154)
            self.assertEqual(a['quality_planning']['N'],17)
            self.assertAlmostEqual(a['quality_planning']['il_share'],0)
            self.assertLess(a['quality_planning']['il_probability'],.02)
            self.assertTrue(all(o['source']=='scored' for t in a['quality_tables'].values() for o in t['opponents']))
            games=[dict(setup_group=k,opponents=t['policy_ids'],result={}) for k,t in a['quality_tables'].items()]
            with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=[1,2,3,4])):
                nxt,report=q.update_scores(s['quality_pool'],games,a['quality_tables'],a['quality_planning'],generation=271,games_done=86720)
            self.assertEqual(report['exposure']['pairwise_comparisons'],616)
            self.assertEqual(report['exposure']['opponent_seats'],308)
            self.assertEqual(report['il']['sampling'],'pfsp')
            for old,new in zip(before['quality_pool']['members'],s['quality_pool']['members']):
                self.assertEqual((old['S'],old['C']),(new['S'],new['C']))
            q.validate_state(json.loads(json.dumps(nxt)),c['opponent_pool'],271)

    def test_same_weight_uniform_floor_normalization_and_retained_counts(self):
        p=pool(17);before=copy.deepcopy(p);p['members'][0].update(S=9.9,C=10.)
        pfsp.seal(p);gate=trigger();plan=q.distribution(p,il_gate=gate)
        self.assertEqual(plan['N'],17);self.assertEqual(plan['il_share'],0.)
        self.assertAlmostEqual(sum(r['p'] for r in plan['members']),1.)
        total=sum(pfsp.weight(m,p['config']) for m in p['members'])
        for m,r in zip(p['members'],plan['members']):
            self.assertAlmostEqual(r['p'],.8*pfsp.weight(m,p['config'])/total+.2/17)
        self.assertGreaterEqual(plan['il_probability'],.2/17)
        self.assertLess(plan['il_probability'],before['next_planning']['il_share'])
        self.assertEqual(p['members'][0]['S'],9.9)
        rng=random.Random(713);draws=[q.draw(p,plan,rng)[1] for _ in range(5000)]
        self.assertTrue(all(d['source']=='scored' for d in draws))
        self.assertAlmostEqual(sum(d['model_id']=='model-0' for d in draws)/len(draws),plan['il_probability'],delta=.007)

    def test_full_pairwise_update_and_restart_with_il_scored(self):
        p=pool(3);plan=q.distribution(p,il_gate=trigger());refs={r['model_id']:r for r in plan['members']}
        opponents=[dict(seat=i+2,model_id=f'model-{i}',source='scored',N=3,p=refs[f'model-{i}']['p'],
            il_share=0.,seat_probability=refs[f'model-{i}']['p']) for i in range(2)]
        table=dict(index=0,learner_seats=[0,1],opponents=opponents,policy_ids=['l','l','model-0','model-1'])
        games=[dict(setup_group='h',opponents=table['policy_ids'],result={})]
        with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=[1,2,3,4])):
            after,report=q.update_scores(p,games,{'h':table},plan,generation=101,games_done=32320)
        self.assertEqual(report['il']['sampling'],'pfsp');self.assertEqual(report['il']['comparisons'],2)
        self.assertEqual(after['members'][0]['S'],p['members'][0]['S']*.5**.2+2.)
        restored=json.loads(json.dumps(after));q.validate_state(restored,restored['config'],101)
        a=random.Random(77);b=random.Random(77)
        self.assertEqual([q.draw(after,q.distribution(after),a) for _ in range(100)],
                         [q.draw(restored,q.distribution(restored),b) for _ in range(100)])
        self.assertEqual(q.distribution(restored)['il_gate'],trigger())
        self.assertIn('including IL',q.sampling_description(after))
        bad=trigger();bad['generation']=20;bad['candidate']='generation-000020'
        with self.assertRaisesRegex(ValueError,'replaced'):q.distribution(after,il_gate=bad)
        bad=copy.deepcopy(table);bad['opponents'][0]['source']='IL'
        with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=[1,2,3,4])),\
             self.assertRaisesRegex(ValueError,'planning probability'):
            q.update_scores(p,games,{'h':bad},plan,generation=101,games_done=32320)

    def test_small_population_and_fallback_preserve_anchor(self):
        p=pool(1);plan=q.distribution(p,il_gate=trigger());self.assertEqual(plan['il_probability'],1.)
        self.assertEqual(q.draw(p,plan,random.Random(1))[0]['generation'],0)
        p=pool(4)
        for m in p['members']:m['S']=m['C']
        plan=q.distribution(p,il_gate=trigger())
        self.assertTrue(plan['all_zero_weights']);self.assertTrue(all(r['p']==.25 for r in plan['members']))


class RecoveryTolerance(unittest.TestCase):
    def test_tolerance_increase_never_lowers_frozen_reference(self):
        baseline=r2.summarize([observe(i) for i in range(1,11)])
        for drop,expected in ((.009999,.75),(.01,.75),(.010001,.5)):
            s=r2.initial(10,.5);s['regime']=dict(source='ppo')
            s['reference']=r2.reference(baseline,10,.5,s['regime']);ref=copy.deepcopy(s['reference'])
            for i in range(11,25):s,w,_,_,report=tick(s,i,bias=math.sqrt(drop))
            self.assertEqual(w,expected);self.assertEqual(s['reference'],ref)
            self.assertEqual(report['recovery_tolerance'],.01)
            self.assertAlmostEqual(report['recover_at_or_above'],baseline['r2']-.01)
            self.assertAlmostEqual(report['reduce_below'],baseline['r2']-.02)
            if expected==.75:self.assertEqual(s['entries'],[])

    def test_g270_evidence_and_compact_fields(self):
        r=read(EXPORT/'iteration-0269/report.json');v=r['value_phase']['critic_rollout']
        self.assertEqual(v['weight_next'],1.)
        self.assertAlmostEqual(v['learner_rolling_r2'],.37544778297935644)
        self.assertEqual(v['weight_recovery']['learner']['games'],306)
        r['panel_gates']=g.initial(270);self.assertEqual(compact(r)['panel_gates'],r['panel_gates'])


if __name__=='__main__':unittest.main()
