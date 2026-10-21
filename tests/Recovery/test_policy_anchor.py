"""Qualified-anchor selection, independent calibration, balanced ratings and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.step5.league import write_json
from brass_ai.recovery import policy_anchor as a,rating_coverage as c,panel_gates as g,il_kl_limit as limits
from brass_ai.recovery import ratings,exploration
from brass_ai.recovery.plackett_luce import IL,V4
from brass_ai.recovery.quality_config import RATING
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.history import compact
from test_auto_il_kl import baseline,measurement,counts


def evaluated(root, generations=(10,20,30), rates=(1.,1.)):
    generation=max(generations);agents={}
    for gen in (0,*generations):
        path=root/f'weights-{gen}.pt';path.write_bytes(f'frozen-{gen}'.encode())
        agents[f'generation-{gen:06d}' if gen else IL]=dict(kind='model',checkpoint=dict(
            path=path.relative_to(root).as_posix(),sha256=sha(path),model_id=f'm{gen}',generation=gen))
    agents[V4]=dict(kind='v4',policy=dict(search_free=True,policy_id='v4'))
    config=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-qualified-anchor.json')
    config['rating']=copy.deepcopy(RATING)
    from brass_ai.step5.league import seeds_for
    seeds,_=seeds_for(RATING['seed'],32,'validation')
    rating=dict(contract=RATING['contract'],config=copy.deepcopy(RATING),agents=agents,seeds=seeds,slots=[],latest_report=None)
    names=[IL,V4]+[f'generation-{gen:06d}' for gen in generations]
    theta=np.array([0.,-.2]+[i+1. for i in range(len(generations))])
    noise=np.random.default_rng(41).normal(0,.05,(200,len(names)));noise[:,0]=0.
    samples=theta+noise
    state=dict(iteration=generation,games_done=320*generation,ratings=rating,
        history=[agents[names[-1]]['checkpoint']],evaluation_anchor=agents[IL]['checkpoint'],
        il_kl_limit=baseline())
    state['il_kl_limit'].update(last_generation=generation,last_update=generation,limit=.8,blocked=True)
    install_fit(root,state,names,theta,samples,rates)
    return state,config,names,theta,samples


def install_fit(root,state,names,theta,samples,rates=(1.,1.)):
    rating=state['ratings'];generation=state['iteration'];candidate=f'generation-{generation:06d}';scale=RATING['elo_scale']
    lineups=[]
    for seed in rating['seeds'][:8]:
        for seat in range(4):
            lineups.extend([dict(seed=seed,seat=seat,kind='v4-panel',agents=[candidate,V4,V4,V4]),
                            dict(seed=seed,seat=seat,kind='il-panel',agents=[candidate,candidate,IL,IL])])
    spec=dict(generation=generation,candidate=candidate,config=RATING,agents=rating['agents'],lineups=lineups)
    panels=dict(candidate=candidate,panels={name:dict(games=32,setups=8,learner_win_share=rate)
        for name,rate in zip(g.NAMES,rates)})
    differences=[]
    for i,x in enumerate(names):
        for j,y in enumerate(names[i+1:],i+1):
            interval=np.quantile(samples[:,j]-samples[:,i],[.025,.975])*scale
            differences.append(dict(a=x,b=y,points_delta=float(theta[j]-theta[i])*scale,points_delta_ci95=interval.tolist()))
    report=dict(generation=generation,candidate=candidate,contract=RATING['contract'],fit=dict(converged=True),
        ratings={n:dict(theta=float(theta[i]),points=float(theta[i])*scale) for i,n in enumerate(names)},differences=differences)
    folder=root/'ratings'/'slots'/f'generation-{generation:06d}';folder.mkdir(parents=True,exist_ok=True)
    for name,data in (('spec',spec),('report',report),('panels',panels)):write_json(folder/(name+'.json'),data)
    with (folder/'bootstrap.npz').open('wb') as f:np.savez(f,theta=samples,agents=np.asarray(names))
    slot=dict(generation=generation,**{k:ratings.receipt(root,folder/(k+('.npz' if k=='bootstrap' else '.json')))
        for k in ('spec','report','panels','bootstrap')})
    rating.update(slots=[slot],latest_report=slot['report'])


class Selection(unittest.TestCase):
    def test_initial_highest_available_qualified_then_only_newer(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,cfg,n,t,draws=evaluated(root)
            chosen,fit,_=a.select(root,s,a.initial(30));self.assertEqual(chosen['agent'],n[-1])
            # Higher point score alone is insufficient: make g30's paired interval straddle zero.
            draws[:80,-1]=-10.;install_fit(root,s,n,t,draws)
            chosen,fit,_=a.select(root,s,a.initial(30));self.assertEqual(chosen['agent'],n[-2])
            ctrl=a.initial(30);ctrl['active']=dict(agent=n[-2],checkpoint=s['ratings']['agents'][n[-2]]['checkpoint'])
            selected,_,_=a.select(root,s,ctrl);self.assertIsNone(selected)
            # A historically stronger but older checkpoint cannot replace it.
            t[2]=20.;draws[:,2]=20.;install_fit(root,s,n,t,draws)
            self.assertIsNone(a.select(root,s,ctrl)[0])
            # Missing highest checkpoint falls back to the next qualified available one.
            (root/s['ratings']['agents'][n[2]]['checkpoint']['path']).unlink()
            self.assertEqual(a.select(root,s,a.initial(30))[0]['agent'],n[-2])

    def test_paired_samples_not_separate_intervals(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,cfg,n,t,draws=evaluated(root)
            shared=np.linspace(-10,10,200);draws[:,2]=shared+1;draws[:,3]=shared+2
            install_fit(root,s,n,t,draws)
            fit={k:s['ratings']['slots'][0][k] for k in ('generation','report','spec','bootstrap')}
            report,samples,names=a.load_fit(root,s['ratings'],fit)
            self.assertGreater(np.quantile(samples[:,2],.975),np.quantile(samples[:,3],.025))
            d=a.paired_difference(report,samples,names,n[3],n[2],RATING['elo_scale'])
            np.testing.assert_allclose(d['points_delta_ci95'],[173.7,173.7])
            report['differences'][next(i for i,r in enumerate(report['differences']) if r['a']==n[2] and r['b']==n[3])]['points_delta_ci95']=[0,1]
            with self.assertRaisesRegex(ValueError,'shared bootstrap'):a.paired_difference(report,samples,names,n[3],n[2],173.7)

    def test_mismatched_fit_bootstrap_or_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,cfg,n,t,draws=evaluated(root)
            path=root/s['ratings']['agents'][n[-1]]['checkpoint']['path'];path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):a.select(root,s,a.initial(30))
            (root/s['ratings']['slots'][0]['bootstrap']['path']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'evidence changed'):a.select(root,s,a.initial(30))


class Calibration(unittest.TestCase):
    def test_dual_probe_uses_one_forward_and_preserves_original_il_arrays(self):
        from test_auto_il_kl import ProbeTests,ToyPolicy
        from brass_ai.recovery import il_kl_probe as probes
        rows=ProbeTests().data();il=ToyPolicy();reference=copy.deepcopy(il);current=copy.deepcopy(il)
        with torch.no_grad():reference.bias.copy_(torch.tensor([2.,1.,0.]));current.bias.copy_(torch.tensor([-1.,3.,2.]))
        update=dict(microbatch_size=3,max_candidate_cells=100)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);source=root/'data.jsonl.gz';source.write_bytes(b'validation')
            manifest=root/'manifest.json';write_json(manifest,dict(entries=[dict(split='validation',path=source.name,sha256=sha(source))]))
            origin=dict(manifest=manifest.name,manifest_sha256=sha(manifest),model_id='il',sha256='a'*64)
            with patch('brass_ai.recovery.demonstrations.load_dataset',return_value=(rows,{})):
                probe,_=probes.load(root,root/'run',origin,il,limits.DEFAULT,update,1)
            arrays={k:v.copy() for k,v in probe.arrays.items()};rng=torch.get_rng_state().clone()
            probabilities=probe.probabilities(reference,update)
            original=probe.measure(current,update)
            with patch.object(current,'forward',wraps=current.forward) as forward:
                dual=probe.measure(current,update,reference=probabilities)
                self.assertEqual(forward.call_count,len(list(probe.parts(update))))
            self.assertEqual(dual['original_il']['kl_mean'],original['kl_mean'])
            self.assertNotEqual(dual['kl_mean'],original['kl_mean'])
            self.assertEqual(probe.measure(reference,update,reference=probabilities)['kl_mean'],0.)
            for key,value in arrays.items():np.testing.assert_array_equal(value,probe.arrays[key])
            for group in limits.GROUPS:
                self.assertEqual(dual['groups'][group]['raw_entropy'],original['groups'][group]['raw_entropy'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            bad=probabilities.copy();bad[0]=-1
            with self.assertRaises(ValueError):probe.measure(current,update,reference=bad)

    def test_reset_preserves_health_and_recomputes_overall_group_block(self):
        old=baseline();old.update(limit=.8,blocked=True,healthy_window=[.7]*8,kl_last_healthy=.7)
        for mean,group,blocked in ((.1,.1,False),(.16,.1,True),(.1,.31,True)):
            measured=measurement(mean);measured['groups']['rail_other']['kl_mean']=group
            new,report=limits.reset_reference(old,measured,generation=10)
            self.assertEqual(new['blocked'],blocked);self.assertEqual(new['limit'],.15)
            self.assertEqual(report['group_limit_after'],.30);self.assertEqual(new['healthy_window'],[])
            for k in ('baseline','baseline_rows','rolling','probe','last_generation'):self.assertEqual(old[k],new[k])
            for i in range(9):
                new,r=limits.advance(new,measurement(.3),counts(),generation=11+i,warmup_done=True)
                self.assertEqual(new['limit'],.15)
            new,r=limits.advance(new,measurement(.3),counts(),generation=20,warmup_done=True)
            self.assertEqual(new['limit'],.1875)
        bad=copy.deepcopy(old);bad['rolling']=[counts(vp_sum=0)]*5
        _,report=limits.reset_reference(bad,measurement(.001),generation=10)
        self.assertFalse(report['lr_anchor_health_permits'])
        self.assertFalse(report['health']['ok'])

    def test_checkpoint_bound_hotfix_is_idempotent_and_does_not_touch_other_state(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,cfg,n,t,draws=evaluated(root)
            s['ratings']['panel_gates']=g.prepare(root,s,cfg)
            s['unrelated']=dict(adam='kept',rng=[1,2,3],gae=.95,r2=.4,lr_multiplier=1.5625)
            before=copy.deepcopy(s);raw=dict(step=30,model_id='m30',training=dict(il_kl_limit=copy.deepcopy(s['il_kl_limit'])))
            write_json(root/'state.json',s)
            model=torch.nn.Linear(1,1)
            class Probe:
                def measure(self,*args,**kw):return dict(measurement(.2),original_il=measurement(1.1))
            with patch.object(a,'cache',return_value=dict(path='cache.json',sha256='a'*64)),\
                 patch.object(a,'values',return_value=np.array([1.])),patch.object(a,'verify_active'):
                _,decision=a.prepare(root,root,s,cfg,Probe(),model)
                self.assertEqual(decision['decision'],'promote');self.assertTrue(s['il_kl_limit']['blocked'])
                self.assertEqual(s['unrelated'],before['unrelated']);self.assertEqual(s['evaluation_anchor'],before['evaluation_anchor'])
                self.assertEqual(s['history'],before['history'])
                a.verify_checkpoint(s,raw)
                snapshot=json.loads(json.dumps(s));payload=(root/'state.json').read_bytes()
                a.prepare(root,root,s,cfg,Probe(),model)
                self.assertEqual(s,snapshot);self.assertEqual(payload,(root/'state.json').read_bytes())
                tampered=copy.deepcopy(s);tampered['il_kl_limit']['baseline']['metrics']['vp_per_seat']=200.
                with self.assertRaises(ValueError):a.verify_checkpoint(tampered,raw)
                next_controller=a.checkpoint_state(s);self.assertIsNone(next_controller['pending'])
                raw['training'].update(policy_anchor=next_controller,il_kl_limit=s['il_kl_limit'])
                s['policy_anchor']=next_controller;a.verify_checkpoint(s,raw)
                _,again=a.prepare(root,root,s,cfg,Probe(),model)
                self.assertEqual(again['decision'],'hold')
                self.assertEqual(compact(dict(iteration=29,policy_anchor=next_controller))['policy_anchor'],next_controller)


class Coverage(unittest.TestCase):
    def test_archived_retirement_qualifies_anchor_and_coverage_without_reopening_panels(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,cfg,n,t,draws=evaluated(root)
            saved=g.prepare(root,s,cfg);old=s['ratings']['slots'][0]
            archive=dict(format=ratings.ARCHIVE_FORMAT,slots=[dict(generation=30,
                panels=read(root/old['panels']['path']),original_sha256={k:old[k]['sha256'] for k in ('panels','spec')})],
                games=[dict(slot=30)])
            path=root/'ratings/archive.json';write_json(path,archive)
            s['ratings'].update(panel_gates=saved,archive=ratings.receipt(root,path))
            s['iteration']=40
            # A newer completed slot has no panels. The old 90% receipt survives.
            s['ratings']['agents']['generation-000040']=copy.deepcopy(s['ratings']['agents'][n[-1]])
            s['ratings']['agents']['generation-000040']['checkpoint']['generation']=40
            install_fit(root,s,n+['generation-000040'],np.append(t,3.1),np.column_stack([draws,draws[:,-1]+.1]))
            slot=s['ratings']['slots'][0]
            panels=dict(candidate='generation-000040',panels={name:dict(games=0,setups=0,
                learner_win_share=None,status='retired',trigger=saved['retired'][name]) for name in g.NAMES})
            spec=read(root/slot['spec']['path']);spec['lineups']=[]
            for key,value in (('panels',panels),('spec',spec)):
                p=root/slot[key]['path'];write_json(p,value);slot[key]=ratings.receipt(root,p)
            self.assertEqual(g.prepare(root,s,cfg)['retired'],saved['retired'])
            self.assertTrue(c.prepare(root,s,cfg)['activated'])
            self.assertIsNotNone(a.select(root,s,a.initial(40))[0])

    def test_80_gate_is_strict_and_independent_of_90_retirement(self):
        for rates,expected in (((.8,1.),False),((1.,.8),False),((.81,.85),True),((1.,1.),True)):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                root=Path(tmp);s,cfg,_,_,_=evaluated(root,rates=rates)
                result=c.prepare(root,s,cfg);self.assertEqual(result['activated'] is not None,expected)
                if expected:
                    s['ratings']['coverage']=result
                    self.assertEqual(c.prepare(root,json.loads(json.dumps(s)),cfg),result)

    def test_128_games_distinct_balanced_nearby_required_and_rotating_reference_seeds(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,cfg,n,t,samples=evaluated(root,generations=tuple(range(10,130,10)))
            rating=s['ratings'];scores=read(root/rating['latest_report']['path'])['ratings']
            candidate=n[-1];anchor=n[2]
            rows,report=c.lineups(rating,rating['agents'],candidate,scores,130,anchor)
            self.assertEqual(len(rows),128);self.assertEqual(report['rated_opponents_per_game'],2.75)
            setups=rows[::4];self.assertEqual(sum(IL in r['agents'] for r in setups),4)
            self.assertEqual(sum(V4 in r['agents'] for r in setups),4)
            self.assertEqual(sum(IL not in r['agents'] and V4 not in r['agents'] for r in setups),24)
            counts=report['rated_setup_appearances'];self.assertEqual(set(counts.values()),{11})
            self.assertIn(anchor,counts);self.assertIn(n[-2],counts)
            for i,row in enumerate(setups):
                self.assertEqual(len(set(row['agents'])),4)
                self.assertEqual([r['seat'] for r in rows[i*4:i*4+4]],[0,1,2,3])
            self.assertEqual((rows,report),c.lineups(rating,rating['agents'],candidate,scores,130,anchor))
            _,next_report=c.lineups(rating,rating['agents'],candidate,scores,140,anchor)
            self.assertNotEqual(report['reference_setup_indices'],next_report['reference_setup_indices'])
            rating['coverage']=c.prepare(root,s,cfg);rating['panel_gates']=g.prepare(root,s,cfg)
            frozen=copy.deepcopy(rating)
            spec=ratings.slot_spec(root,root,rating,dict(rating['agents'][candidate]['checkpoint'],generation=130),cfg,anchor_agent=anchor)
            self.assertEqual(len(spec['lineups']),128)
            self.assertEqual(spec['coverage']['rated_opponents_per_game'],2.75)
            self.assertEqual(rating,frozen)
            jobs=ratings.jobs_for(spec,root)
            self.assertTrue(all(p.get('exploration',0)==0. for j in jobs for p in j['policies']))

    def test_duplicate_models_and_small_pool_fallback_explicit(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);s,cfg,n,t,samples=evaluated(root,generations=(10,20,30))
            rating=s['ratings'];scores=read(root/rating['latest_report']['path'])['ratings']
            rating['agents'][n[2]]['checkpoint']['model_id']=rating['agents'][n[3]]['checkpoint']['model_id']
            rows,report=c.lineups(rating,rating['agents'],n[-1],scores,40)
            self.assertEqual(len(report['nearest']),1);self.assertTrue(report['startup_fallback_seats'])
            self.assertEqual(len(rows),128)


class Configuration(unittest.TestCase):
    def test_g270_reviewed_continuation_preserves_progress_and_rejects_other_recipe_edits(self):
        from brass_ai.recovery.resume import check_signature,commit_transition
        from brass_ai.step5.league import implementation_id
        from test_panel_gates import exported
        state=exported();before=copy.deepcopy(state)
        cfg=validate_config(a.configure(state['signature']['config']))
        signature=dict(copy.deepcopy(state['signature']),implementation=implementation_id(),config=cfg)
        self.assertTrue(check_signature(state,signature,reviewed=True))
        wrong=copy.deepcopy(signature);wrong['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,wrong,reviewed=True)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            after=commit_transition(Path(tmp),state,signature)
        for key in before:
            if key not in ('signature','source_transitions','v4_share_gate'):self.assertEqual(after[key],before[key])
        self.assertEqual(state,before)

    def test_retention_pins_active_checkpoint_cache_and_promotion_evidence(self):
        from brass_ai.recovery.retention import references
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            def put(name,value):
                path=run/name;path.parent.mkdir(parents=True,exist_ok=True)
                if isinstance(value,bytes):path.write_bytes(value)
                else:write_json(path,value)
                return ratings.receipt(root,path)
            cp=put('population/weights-active.pt',b'model')
            arrays=put('policy-anchor/probabilities/cache.npz',b'cached probabilities')
            cache=put('policy-anchor/probabilities/cache.json',dict(probabilities=arrays))
            fit={key:put('ratings/slots/generation-000030/'+key+'.json',{}) for key in ('report','spec')}
            fit['bootstrap']=put('ratings/slots/generation-000030/bootstrap.npz',b'paired draws')
            state=dict(format='step5-recovery-league/2.2.0',signature=dict(config={}),
                policy_anchor=dict(active=dict(checkpoint=cp,cache=cache,fit=fit)))
            write_json(run/'state.json',state)
            pinned=references(root,run)
            for ref in [cp,cache,arrays,*fit.values()]:self.assertIn(root/ref['path'],pinned)

    def test_only_explicit_blocks_change_and_epsilon_behavior_identity(self):
        before=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-g270.json')
        cfg=validate_config(a.configure(before))
        self.assertEqual({k for k in cfg if cfg[k]!=before.get(k)},{'policy_anchor','rating_coverage','exploration'})
        self.assertEqual(cfg['exploration'],.02)
        self.assertEqual(latest_model_config(cfg,cfg['imitation_checkpoint'])['exploration'],.02)
        from brass_ai.recovery.league import Behavior
        settings=exploration.effective(cfg,exploration.initial(cfg,0),100)
        old=Behavior('fixed-model',.03,settings);new=Behavior('fixed-model',.02,settings)
        self.assertNotEqual(old.policy_id,new.policy_id)
        for field in ('initial_limit','initial_group_limit'):
            bad=copy.deepcopy(cfg);bad['policy_anchor'][field]*=2
            with self.assertRaises(ValueError):validate_config(bad)
        for field in ('policy_anchor','rating_coverage'):
            bad=copy.deepcopy(cfg);bad.pop(field)
            with self.assertRaises(ValueError):validate_config(bad)


if __name__=='__main__':unittest.main()
