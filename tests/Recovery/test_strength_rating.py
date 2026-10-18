"""PL likelihood/uncertainty and immutable seat-rotated evaluation lifecycle."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.step5.league import seeds_for
from brass_ai.recovery import ratings as r,plackett_luce as pl
from brass_ai.recovery.quality_config import RATING


AGENTS=[pl.IL,pl.V4,'generation-000010','generation-000020']


def simulated(theta=(0.,-.3,.4,.8),seat=(.2,-.15,.1,-.15),setups=32,seed=719):
    rng=np.random.default_rng(seed);games=[]
    for setup in range(setups):
        for rotation in range(4):
            indices=np.roll(np.arange(4),rotation);agents=[AGENTS[i] for i in indices]
            utility=np.asarray(theta)[indices]+np.asarray(seat)
            order=np.argsort(-(utility+rng.gumbel(size=4))).tolist()
            games.append(dict(slot=10,setup_id=f'setup-{setup:04d}',candidate_seat=rotation,
                game_id=f'game-{setup:04d}-{rotation}',agents=agents,order=order))
    return games


def fixture_rows(spec):
    result=[]
    for i,row in enumerate(spec['lineups']):
        agents=[row['agents'][(s-row['seat'])%4] for s in range(4)]
        order=np.roll(np.arange(4),i%4).tolist();ranks=[1+order.index(s) for s in range(4)]
        result.append(dict(slot=spec['generation'],setup_id=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=row['seed'])),
            setup_seed=row['seed'],candidate_seat=row['seat'],kind=row['kind'],agents=agents,order=order,
            official_placement=ranks,vp=[5*rank for rank in ranks],win_share=[int(rank==1) for rank in ranks],
            spec_digest=digest(spec),game_id=digest(dict(spec=digest(spec),index=i))))
    return result


class PlackettLuce(unittest.TestCase):
    def test_analytic_gradient_including_repeated_agents_and_seat_constraint(self):
        ordered=np.array([[0,1,2,2],[2,2,0,1]]);orders=np.array([[0,1,2,3],[3,2,0,1]])
        x=np.array([.3,-.2,.1,-.1,.2]);weights=np.array([2.,1.])
        fn=lambda z:pl.objective(z,ordered,orders,weights,3,2.,.5)
        value,grad=fn(x);eps=1e-6
        numeric=np.array([(fn(x+np.eye(5)[i]*eps)[0]-fn(x-np.eye(5)[i]*eps)[0])/(2*eps) for i in range(5)])
        np.testing.assert_allclose(grad,numeric,rtol=1e-6,atol=1e-7)
        self.assertTrue(np.isfinite(value))

    def test_recovers_known_theta_and_seats_and_keeps_IL_zero(self):
        theta=np.array([0.,-.3,.4,.8]);seat=np.array([.2,-.15,.1,-.15])
        result=pl.fit(simulated(theta,seat,setups=1500),AGENTS,RATING)
        actual=dict(zip(result['agents'],result['theta']))
        np.testing.assert_allclose([actual[a] for a in AGENTS],theta,atol=.09)
        np.testing.assert_allclose(result['seat'],seat,atol=.07)
        self.assertEqual(actual[pl.IL],0.);self.assertAlmostEqual(sum(result['seat']),0.)

    def test_bootstrap_refit_is_order_independent_and_difference_intervals_used(self):
        games=simulated();c=dict(RATING,bootstrap=25)
        before=torch.get_rng_state().clone()
        first,samples=pl.estimate(games,AGENTS,c)
        second,again=pl.estimate(list(reversed(games)),list(reversed(AGENTS)),c)
        self.assertEqual(first,second);np.testing.assert_array_equal(samples,again)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(first['ratings'][pl.IL]['theta_ci95'],[0.,0.])
        self.assertEqual(first['bootstrap_refits'],25)
        for d in first['differences']:
            self.assertEqual(d['b_stronger'],d['points_delta_ci95'][0]>0)
            self.assertEqual(d['a_stronger'],d['points_delta_ci95'][1]<0)

    def test_bootstrap_resamples_setup_once_across_slots_and_rotations(self):
        games=simulated(setups=3);later=[dict(g,slot=20,game_id=g['game_id']+'later') for g in games]
        calls=[]
        def fit(ordered,orders,count,config,weights=None):
            calls.append(weights);return np.zeros(count),np.zeros(4),dict(converged=True)
        with patch.object(pl,'fit_encoded',side_effect=fit):pl.estimate(games+later,AGENTS,dict(RATING,bootstrap=4))
        ordered_games=pl.encode(games+later,AGENTS)[0]
        for weights in calls[1:]:
            for setup in {g['setup_id'] for g in ordered_games}:
                self.assertEqual(len(set(weights[i] for i,g in enumerate(ordered_games) if g['setup_id']==setup)),1)
            self.assertEqual(float(weights.sum()),24.)


class Slots(unittest.TestCase):
    def test_frozen_reference_keeps_bank_admission_distinct_from_game_purpose(self):
        config=dict(rating=copy.deepcopy(RATING),legacy_budget={},legacy_opponent_work_budget={})
        state=dict(population=[dict(generation=0,model_id='il',path='initial.pt',sha256='a'*64)])
        with patch('brass_ai.recovery.legacy_search.legacy_spec',return_value=dict(search_free=True)) as legacy:
            result=r.initialize(SOURCE_ROOT,SOURCE_ROOT,state,config)
        self.assertEqual(legacy.call_args.kwargs['purpose'],'training')
        self.assertTrue(legacy.call_args.kwargs['search_free']);self.assertEqual(len(result['seeds']),32)
        # The actual pinned Balanced reference is admitted to the training
        # bank; "evaluation" bank admission is reserved for Pottery.
        from brass_ai.bank import admission
        admission(SOURCE_ROOT,RATING['v4_profile'],'training')

    def state(self,root,config):
        path=root/'initial.pt';path.write_bytes(b'initial');candidate=root/'candidate.pt';candidate.write_bytes(b'candidate')
        def entry(path,g):return dict(path=path.name,sha256=sha(path),generation=g,model_id=f'model-{g}')
        anchor=entry(path,0);new=entry(candidate,10);seeds,_=seeds_for(config['seed'],config['setups'],'validation')
        rating=dict(contract=config['contract'],config=config,seeds=seeds,slots=[],latest_report=None,
            agents={pl.IL:dict(kind='model',checkpoint=anchor),pl.V4:dict(kind='v4',policy=dict(kind='legacy',policy_id='v4'))})
        return dict(format='step5-recovery-league/2.2.0',ratings=rating,iteration=10,history=[anchor,new],population=[anchor],
            recent_training=[dict(iteration=9,il_kl_limit=dict(health=dict(ok=True)))])

    def test_pairing_reference_balance_rotation_startup_and_no_search(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);state=self.state(root,dict(RATING))
            c=dict(rating=RATING,host='host',execution={})
            spec=r.slot_spec(root,root,state['ratings'],state['history'][-1],c);jobs=r.jobs_for(spec,root)
            main=[row for row in spec['lineups'] if row['kind']=='rating']
            self.assertEqual(len(main),128);self.assertTrue(spec['startup_reference_fallback'])
            self.assertEqual(sum(row['agents'][1]==pl.IL for row in main),64)
            for seed in state['ratings']['seeds']:
                group=[row for row in main if row['seed']==seed]
                self.assertEqual({g['seat'] for g in group},{0,1,2,3})
                self.assertTrue(all(g['agents']==group[0]['agents'] for g in group))
            for job in jobs:
                self.assertEqual(job['provenance']['purpose'],'evaluation')
                for policy in job['policies']:
                    if policy['kind']=='model':self.assertEqual(policy['exploration'],0);self.assertEqual(policy['focus'],'neutral')

    def test_official_ties_have_seeded_uniform_order_without_training_rng(self):
        row=dict(seed=42,seat=1,kind='rating',agents=AGENTS)
        spec=dict(config=RATING,generation=10,lineups=[row]);game=dict(game_id='game',setup_group='s',result={})
        targets=dict(placement=[1,1,3,4],vp=[20,20,10,0],win_share=[.5,.5,0,0])
        before=torch.get_rng_state().clone()
        with patch('brass_ai.recovery.ratings.terminal_targets',return_value=targets):
            first=r.game_record(game,row,spec);self.assertEqual(first,r.game_record(game,row,spec))
        self.assertEqual(set(first['order'][:2]),{0,1});self.assertEqual(first['order'][2:],[2,3])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_slot_commit_restart_is_idempotent_store_append_only_and_refit_reproducible(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);c=dict(RATING,setups=2,panel_setups=1,bootstrap=4)
            state=self.state(root,c);before=copy.deepcopy(state);cfg=dict(rating=c,host='host',execution={})
            def collect(root,directory,spec,*args):return fixture_rows(spec)
            def plot(path,*args):path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,old=None:e),\
                 patch.object(r,'collect_slot',side_effect=collect) as collector,patch.object(r,'plot_report',side_effect=plot):
                first=r.schedule(root,root,state,cfg)
                self.assertEqual(first['games'],8);self.assertEqual(first['protocol']['panel_games'],8)
                self.assertIsNone(first['alerts']['deployment_choice'])
                slot=state['ratings']['slots'][0];raw=(root/slot['records']['path']).read_bytes()
                self.assertIsNone(r.schedule(root,root,state,cfg))
                restored=copy.deepcopy(before);again=r.schedule(root,root,restored,cfg)
                self.assertEqual(first,again);self.assertEqual(state,restored);self.assertEqual(collector.call_count,1)
                self.assertEqual(raw,(root/slot['records']['path']).read_bytes())
                (root/slot['records']['path']).write_bytes(raw+b'{}\n')
                with self.assertRaisesRegex(ValueError,'evidence changed'):r.schedule(root,root,state,cfg)

    def test_regression_watch_and_pending_deployment_do_not_change_sampler(self):
        ratings={pl.IL:dict(points=0),pl.V4:dict(points=20),
            'generation-000010':dict(points=100,points_ci95=[80,120]),
            'generation-000020':dict(points=20,points_ci95=[0,40])}
        state=dict(agents={a:dict(checkpoint=dict(generation=int(a.split('-')[1]))) for a in ratings if a not in (pl.IL,pl.V4)})
        panel=lambda win:dict(panels={'v4-panel':dict(learner_win_share=win)})
        result=r.alerts(dict(ratings=ratings),state,panel(.1),[panel(.3),panel(.2)])
        self.assertTrue(result['regression']);self.assertTrue(result['v4_watch'])
        self.assertIsNone(result['deployment_choice'])

    def test_repeated_model_uses_existing_rated_cache_outside_training_pool(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);c=dict(RATING,setups=2,panel_setups=1,bootstrap=4)
            state=self.state(root,c);cfg=dict(rating=c,host='host',execution={})
            def slim(root,run,entry,old=None):return dict(entry,weights_only=True)
            with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=slim) as cached,\
                 patch.object(r,'collect_slot',side_effect=lambda root,directory,spec,*args:fixture_rows(spec)),\
                 patch.object(r,'plot_report',side_effect=lambda path,*args:path.write_text('<svg/>')):
                r.schedule(root,root,state,cfg)
                frozen=state['ratings']['agents']['generation-000010']['checkpoint']
                state['iteration']=20;state['history'].append(dict(state['history'][-1],generation=20))
                r.schedule(root,root,state,cfg)
            self.assertEqual(cached.call_args.args[-1],frozen)
            self.assertEqual(len(state['ratings']['slots']),2)

    def test_offline_refit_command_reads_only_committed_main_games(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';c=dict(RATING,setups=2,panel_setups=1,bootstrap=4)
            state=self.state(root,c);cfg=dict(rating=c,host='host',execution={});state['signature']=dict(config=cfg)
            with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,old=None:e),\
                 patch.object(r,'collect_slot',side_effect=lambda root,directory,spec,*args:fixture_rows(spec)) as collector,\
                 patch.object(r,'plot_report',side_effect=lambda path,*args:path.write_text('<svg/>')):
                first=r.schedule(root,run,state,cfg)
                before={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
                spec=importlib.util.spec_from_file_location('strength_rating_command',SOURCE_ROOT/'scripts/strength-ratings.py')
                command=importlib.util.module_from_spec(spec);spec.loader.exec_module(command);command.ROOT=root
                with patch.object(sys,'argv',['strength-ratings.py','--run','artifacts/run','--output','artifacts/review']):
                    command.main()
                report=json.loads((root/'artifacts/review/report.json').read_text())
                self.assertEqual(report['games'],8);self.assertEqual(report['ratings'],first['ratings'])
                self.assertIsNone(report['alerts']['deployment_choice']);self.assertEqual(collector.call_count,1)
                self.assertTrue(all(path.read_bytes()==raw for path,raw in before.items()))


if __name__=='__main__':unittest.main()
