"""Bounded analytic, lifecycle and receipt checks; no campaign or training."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.step5.league import write_json
from brass_ai.step5.policy import context_for
from brass_ai.recovery import ratings
from brass_ai.recovery import context_evaluation_config as C,context_evaluation_spec as S
from brass_ai.recovery import context_evaluation_collect as E,context_evaluation_analysis as A
from brass_ai.recovery import context_evaluation as L
from brass_ai.recovery.domino_config import NAMES
from brass_ai.recovery.domino_features import width,manifest

RUN='results/joint7-v11-2-grouped-lr-critic-g1330-20261031T024940047516Z'


def recipe():
    return read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-domino-entropy010-g1326.json')


def parent():
    return read(SOURCE_ROOT/RUN/'ratings/slots/generation-001330/spec.json')


def fixture_spec(setups=None):
    c=C.configure(recipe());p=parent()
    with patch.object(S,'runtime_receipt',return_value=dict(path='fake',runtime_sha256='r',host_sha256='h')):
        return S.specification_for(SOURCE_ROOT,p,c,test_setups=setups)


def fixture_rows(spec,identical=False):
    rows=[];md=manifest();w=width()
    for z in range(7):
        for j,s in enumerate(spec['selection']['setups']):
            for seat,index in enumerate(s['indices']):
                # Shared setup variation cancels from paired quality differences.
                d=0 if identical else z
                vector=np.zeros(4*w);vector[0]=j/100+seat/400;vector[5]=d/100
                vector[7]=d/40;vector[7+18]=d/50
                rows.append(dict(game_id=f'{z}:{j}:{seat}',panel_digest=digest(spec),parent_digest=spec['parent_digest'],
                    parent_index=index,context_id=z,context=NAMES[z],setup_seed=s['seed'],seat=seat,
                    win_share=.25+d*.02,vp=80+j+seat+d,placement=2.5-d*.1,outrank_share=.5+d*.01,
                    task_return=.4+j/100+d/100,insolvent=0,zero_vp=0,insolvency_events=0,passes=2,main_actions=10,
                    actions={'build':4,'loan':4,'pass':2},industry={'build':{'CottonMill':4},'develop':{}},
                    descriptor=vector.tolist(),phase_counts=[4,2,2,2],opponents=['neutral']*4,
                    replay={'path':'authored','sha256':'authored'},trace=[],
                    setup_id=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=s['seed']))))
    return rows


class ConfigurationSelection(unittest.TestCase):
    def test_absence_off_strict_config_and_fresh_only_opt_in(self):
        old=recipe();original=copy.deepcopy(old)
        self.assertFalse(C.enabled(old));new=C.configure(old)
        self.assertEqual({k for k in new if new[k]!=old.get(k)},{'context_evaluation'})
        self.assertEqual(C.validate(new['context_evaluation'],new),C.defaults());self.assertEqual(old,original)
        from brass_ai.recovery.config import validate_config
        validate_config(new)
        for change in ({'setups':8},{'seat_rotations':True},{'bootstrap_samples':2000.0},{'extra':0},
            {'overhead_review_fraction':float('nan')},{'reuse_main_neutral':False},{'mode':'apply'},{'every_iterations':30}):
            with self.subTest(change=change),self.assertRaises(ValueError):C.validate(dict(C.defaults(),**change),new)
        for update in ({'every_iterations':30},{'setups':8},{'enabled':False}):
            c=copy.deepcopy(new);c['rating'].update(update)
            with self.assertRaises(ValueError):C.validate(c['context_evaluation'],c)
        from brass_ai.recovery.domino_config import configure
        self.assertTrue(C.enabled(configure(old)))
        self.assertFalse(C.enabled(configure(old,dict(iteration=1330))))

    def test_selection_is_outcome_blind_16_strata_and_balanced_rotations(self):
        p=parent();before=copy.deepcopy(p);s=S.select(p)
        self.assertEqual(p,before);self.assertEqual(s,S.select(p))
        self.assertEqual(s['strata'],dict(il=2,v4=2,competitive=12));self.assertEqual(len(s['setups']),16)
        self.assertEqual(len({i for r in s['setups'] for i in r['indices']}),64)
        self.assertEqual(s['preferred_shortage'],[])
        from collections import Counter
        self.assertEqual(Counter(p['lineups'][i]['seat'] for r in s['setups'] for i in r['indices']),{0:16,1:16,2:16,3:16})
        p['outcomes']={'ignore':'a losing main slot'}
        self.assertEqual(s,S.select(p))

    def test_startup_fallback_and_same_model_not_duplicated_by_contexts(self):
        p=parent();p.pop('coverage');p['startup_reference_fallback']=True
        for i,r in enumerate(p['lineups']):r['agents'][1]='generation-0' if (i//4)%2==0 else 'v4'
        s=S.select(p);self.assertEqual(s['strata'],dict(il=8,v4=8));self.assertTrue(s['inherited_startup_fallback'])
        spec=fixture_spec();jobs=S.jobs_for(spec,Path('dummy'));self.assertEqual(len(jobs),384)
        self.assertEqual(len({j['stable_game_id'] for j in jobs}),384)
        self.assertEqual(len({j['policies'][0]['model_id'] for j in jobs}),1)
        self.assertEqual(len({j['policies'][0]['policy_id'] for j in jobs}),6)

    def test_exact_main_seed_rotation_opponent_policies_candidate_context_only(self):
        s=fixture_spec();base=ratings.jobs_for(s['parent'],Path('base'))
        for j in S.jobs_for(s,Path('panel')):
            i=j['provenance']['parent_index'];z=j['provenance']['context_id'];old=base[i]
            self.assertEqual(j['policies'][1:],old['policies'][1:])
            self.assertEqual(j['policies'][0]['focus'],NAMES[z]);self.assertEqual(j['policies'][0]['exploration'],0.)
            for k in ('seed','sampling_seed','rotation'):self.assertEqual(j[k],old[k])
            self.assertEqual(j['rng_index'],0);self.assertNotIn('branching',j);self.assertNotIn('opening',j['policies'][0])
            self.assertNotIn('diversity_assignment',j['provenance'])
            actual=[j['policies'][(seat-j['rotation'])%4] for seat in range(4)]
            self.assertEqual(actual[j['rotation']]['focus'],NAMES[z])
        from brass_ai.rust_backend.lab_collection import route,frozen_id
        pure=next(j for j in S.jobs_for(s,Path('x')) if all(p['kind']=='neural' for p in j['policies']))
        self.assertEqual(route(pure)[0],'rust');self.assertEqual(frozen_id(pure),frozen_id(dict(pure,path='elsewhere')))
        from brass_ai.recovery.execution import _collection_recipe
        _collection_recipe(S.jobs_for(s,Path('x')),s['execution'])


class Statistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=fixture_spec();cls.rows=fixture_rows(cls.spec);cls.report=A.analyze(cls.spec,cls.rows)

    def test_known_paired_effect_and_joint_mixture(self):
        r=self.report
        self.assertEqual(r['support']['total_games'],448);self.assertEqual(r['statistics']['independent_clusters'],16)
        for z,row in enumerate(r['contexts']):
            if z==0:self.assertIsNone(row['paired']);continue
            expected=z if z<7 else 3
            self.assertAlmostEqual(row['paired']['vp']['mean'],expected)
            np.testing.assert_allclose(row['paired']['vp']['ci95'],[expected,expected],atol=1e-10)
        self.assertIsNone(r['contexts'][-1]['counts']);self.assertIsNone(r['contexts'][-1]['games'])
        self.assertEqual(self.report,A.analyze(self.spec,list(reversed(self.rows))))

    def test_rotation_clusters_strata_and_determinism(self):
        labels=[s['stratum'] for s in self.spec['selection']['setups']]
        indices=A.bootstrap_indices(labels,100,42)
        from collections import Counter
        for row in indices:self.assertEqual(Counter(labels[i] for i in row),Counter(labels))
        np.testing.assert_array_equal(indices,A.bootstrap_indices(labels,100,42))
        self.assertTrue(any(len(set(r))<16 for r in indices))

    def test_identical_contexts_null_zero_rare_events_not_zero_risk(self):
        r=A.analyze(self.spec,fixture_rows(self.spec,True))
        for key in ('full','without_industry'):
            self.assertAlmostEqual(r['descriptors'][key]['separation'],0.)
            self.assertEqual(r['descriptors'][key]['null']['p_exploratory'],1.)
        self.assertTrue(any('zero risk' in w for w in r['warnings']))
        self.assertGreater(r['descriptors']['full']['mean_within_context_rms'],0)

    def test_ablation_removes_both_industry_families_without_rescaling(self):
        keep=A.projection_indices();self.assertEqual(len(keep),4*(width()-30))
        for phase in range(4):
            self.assertTrue(all(phase*width()+i not in keep for i in range(7,37)))
            self.assertIn(phase*width()+5,keep)
        a=np.asarray(self.report['descriptors']['full']['distance_matrix'])
        b=np.asarray(self.report['descriptors']['without_industry']['distance_matrix'])
        self.assertTrue((b<=a+1e-12).all());self.assertGreater(a[0,6],b[0,6])

    def test_incomplete_or_duplicate_support_rejected(self):
        for rows in (self.rows[:-1],self.rows+self.rows[:1]):
            with self.assertRaisesRegex(ValueError,'Incomplete'):A.analyze(self.spec,rows)


class ReplaySummary(unittest.TestCase):
    def make(self,z=2,rotation=3):
        spec=fixture_spec(1);jobs=S.jobs_for(spec,Path('raw'));j=next(j for j in jobs if j['provenance']['context_id']==z and j['rotation']==rotation)
        index=j['provenance']['parent_index'];records=[]
        from test_domino import descriptor_row
        for step in range(4):
            for seat in range(4):
                p=j['policies'][(seat-rotation)%4]
                if p['kind']!='neural':continue
                row=descriptor_row(seat,step);style,intent=context_for(p['focus'])
                row.update(step=len(records),style=style,intent=intent,teacher=None,
                    behavior=dict(kind='model',policy_id=p['policy_id'],probability=1.,temperature=1.))
                records.append(row)
        result=dict(seats=[dict(seat=s,vp=20,income=2,cash=5) for s in range(4)])
        # terminal_targets requires the declared official order and winners.
        result.update(ranking=[0,1,2,3],winners=[0,1,2,3])
        game=dict(records=records,result=result,game_id='authored',setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=j['seed'])),
            opponents=[j['policies'][(s-rotation)%4]['policy_id'] for s in range(4)])
        return spec,j,index,game

    def test_context_admission_and_missing_phases(self):
        spec,j,i,g=self.make()
        from brass_ai.learning.replay import terminal_targets
        targets=dict(vp=[20]*4,win_share=[.25]*4,placement=[1]*4)
        with patch('brass_ai.recovery.context_evaluation_collect.terminal_targets',return_value=targets),\
             patch('brass_ai.learning.reward.terminal_targets',return_value=targets):
            r=E.summary(g,j,spec,2,i,{'path':'fake','sha256':'x'})
            self.assertEqual(r['win_share'],.25);self.assertAlmostEqual(r['outrank_share'],.5)
            self.assertAlmostEqual(r['task_return'],.25+.1*.5+.001*20)
            self.assertEqual(r['phase_counts'],[1]*4);self.assertEqual(r['missing_phases'],[])
            for mut in (lambda g:g['records'][3].update(style=[1.]*4),
                lambda g:g.update(branches=[]),lambda g:g['records'][0]['behavior'].update(temperature=2.)):
                bad=copy.deepcopy(g);mut(bad)
                with self.assertRaises(ValueError):E.summary(bad,j,spec,2,i,{'path':'fake','sha256':'x'})
            g['records']=[r for r in g['records'] if r['view']['context']['era']=='CanalEra']
            r=E.summary(g,j,spec,2,i,{'path':'fake','sha256':'x'})
            self.assertEqual(r['missing_phases'],['rail_early','rail_late']);self.assertTrue(all(v==0 for v in r['descriptor'][2*width():]))

    def test_snapshot_context_and_reward_receipts_fail_closed(self):
        spec,j,i,g=self.make();targets=dict(vp=[20]*4,win_share=[.25]*4,placement=[1]*4)
        with patch('brass_ai.recovery.context_evaluation_collect.terminal_targets',return_value=targets),\
             patch('brass_ai.learning.reward.terminal_targets',return_value=targets):r=E.summary(g,j,spec,2,i,{'path':'raw','sha256':'x'})
        for key,value in [('runtime',{}),('implementation','x'),('parent_digest','x'),('sampling_seed',99),('candidate',{}),('opponents',[])]:
            with self.subTest(key=key),self.assertRaises(ValueError):E.verify_summary(dict(r,**{key:value}),j,spec,2,i)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp);E.save_summary(path,r)
            restored=E.load_summaries(path,spec);self.assertEqual(restored,[r])
            (path/'evidence'/f'context-2-parent-{i:06d}.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'receipt'):E.load_summaries(path,spec)

    def test_development_continuation_counts_industry_but_not_another_action(self):
        spec,j,i,g=self.make();seat=j['rotation']
        main=next(r for r in g['records'] if r['seat']==seat)
        main['choices'][main['chosen']].update(kind='develop',industry='Manufacturer')
        other=copy.deepcopy(main);other['view']['context']['phase']='continuation'
        other['choices'][other['chosen']]['industry']='Brewery';g['records'].append(other)
        targets=dict(vp=[20]*4,win_share=[.25]*4,placement=[1]*4)
        with patch('brass_ai.recovery.context_evaluation_collect.terminal_targets',return_value=targets),\
             patch('brass_ai.learning.reward.terminal_targets',return_value=targets):
            r=E.summary(g,j,spec,2,i,{'path':'fake','sha256':'x'})
        self.assertEqual(r['industry']['develop'],dict(Manufacturer=1,Brewery=1))
        self.assertEqual(r['actions']['develop'],1);self.assertEqual(r['main_actions'],4)


class Lifecycle(unittest.TestCase):
    def test_activation_restart_off_and_rng_error_isolation(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);state=dict(iteration=1330,weights='unchanged',optimizer='unchanged');c=C.configure(recipe())
            L.prepare(SOURCE_ROOT,run,state,c);self.assertEqual(state['context_evaluation']['next_due_generation'],1380)
            before=copy.deepcopy(state);state['iteration']=1339;L.prepare(SOURCE_ROOT,run,state,c)
            self.assertEqual(state['context_evaluation'],before['context_evaluation'])
            state=read(run/'state.json');self.assertEqual(state['context_evaluation']['activation_generation'],1330)
            fresh=dict(iteration=0);L.prepare(SOURCE_ROOT,run,fresh,c);self.assertEqual(fresh['context_evaluation']['next_due_generation'],50)
            off=dict(iteration=100);self.assertIsNone(L.prepare(SOURCE_ROOT,run,off,recipe()));self.assertEqual(off,dict(iteration=100))
        p=random.getstate();n=np.random.get_state();t=torch.get_rng_state().clone();model=torch.nn.Linear(2,2);t=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with L.isolated_rng(model):random.random();np.random.random();torch.rand(2);raise RuntimeError('expected')
        self.assertEqual(p,random.getstate());np.testing.assert_equal(n,np.random.get_state());self.assertTrue(torch.equal(t,torch.get_rng_state()));self.assertTrue(model.training)

    def test_evaluation_only_compatibility_does_not_allow_recipe_changes(self):
        old=recipe();new=C.configure(old);self.assertTrue(C.compatible(old,new))
        for k in ('exploration','learning_rate'):
            bad=copy.deepcopy(new);bad[k]*=2;self.assertFalse(C.compatible(old,bad))

    def test_missing_neutral_never_launches_replacement_games(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();c=C.configure(recipe());state=dict(iteration=1280,ratings={'slots':[]})
            L.prepare(root,run,state,c);state['iteration']=1330;p=parent()
            path=run/'main/spec.json';ratings.write_once(path,p)
            with patch.object(L,'specification_for',return_value=fixture_spec()):L.seal(root,run,state,c,p,path)
            with patch.object(L,'validate_runtime'),patch.object(L,'collect') as collector:
                result=L.run_pending(root,run,state,c)
            self.assertNotEqual(result['status'],'complete');collector.assert_not_called();self.assertEqual(state['iteration'],1330)


if __name__=='__main__':unittest.main()
