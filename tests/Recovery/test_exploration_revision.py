import copy
import json
import random
import unittest
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.step5.policy import exploration_distribution,INDUSTRIES,NeuralPolicy
from brass_ai.recovery import exploration as E
from brass_ai.recovery.branching import alternative,eligible,seed_for,family
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.execution import neural_spec,_collection_recipe
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.updates import entropy_term,row_probabilities,verify_behavior_metadata


def config(staged=False):
    c=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-ppg-window.json')
    return E.configure(c,staged)


def measured(value):
    return dict(groups={g:dict(raw_entropy=value,normalized_entropy=value/4) for g in E.GROUPS})


class DistributionTests(unittest.TestCase):
    def choices(self):
        return [dict(kind=k) for k in E.Q['action_family_weights']]+[dict(kind='pass')]

    def test_exact_family_shares_and_pass(self):
        q=exploration_distribution(self.choices(),[0]*8,3,'action')
        np.testing.assert_allclose(q,[4/17,4/17,4/17,2/17,2/17,1/17,0],rtol=0,atol=1e-16)
        np.testing.assert_equal(exploration_distribution([dict(kind='pass')],None,3,'action'),[1])

    def test_industry_focus_stays_within_family(self):
        choices=[dict(kind='build',industry='Pottery')]*5+[dict(kind='build',industry='CoalMine')]+[dict(kind='loan')]
        intent=[float(i=='Pottery') for i in INDUSTRIES]+[0,0]
        q=exploration_distribution(choices,intent,3,'action')
        self.assertAlmostEqual(q[:6].sum(),4/6);self.assertAlmostEqual(q[:5].sum(),2*q[5])
        self.assertAlmostEqual(q.sum(),1);self.assertTrue(np.all(q>0))
        for phase in ('debt','continuation','bonus','payment'):
            np.testing.assert_array_equal(exploration_distribution(choices,intent,3,phase),exploration_distribution(choices,intent))

    def test_missing_and_unknown_families(self):
        q=exploration_distribution([dict(kind='network'),dict(kind='new_action'),dict(kind='pass')],None,3,'action')
        np.testing.assert_allclose(q,[.8,.2,0])
        with self.assertRaises(ValueError):exploration_distribution(self.choices(),version=3)
        with self.assertRaises(ValueError):exploration_distribution(self.choices(),version=4,phase='action')

    def test_legal_mask_and_raw_entropy_gradient(self):
        logits=torch.tensor([[.8,-.4,float('-inf')],[.3,.1,-.2]],requires_grad=True)
        q=torch.tensor([[.6,.4,0],[.1,.2,.7]])
        mixed=.97*torch.softmax(logits/1.5,-1)+.03*q
        ds=SimpleNamespace(rows=[dict(view=dict(context=dict(era='CanalEra',phase='action'))),
                                      dict(view=dict(context=dict(era='RailEra',phase='bonus')))])
        behavior=SimpleNamespace(raw_entropy=True,entropy_alphas=dict.fromkeys(E.GROUPS,.005))
        behavior.entropy_alphas['rail_other']=.05
        h,loss=entropy_term(logits,mixed,ds,[0,1],behavior,dict(entropy_weight=.005))
        raw=logits.softmax(-1);expected=-(raw*raw.clamp_min(1e-30).log()).sum(-1)
        torch.testing.assert_close(h,expected)
        torch.testing.assert_close(loss,expected*torch.tensor([.005,.05]))
        loss.sum().backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertEqual(float(logits.grad[0,2]),0)


class ControllerTests(unittest.TestCase):
    def test_strict_configuration_and_fresh_defaults(self):
        c=config();validate_config(c)
        new=latest_model_config(c,c['imitation_checkpoint']);self.assertEqual(new['exploration'],.02)
        for key in E.BLOCKS:self.assertEqual(new[key],c[key])
        for key in E.BLOCKS:
            bad=copy.deepcopy(c);bad[key]['surprise']=1
            with self.assertRaises(ValueError):validate_config(bad)

    def test_stages_and_restart_origin(self):
        c=config(True);s=E.initial(c,185)
        self.assertIsNone(E.effective(c,s,185)['opening'])
        self.assertIsNotNone(E.effective(c,s,186)['opening'])
        self.assertFalse(E.effective(c,s,186)['raw_entropy'])
        self.assertTrue(E.effective(c,s,187)['raw_entropy'])
        self.assertIsNone(E.effective(c,s,204)['branching'])
        self.assertIsNone(E.effective(c,s,205)['branching'])
        s['baseline']=dict.fromkeys(E.GROUPS,2.)
        self.assertIsNotNone(E.effective(c,s,205)['branching'])
        s=E.initial(c,185)
        s['generation']=193;state=dict(iteration=193,exploration=s)
        self.assertEqual(E.restore(c,state,copy.deepcopy(s)),s)
        with self.assertRaises(ValueError):E.restore(c,state,dict(s,origin=0))

    def arm(self,c=None):
        c=c or config();s=E.initial(c,0)
        for i in range(1,11):
            s,r=E.advance(s,c,measured(2),generation=i,warmup_done=True,healthy=True)
        return c,s

    def test_warmup_baseline_and_no_rng(self):
        c=config();s=E.initial(c,0);rng=torch.get_rng_state()
        s,r=E.advance(s,c,measured(2),generation=1,warmup_done=False,healthy=True)
        self.assertEqual(s['baseline_rows'],[]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for i in range(2,12):s,r=E.advance(s,c,measured(2),generation=i,warmup_done=True,healthy=True)
        self.assertEqual(s['armed_at'],11);self.assertEqual(s['baseline'],dict.fromkeys(E.GROUPS,2))

    def test_raise_relax_cap_hold_decay_and_exact_restart(self):
        c,s=self.arm()
        for i in range(11,31):
            restored=json.loads(json.dumps(s))
            a,r=E.advance(s,c,measured(.5),generation=i,warmup_done=True,healthy=True)
            b,_=E.advance(restored,c,measured(.5),generation=i,warmup_done=True,healthy=True)
            self.assertEqual(a,b);s=a
        self.assertEqual(s['groups']['canal_action']['alpha'],.05)
        self.assertIn('alpha_at_cap_10_iterations',r['groups']['canal_action']['alarms'])
        held,r=E.advance(s,c,measured(4),generation=31,warmup_done=True,healthy=False)
        self.assertEqual(held['groups']['canal_action']['alpha'],.05)
        for i in range(32,48):held,r=E.advance(held,c,measured(4),generation=i,warmup_done=True,healthy=True)
        self.assertEqual(held['groups']['canal_action']['alpha'],.005)
        self.assertIn('entropy_above_1.3_baseline',r['groups']['canal_action']['alarms'])
        held['generation']=999
        held,r=E.advance(held,c,measured(1.21),generation=1000,warmup_done=True,healthy=True)
        self.assertEqual(r['groups']['canal_action']['floor'],1.9)

    def test_hysteresis_and_shadow_do_not_apply_alpha(self):
        c,s=self.arm()
        for i in range(11,14):s,r=E.advance(s,c,measured(1.92),generation=i,warmup_done=True,healthy=True)
        self.assertEqual(r['groups']['canal_action']['reason'],'hysteresis')
        c['entropy_control']['mode']='shadow'
        for i in range(14,25):s,r=E.advance(s,c,measured(.1),generation=i,warmup_done=True,healthy=True)
        self.assertEqual(E.effective(c,s,25)['alphas'],dict.fromkeys(E.GROUPS,.005))
        self.assertGreater(s['groups']['canal_action']['alpha'],.005)

    def test_saved_evidence_is_strict_and_rollout_is_not_restarted(self):
        c,s=self.arm()
        self.assertEqual(E.continuation_config(c),c)
        state=dict(iteration=10,exploration=s)
        self.assertEqual(E.restore(c,state,copy.deepcopy(s)),s)
        for field in ('alpha','window','cap_streak'):
            bad=copy.deepcopy(s)
            bad['groups']['canal_action'][field]={'alpha':.1,'window':[float('nan')],'cap_streak':-1}[field]
            with self.assertRaises(ValueError):E.restore(c,dict(state,exploration=bad),copy.deepcopy(bad))
        bad=copy.deepcopy(s);bad['baseline']['canal_action']=3.
        with self.assertRaises(ValueError):E.restore(c,dict(state,exploration=bad),copy.deepcopy(bad))

    def test_lr_streak_resets_for_stage_change_not_random_branch_counts(self):
        from brass_ai.recovery.lr_controller import collection_signature
        c=config(True);s=E.initial(c,185)
        sampling=dict(by_composition=dict(selfplay=dict(tables=160,learner_trajectories=640)))
        def key(age,branch_count):
            effective=E.effective(c,s,185+age)
            return collection_signature(c,dict(sampling,branch_count=branch_count,
                exploration_regime={k:effective[k] for k in ('version','opening','raw_entropy','branching')}))
        self.assertEqual(key(0,1),key(0,80))
        self.assertNotEqual(key(0,0),key(1,0))
        s['baseline']=dict.fromkeys(E.GROUPS,2.)
        self.assertNotEqual(key(19,0),key(20,0))


class BehaviorTests(unittest.TestCase):
    def policy(self):
        return NeuralPolicy(None,exploration=.03,focus='neutral',evaluator=SimpleNamespace(model_id='model'),
            exploration_version=3,opening=E.OPENING)

    def test_openings_independent_seats_and_subdecisions(self):
        p=self.policy();rngs=[random.Random(i) for i in range(4)];choices=[dict(kind='loan'),dict(kind='pass')]
        records=[]
        for phase in ('action','continuation','action','action','debt'):
            for seat,rng in enumerate(rngs):
                view=dict(context=dict(phase=phase))
                p.sample_training(view,choices,rng,np.array([.7,.3]),[0]*8)
                records.append((seat,phase,dict(p.last_behavior[rng])))
        for seat in range(4):
            rows=[b for s,phase,b in records if s==seat]
            self.assertEqual([b['temperature'] for b in rows],[2.,1.,2.,1.,1.])
            self.assertEqual([b['action_decision'] for b in rows],[0,None,1,2,None])
        spec=neural_spec('model',.03,'neutral',version=3,opening=E.OPENING)
        self.assertEqual(p.policy_id,spec['policy_id'])
        self.assertNotEqual(spec['policy_id'],neural_spec('model',.03,'neutral')['policy_id'])

    def test_probability_reconstruction_and_metadata_rejection(self):
        p=self.policy();rng=random.Random(98);choices=[dict(kind='build',industry='CoalMine'),dict(kind='pass')]
        logits=torch.tensor([[.9,-.3]])
        chosen,prob,_=p.sample_training(dict(context=dict(phase='action')),choices,rng,logits.softmax(-1)[0].numpy(),[0]*8)
        b=dict(p.last_behavior[rng],probability=prob,kind='model',policy_id=p.policy_id)
        row=dict(behavior=b,view=dict(context=dict(phase='action')))
        ds=SimpleNamespace(rows=[row],temperatures=torch.tensor([[2.]]))
        behavior=SimpleNamespace(exploration=.03,temperature=1.,exploration_version=3,opening=E.OPENING)
        q=torch.tensor(np.array([exploration_distribution(choices,[0]*8,3,'action')]),dtype=torch.float32)
        actual=row_probabilities(ds,[0],logits,q,behavior)
        self.assertAlmostEqual(prob,float(actual[0,chosen]),places=6);verify_behavior_metadata(ds,behavior)
        b['exploration_version']=2
        with self.assertRaises(ValueError):verify_behavior_metadata(ds,behavior)
        b['exploration_version']=3;b['temperature']=1.
        with self.assertRaises(ValueError):verify_behavior_metadata(ds,behavior)

    def test_full_plan_all_neural_seats_and_cached_recipe_identity(self):
        from brass_ai.bank import sha
        from brass_ai.recovery.ordinary import plan_collection
        from brass_ai.recovery import quality_pool
        from test_pfsp_pool import pool
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            root=Path(temporary);path=root/'weights.pt';path.write_bytes(b'planning fixture')
            c=config(True);value=pool(4,generation=185)
            fp=dict(values=[1/8]*16+[1/6]*12)
            for m in value['members']:
                m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),fingerprint=fp)
            state=dict(iteration=185,games_done=59200,next_seed=c['seed'],context_counter=0,
                quality_pool=value,population=[quality_pool.reference(m) for m in value['members']],
                history=[quality_pool.reference(value['members'][0])],
                evaluation_anchor=quality_pool.reference(value['members'][0]))
            service=Mock();service.models={m['model_id']:None for m in value['members']}
            profiles=[f'v4-{i}' for i in range(5)]
            bots={p:dict(kind='legacy',policy_id=p,search_free=True) for p in profiles}
            rollout=E.initial(c,185)
            def plan(settings):
                return plan_collection(SOURCE_ROOT,root,state,c,dict(config=c),service,'current',
                    profiles,bots,bots,None,fingerprints=dict(learner=fp,rule_bot=fp),exploration=settings)
            first=plan(E.effective(c,rollout,185));opened=plan(E.effective(c,rollout,186))
            self.assertEqual(first['composition'],dict(selfplay=160,historical=144,legacy=16))
            self.assertEqual(opened,plan(E.effective(c,json.loads(json.dumps(rollout)),186)))
            self.assertNotEqual(_collection_recipe(first['jobs'],c['execution']),_collection_recipe(opened['jobs'],c['execution']))
            # A previous-stage shard renamed to the new cache prefix still
            # fails admission before any actor starts.
            import gzip
            from brass_ai.recovery.execution import ActorPool
            old_recipe=_collection_recipe(first['jobs'][:1],c['execution'])
            new_recipe=_collection_recipe(opened['jobs'][:1],c['execution'])
            path=root/f'games-{digest(new_recipe)[:16]}-part-000000.jsonl.gz'
            with gzip.open(path,'wt',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(provenance=dict(collection=digest(old_recipe),recipe=old_recipe)))+'\n')
            actor_pool=ActorPool(SOURCE_ROOT,c['host'],service,c['execution'])
            try:
                with patch.object(actor_pool,'start',side_effect=AssertionError('Actor started on stale cache')):
                    with self.assertRaisesRegex(ValueError,'Cached collection differs'):
                        actor_pool._collect(opened['jobs'][:1])
            finally:actor_pool.close()
            for job in opened['jobs']:
                for p in job['policies']:
                    if p['kind']=='neural':
                        self.assertEqual(p['exploration'],.02);self.assertEqual(p['exploration_version'],3)
                        self.assertEqual(p['opening'],E.OPENING)
                    else:self.assertEqual(p,bots[p['policy_id']])
                self.assertNotIn('branching',job)
            rollout['baseline']=dict.fromkeys(E.GROUPS,2.)
            enabled=plan(E.effective(c,rollout,205))
            self.assertEqual(sum('branching' in j for j in enabled['jobs']),160)
            self.assertTrue(all(j['provenance']['kind']=='selfplay' for j in enabled['jobs'] if 'branching' in j))

    def test_branch_move_can_never_be_admitted_as_on_policy(self):
        b=dict(kind='model',policy_id='model',probability=.5,temperature=1.,
            exploration=.03,exploration_version=3,component='branch_move',action_decision=4)
        E.check_behavior(b,'action',branch_move=True)
        with self.assertRaises(ValueError):E.check_behavior(b,'action')

    def test_pass_only_multiple_card_choices_have_explicit_metrics(self):
        from brass_ai.recovery.exploration_metrics import collect,merge
        row=dict(seat=0,view=dict(context=dict(phase='action')),chosen=0,
            choices=[dict(kind='pass',id='discard-a'),dict(kind='pass',id='discard-b')],
            behavior=dict(exploration_version=3,component='exploration',temperature=1.,action_decision=5))
        report=merge([collect([dict(records=[row])])])
        self.assertEqual(report['pass_only_exploration_picks'],1)
        self.assertEqual(report['pass_with_alternative_exploration_picks'],0)


class BranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_move_distribution_probability_and_seed(self):
        raw=np.array([.7,.2,.1]);q=np.array([.6,.4,0])
        for i in range(100):
            a=alternative(raw,q,0,random.Random(i));b=alternative(raw,q,0,random.Random(i))
            self.assertEqual(a,b);self.assertNotEqual(a['chosen'],0)
            weights={'temperature_1':raw,'temperature_2':np.sqrt(raw),'exploration_q':q}
            values=weights[a['component']].copy();values[0]=0;values/=values.sum()
            self.assertAlmostEqual(a['probability'],values[a['chosen']])
        # No q support after exclusion: discard that component, not the branch.
        a=alternative(np.array([.9,.1]),np.array([1.,0.]),0,random.Random(5))
        self.assertEqual(a['chosen'],1);self.assertEqual(a['marginal_probability'],1.)
        self.assertEqual(seed_for('parent',33),seed_for('parent',33))
        self.assertNotEqual(seed_for('parent',33),seed_for('parent',34))

    def test_eligibility(self):
        view=dict(context=dict(phase='action',era='RailEra'),deck_count=0,own_hand=[1]*4)
        choices=[dict(kind='loan'),dict(kind='pass')]
        self.assertFalse(eligible(view,choices,2));view['own_hand'].append(1)
        self.assertTrue(eligible(view,choices,2));self.assertFalse(eligible(view,choices,1))
        view['context']['phase']='continuation';self.assertFalse(eligible(view,choices,3))

    def test_parent_holdout_inherited_and_no_population_skew(self):
        from brass_ai.recovery.critic_holdout import select,DEFAULT,validate_selection
        games={f'g{i}':'selfplay' for i in range(20)}
        base=select(games,.05,DEFAULT)
        for g in list(games):games[g+'/branch/seed']='selfplay'
        selection=select(games,.05,DEFAULT);validate_selection(selection)
        for g in games:self.assertEqual(g in selection['heldout_games'],family(g) in base['heldout_games'])
        self.assertEqual(set(family(g) for g in selection['heldout_games']),set(base['heldout_games']))

    def test_branch_cache_preserves_family_and_ev_gate_excludes_branch_evidence(self):
        from test_gae_value_phase import fixture
        from test_value_buffer import source
        from brass_ai.recovery import value_phase,value_buffer,updates
        from brass_ai.step5.league import write_json
        model,_,cfg,(behavior,data)=fixture(games=4)
        cfg=value_phase.refined_options(cfg);cfg['value_phase']['buffer_iterations']=2
        children=[(dict(row,_branch=True),target,(identity[0]+'/branch/seed',identity[1]),encoded)
            for row,target,identity,encoded in data]
        all_rows=data+children;ds=updates.Dataset(all_rows,'cpu')
        train,held,split=value_phase.split(all_rows,.05,cfg['value_phase']['holdout'])
        self.assertEqual(split['heldout_games'],2*split['heldout_parent_games'])
        evidence=value_phase.metrics(model,ds,held,cfg['update'],evidence=True)
        ordinary=evidence['groups']['source:ordinary'];branch=evidence['groups']['source:branch']
        self.assertEqual(ordinary['rows'],branch['rows'])
        self.assertEqual(evidence['groups']['table:selfplay/role:learner']['games'],ordinary['games'])
        self.assertEqual(evidence['explained_variance'],ordinary['explained_variance'])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            run=Path(temporary);buf=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=0)
            buf.stage(ds,train,source(behavior),model);receipt=copy.deepcopy(buf.pending)
            write_json(run/'state.json',dict(iteration=1));buf.committed(1)
            restored=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=1,receipt=receipt)
            with restored.previous() as disk:
                self.assertEqual([disk[i][2] for i in range(len(disk))],[all_rows[i][2] for i in train])
                self.assertTrue(any('/branch/' in g for g in receipt['game_ids']))
                self.assertFalse(set(receipt['game_ids']) & set(split['selection']['heldout_games']))

    def test_temperature_check_is_paired_and_evaluation_only(self):
        from brass_ai.recovery.temperature_rating import specification
        from brass_ai.recovery.ratings import jobs_for
        from pathlib import Path
        base=dict(candidate='g200',agents={'g200':dict(kind='model',checkpoint=dict(model_id='m')),
            'il':dict(kind='model',checkpoint=dict(model_id='il'))},
            lineups=[dict(seed=22,seat=2,kind='rating',agents=['g200','il','il','il'])])
        spec=specification(base);jobs=jobs_for(spec,Path('unused'))
        self.assertEqual(jobs[0]['policies'][0]['temperature'],.7)
        self.assertEqual(jobs[0]['policies'][0]['exploration'],0.)
        self.assertNotIn('opening',jobs[0]['policies'][0])
        self.assertEqual(jobs[0]['provenance']['purpose'],'evaluation')
        self.assertEqual(base['lineups'][0]['agents'][0],'g200')

    def test_temperature_schedule_restart_and_retention_are_separate(self):
        from test_rating_retention import RatingRetention
        from brass_ai.recovery import temperature_rating,ratings,retention
        from brass_ai.bank import sha
        case=RatingRetention(methodName='runTest');case.setUp()
        self.addCleanup(case.doCleanups)
        case.through(50)
        before=copy.deepcopy(case.state['ratings'])
        # Archived signed recipes remain reproducible; fresh defaults disable it.
        case.cfg['entropy_control']=copy.deepcopy(E.LEGACY_ENTROPY)
        case.state['exploration']=dict(origin=0)
        report=temperature_rating.schedule(case.root,case.run,case.state,case.cfg,device='cpu')
        self.assertEqual(case.state['ratings'],before)
        self.assertIsNone(report['deployment_qualified'])
        self.assertEqual(report['generation'],50)
        reference=case.state['temperature_ratings'][0]['report']
        self.assertEqual(sha(case.root/reference['path']),reference['sha256'])
        pins=retention.references(case.root,case.run)
        for item in (reference,report['spec'],report['records']):
            self.assertIn((case.root/item['path']).resolve(),pins)
        restored=read(case.run/'state.json')
        with patch.object(ratings,'collect_slot',side_effect=AssertionError('Must not recollect')):
            self.assertIsNone(temperature_rating.schedule(case.root,case.run,restored,case.cfg,device='cpu'))


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_g185_review_changes_only_requested_blocks(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.recovery.gae import health_options
        from brass_ai.recovery.value_phase import interval_options
        from brass_ai.step5.league import implementation_id
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0185-20261020T090519451283Z/state.json')
        previous=copy.deepcopy(state)
        new=E.continuation_config(interval_options(health_options(state['signature']['config'])))
        target=dict(state['signature'],config=new,implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        self.assertEqual(state,previous)
        self.assertEqual(E.restore(new,state,None)['origin'],185)
        self.assertEqual(set(new)-set(state['signature']['config']),set(E.BLOCKS))
        with self.assertRaises(ValueError):check_signature(state,target)
        for key in ('learning_rate','exploration'):
            bad=copy.deepcopy(target);bad['config'][key]*=1.1
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)

    def test_adoption_preserves_both_optimizers_rng_and_checkpoint_state(self):
        from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.retention import _same_payload
        from brass_ai.recovery import gae
        from test_gae_value_phase import fixture
        from test_ppg_window import upgraded
        model,adam,old,_=fixture();old=upgraded(old)
        phase=TrainingPhases(model,old)
        p=next(model.parameters());p.grad=torch.ones_like(p)
        phase.optimizer.step();phase.optimizer.zero_grad(set_to_none=True)
        phase.state=gae.initial(old['advantage'],185);phase.critic_state['generation']=185
        from brass_ai.recovery import snapshot_r2
        phase.critic_state['weight_recovery']=snapshot_r2.initial(185,phase.critic_state['snapshot_weight'])
        new=E.continuation_config(old)
        state=dict(iteration=185,games_done=59200,next_seed=123,context_counter=177600,
            population=['kept'],history=['kept'],origin=dict(imitation='kept'))
        before=copy.deepcopy(state);weights=copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            path=Path(temporary)/'generation-0185.pt'
            save_checkpoint(path,model,adam,step=185,training={KEY:phase.payload()})
            net,raw=load_checkpoint(path);restored=torch.optim.AdamW(net.parameters())
            resume_optimizer(raw,restored);rng=torch.get_rng_state().clone()
            phases=TrainingPhases(net,new,raw=raw,generation=185)
            rollout=E.restore(new,state,raw['training'].get('exploration'))
            self.assertEqual(rollout['origin'],185);self.assertEqual(state,before)
            self.assertTrue(_same_payload(dict(net.state_dict()),dict(weights)))
            self.assertTrue(_same_payload(adam.state_dict(),restored.state_dict()))
            self.assertTrue(_same_payload(phase.payload(),phases.payload()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for generation in range(186,190):
                rollout,_=E.advance(rollout,new,measured(2.),generation=generation,warmup_done=True,healthy=True)
            state.update(iteration=189,exploration=rollout)
            self.assertEqual(E.restore(new,state,json.loads(json.dumps(rollout))),rollout)


if __name__=='__main__':unittest.main()
