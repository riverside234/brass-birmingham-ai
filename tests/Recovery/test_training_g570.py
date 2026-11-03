"""g570 continuation, branch behavior, v4 PFSP and complete frozen snapshots."""
import copy
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import exploration as E,opening_objective as O,branching as B,updates
from brass_ai.recovery import quality_pool as Q,pfsp_pool as P,v4_history as H,rating_coverage as R,ratings,gae
from brass_ai.recovery.training_revision_g570 import configure
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload,slim_checkpoint,preserve_available_companions
from brass_ai.step5.league import write_json,implementation_id

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z'


def bots(state):
    return {p:dict(kind='lab-legacy',parent=p,policy_id='fast-'+str(i),search_free=True)
        for i,p in enumerate(sorted(state['signature']['legacy_pool']))}


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.state=read(BUNDLE/'state.json');cls.old=cls.state['signature']['config'];cls.new=configure(cls.old)

    def test_config_precise_scope_and_strict_versions(self):
        validate_config(self.new);self.assertEqual(configure(self.new),self.new)
        changed={k for k in self.new if self.new[k]!=self.old.get(k)}
        self.assertEqual(changed,{'branching','exploration','update','rating_coverage','v4_history'})
        self.assertEqual(self.new['opening'],E.THREE_OPENING)
        self.assertEqual(self.new['branching']['continuation_opening'],E.FORK_OPENING)
        self.assertEqual(self.new['update']['fixed_entropy']['weights'],dict.fromkeys(E.GROUPS,.005))
        for key in ('value_phase','advantage','insolvency','ppo_lr_controller','learning_rate_warmup','local_scoring_auxiliary'):
            self.assertEqual(self.new[key],self.old[key])
        for section,key,value in [('branching','skip_opening_decisions',2),('branching','rate_per_decision',.01),
            ('v4_history','window_iterations',9),('update','entropy_weight',.0025)]:
            c=copy.deepcopy(self.new);c[section][key]=value
            with self.assertRaises(ValueError):validate_config(c)
        auto=configure(self.old,manual_generation=None);validate_config(auto)

    def test_branch_sampling_and_restore_parent_opening(self):
        from brass_ai.step5.policy import NeuralPolicy
        view=dict(context=dict(phase='action',era='CanalEra'),deck_count=1,own_hand=[1])
        self.assertEqual([B.eligible(view,[{},{}],i,3) for i in range(4)],[False,False,False,True])
        p=NeuralPolicy(None,exploration=.02,focus='neutral',evaluator=SimpleNamespace(model_id='m'),
            exploration_version=3,opening=E.THREE_OPENING)
        rng=random.Random(2);temperatures=[]
        for _ in range(4):
            p.sample_training(view,[dict(kind='loan'),dict(kind='pass')],rng,np.array([.7,.3]),[0.]*8)
            temperatures.append(p.last_behavior[rng]['temperature'])
        self.assertEqual(temperatures,[1.3,1.3,1.3,1.])
        row=dict(step=25,seat=2,view=view,choices=[dict(id=0),dict(id=1)],chosen=0,style=[0.]*4,intent=[0.]*8,teacher=None,seconds=0.,
            behavior=dict(probability=.5,component='policy',temperature=1.,action_decision=3))
        meta=dict(contract=E.FORK_T15_BRANCH['contract'],seed='12345678',decision=25,chosen=1,marginal_probability=.3)
        packet=dict(row=row,metadata=meta,counts=[3,4,4,2],contexts=[{} for _ in range(4)])
        assigned=[NeuralPolicy(None,exploration=.02,focus='neutral',evaluator=SimpleNamespace(model_id='m'),
            exploration_version=3,opening=E.THREE_OPENING) for _ in range(4)]
        host=Mock();host.call.return_value=view['context'];host.apply.return_value={'terminal':True}
        parent=dict(game_id='p',setup_group='s',split='train',opponents=['m']*4)
        def turns(host,job,policies,timing,cap,context,game,rngs,encoded,encoder,lease):
            for seat,policy in enumerate(policies):
                self.assertEqual(policy.action_counts[rngs[seat]],0)
                ts=[]
                for _ in range(3):
                    policy.sample_training(view,[dict(kind='loan'),dict(kind='pass')],rngs[seat],np.array([.7,.3]),[0.]*8)
                    ts.append(policy.last_behavior[rngs[seat]]['temperature'])
                self.assertEqual(ts,[1.5,1.5,1.])
            return game
        with patch('brass_ai.recovery.execution._run_turns',side_effect=turns):
            B.continue_branch(host,dict(branching=E.FORK_T15_BRANCH,provenance={}),parent,packet,assigned,Mock(),20,Mock(),Mock())
        self.assertTrue(all(p.opening==E.THREE_OPENING for p in assigned))
        for seed in range(20):
            sample=B.alternative(np.array([.5,.4,.1]),np.array([1.,0.,0.]),0,random.Random(seed),contract=E.FORK_T15_BRANCH['contract'])
            self.assertEqual(sample['component'],'temperature_2');self.assertEqual(sample['component_probability'],1.)

    def test_real_behavior_reconstruction_decoupled_branch_PPO_and_KL(self):
        from test_gae_value_phase import fixture
        from test_opening_ppo import opening_rows
        from brass_ai.recovery.telemetry import Timings
        from brass_ai.recovery.review_metrics import CategoricalProbe
        model,_,c,(_,data)=fixture(games=3,decisions=4)
        parent=[d for d in data if d[2]==data[1][2]]
        branch=[(dict(r,_branch=True),t,(g+'/branch/test',s),e) for r,t,(g,s),e in parent]
        behavior,normal=opening_rows(model,parent,.02,opening=E.THREE_OPENING,objective=O.FORK_T15)
        _,branched=opening_rows(model,branch,.02,opening=E.FORK_OPENING,objective=O.FORK_T15)
        for row,*_ in branched:row['behavior']['policy_id']=behavior.policy_id
        ds=updates.Dataset(normal+branched,'cpu');ds.opening_reference=O.Reference(ds,behavior,O.FORK_T15)
        self.assertEqual(ds.opening_reference.mask.tolist(),[True,True,True,False,True,True,False,False])
        updates.policy_measure(model,ds,range(len(ds.rows)),behavior,c['update'],Timings(),verify=True)
        self.assertTrue(ds.opening_reference.frozen)
        probe=CategoricalProbe(model,ds,list(range(len(ds.rows))),behavior,dict(c['update'],opening_objective=O.FORK_T15))
        self.assertEqual(probe.opening_positions,5)
        self.assertAlmostEqual(probe.measure(model)['categorical_kl'],0.,places=6)
        bad=copy.deepcopy(normal+branched);bad[6][0]['behavior']['temperature']=1.5
        with self.assertRaises(ValueError):updates.verify_behavior_metadata(updates.Dataset(bad,'cpu'),behavior)
        old=copy.deepcopy(branched);old[0][0][O.ROW_KEY]=O.T13
        with self.assertRaises(ValueError):O.Reference(updates.Dataset(old,'cpu'),behavior,O.FORK_T15)
        with self.assertRaises(ValueError):updates.verify_behavior_metadata(ds,SimpleNamespace(exploration_version=3,opening=E.THREE_OPENING,temperature=1.,exploration=.05))

    def test_manual_v4_probability_and_exact_restart_receipt(self):
        old=self.state['quality_pool'];before=copy.deepcopy(old)
        migrated,receipt=H.migrate(old,self.new,bots(self.state))
        self.assertEqual(old,before);P.validate_state(migrated,migrated['config'],570)
        plan=P.distribution(migrated);member=migrated['members'][-1]
        self.assertEqual(next(r['p'] for r in plan['members'] if r['model_id']==member['model_id']),plan['il_probability'])
        self.assertEqual((member['S'],member['C']),(old['members'][0]['S'],old['members'][0]['C']))
        self.assertIsNone(migrated[H.KEY]['activated']['measured_win_share'])
        again,empty=H.migrate(json.loads(json.dumps(migrated)),self.new,bots(self.state))
        self.assertEqual(again,migrated);self.assertIsNone(empty)
        self.assertEqual(receipt['from_digest'],Q.state_digest(old));self.assertEqual(receipt['to_digest'],Q.state_digest(again))
        for key in ('health_window','planning','events','games_done','generation'):self.assertEqual(migrated[key],old[key])
        bad=copy.deepcopy(migrated);bad['members'][-1]['policies'].pop(next(iter(bad['members'][-1]['policies'])))
        with self.assertRaises(ValueError):P.validate_state(bad,bad['config'])
        # New neural admission compares only neural probes; the bot does not
        # pretend to have a model distribution or consume a neural pool slot.
        candidate=dict(model_id='new-neural',generation=570,path='new.pt',sha256='a'*64)
        migrated['health_window']=[True]*5
        distances={m['model_id']:.1 for m in migrated['members'] if not H.bot(m)}
        _,event=P.admission(migrated,candidate,dict(ok=True),distances,lambda a,b:.1)
        self.assertNotEqual(event['reason'],'probe_unavailable')
        self.assertNotIn(member['model_id'],event.get('eligible_members',[]))
        report=P.report_finish({},migrated,event)
        self.assertEqual(report['v4_history']['probability'],next(
            r['p'] for r in report['next_distribution']['members'] if r['model_id']==member['model_id']))

    def test_auto_gate_requires_ten_complete_iterations_ties_fail_and_latches(self):
        cfg=configure(self.old,manual_generation=None)
        pool,_=H.migrate(self.state['quality_pool'],cfg,bots(self.state))
        game=dict(setup_group='x',opponents=['learner','learner','bot','bot'],result={})
        for g in range(571,591):
            pool.update(generation=g,games_done=g*320)
            win=[.5,.5,0.,0.] if g!=580 else [.5,0.,.5,0.]
            with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(win_share=win)):
                H.advance(pool,[game],{'x':'legacy'},'learner')
            self.assertEqual(H.active(pool),g==590)
        self.assertEqual(pool[H.KEY]['activated']['source'],'measured_ordinary_tables')
        self.assertEqual([r['generation'] for r in pool[H.KEY]['window']],list(range(581,591)))
        pool.update(generation=591,games_done=591*320);H.advance(pool,[],{},'learner')
        self.assertTrue(H.active(pool));self.assertEqual(pool[H.KEY]['activated']['generation'],590)

    def test_actual_planner_166_154_0_bot_eligibility_profile_balance_and_PFSP(self):
        from brass_ai.recovery.ordinary import plan_collection
        from brass_ai.recovery.history import compact
        c=copy.deepcopy(self.new);c['local_scoring_auxiliary']['mode']='off'
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'weights.pt';path.write_bytes(b'planning')
            state=copy.deepcopy(self.state)
            for m in state['quality_pool']['members']:
                m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),S=5.,C=10.)
            P.seal(state['quality_pool'])
            state['quality_pool'],_=H.migrate(state['quality_pool'],c,bots(state))
            state['population']=[Q.reference(m) for m in state['quality_pool']['members'] if not H.bot(m)]
            state['signature']['config']=c
            service=Mock();service.models={m['model_id']:None for m in state['population']}
            values=bots(state);settings=E.effective(c,E.restore(c,state,state['exploration']),570)
            def plan():return plan_collection(SOURCE_ROOT,Path(temp),state,c,state['signature'],service,'current',
                sorted(values),values,values,None,exploration=settings)
            result=plan();self.assertEqual(result,plan())
            self.assertEqual(result['composition'],dict(selfplay=166,historical=154))
            self.assertEqual(len(result['quality_tables']),154)
            profiles=Counter();games=[];bot_seats=0
            for job in result['jobs']:
                table=job['provenance']['critic_table']
                for seat,r in enumerate(table['seats']):
                    if r['role']=='rule_bot':
                        policy=job['policies'][(seat-job['rotation'])%4];profiles[policy['parent']]+=1;bot_seats+=1
                        self.assertNotIn('model_id',policy)
                group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
                games.append(dict(setup_group=group,opponents=[job['policies'][(s-job['rotation'])%4]['policy_id'] for s in range(4)],result={}))
            self.assertGreater(bot_seats,0);self.assertLessEqual(max(profiles.values())-min(profiles.values()),1)
            with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=[1,2,3,4])):
                new,report=Q.update_scores(state['quality_pool'],games,result['quality_tables'],result['quality_planning'],generation=571,games_done=571*320)
            self.assertEqual(new['members'][-1]['profile_seats'],bot_seats)
            report['v4_history']=H.advance(new,games,result['kind_by_group'],'unused')
            self.assertEqual(compact(dict(quality_pool=report))['quality_pool'],report)
            self.assertEqual(report['exposure']['pairwise_comparisons'],616)

    def test_v4_EV_gate_is_retired_without_changing_history_or_main_GAE(self):
        from test_ppg_window import tick
        s=gae.initial(self.new['advantage'],570);before=copy.deepcopy(s)
        with patch('brass_ai.recovery.ev_evidence.pool',wraps=__import__('brass_ai.recovery.ev_evidence',fromlist=['pool']).pool) as pooled:
            value=dict(optimizer_steps=1,kl_guard_tripped=False,heldout_after=dict(explained_variance=.7,groups={}))
            new,report=gae.advance(s,self.new['advantage'],generation=571,value_report=value,health_ok=True,rating=dict(ok=True),retired_tables=('legacy',))
        self.assertEqual(report['table_gate']['legacy']['status'],'retired')
        self.assertEqual(new['table_enabled']['legacy'],before['table_enabled']['legacy'])
        self.assertEqual(new['table_streaks']['legacy'],before['table_streaks']['legacy'])
        self.assertEqual(new['active_lambda'],.95)
        self.assertEqual(pooled.call_count,2)  # History/selfplay only; no legacy EV pooling.

    def test_PL_lineups_seed_reference_distinctness_rotations_and_float_rating(self):
        from brass_ai.recovery import plackett_luce as pl
        from brass_ai.recovery.quality_config import RATING
        from test_strength_rating import fixture_rows
        c=self.new['rating'];agents={pl.IL:dict(kind='model',checkpoint=dict(model_id='il',generation=0)),pl.V4:dict(kind='v4'),
            R.SEED:dict(kind='model',checkpoint=dict(model_id='seed',generation=0))}
        scores={pl.IL:dict(theta=0.),pl.V4:dict(theta=-.1)}
        for g in range(10,91,10):
            a=f'generation-{g:06d}';agents[a]=dict(kind='model',checkpoint=dict(model_id=a,generation=g));scores[a]=dict(theta=g/100)
        candidate='generation-000100';agents[candidate]=dict(kind='model',checkpoint=dict(model_id='candidate',generation=100))
        rating=dict(config=c,seeds=list(range(32)))
        rows,coverage=R.lineups(rating,agents,candidate,scores,100,coverage_config=R.WITH_SEED)
        self.assertEqual(len(rows),128);self.assertEqual(Counter(coverage['reference_setup_indices'].values()),{pl.IL:2,pl.V4:2,R.SEED:4})
        self.assertEqual(Counter(r['agents'][1] for r in rows if r['agents'][1] in (pl.IL,pl.V4,R.SEED)),{pl.IL:8,pl.V4:8,R.SEED:16})
        for i in range(0,128,4):
            self.assertEqual(len(set(rows[i]['agents'])),4)
            self.assertEqual([r['seat'] for r in rows[i:i+4]],list(range(4)))
            self.assertTrue(all(r['agents']==rows[i]['agents'] for r in rows[i:i+4]))
        spec=dict(generation=100,lineups=rows);data=fixture_rows(spec)
        # A synthetic strength ordering exercises estimation only, not game strength.
        for row in data:row['order']=sorted(range(4),key=lambda s:row['agents'][s]!=R.SEED)
        result,samples=pl.estimate(data,agents,dict(RATING,bootstrap=4))
        self.assertEqual(result['ratings'][pl.IL]['theta'],0.)
        self.assertGreater(result['ratings'][R.SEED]['theta'],0.)
        self.assertGreater(result['ratings'][R.SEED]['theta_ci95'][1],0.)
        fallback={k:v for k,v in agents.items() if k in (pl.IL,pl.V4,R.SEED,candidate)}
        small,report=R.lineups(rating,fallback,candidate,{},100,coverage_config=R.WITH_SEED)
        self.assertEqual(len(small),128);self.assertTrue(report['startup_fallback_seats'])

    def test_complete_companion_snapshot_save_restart_collision_and_missing_legacy(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
        from brass_ai.recovery.local_scoring_auxiliary import Companion,KEY
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.learning.batching import model_id
        model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=self.new['learning_rate'])
        migrate(model,opt,reward_spec(self.new));aux=Companion(model,self.new[KEY],raw=dict(model_id=model_id(model)))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);full=run/'candidate.pt'
            def save():
                identity=save_checkpoint(full,model,opt,step=1,training={KEY:aux.payload(1)})
                return dict(path=full.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(full),model_id=identity,generation=1)
            entry=save();rng=torch.get_rng_state().clone();frozen=slim_checkpoint(SOURCE_ROOT,run,entry)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            raw=load_checkpoint(SOURCE_ROOT/frozen['path'])[1]
            self.assertTrue(_same_payload(raw['training'][KEY],aux.payload(1)))
            self.assertTrue(_same_payload(raw['state_dict'],dict(model.state_dict())))
            self.assertIsNone(raw['optimizer']);self.assertIn('auxiliary_heads',frozen)
            old_sha=frozen['sha256']
            with torch.no_grad():next(aux.heads.parameters()).add_(.01)
            entry2=save();frozen2=slim_checkpoint(SOURCE_ROOT,run,entry2,frozen)
            self.assertNotEqual(frozen2['path'],frozen['path']);self.assertEqual(sha(SOURCE_ROOT/frozen['path']),old_sha)
            self.assertEqual(slim_checkpoint(SOURCE_ROOT,run,entry2),frozen2)
            legacy=run/'legacy.pt';save_checkpoint(legacy,model,training={})
            old=dict(entry,path=legacy.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(legacy),generation=0)
            no_heads=slim_checkpoint(SOURCE_ROOT,run,old)
            self.assertNotIn(KEY,load_checkpoint(SOURCE_ROOT/no_heads['path'])[1]['training'])
            self.assertNotIn('auxiliary_heads',no_heads)

    def test_reviewed_pool_resume_before_first_commit_and_baseline_hash(self):
        from brass_ai.recovery.resume import check_signature,commit_transition,prepare_opponent_pool,gae_floor_receipt_preserved
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);s=copy.deepcopy(self.state);entry=s['history'][-1]
            raw=dict(training=dict(quality_pool_digest=Q.state_digest(s['quality_pool']),signature=digest(s['signature'])))
            sig=dict(s['signature'],config=self.new,implementation=implementation_id())
            check_signature(s,sig,reviewed=True)
            self.assertTrue(gae_floor_receipt_preserved(s,sig,raw,entry))
            migrated,receipt=H.migrate(s['quality_pool'],self.new,bots(s))
            write_json(run/'state.json',s)
            committed=commit_transition(run,s,sig,opponent_pool=migrated,pool_migration=receipt)
            with patch.object(H,'policies',return_value=bots(s)):
                again,receipt2=prepare_opponent_pool(SOURCE_ROOT,run,committed,raw,self.new)
            self.assertEqual(again,migrated);self.assertIsNone(receipt2)
            seed=run/'seed.pt';seed.write_bytes(b'trained start')
            s['origin']['seed_checkpoint'].update(path=seed.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(seed))
            baseline=R.seed_reference(SOURCE_ROOT,s,self.new)
            self.assertEqual(baseline['checkpoint']['generation'],0);self.assertFalse(baseline['fixed'])
            self.assertEqual(baseline['source_generation'],1290)
            seed.write_bytes(b'changed')
            with self.assertRaises(ValueError):R.seed_reference(SOURCE_ROOT,s,self.new)

    def test_config_helper_is_read_only_idempotent_and_rejects_stale_or_changed_artifacts(self):
        spec=importlib.util.spec_from_file_location('g570_helper',SOURCE_ROOT/'scripts/configure-training-g570.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0570.pt';checkpoint.write_bytes(b'committed full fixture')
            seed=run/'seed.pt';seed.write_bytes(b'original seed fixture')
            state=copy.deepcopy(self.state)
            state['history'][-1].update(path='artifacts/run/generation-0570.pt',sha256=sha(checkpoint))
            state['origin']['seed_checkpoint'].update(path='artifacts/run/seed.pt',sha256=sha(seed))
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes()
            output='configs/step5/test-g570.json'
            with patch.object(H,'policies',return_value=bots(state)):
                first=module.generate(root,'artifacts/run',output,570)
                self.assertEqual(module.generate(root,'artifacts/run',output,570),first)
                self.assertEqual((run/'state.json').read_bytes(),before)
                from brass_ai.recovery import entropy_schedule as S
                self.assertEqual(read(root/output),S.configure(self.new,state,'on',start='next'))
                with self.assertRaises(ValueError):module.generate(root,'artifacts/run',output,571)
                checkpoint.write_bytes(b'changed')
                with self.assertRaises(ValueError):module.generate(root,'artifacts/run',output,570)

    def test_bounded_PPG_smoke_keeps_Adam_cache_and_filters_old_v4_holdouts(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from brass_ai.learning.batching import model_id
        from test_ppg_g210 import prepare
        old=copy.deepcopy(self.old);old['model_size']='small';old['update']['preparation_workers']=0
        old['value_phase'].update(batch_size=128,clone_rows=24,clone_batch_size=8)
        model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
        migrate(model,opt,reward_spec(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);phase=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            phase.local_scoring=Companion(model,old['local_scoring_auxiliary'],raw=dict(model_id=model_id(model)))
            prepare(model,phase,1,companion=True)
            phase.buffer.stage(phase.critic_dataset,phase.train_ids,phase.heldout_ids,phase.actor_ids,
                phase.collection_source,model,phase.split['selection'])
            phase.state=gae.initial(old['advantage'],1);phase.critic_state['generation']=1
            raw=dict(training={KEY:phase.payload(),'local_scoring_auxiliary':phase.local_scoring.payload(1)})
            write_json(run/'state.json',dict(iteration=1));phase.buffer.committed(1)
            before=copy.deepcopy((model.state_dict(),opt.state_dict(),phase.optimizer.state_dict(),phase.state,torch.get_rng_state()))
            config=configure(old);resumed=TrainingPhases(model,config,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),resumed.optimizer.state_dict(),resumed.state,torch.get_rng_state())))
            # Stage a small second collection, then retire legacy validation.
            # The retained first-iteration cache still has its original labels.
            resumed.local_scoring=Companion(model,config['local_scoring_auxiliary'],raw=raw,generation=1)
            prepare(model,resumed,2,companion=True)
            resumed.retired_tables=('legacy',)
            resumed.heldout_ids=[i for i in resumed.heldout_ids if resumed.critic_dataset.tables[i]!='legacy']
            result=resumed.fit(model)
            self.assertEqual(result['retired_table_checks'],['legacy'])
            self.assertGreater(result['attempted_steps'],0)
            self.assertLessEqual(result['probe_kl'],.002)
            print('g570 isolated PPG smoke:',{k:result[k] for k in ('optimizer_steps','attempted_steps','probe_kl','early_stop_reason')})


if __name__=='__main__':unittest.main()
