"""Decoupled opening objective, actual behavior and reviewed PPG continuation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import opening_objective as O,ppg_config as P,exploration as E,updates,value_phase,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.review_metrics import CategoricalProbe,categorical_kl
from test_gae_value_phase import fixture,examples,ppo,finish

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0503-20261022T051624803527Z'
spec=importlib.util.spec_from_file_location('g503_openings',SOURCE_ROOT/'scripts/joint7-g503-ppg.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def opening_rows(model,data,epsilon=0.,opening=E.OPENING):
    """Authored complete-policy fixtures, with actual T2/T1 behavior retained."""
    behavior=Behavior(model_id(model),epsilon,dict(version=3,opening=opening,
        raw_entropy=True,alphas=dict.fromkeys(E.GROUPS,.005)))
    counters={};result=[]
    for row,t,identity,e in data:
        row=copy.deepcopy(row);count=counters.get(identity,0)
        action=row['view']['context']['phase']=='action'
        b=dict(kind='model',policy_id=behavior.policy_id,probability=1.,temperature=2. if opening and action and count<2 else 1.,
            exploration=epsilon,exploration_version=3,component='policy',action_decision=count if action else None)
        row.update(behavior=b,**{O.ROW_KEY:copy.deepcopy(O.DEFAULT)})
        if action:counters[identity]=count+1
        result.append((row,t,identity,e))
    ds=updates.Dataset(result,'cpu');model.eval()
    with torch.no_grad():
        for part in ds.microbatches(range(len(result)),32,131072):
            ix,inputs,q=ds.batch(part,epsilon)
            p=updates.row_probabilities(ds,ix,model(*inputs)[0],q,behavior)
            for j,i in enumerate(part):result[i][0]['behavior']['probability']=float(p[j,result[i][0]['chosen']])
    return behavior,result


class Objective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_clipped_objective_positive_negative_and_uncorrected_rows(self):
        b=torch.tensor([.1,.6,.4,.2],requires_grad=True)
        old=torch.tensor([.3,.3,.6,.3],requires_grad=True)
        new=torch.tensor([.45,.15,.8,.5],requires_grad=True)
        mixed=torch.tensor([.2,.5,.5,.3],requires_grad=True)
        a=torch.tensor([2.,-3.,4.,-1.]);mask=torch.tensor([True,True,False,False])
        loss,r,w=O.losses(mixed,b,new,old,mask,a,.2)
        expected_r=torch.tensor([1.5,.5,1.25,1.5]);expected_w=torch.tensor([2.,.5,1.,1.])
        expected=-expected_w*torch.minimum(expected_r*a,expected_r.clamp(.8,1.2)*a)
        torch.testing.assert_close(loss,expected);torch.testing.assert_close(r,expected_r);torch.testing.assert_close(w,expected_w)
        self.assertNotAlmostEqual(float(w.mean()),1.)
        self.assertFalse(w.requires_grad);loss.mean().backward()
        self.assertIsNone(old.grad);self.assertIsNone(b.grad)
        self.assertEqual(float(new.grad[2:].abs().sum()),0.)
        self.assertEqual(float(mixed.grad[:2].abs().sum()),0.)
        self.assertTrue(torch.isfinite(new.grad).all())

    def test_importance_denominator_includes_temperature_exploration_and_legal_mask(self):
        logits=torch.tensor([[2.,-.5,float('-inf')],[.1,.9,1.6]])
        q=torch.tensor([[.9,.1,0],[.1,.2,.7]])
        raw=logits.softmax(-1);actual=updates.probabilities(logits,q,.2,2.)
        self.assertEqual(float(actual[0,2]),0.);torch.testing.assert_close(actual.sum(-1),torch.ones(2))
        old=raw[:,0];b=actual[:,0]
        loss,r,w=O.losses(b,b,old,old,torch.ones(2,dtype=torch.bool),torch.ones(2),.2)
        torch.testing.assert_close(w,(old/b).clamp(max=2.));torch.testing.assert_close(loss,-w)
        self.assertFalse(torch.allclose(w,(old/logits.div(2).softmax(-1)[:,0]).clamp(max=2.)))
        torch.testing.assert_close(r,torch.ones(2))

    def test_unclipped_opening_gradient_uses_frozen_capped_weight(self):
        actual=torch.tensor([.2,.1],requires_grad=True);old=torch.tensor([.1,.5],requires_grad=True)
        new=torch.tensor([.11,.51],requires_grad=True);behavior=torch.tensor([.3,.2],requires_grad=True)
        loss,_,w=O.losses(behavior,actual,new,old,torch.ones(2,dtype=torch.bool),torch.tensor([2.,1.]),.2)
        loss.sum().backward()
        torch.testing.assert_close(w,torch.tensor([.5,2.]))
        torch.testing.assert_close(new.grad,torch.tensor([-10.,-4.]))
        self.assertIsNone(old.grad);self.assertIsNone(actual.grad)
        torch.testing.assert_close(behavior.grad,torch.zeros(2))

    def test_reference_freeze_and_actual_behavior_verification(self):
        model,_,cfg,(_,data)=fixture(games=3,decisions=3)
        behavior,data=opening_rows(model,data,.02)
        ds=updates.Dataset(data,'cpu');ds.opening_reference=O.Reference(ds,behavior,O.DEFAULT)
        actual=ds.old.clone();rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,'frozen'):ds.opening_reference.loss([0],torch.zeros(1,2),actual[:1],torch.ones(1),.2)
        updates.policy_measure(model,ds,range(len(data)),behavior,cfg['update'],Timings(),verify=True)
        self.assertTrue(ds.opening_reference.frozen);self.assertTrue(torch.equal(actual,ds.old))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(ds.opening_reference.old.requires_grad)
        self.assertTrue(ds.opening_reference.mask.any());self.assertTrue((~ds.opening_reference.mask).any())
        with self.assertRaisesRegex(ValueError,'already frozen'):ds.opening_reference.capture([0],torch.zeros(1,2))
        with torch.no_grad():ds.old[0]=.001
        with self.assertRaisesRegex(ValueError,'Wrong/stale behavior probabilities'):
            ds.opening_reference=None
            updates.policy_measure(model,ds,range(len(data)),behavior,cfg['update'],Timings(),verify=True)

    def test_stale_provenance_epsilon_temperature_and_prefix_rejected(self):
        model,_,cfg,(_,data)=fixture(games=3,decisions=3);b,data=opening_rows(model,data)
        for field,value in [(O.ROW_KEY,None),('_copied_prefix',True)]:
            bad=copy.deepcopy(data);bad[0][0][field]=value;ds=updates.Dataset(bad,'cpu')
            with self.assertRaises(ValueError):O.Reference(ds,b,O.DEFAULT)
        for field,value in [('exploration',.02),('temperature',1.),('component','branch_move')]:
            bad=copy.deepcopy(data);bad[0][0]['behavior'][field]=value
            with self.assertRaises(ValueError):updates.verify_behavior_metadata(updates.Dataset(bad,'cpu'),b)
        bad=copy.deepcopy(data);bad[0][0]['behavior']['action_decision']=2
        with self.assertRaises(ValueError):O.Reference(updates.Dataset(bad,'cpu'),b,O.DEFAULT)

    def test_raw_t1_probe_and_behavior_kl_are_separate_complete_distributions(self):
        model,_,cfg,(_,data)=fixture(games=3,decisions=3);b,data=opening_rows(model,data,.02)
        ds=updates.Dataset(data,'cpu');ids=[i for i,r in enumerate(ds.rows) if len(r['choices'])>1]
        probe=CategoricalProbe(model,ds,ids,b,dict(cfg['update'],opening_objective=O.DEFAULT))
        class Shift(torch.nn.Module):
            def forward(self,*inputs):
                result=list(model(*inputs));result[0]=result[0]+torch.arange(result[0].shape[1])*.4
                return result
        changed=Shift();result=probe.measure_step(changed);before=probe.measure_step(model)
        self.assertAlmostEqual(before['categorical_kl'],0.,places=7)
        self.assertGreater(result['temperature1_categorical_kl'],result['behavior_categorical_kl'])
        expected_raw=expected_behavior=0.
        with torch.no_grad():
            for index in range(len(probe.parts)):
                _,old,ix,inputs,q=probe._batch(index);logits=changed(*inputs)[0]
                expected_raw+=float(categorical_kl(probe.raw_parts[index],logits.softmax(-1)))
                expected_behavior+=float(categorical_kl(old,updates.row_probabilities(ds,ix,logits,q,b)))
        self.assertAlmostEqual(result['categorical_kl'],expected_raw/len(ids),places=6)
        self.assertAlmostEqual(result['behavior_categorical_kl'],expected_behavior/len(ids),places=6)
        final=probe.measure(changed)
        self.assertAlmostEqual(final['categorical_kl'],result['categorical_kl'],places=6)
        self.assertEqual(final['opening_positions'],sum(ds.rows[i]['behavior']['temperature']==2 for i in ids))

    def test_branch_opening_rows_use_their_own_counters_and_exclude_fork(self):
        model,_,cfg,(_,data)=fixture(games=3,decisions=4)
        parent=[d for d in data if d[2]==data[0][2]]
        branch=[(dict(r,_branch=True),t,(g+'/branch/test',s),e) for r,t,(g,s),e in parent]
        b,rows=opening_rows(model,parent+branch)
        ref=O.Reference(updates.Dataset(rows,'cpu'),b,O.DEFAULT)
        self.assertEqual([r['behavior']['action_decision'] for r,_,_,_ in rows],[0,1,2,3]*2)
        self.assertEqual(ref.mask.tolist(),[True,False,False,False]*2)  # second action is forced for this seat
        self.assertTrue(rows[len(parent)][0]['_branch'])
        rows[len(parent)][0]['behavior']['component']='branch_move'
        with self.assertRaisesRegex(ValueError,'Fork moves'):O.Reference(updates.Dataset(rows,'cpu'),b,O.DEFAULT)

    def test_end_to_end_ppo_preserves_behavior_and_other_losses(self):
        model,opt,cfg,(_,data)=fixture(games=3,decisions=3);b,data=opening_rows(model,data)
        cfg['update']['opening_objective']=copy.deepcopy(O.DEFAULT)
        before=[copy.deepcopy(r['behavior']) for r,_,_,_ in data]
        report=ppo(model,opt,cfg,b,data)
        self.assertGreater(report['optimizer_steps'],0)
        self.assertEqual(before,[r['behavior'] for r,_,_,_ in data])
        self.assertTrue(report['opening_objective']['old_t1_frozen'])
        self.assertEqual(report['kl_guard']['contract'],'ppo-categorical-kl-guard/2.0.0')
        self.assertIn('behavior_categorical_kl',report['categorical_probe'])
        self.assertIn('utility_prediction',report['epochs'][0])
        from brass_ai.recovery.history import compact
        self.assertEqual(compact(dict(actor=report))['actor']['opening_objective'],report['opening_objective'])

    def test_both_exact_guards_stop_independently(self):
        for metric,reason in [('categorical_kl','temperature1_categorical_probe'),
                              ('behavior_categorical_kl','behavior_categorical_probe')]:
            model,opt,cfg,(_,data)=fixture(games=3,decisions=3);b,data=opening_rows(model,data)
            cfg['update']['opening_objective']=copy.deepcopy(O.DEFAULT)
            measured=dict(categorical_kl=0.,behavior_categorical_kl=0.,sampled_kl=0.)
            measured[metric]=.02
            with patch.object(CategoricalProbe,'measure_step',return_value=measured):
                report=ppo(model,opt,cfg,b,data)
            self.assertEqual(report['optimizer_steps'],1);self.assertEqual(report['kl_guard']['stop_reason'],reason)

    def test_no_opening_rows_leave_ppo_weights_optimizer_advantages_and_rng_identical(self):
        model,opt,cfg,(_,data)=fixture(games=3,decisions=3);b,data=opening_rows(model,data,opening=None)
        other=copy.deepcopy(model);other_opt=torch.optim.AdamW(other.parameters());other_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        changed=copy.deepcopy(cfg);changed['update']['opening_objective']=copy.deepcopy(O.DEFAULT)
        rng=torch.get_rng_state().clone();before=ppo(model,opt,cfg,b,data);after_rng=torch.get_rng_state().clone()
        torch.set_rng_state(rng);after=ppo(other,other_opt,changed,b,data)
        self.assertTrue(_same_payload(model.state_dict(),other.state_dict()))
        self.assertTrue(_same_payload(opt.state_dict(),other_opt.state_dict()))
        self.assertTrue(torch.equal(after_rng,torch.get_rng_state()))
        self.assertEqual(before['advantage'],after['advantage']);self.assertEqual(after['opening_objective']['opening_rows'],0)
        for key in ('actor','value','entropy','clip_fraction','utility_prediction'):
            self.assertEqual(before['epochs'][0][key],after['epochs'][0][key])

    def test_provenance_survives_shard_reader_and_compaction(self):
        from review_fixture import current_game
        from brass_ai.recovery.replay import write_shard,read_shards
        from brass_ai.recovery.ingress import _prepare_inner
        from brass_ai.learning.replay import split_for
        from brass_ai.step5.league import seeds_for
        episode=current_game();episode=copy.deepcopy(episode)
        seed=seeds_for(17100000,1)[0][0];episode['setup_group']=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
        episode['split']=split_for(episode['setup_group']);episode['opponents']=['learner']*4
        for row in episode['records']:row['behavior']=dict(kind='model',policy_id='learner',probability=1.,temperature=1.)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'games.jsonl.gz'
            write_shard(path,[episode],dict(purpose='training',ppo_opening_objective=O.DEFAULT),cache=False)
            game=read_shards([path])[0];self.assertEqual(game.ppo_opening_objective,O.DEFAULT)
            record=dict(path=str(path),sha256=sha(path),game_id=episode['game_id'])
            with patch('brass_ai.recovery.ingress.summarize',return_value={}):
                _,data,_,_=_prepare_inner(([record],'learner',{episode['setup_group']:'selfplay'},None,None))
            self.assertTrue(all(row.get(O.ROW_KEY)==O.DEFAULT for row,_,_,_ in data))

    def test_changed_objective_or_epsilon_rejects_cached_collection(self):
        import gzip,json
        from brass_ai.recovery.execution import neural_spec,_collection_recipe,ActorPool
        from unittest.mock import Mock
        c=read(EXPORT/'state.json')['signature']['config'];service=Mock()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp)
            old=dict(index=0,seed=321,path=str(directory/'actor.jsonl.gz'),
                policies=[neural_spec('learner',.02,'mixed',version=3,opening=E.OPENING)]*4,
                provenance=dict(purpose='training'))
            for new in (dict(old,provenance=dict(purpose='training',ppo_opening_objective=O.DEFAULT)),
                        dict(old,policies=[neural_spec('learner',0.,'mixed',version=3,opening=E.OPENING)]*4)):
                before=_collection_recipe([old],c['execution']);after=_collection_recipe([new],c['execution'])
                self.assertNotEqual(before,after)
                path=directory/f'games-{digest(after)[:16]}-part-000000.jsonl.gz'
                with gzip.open(path,'wt',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(provenance=dict(collection=digest(before),recipe=before)))+'\n')
                pool=ActorPool(SOURCE_ROOT,c['host'],service,c['execution'])
                try:
                    with patch.object(pool,'start',side_effect=AssertionError('Started before admission')):
                        with self.assertRaisesRegex(ValueError,'Cached collection differs'):pool._collect([new])
                finally:pool.close()


class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.state=read(EXPORT/'state.json');cls.old=cls.state['signature']['config']

    def test_exact_config_and_original_entropy_baseline(self):
        c=helper.configure(self.old);validate_config(c);self.assertEqual(c,helper.configure(c))
        self.assertEqual({k for k in c if c[k]!=self.old[k]},
            {'insolvency','search','branching','critic_data','value_phase','exploration','update'})
        self.assertEqual(c['value_phase'],dict(self.old['value_phase'],contract=P.SHORT_CONTRACT,enabled=True,acceptance_steps=16,patience=1))
        self.assertEqual(c['update'],dict(self.old['update'],opening_objective=O.DEFAULT))
        self.assertEqual(c['exploration'],0.)
        self.assertEqual(E.restore(c,self.state,self.state['exploration']),self.state['exploration'])
        self.assertEqual(c['entropy_control'],self.old['entropy_control'])
        fresh=latest_model_config(self.old,self.old['imitation_checkpoint'])
        self.assertEqual(fresh['exploration'],0.);self.assertTrue(fresh['value_phase']['enabled'])
        self.assertEqual(fresh['value_phase']['acceptance_steps'],16);self.assertEqual(fresh['value_phase']['patience'],1)
        for key,val in [('importance_cap',3),('target_temperature',2),('normalization','batch')]:
            bad=copy.deepcopy(c);bad['update']['opening_objective'][key]=val
            with self.assertRaises(ValueError):validate_config(bad)
        for key,val in [('patience',2),('acceptance_steps',17)]:
            bad=copy.deepcopy(c);bad['value_phase'][key]=val
            with self.assertRaises(ValueError):validate_config(bad)

    def test_reenable_preserves_both_adams_gae_and_starts_new_eight_iteration_schedule(self):
        from test_ppg_eight import setup
        from brass_ai.recovery import gae,snapshot_r2
        model,opt,cfg=setup();old=value_phase.disabled_options(cfg)
        phase=TrainingPhases(model,old)
        for optimizer in (opt,phase.optimizer):
            p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
        phase.state=gae.initial(old['advantage'],503)
        phase.critic_state['generation']=503;phase.critic_state['weight_recovery']=snapshot_r2.initial(503,phase.critic_state['snapshot_weight'])
        phase.critic_evidence['origin']=503;phase.beta=8.
        saved=phase.payload();weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        new=critic_data.fixed_config(P.short_options(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp)
            first=TrainingPhases(model,new,raw=dict(training={KEY:saved}),generation=503,root=SOURCE_ROOT,run=run)
            second=TrainingPhases(model,new,raw=dict(training={KEY:saved}),generation=503,root=SOURCE_ROOT,run=run)
            for field in ('state','critic_state','schedule','options','epoch_control','critic_evidence'):
                self.assertTrue(_same_payload(getattr(first,field),getattr(second,field)),field)
            self.assertEqual(first.state,phase.state);self.assertEqual(first.beta,8.)
            self.assertTrue(_same_payload(first.optimizer.state_dict(),phase.optimizer.state_dict()))
            self.assertTrue(_same_payload(weights,model.state_dict()));self.assertTrue(_same_payload(adam,opt.state_dict()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(first.critic_state['snapshot_weight'],1.)
            clock=first.schedule;due=[]
            for g in range(504,524):
                ran=P.due(clock)
                if ran:due.append(g)
                clock=P.advance(clock,g,ran)
            self.assertEqual(due,[511,519]);self.assertEqual(first.buffer.receipts,[])

    def test_ppg_safeguards_evidence_and_sixteen_step_rate(self):
        c=P.short_options(self.old)['value_phase'];P.validate(c)
        self.assertEqual([P.weight(i,c) for i in range(8)],[1.]*8)
        self.assertAlmostEqual(P.phase_rate(c,1,64),c['learning_rate']*.1)
        self.assertAlmostEqual(P.phase_rate(c,16,64),c['learning_rate'])
        self.assertAlmostEqual(P.phase_rate(c,64,64),c['learning_rate']*.3)
        work=dict(source='ppo',optimizer_steps=5,distinct_rows=42)
        self.assertEqual(value_phase.critic_work(dict(contract=P.SHORT_CONTRACT,enabled=True,gate_critic_training=work)),work)
        self.assertEqual(P.configure(self.old)['value_phase'],dict(self.old['value_phase'],enabled=True))

    def test_sixteen_step_check_and_patience_one_keep_best_value_adam(self):
        from test_ppg_eight import setup,stage_only
        from brass_ai.recovery import ppg_auxiliary
        from brass_ai.step5.league import write_json
        model,opt,cfg=setup();cfg=P.short_options(cfg)
        cfg['value_phase']['batch_size']=24  # Two full 16-step intervals before epoch end.
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(model,phase,g)
                if g<8:write_json(run/'state.json',dict(iteration=g));phase.buffer.committed(g)
            calls=[];saved={};step=phase.optimizer.step
            def recording_step(*args,**kwargs):
                result=step(*args,**kwargs);calls.append(1)
                if len(calls)==16:
                    saved['weights']=copy.deepcopy(model.state_dict());saved['adam']=copy.deepcopy(phase.optimizer.state_dict())
                return result
            checks=[]
            def compare(before,after):
                checks.append(len(calls));self.assertEqual(len(before),48)
                return dict(games=48,mean_change=-.1 if len(checks)==1 else .01,standard_error=.02,
                    reject=False,improved=len(checks)==1)
            with patch.object(phase.optimizer,'step',recording_step),patch.object(ppg_auxiliary,'paired',compare):
                report=ppg_auxiliary.fit(model,phase.optimizer,phase.buffer,phase.critic_dataset,
                    phase.train_ids,phase.heldout_ids,cfg['value_phase'],cfg['update'],beta=1.,
                    snapshot_weight=1.,dropout=None,run_phase=True,ppo_work=phase.ppo_work)
            self.assertEqual(checks,[16,32]);self.assertEqual(report['attempted_steps'],32)
            self.assertEqual(report['optimizer_steps'],16);self.assertEqual(report['early_stop_reason'],'validation_patience')
            self.assertTrue(_same_payload(model.state_dict(),saved['weights']))
            self.assertTrue(_same_payload(phase.optimizer.state_dict(),saved['adam']))

    def test_objective_change_resets_only_lr_measurement_identity(self):
        from brass_ai.recovery.lr_controller import collection_signature
        c=copy.deepcopy(self.old);sampling=dict(by_composition=dict(selfplay=dict(tables=160,learner_trajectories=640)))
        self.assertNotEqual(collection_signature(c,sampling),collection_signature(O.configure(c),sampling))

    def test_actual_signature_and_config_helper(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id,write_json
        c=helper.configure(self.old);signature=dict(self.state['signature'],config=c,implementation=implementation_id())
        self.assertTrue(check_signature(self.state,signature,reviewed=True))
        with self.assertRaises(ValueError):check_signature(self.state,signature)
        bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(self.state,bad,reviewed=True)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True);write_json(run/'state.json',self.state)
            original=sha(run/'state.json');output='configs/step5/openings.json'
            for _ in range(2):
                result=helper.generate(root,run,output,503);self.assertFalse(result['run_modified'])
                self.assertEqual(result['first_auxiliary_generation'],511)
                self.assertEqual(read(root/output),c);self.assertEqual(sha(run/'state.json'),original)
            with self.assertRaisesRegex(ValueError,'Unexpected committed'):helper.generate(root,run,output,502)

    def test_interrupted_adoption_preserves_checkpoint_and_repeats_exactly(self):
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.step5.league import implementation_id,write_json
        from brass_ai.recovery.resume import load_resume,commit_transition
        from brass_ai.recovery.interrupted import archive_uncommitted_iteration
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.recovery import snapshot_r2
        model,opt,_,_=fixture();old=copy.deepcopy(self.old);old['model_size']='small'
        signature=dict(copy.deepcopy(self.state['signature']),config=old)
        phase=TrainingPhases(model,old);phase.state['generation']=1
        phase.state['table_evidence']['origin']=phase.state['stage_evidence']['origin']=1
        phase.critic_state.update(generation=1,snapshot_weight=0.,snapshots_active=True,activation_generation=1,
            weight_recovery=snapshot_r2.initial(1,0.))
        for optimizer in (opt,phase.optimizer):
            p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True);path=run/'generation-0001.pt'
            mid=save_checkpoint(path,model,opt,step=1,training={KEY:phase.payload(),
                'signature':digest(signature),'fresh_run':run.relative_to(root).as_posix()})
            original=sha(path);entry=dict(path=path.relative_to(root).as_posix(),sha256=original,model_id=mid,generation=1)
            state=dict(format='step5-recovery-league/2.2.0',iteration=1,games_done=320,signature=signature,
                history=[entry],population=[],evaluation_anchor=entry)
            write_json(run/'state.json',state);pending=run/'iteration-0001';pending.mkdir()
            (pending/'old.bin').write_bytes(b'old behavior collection')
            new=helper.configure(old);target=dict(signature,config=new,implementation=implementation_id())
            with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):load_resume(root,run,state,target,reviewed=True)
            receipt=archive_uncommitted_iteration(root,run,expected_iteration=1,apply=True)
            self.assertEqual((run/receipt['archive']/'iteration-0001/old.bin').read_bytes(),b'old behavior collection')
            def restore():
                net,raw,_=load_resume(root,run,state,target,reviewed=True)
                adam=torch.optim.AdamW(net.parameters());resume_optimizer(raw,adam)
                migrate(net,adam,reward_spec(new))
                restored=TrainingPhases(net,new,raw=raw,generation=1,root=root,run=run)
                return net,adam,restored,torch.get_rng_state().clone()
            first,adam,a,rng=restore();state=commit_transition(run,state,target)
            second,other,b,rng2=restore()
            self.assertTrue(_same_payload(first.state_dict(),second.state_dict()))
            self.assertTrue(_same_payload(adam.state_dict(),other.state_dict()))
            self.assertTrue(_same_payload(a.optimizer.state_dict(),b.optimizer.state_dict()))
            self.assertTrue(torch.equal(rng,rng2));self.assertEqual(a.state,b.state)
            self.assertEqual(a.critic_state,b.critic_state);self.assertEqual(a.schedule,b.schedule)
            self.assertEqual(a.schedule['origin'],1);self.assertEqual(a.critic_state['snapshot_weight'],1.)
            self.assertEqual(sha(path),original);self.assertEqual(state['iteration'],1);self.assertEqual(state['games_done'],320)


if __name__=='__main__':unittest.main()
