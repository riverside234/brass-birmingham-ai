"""Qualified on-mode and real restore/head-Adam regressions; no games."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.learning.search_value import ordinary_parameters
from brass_ai.recovery import resume,teacher_gradient_qualification as Q
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.search_value_iteration import SearchValueIteration,stamp
from brass_ai.recovery.teacher_phase import TeacherPhase,plan_jobs
from brass_ai.recovery.teacher_term import TeacherTerm,forward_policy
from brass_ai.recovery.value_preservation import TrunkGradients
from brass_ai.rust_backend import lab_teacher_activation as A,lab_shadow_config as C,lab_backend as B,lab_replay_compat as R
from brass_ai.step5.league import write_json
import test_b07_shadow_resume as resume_fixture

RESULTS=ROOT/'results/joint7-v11-2-g1290-trained-restart-g1261-20261108T070035419646Z'


class Activation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.control=read(RESULTS/'state.json')

    def setUp(self):
        # Reuse the existing real-checkpoint/Adam fixture; only scale the model
        # allocation. Preserve the actual four-iteration receipt and lineage.
        f=resume_fixture.PendingResume('test_real_loader_restores_pending_upgrade_with_exact_weights_adam_rng_and_first_measurement')
        f.setUp();self.addCleanup(f.doCleanups);self.f=f
        before=copy.deepcopy(f.after);before['config']=C.configure(read(ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json'))
        f.before=before;f.after=dict(before,config=A.configure(before['config']))
        f.pending.rename(f.run/'iteration-1261');f.pending=f.run/'iteration-1261'
        f.state.update(iteration=1261,games_done=403520,signature=before,
            exploration=copy.deepcopy(self.control['exploration']),teacher_seed_cursor=self.control['teacher_seed_cursor'])
        f.phase=copy.deepcopy(self.control['teacher_phase']);f.phase['buffer']={}
        f.state['teacher_phase']=copy.deepcopy(f.phase)
        f.state['native_backend']['config_sha256']=digest(before['config'])
        shadow=SearchValueIteration(f.run,before['config'],dict(iteration=1260,
            history=[dict(generation=1260,model_id=model_id(f.model),sha256='a'*64)]),f.model)
        value=shadow.payload(1261);f.state['search_value']=stamp(value)
        f.checkpoint=f.run/'generation-1261.pt'
        execution=read(ROOT/'results/rust-lab-qualification/20261108T063358240043Z-20261108T060920000950Z-shadow-training-05484c69-61fd605f/evidence/qualification.json')['execution_source']
        identity=save_checkpoint(f.checkpoint,f.model,f.optimizer,step=1261,training=dict(mode='step5-population-ppo',
            fresh_run=f.run.relative_to(ROOT).as_posix(),signature=digest(before),teacher_phase=f.phase,
            search_value=value,native_backend=f.state['native_backend'],native_backend_execution=execution))
        old=[dict(e,path=(f.run/('retired-'+str(e['generation'])+'.pt')).relative_to(ROOT).as_posix())
            for e in self.control['history'] if 1257<=e['generation']<=1260]
        entry=dict(path=f.checkpoint.relative_to(ROOT).as_posix(),sha256=sha(f.checkpoint),model_id=identity,generation=1261)
        f.state.update(history=old+[entry],population=[entry],evaluation_anchor=entry)
        write_json(f.run/'state.json',f.state);f.state_bytes=(f.run/'state.json').read_bytes()
        f.files=f.pending_bytes()

    def test_actual_qualification_fixed_beta_and_runtime_anneal_completion(self):
        q=A.authorize(self.control,A.configure(self.control['signature']['config']))
        self.assertEqual(q['beta'],0.16110139609854138);self.assertEqual(sum(o['valid_steps'] for o in q['observations']),184)
        self.assertFalse(A.annealing(self.f.after['config'],self.f.state))
        bad=copy.deepcopy(self.f.state);schedule=bad['exploration']['entropy_schedule'];schedule['progress']=99
        from brass_ai.recovery import entropy_schedule as S
        S.seal(schedule)
        self.assertTrue(A.annealing(self.f.after['config'],bad))
        with self.assertRaisesRegex(ValueError,'Complete entropy'):A.authorize(bad,self.f.after['config'])

    def test_resealed_bad_beta_median_and_wrong_lineage_do_not_authorize(self):
        for field in ('median','lineage','missing'):
            state=copy.deepcopy(self.f.state);q=state['teacher_phase']['weight']['gradient_qualification']
            if field=='median':q.update(raw_median=2.,beta=.125)
            elif field=='lineage':q['observations'][0]['context']['model_id']='f'*64
            else:q['passed']=False
            state['teacher_phase']['weight']['gradient_qualification']=Q.seal(q)
            with self.subTest(field=field),self.assertRaises(ValueError):A.authorize(state,self.f.after['config'])

    def test_unqualified_activation_stage_skips_and_source_changes_remain_blocked(self):
        f=self.f
        state=copy.deepcopy(f.state);state['teacher_phase']['weight'].pop('gradient_qualification')
        with self.assertRaisesRegex(ValueError,'four qualified'):C.approved(state,f.after)
        for field,value in [('stage','B'),('w',.05),('scale',.1)]:
            bad=copy.deepcopy(f.after)
            options=bad['config']['teacher_phase']['search_value']
            if field=='stage':options['stage']=value
            elif field=='w':options['blend']['w']=value
            else:options['trunk']['scale']=value
            self.assertFalse(A.is_on(bad['config']));self.assertFalse(C.signature_transition(f.before,bad))
        with patch.object(R,'implementation_id',return_value='f'*64):self.assertFalse(C.approved(f.state,f.after))

    def test_explicit_not_ready_blocks_activation_but_defers_an_existing_on_run(self):
        f=self.f;state=copy.deepcopy(f.state);state['teacher_phase']['readiness']['ready']=False
        with self.assertRaisesRegex(ValueError,'utility readiness'):A.authorize(state,f.after['config'])
        # Existing valid labels/beta remain bound; a missing new critic fit
        # defers collection instead of terminating ordinary training.
        state['signature']=f.after
        self.assertEqual(A.authorize(state,f.after['config'])['beta'],0.16110139609854138)

    def test_real_loader_activation_commit_and_restart_keep_full_state_and_pending(self):
        f=self.f;model,raw,changed=f.load();self.assertTrue(changed);f.unchanged()
        optimizer=torch.optim.AdamW(ordinary_parameters(model));resume_optimizer(raw,optimizer)
        self.assertTrue(_same_payload(optimizer.state_dict(),f.optimizer.state_dict()))
        with patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value={'receipt':{'existing':'runtime'}}):
            updated=resume.commit_transition(f.run,f.state,f.after)
        with patch.object(B,'checked_pin',return_value={'passed':True,'implementation':R.QUALIFIED}):
            B.validate_state(updated,raw,check_runtime=False)
        again,retry,revised=f.load(state=updated)
        self.assertFalse(revised);self.assertTrue(_same_payload(model.state_dict(),again.state_dict()))
        self.assertEqual(updated['teacher_phase'],f.phase);self.assertEqual(f.pending_bytes(),f.files)
        self.assertEqual(updated['native_backend']['teacher']['beta'],0.16110139609854138)
        self.assertEqual(retry['training']['search_value'],raw['training']['search_value'])

    def test_constructor_retains_collection_identity_and_frozen_jobs_until_post_ppg(self):
        f=self.f;model,raw,_=f.load();identity=model_id(model);rng=torch.get_rng_state().clone()
        phase=TeacherPhase(ROOT,f.run,f.after['config'],f.state,model,raw)
        self.assertFalse(hasattr(model,'search_value_head'));self.assertIsNone(phase.search_value.phase.companion)
        self.assertEqual(model_id(model),identity);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        args=dict(model_id=identity,export_hash=f.state['history'][-1]['sha256'],epoch=1261,ready=True)
        old=plan_jobs(f.before['config'],f.state,**args);new=plan_jobs(f.after['config'],f.state,**args)
        self.assertEqual(old,new);self.assertEqual(len(new[0]),32)
        self.assertEqual(C.collection_signature(f.before),C.collection_signature(f.after))
        phase.begin(f.state,model,export_hash=args['export_hash'],epoch=1261)
        self.assertTrue(all(j['expected']['model_id']==f.state['history'][-1]['model_id'] for j in phase.jobs))

    def test_post_ppg_head_trains_32_steps_without_main_weight_adam_rng_changes_and_reloads(self):
        from test_b045 import value_fixture
        f=self.f;model,raw,_=f.load();optimizer=torch.optim.AdamW(ordinary_parameters(model));resume_optimizer(raw,optimizer)
        phase=SearchValueIteration(f.run,f.after['config'],f.state,model,raw)
        self.assertFalse(hasattr(model,'search_value_head'))
        row,_,_=value_fixture(model,generation=1261)
        row.update(reward_id=phase.phase.reward_id,terminal_utility=row['u'])
        phase.phase.buffer.publish(1261,'train',[row],game_id=row['game_id'])
        weights={k:v.clone() for k,v in model.state_dict().items()};adam=copy.deepcopy(optimizer.state_dict())
        rng=torch.get_rng_state().clone()
        phase.prepare_ppg(None,1262,annealing=A.annealing(f.after['config'],f.state))
        result=phase.finish_ppg();self.assertTrue(result['updated']);self.assertEqual(result['update']['steps'],32)
        self.assertEqual(phase.phase.machine.state['w'],0);self.assertEqual(phase.phase.machine.state['scale'],0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in weights.items():self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        self.assertTrue(_same_payload(adam,optimizer.state_dict()))
        head={id(p) for p in model.search_value_head.parameters()}
        self.assertFalse(head & {id(p) for g in optimizer.param_groups for p in g['params']})
        payload=phase.payload(1262);path=f.run/'generation-1262.pt'
        save_checkpoint(path,model,optimizer,step=1262,training={'search_value':payload})
        restored,saved=load_checkpoint(path)
        resumed=SearchValueIteration(f.run,f.after['config'],dict(iteration=1262,search_value=stamp(payload),
            history=[dict(generation=1262,model_id=model_id(restored),sha256=sha(path))]),restored,saved)
        self.assertEqual(resumed.phase.companion.steps,32)
        self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()))

    def test_fresh_head_failure_restores_head_adam_and_stage_for_retry(self):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        from test_b045 import value_fixture
        f=self.f;model,raw,_=f.load()
        phase=SearchValueIteration(f.run,f.after['config'],f.state,model,raw)
        phase.prepare_ppg(None,1262,annealing=False)
        before=copy.deepcopy(phase.phase.machine.state)
        row,_,_=value_fixture(model,generation=1261)
        def fail(*args,**kwargs):
            phase.phase.companion.head_only([row],generation=1261,steps=1)
            raise RuntimeError('debug interrupted head step')
        with patch.object(SearchValuePhase,'after_ppg',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):phase.finish_ppg()
        self.assertEqual(phase.phase.companion.steps,0);self.assertEqual(phase.phase.machine.state,before)
        self.assertTrue(all(torch.count_nonzero(p)==0 for p in model.search_value_head[-1].parameters()))

    def test_live_on_barrier_keeps_32_tables_cold_recovery_identity_and_rng(self):
        from brass_ai.rust_backend import lab_shadow as S,lab_teacher as T
        f=self.f;model,raw,_=f.load();state=copy.deepcopy(f.state)
        fixture=f.run/'b01/inputs/fixtures.json'
        write_json(fixture,{'roots':[{'view':{'context':{'phase':'action'}},'expected':{'choices':[1,2],'state':{}}}]})
        build={'binary':{'path':'fake'},'core_source':'qualified','bootstrap':{'binary':{'path':'fake'}},'receipt':{}}
        updated=copy.deepcopy(state);updated['signature']=f.after
        with patch.object(S,'teacher_build',return_value=build):C.rebind_backend(state,updated)
        state=updated;phase=TeacherPhase(ROOT,f.run,f.after['config'],state,model,raw)
        jobs,_=plan_jobs(f.after['config'],state,model_id=model_id(model),export_hash=state['history'][-1]['sha256'],
            epoch=1261,ready=True,native_source=S.source_fingerprint())
        def admit(results,model,*,verifier=None):
            if verifier is not None:
                for result,job in zip(results,jobs):verifier.take(result,job,model)
            phase.metrics={'tables':{'train':28,'heldout':4}}
        phase.admit_collection=Mock(side_effect=admit)
        def collector(*args,**kwargs):
            worker=SimpleNamespace(verifier=SimpleNamespace(cache={}))
            def take(result,job,model):worker.verifier.cache.pop(result['job_id']);return [],{}
            worker.verifier.take=take
            def collect(batch,*,on_result):
                output=[]
                for job in batch:
                    result=dict(job_id=job['expected']['job_id'],rows=[],metrics={'turns':[{'profile':'full','post_search_ns':0}]})
                    worker.verifier.cache[result['job_id']]={}
                    result['metrics']['shadow_cache_seconds']=on_result(result,job);output.append(result)
                return output
            worker.collect=collect;return worker
        service=SimpleNamespace(metrics={},cpu=SimpleNamespace(capacity=60))
        with patch.object(S,'teacher_build',return_value=build),patch.object(B,'checked_pin',return_value={'b01':{'path':str(f.run/'b01/qualification.json')}}),\
                patch.object(T,'calibrate',side_effect=lambda *a:dict(estimate=float(torch.rand(())))),\
                patch.object(T,'TeacherCollector',side_effect=collector) as native:
            runtime=S.ShadowCollector(ROOT,f.run,state);rng=torch.get_rng_state().clone()
            runtime.collect(phase,state,model,service,f.run/'iteration-1261')
            self.assertEqual(phase.metrics['new_tables'],32);self.assertFalse(phase.metrics['shadow'])
            runtime.collect(phase,state,model,service,f.run/'iteration-1261')
            self.assertEqual(phase.metrics['cached_tables'],32);self.assertEqual(native.call_count,1)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(hasattr(model,'search_value_head'))

    def test_on_policy_term_uses_fixed_beta_and_never_updates_critic_or_search_head(self):
        from test_b025 import example
        from brass_ai.learning.search_value import install
        f=self.f;install(f.model)
        row,_=example(f.model);term=TeacherTerm(f.model,f.after['config']['teacher_phase'],[(0,row)],
            seed=7,generation=1261,state=f.phase['weight'])
        loss=-forward_policy(f.model,[row]).log_softmax(-1)[0,1]
        actor=TrunkGradients(f.model);actor.add_actor(loss);loss.backward()
        before={n:p.grad.clone() if p.grad is not None else None for n,p in f.model.named_parameters()}
        term.apply(f.model,actor,1);self.assertEqual(term.beta_used,0.16110139609854138)
        self.assertTrue(any(before[n] is not None and not torch.equal(before[n],p.grad)
            for n,p in f.model.named_parameters() if n.startswith('state_encoder.')))
        for n,p in f.model.named_parameters():
            if n.startswith(('utility_head.','outcomes.','search_value_head.')):
                self.assertTrue(p.grad is None if before[n] is None else torch.equal(p.grad,before[n]),n)
        self.assertEqual(term.finish()['gradient_qualification'],f.phase['weight']['gradient_qualification'])

    def launcher(self):
        spec=importlib.util.spec_from_file_location('on_launcher',ROOT/'scripts/continue-rust-shadow.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

    def test_preparation_keeps_pending_and_primary_checkpoint_and_never_runs_extra_iteration(self):
        f=self.f;launcher=self.launcher()
        destination=ROOT/'configs/step5'/(f.run.name+'-rust-teacher-on-stage-a-share025.json')
        self.addCleanup(destination.unlink,missing_ok=True)
        with patch.object(B,'validate_state'),patch('brass_ai.rust_backend.lab_shadow.teacher_build'):
            state,config,remaining=launcher.prepare(ROOT,str(f.run),1261,1,mode='on')
        self.assertEqual(remaining,1);self.assertTrue(A.is_on(read(config)));f.unchanged()
        self.assertEqual(state['history'][-1]['sha256'],sha(f.checkpoint))

    def test_public_on_mode_command_requests_one_iteration_with_imitation_and_exports_receipt(self):
        import sys,os
        f=self.f;launcher=self.launcher();config=f.run/'on.json';write_json(config,f.after['config'])
        final=copy.deepcopy(f.state);final.update(iteration=1262,signature=f.after)
        final['history'][-1]=dict(final['history'][-1],generation=1262)
        def train(argv,log):
            self.assertEqual(argv[argv.index('--iterations')+1],'1')
            self.assertEqual(argv[argv.index('--expected-generation')+1],'1261')
            self.assertIn('--imitation-run',argv);self.assertIn('--resume-reviewed-source',argv)
            write_json(f.run/'state.json',final)
        with patch.object(launcher,'ROOT',f.run),patch.object(launcher,'prepare',return_value=(f.state,config,1)),\
                patch.object(launcher,'run_training',side_effect=train) as invoke,patch('rust_lab_results.export',return_value=f.run/'results'),\
                patch.object(sys,'argv',['continue-rust-teacher.py','--run',str(f.run),'--expected-generation','1261']),\
                patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            launcher.main(default_mode='on');invoke.assert_called_once()
        report=read(next((f.run/'artifacts/rust-backend').glob('*/qualification.json')))
        self.assertTrue(report['passed']);self.assertEqual(report['teacher_mode'],'on')


if __name__=='__main__':unittest.main()
