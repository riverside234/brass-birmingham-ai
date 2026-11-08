"""Reviewed branch-off/full shadow wiring and fixed-beta debug regressions."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.recovery import teacher_gradient_qualification as Q
from brass_ai.recovery.teacher_term import TeacherTerm,forward_policy
from brass_ai.recovery.value_preservation import TrunkGradients
from brass_ai.rust_backend import lab_shadow_config as C,lab_replay_compat as R
from brass_ai.rust_backend.lab_shadow import TableCache
from brass_ai.step5.league import write_json


class ShadowConfig(unittest.TestCase):
    def setUp(self):
        self.base=read(ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json')
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup);self.run=Path(self.tmp.name)

    def test_all_requested_targets_validate_and_only_requested_recipe_fields_change(self):
        from brass_ai.recovery.config import validate_config
        for target in Q.TARGETS:
            config=C.configure(self.base,target);validate_config(config);self.assertTrue(C.is_shadow(config))
            self.assertEqual(config['branching']['rate_per_decision'],0.)
            self.assertEqual(config['teacher_phase']['tables'],{'train':28,'held_out':4})
            self.assertEqual(config['teacher_phase']['mode'],'shadow')
            self.assertEqual(config['teacher_phase']['term']['share_target'],target)
            self.assertEqual(config['teacher_phase']['term']['shadow_iterations'],4)
            self.assertFalse(config['teacher_value']['enabled']);self.assertFalse(config['teacher_enabled'])
            self.assertEqual({k:v for k,v in config.items() if k not in ('branching','teacher_value','teacher_phase')},
                {k:v for k,v in self.base.items() if k not in ('branching','teacher_value','teacher_phase')})
        bad=C.configure(self.base);bad['learning_rate']*=2;self.assertFalse(C.is_shadow(bad))
        bad=C.configure(self.base);bad['teacher_phase']['mode']='on';self.assertFalse(C.is_shadow(bad))
        for count in (2.0,4.0,True,3):
            bad=C.configure(self.base);bad['teacher_phase']['term']['shadow_iterations']=count
            self.assertFalse(C.is_shadow(bad))
        with self.assertRaises(ValueError):C.configure(self.base,.3)

    def test_reviewed_source_mapping_keeps_new_config_and_requires_explicit_resume_flag(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.learning.contracts import compatibility
        before=dict(config=self.base,implementation=R.QUALIFIED,compatibility=compatibility(),bindings='same')
        state=dict(format='step5-recovery-league/2.2.0',iteration=1257,signature=before,
            games_done=402240,native_backend=dict(mode='rust',implementation=R.QUALIFIED))
        boost=self.base.get('legacy_boost')
        if boost:state['legacy_boost']=dict(start_games=0,until_games=sum(s['games'] for s in boost['stages']),config=digest(boost))
        incoming=dict(before,config=C.configure(self.base),implementation=R.execution()['implementation'])
        target=R.training_signature(state,incoming)
        self.assertEqual(target['implementation'],R.QUALIFIED);self.assertEqual(target['config'],incoming['config'])
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,target)
        self.assertTrue(check_signature(state,target,reviewed=True))
        bad=copy.deepcopy(incoming);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):R.training_signature(state,bad)
        with patch.object(R,'implementation_id',return_value='f'*64):
            with self.assertRaises(ValueError):R.training_signature(state,incoming)

    def test_pending_archive_preserves_every_byte_and_old_state_and_never_overwrites(self):
        state=dict(iteration=1257,signature={'config':self.base},history=[dict(generation=1257,sha256='a'*64)])
        path=self.run/'iteration-1257/native-ordinary/replay/frozen.gz';path.parent.mkdir(parents=True);path.write_bytes(b'sealed original')
        receipt=C.archive_pending(ROOT,self.run,state)
        record=read(receipt);directory=Path(receipt).parent
        saved=directory/'iteration-1257/native-ordinary/replay/frozen.gz'
        self.assertEqual(saved.read_bytes(),b'sealed original');self.assertEqual(record['files'][0]['sha256'],sha(saved))
        self.assertEqual(read(directory/'state-before.json'),state);self.assertFalse((self.run/'iteration-1257').exists())
        self.assertIsNone(C.archive_pending(ROOT,self.run,state))
        path.parent.mkdir(parents=True);path.write_bytes(b'sealed original')
        with self.assertRaisesRegex(ValueError,'overwrite'):C.archive_pending(ROOT,self.run,state)
        self.assertEqual(path.read_bytes(),b'sealed original')

    def test_uncommitted_checkpoint_prevents_archive_and_generation_change(self):
        state=dict(iteration=1257);pending=self.run/'iteration-1257';pending.mkdir()
        checkpoint=self.run/'generation-1258.pt.partial';checkpoint.write_bytes(b'pending checkpoint')
        with self.assertRaisesRegex(ValueError,'Uncommitted checkpoint'):C.archive_pending(ROOT,self.run,state)
        self.assertTrue(pending.exists());self.assertEqual(checkpoint.read_bytes(),b'pending checkpoint')

    def test_backend_rebind_preserves_original_qualification_and_exact_checkpoint_switch(self):
        from brass_ai.rust_backend import lab_backend as B
        old=dict(contract=B.CONTRACT,mode='rust',implementation=R.QUALIFIED,
            config_sha256=digest(self.base),qualification={'original':'pin'},build={'original':'binary'},rollback={})
        before=dict(iteration=1257,history=[{'generation':1257}],signature={'config':self.base,'implementation':R.QUALIFIED},native_backend=old)
        updated=copy.deepcopy(before);updated['signature']['config']=C.configure(self.base)
        with patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value={'receipt':{'teacher':'qualified'}}):C.rebind_backend(before,updated)
        self.assertEqual(updated['native_backend']['qualification'],old['qualification'])
        self.assertEqual(updated['native_backend']['build'],old['build'])
        self.assertEqual(updated['native_backend']['config_sha256'],digest(updated['signature']['config']))
        self.assertEqual(updated['native_backend_switch']['before'],old)
        self.assertEqual(updated['native_backend_switch']['checkpoint'],before['history'][-1])

    def test_continuation_helper_preserves_partial_work_and_never_starts_extra_generations(self):
        spec=importlib.util.spec_from_file_location('shadow_continue',ROOT/'scripts/continue-rust-shadow.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        root=self.run;run=root/'artifacts/run';run.mkdir(parents=True)
        checkpoint=run/'generation-1257.pt';checkpoint.write_bytes(b'checkpoint bytes')
        state=dict(iteration=1257,history=[dict(generation=1257,path='artifacts/run/generation-1257.pt',sha256=sha(checkpoint))],
            signature=dict(config=self.base,implementation=R.QUALIFIED,imitation_source={'bc_run':'artifacts/il'}),native_backend={})
        write_json(run/'state.json',state);pending=run/'iteration-1257/native-ordinary/part';pending.parent.mkdir(parents=True);pending.write_bytes(b'old')
        with patch('brass_ai.rust_backend.lab_backend.validate_state'),patch('brass_ai.rust_backend.lab_shadow.teacher_build'):
            initial,config,left=launcher.prepare(root,'artifacts/run',1257,2)
            self.assertEqual(left,2);self.assertEqual(read(run/'state.json'),state);self.assertFalse(pending.exists())
            state['signature']['config']=read(config)
            for generation,expected_left in ((1258,1),(1259,0)):
                checkpoint=run/f'generation-{generation}.pt';checkpoint.write_bytes(b'next')
                state['iteration']=generation;state['history'].append(dict(generation=generation,
                    path=checkpoint.relative_to(root).as_posix(),sha256=sha(checkpoint)))
                write_json(run/'state.json',state)
                _,_,left=launcher.prepare(root,'artifacts/run',1257,2);self.assertEqual(left,expected_left)
            state['iteration']=1260;write_json(run/'state.json',state)
            with self.assertRaisesRegex(ValueError,'outside'):launcher.prepare(root,'artifacts/run',1257,2)

    def test_actual_g1257_readiness_uses_last_nonnull_measured_fit_without_null_exception(self):
        from brass_ai.rust_backend.lab_shadow import readiness_seed
        path=ROOT/'results/joint7-v11-2-g1290-trained-restart-g1257-20261107T170030753211Z/state.json'
        if not path.exists():self.skipTest('Pull the pinned g1257 results for this evidence check')
        state=read(path);seed=readiness_seed(ROOT,state)
        self.assertTrue(seed['receipt']['ready']);self.assertEqual(seed['fit_checkpoint']['generation'],1254)
        self.assertTrue(all(r['receipt'] is None for r in seed['intervening_receipts']))
        state['teacher_phase']={'readiness':dict(seed['receipt'],ready=False)}
        self.assertIsNone(readiness_seed(ROOT,state))

    def test_public_scheduler_accepts_exact_shadow_transition_and_keeps_generation_guard(self):
        from brass_ai.recovery import entropy_schedule as S,league
        from brass_ai.learning.contracts import compatibility
        schedule_config=copy.deepcopy(self.base['update'][S.KEY])
        schedule=S.seal(dict(contract=S.STATE,config=schedule_config,started_at=schedule_config['start_generation'],
            progress=100,held=0,last_generation=1257,insolvency=copy.deepcopy(self.base['insolvency']),
            insolvency_steps_applied=S.step_count(schedule_config,self.base['insolvency']),reset_count=0))
        state=dict(format='step5-recovery-league/2.2.0',iteration=1257,games_done=402240,
            signature=dict(config=self.base,implementation=R.QUALIFIED,compatibility=compatibility()),
            native_backend=dict(mode='rust',implementation=R.QUALIFIED),exploration={S.KEY:schedule})
        write_json(self.run/'state.json',state);after=C.configure(self.base);calls=[]
        class StopBeforeModel(Exception):pass
        def stop(root,run,config,**kwargs):calls.append((config,kwargs));raise StopBeforeModel()
        with patch.object(league,'_train',side_effect=stop):
            with self.assertRaises(StopBeforeModel):league.train(ROOT,str(self.run),after,
                expected_generation=1257,iterations=2,resume_reviewed_source=True)
        self.assertEqual(calls[0][0],after);self.assertEqual(calls[0][1]['iterations'],2)
        self.assertEqual(read(self.run/'state.json'),state)

    def test_public_continuation_command_launches_only_requested_iterations_and_exports_fixed_beta(self):
        import sys
        import os
        spec=importlib.util.spec_from_file_location('shadow_main',ROOT/'scripts/continue-rust-shadow.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        run=self.run/'artifacts/run';run.mkdir(parents=True)
        first=dict(iteration=1257,signature={'imitation_source':{'bc_run':'artifacts/il'}})
        final=dict(iteration=1261,history=[dict(generation=1261,sha256='a'*64)],
            signature={'config':{'teacher_phase':{'mode':'shadow'}}},
            teacher_phase={'weight':{'gradient_qualification':{'passed':True,'beta':.125}}})
        write_json(self.run/'config.json',C.configure(self.base))
        def train(argv,**kwargs):
            self.assertIn('--resume-reviewed-source',argv)
            self.assertEqual(argv[argv.index('--iterations')+1],'4')
            self.assertEqual(argv[argv.index('--expected-generation')+1],'1257')
            write_json(run/'state.json',final)
        with patch.object(launcher,'ROOT',self.run),patch.object(launcher,'prepare',return_value=(first,self.run/'config.json',4)),\
                patch.object(launcher,'run_training',side_effect=lambda argv,log:train(argv)) as invoke,\
                patch('rust_lab_results.export',return_value=self.run/'results') as export,\
                patch.object(sys,'argv',['continue-rust-shadow.py','--run','artifacts/run','--expected-generation','1257']),\
                patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            launcher.main();invoke.assert_called_once();export.assert_called_once()
        report=read(next((self.run/'artifacts/rust-backend').glob('*/qualification.json')))
        self.assertTrue(report['training_passed']);self.assertTrue(report['passed']);self.assertEqual(report['teacher_mode'],'shadow')
        self.assertEqual(report['gradient_qualification']['beta'],.125)

    def test_four_iteration_upgrade_keeps_frozen_jobs_pending_work_and_completed_measurement(self):
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        before=C.configure(self.base);before['teacher_phase']['term']['shadow_iterations']=2
        after=C.configure(self.base)
        self.assertTrue(C.is_shadow(before));self.assertTrue(C.qualification_upgrade(before,after))
        a=dict(config=before,implementation=R.QUALIFIED);b=dict(a,config=after)
        self.assertTrue(C.signature_transition(a,b));self.assertFalse(C.signature_transition(b,a))
        self.assertEqual(C.collection_signature(a),C.collection_signature(b))
        self.assertEqual(buffer_identity(before),buffer_identity(after))
        args=dict(model_id='frozen',export_hash='a'*64,epoch=1258,ready=True)
        self.assertEqual(plan_jobs(before,{'iteration':1258},**args),plan_jobs(after,{'iteration':1258},**args))
        spec=importlib.util.spec_from_file_location('shadow_upgrade',ROOT/'scripts/continue-rust-shadow.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        run=self.run/'artifacts/run';run.mkdir(parents=True)
        checkpoint=run/'generation-1258.pt';checkpoint.write_bytes(b'completed full checkpoint')
        state=dict(iteration=1258,history=[dict(generation=1258,path=checkpoint.relative_to(self.run).as_posix(),sha256=sha(checkpoint))],
            signature=a,teacher_phase={'weight':{'gradient_qualification':{'observations':[{'generation':1257}]}}})
        write_json(run/'state.json',state)
        pending=run/'iteration-1258/teacher-shadow/table.json.gz';pending.parent.mkdir(parents=True);pending.write_bytes(b'completed table')
        ordinary=run/'iteration-1258/native-ordinary/game.gz';ordinary.parent.mkdir(parents=True);ordinary.write_bytes(b'completed ordinary family')
        config_path=self.run/'configs/step5/run-rust-shadow32-no-branch-share025.json'
        write_json(config_path,before)
        with patch('brass_ai.rust_backend.lab_backend.validate_state'),patch('brass_ai.rust_backend.lab_shadow.teacher_build'),\
                patch.object(C,'archive_pending',side_effect=AssertionError('Diagnostic upgrade must not archive')):
            restored,output,remaining=launcher.prepare(self.run,'artifacts/run',1257,4)
        self.assertEqual(remaining,3);self.assertEqual(restored,state);self.assertEqual(read(run/'state.json'),state)
        self.assertEqual(read(output)['teacher_phase']['term']['shadow_iterations'],4)
        self.assertEqual(pending.read_bytes(),b'completed table');self.assertEqual(ordinary.read_bytes(),b'completed ordinary family')

    def test_routed_native_pool_uses_preserved_collection_binding_with_new_measurement_config(self):
        from brass_ai.rust_backend import lab_backend as B,lab_pool as P,lab_perf_runtime as RUNTIME
        after=C.configure(self.base);before=copy.deepcopy(after);before['teacher_phase']['term']['shadow_iterations']=2
        signature=dict(config=after,implementation=R.QUALIFIED)
        expected=digest(dict(signature,config=before))
        state={'signature':signature};jobs=[{'provenance':{'purpose':'training','spec':expected}}]
        with patch('brass_ai.recovery.execution.ActorPool'),patch.object(B,'validate_state',return_value={'mode':'rust','build':{}}),\
                patch.object(B,'check_build',return_value={}),patch.object(RUNTIME,'resolve',return_value={}),patch.object(P,'NativePool') as native:
            pool=B.RoutedPool(ROOT,'host',Mock(),after,state)
            pool.collect(jobs)
            args=native.call_args.args
            self.assertEqual(args[4],after);self.assertEqual(args[5],expected)
            native.return_value.__enter__.return_value.collect.assert_called_once_with(jobs,on_records=None)


class PendingValues(unittest.TestCase):
    def test_expected_staged_bytes_never_enter_receipts_without_republication_and_foreign_files_fail(self):
        from brass_ai.recovery.search_value_rows import ValueBuffer
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            run=Path(temp);values=run/'search-values';values.mkdir()
            game='teacher/game';name='001257-train-'+digest(game)+'.json.gz';path=values/name
            path.write_bytes(b'unvalidated stage')
            buffer=ValueBuffer(values,'reward',pending_files=[name],pending_directory=run/'iteration-1257/teacher-shadow/value-partials')
            self.assertEqual(buffer.payload()['receipts'],[])
            row=dict(generation=1257,split='train',reward_id='reward',game_id=game,index=0)
            with self.assertRaisesRegex(ValueError,'Uncommitted value buffer differs'):buffer.publish(1257,'train',[row],game_id=game)
            self.assertEqual(buffer.payload()['receipts'],[]);self.assertEqual(path.read_bytes(),b'unvalidated stage')
            (values/'foreign.json.gz').write_bytes(b'foreign')
            with self.assertRaisesRegex(ValueError,'Unknown'):ValueBuffer(values,'reward',pending_files=[name])

    def test_interrupted_partial_is_preserved_outside_value_buffer_and_can_be_republished(self):
        from brass_ai.recovery.search_value_rows import ValueBuffer
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            run=Path(temp);values=run/'search-values';values.mkdir()
            game='teacher/game';name='001257-train-'+digest(game)+'.json.gz';partial=(values/name).with_suffix('.partial')
            partial.write_bytes(b'incomplete original bytes')
            buffer=ValueBuffer(values,'reward',pending_files=[name],pending_directory=run/'iteration-1257/teacher-shadow/value-partials')
            self.assertFalse(partial.exists());self.assertEqual(Path(buffer.recovered_partials[0]).read_bytes(),b'incomplete original bytes')
            self.assertEqual(buffer.payload()['receipts'],[])
            row=dict(generation=1257,split='train',reward_id='reward',game_id=game,index=0)
            receipt=buffer.publish(1257,'train',[row],game_id=game)
            restored=ValueBuffer(values,'reward',receipts=[receipt]);self.assertEqual(restored._read(receipt),[row])


class FixedBeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        from test_b025 import config,model_fixture,example
        from brass_ai.recovery.search_value_config import adopt
        model=model_fixture();row,_=example(model)
        options=adopt(config())['teacher_phase'];options['mode']='shadow'
        context=dict(tables={'train':28,'heldout':4},labels_by_phase={p:1 for p in
            ('canal_early','canal_late','rail_early','rail_late')},model_id='frozen',export_hash='a'*64)
        return model,row,options,context

    def measure(self,model,row,options,context,generation,state=None):
        model.zero_grad(set_to_none=True)
        term=TeacherTerm(model,options,[(0,row)],seed=5,generation=generation,state=state)
        term.qualification_context=context
        logits=forward_policy(model,[row]);loss=-logits.log_softmax(-1)[0,1]
        actor=TrunkGradients(model);actor.add_actor(loss);loss.backward()
        before=[p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
        weights=[p.detach().clone() for p in model.parameters()];rng=torch.get_rng_state().clone()
        term.apply(model,actor,1)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for p,w,g in zip(model.parameters(),weights,before):
            self.assertTrue(torch.equal(p,w));self.assertTrue(p.grad is None if g is None else torch.equal(p.grad,g))
        return term,term.finish()

    def test_four_full_shadow_gradients_qualify_fixed_beta_and_exact_restart(self):
        model,row,options,context=self.fixture()
        first,state=self.measure(model,row,options,context,1257)
        self.assertFalse(state['gradient_qualification']['passed'])
        second,state=self.measure(model,row,options,context,1258,state)
        self.assertFalse(state['gradient_qualification']['passed'])
        _,state=self.measure(model,row,options,context,1259,state)
        self.assertFalse(state['gradient_qualification']['passed'])
        _,state=self.measure(model,row,options,context,1260,state)
        q=state['gradient_qualification'];self.assertTrue(q['passed']);self.assertEqual(q['qualified_generation'],1261)
        self.assertEqual(q['required_iterations'],4);self.assertEqual(len(q['observations']),4)
        raw=[x for o in q['observations'] for x in o['raw_shares']]
        import statistics
        self.assertEqual(q['beta'],Q.fixed_beta(.25,statistics.median(raw)))
        # A later, very different shadow ratio is monitoring, not recalibration.
        later=TeacherTerm(model,options,[],seed=5,generation=1261,state=state)
        later.steps=[dict(raw_share=100.,share=100.)]
        same=later.finish();self.assertEqual(same['gradient_qualification'],q);self.assertEqual(same['beta'],q['beta'])
        on=copy.deepcopy(options);on['mode']='on'
        ready=TeacherTerm(model,on,[],seed=5,generation=1262,state=same)
        self.assertEqual(ready.beta_used,q['beta'])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            path=Path(temp)/'weight.pt';torch.save(state,path)
            loaded=torch.load(path,weights_only=True)
            self.assertEqual(TeacherTerm(model,on,[],seed=5,generation=1262,state=loaded).beta_used,q['beta'])

    def test_manual_targets_rescale_only_from_frozen_measurements_and_receipt_tampering_fails(self):
        model,row,options,context=self.fixture();_,state=self.measure(model,row,options,context,1257)
        for generation in (1258,1259,1260):_,state=self.measure(model,row,options,context,generation,state)
        q=state['gradient_qualification']
        for target in (.4,.5):
            changed=copy.deepcopy(options);changed['term']['share_target']=target
            term=TeacherTerm(model,changed,[],seed=5,generation=1261,state=state);after=term.finish()
            self.assertEqual(after['beta'],Q.fixed_beta(target,q['raw_median']))
            self.assertEqual(after['gradient_qualification']['observations'],q['observations'])
        bad=copy.deepcopy(state);bad['gradient_qualification']['beta']*=2
        with self.assertRaisesRegex(ValueError,'qualification changed'):TeacherTerm(model,options,[],seed=5,generation=1259,state=bad)

    def test_existing_two_iteration_receipts_extend_without_losing_observations_or_premature_beta(self):
        model,row,options,context=self.fixture();old=copy.deepcopy(options);old['term']['shadow_iterations']=2
        _,state=self.measure(model,row,old,context,1257)
        legacy=copy.deepcopy(state);legacy['gradient_qualification'].pop('required_iterations')
        legacy['gradient_qualification']=Q.seal(legacy['gradient_qualification'])
        # Simulate the actual g1258 checkpoint, written before a window field existed.
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            checkpoint=Path(temp)/'checkpoint.pt';torch.save(legacy,checkpoint)
            loaded=torch.load(checkpoint,weights_only=True)
        _,continued=self.measure(model,row,options,context,1258,loaded)
        self.assertEqual(continued['gradient_qualification']['observations'][0],legacy['gradient_qualification']['observations'][0])
        self.assertFalse(continued['gradient_qualification']['passed'])
        _,passed2=self.measure(model,row,old,context,1258,state)
        self.assertTrue(passed2['gradient_qualification']['passed'])
        pending=TeacherTerm(model,options,[],seed=5,generation=1259,state=passed2)
        self.assertFalse(pending.calibrated);self.assertEqual(pending.beta,1.)
        self.assertEqual(pending.qualification['observations'],passed2['gradient_qualification']['observations'])
        on=copy.deepcopy(options);on['mode']='on'
        with self.assertRaisesRegex(ValueError,'4 full shadow'):TeacherTerm(model,on,[],seed=5,generation=1259,state=passed2)
        for generation in (1259,1260):_,continued=self.measure(model,row,options,context,generation,continued)
        self.assertTrue(continued['gradient_qualification']['passed'])

    def test_incomplete_phase_or_nonconsecutive_or_nonfinite_measurements_never_qualify(self):
        model,row,options,context=self.fixture();_,state=self.measure(model,row,options,context,1257)
        _,gap=self.measure(model,row,options,context,1259,state);self.assertFalse(gap['gradient_qualification']['passed'])
        term=TeacherTerm(model,options,[],seed=5,generation=1258,state=state)
        term.qualification_context=dict(context,tables={'train':1,'heldout':1});term.steps=[dict(raw_share=1.,share=1.,cosine=0.,actor_norm=1.,teacher_norm=1.)]
        self.assertFalse(term.finish()['gradient_qualification']['passed'])
        term.steps=[dict(raw_share=float('nan'),share=1.)]
        with self.assertRaisesRegex(ValueError,'Nonfinite'):term.finish()


class Cache(unittest.TestCase):
    def test_keyboard_interrupt_signals_teacher_cancel_before_pool_join_and_releases_lease(self):
        from brass_ai.rust_backend import lab_teacher as T
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.learning.batching import model_id
        from test_b025 import config,model_fixture
        model=model_fixture();c=config();c['teacher_phase']['mode']='shadow'
        state={'iteration':10};job=plan_jobs(c,state,model_id=model_id(model),export_hash='a'*64,epoch=10,
            ready=True,native_source=T.source_fingerprint())[0][0]
        job['readiness']={'ready':True,'reward_id':job['expected']['reward_id']}
        endpoint=Mock();lease=Mock();lease.acquire.return_value=endpoint
        service=SimpleNamespace(device='cpu',channels=Mock(return_value=lease),release_channels=Mock())
        calibration=dict(contract=T.CONTRACT,source=T.source_fingerprint(),core_source='core',model_id=model_id(model),
            export_hash='a'*64,reward_id=job['expected']['reward_id'],device='cpu',profile=digest(job['search']),samples=[{}],estimate_ns=1)
        collector=T.TeacherCollector('unused','core',['unused'],service,calibration,lanes=1,model=model,
            history=[dict(generation=10,model_id=model_id(model))])
        case=self
        class Pool:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def map(self,*args):raise KeyboardInterrupt('debug Ctrl-C')
            def __exit__(self,*args):case.assertTrue(collector.cancel.is_set())
        with patch.object(T,'ThreadPoolExecutor',Pool):
            with self.assertRaisesRegex(KeyboardInterrupt,'debug Ctrl-C'):collector.collect([job])
        endpoint.close.assert_called_once();service.release_channels.assert_called_once_with(lease)

    def test_sealed_results_recover_exact_values_and_reject_stale_or_changed_tables(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            run=Path(temp);job=dict(expected={'job_id':'a'*64},generation=1257)
            cache=TableCache(run,run/'iteration-1257/teacher-shadow',[job])
            result=dict(job_id='a'*64,rows=[{'x':.1234567890123456}],metrics={'turns':[{'post_search_ns':0,'profile':'full'}]})
            spent=cache.publish(result,job);restored=cache.load(job)
            self.assertGreaterEqual(spent,0);self.assertEqual(restored['rows'],result['rows'])
            path,_=cache.paths(job);raw=path.read_bytes();path.write_bytes(raw+b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):cache.load(job)
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'Unknown'):cache.load(dict(job,generation=1258))

    def test_late_verified_publication_is_sealed_without_verification_deadline(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            run=Path(temp);job={'expected':{'job_id':'a'*64}};cache=TableCache(run,run/'shadow',[job])
            result=dict(job_id='a'*64,rows=[],metrics={'turns':[{'post_search_ns':2_000_000_000,'profile':'full'}]})
            cache.publish(result,job)
            self.assertEqual(cache.load(job)['job_id'],result['job_id'])
            self.assertFalse(list((run/'shadow').glob('*.overrun')))

    def test_32_table_post_accounting_charges_each_table_and_its_own_split(self):
        from brass_ai.recovery.teacher_phase import TeacherPhase
        phase=TeacherPhase.__new__(TeacherPhase);phase.generation=1257;phase.reward_id='reward';phase.jobs=[]
        results=[];rows={}
        for i in range(32):
            key=str(i);split='train' if i<28 else 'heldout'
            phase.jobs.append(dict(expected={'job_id':key,'model_id':'model'},split=split,
                reservations={'0':{'canal_early':1}}))
            label=dict(kind='root',phase='canal_early',seat=0,completed_trajectories=1,effective_k=8,
                completed_rounds=1,deadline_hit=False,overrun_seconds=0.,fallback_played=False)
            rows[key]=SimpleNamespace(row={'view':{'context':{'round':1}}},label=label)
            results.append(dict(job_id=key,metrics={'reserved_turns':1,'turns':[{'profile':'fast','post_search_ns':300_000_000}]}))
        phase.buffer=Mock();phase.search_value=Mock()
        verifier=Mock();verifier.take.side_effect=lambda result,job,model:([rows[result['job_id']]],{})
        ticks=iter(i*.01 for i in range(1000))
        with patch('brass_ai.recovery.teacher_phase.time.perf_counter',side_effect=lambda:next(ticks)):
            phase.admit_collection(results,None,verifier=verifier)
        self.assertGreater(phase.admission_seconds,.5)
        spent=phase.search_value.admit.call_args.kwargs['post_spent']
        self.assertEqual(set(spent),set(rows));self.assertLess(max(spent.values()),.2)
        self.assertTrue(phase.metrics['root_label_yield_pass'])

    def test_full_32_table_runtime_barrier_reuses_sealed_tables_and_preserves_rng(self):
        from brass_ai.rust_backend import lab_shadow as S,lab_teacher as T,lab_backend as B
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.learning.batching import model_id
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        from brass_ai.recovery import entropy_schedule as E
        config=C.configure(read(ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json'))
        schedule=copy.deepcopy(config['update'][E.KEY])
        state=dict(iteration=1257,signature={'config':config},native_backend={'build':{}},
            history=[dict(generation=1257,sha256='a'*64)],exploration={E.KEY:E.seal(dict(contract=E.STATE,
                config=schedule,started_at=schedule['start_generation'],progress=100,held=0,last_generation=1257,
                insolvency=copy.deepcopy(config['insolvency']),insolvency_steps_applied=E.step_count(schedule,config['insolvency']),reset_count=0))})
        model=UtilityNet(utility_config(ModelConfig.preset('small'),config['teacher_phase']['search']['reward']))
        state['history'][-1]['model_id']=model_id(model)
        jobs,_=plan_jobs(config,state,model_id=model_id(model),export_hash=state['history'][-1]['sha256'],
            epoch=1257,ready=True,native_source=source_fingerprint())
        for job in jobs:job['readiness']={'ready':True,'reward_id':job['expected']['reward_id']}
        self.assertEqual([j['split'] for j in jobs].count('train'),28);self.assertEqual(len(jobs),32)
        phase=SimpleNamespace(options={'mode':'shadow'},receipt={'ready':True},jobs=jobs,search_value=None,metrics={})
        phase.begin=Mock(return_value=jobs)
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
                    result={'job_id':job['expected']['job_id'],'rows':[],
                        'metrics':{'turns':[{'profile':'full','post_search_ns':0}]}}
                    worker.verifier.cache[result['job_id']]={}
                    result['metrics']['shadow_cache_seconds']=on_result(result,job);output.append(result)
                return output
            worker.collect=collect;return worker
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            run=Path(temp);fixture=run/'b01/inputs/fixtures.json'
            write_json(fixture,{'roots':[{'view':{'context':{'phase':'action'}},'expected':{'choices':[1,2],'state':{}}}]})
            build={'binary':{'path':'fake'},'core_source':'qualified','bootstrap':{'binary':{'path':'fake'}}}
            service=SimpleNamespace(metrics={},cpu=SimpleNamespace(capacity=60))
            with patch.object(S,'teacher_build',return_value=build),patch.object(B,'checked_pin',return_value={'b01':{'path':str(run/'b01/qualification.json')}}),\
                    patch.object(T,'calibrate',side_effect=lambda *a:dict(estimate=int(torch.rand(()).item()*100))),\
                    patch.object(T,'TeacherCollector',side_effect=collector) as created:
                runtime=S.ShadowCollector(ROOT,run,state);rng=torch.get_rng_state().clone()
                runtime.collect(phase,state,model,service,run/'iteration-1257')
                self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(phase.metrics['new_tables'],32)
                runtime.collect(phase,state,model,service,run/'iteration-1257')
                self.assertEqual(phase.metrics['cached_tables'],32);self.assertEqual(phase.metrics['new_tables'],0)
                self.assertEqual(created.call_count,1);self.assertEqual(phase.admit_collection.call_count,2)
                self.assertNotIn('verifier',phase.admit_collection.call_args.kwargs)


if __name__=='__main__':unittest.main()
