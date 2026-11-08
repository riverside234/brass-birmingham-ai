"""Short debug regressions: uncapped validation, frozen reuse and failure logs."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import importlib.util
import io
import json
import queue
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
import torch
from brass_ai.bank import read
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.recovery.teacher_phase_config import TIMING
from brass_ai.rust_backend import lab_teacher_finalization as F
from brass_ai.rust_backend.lab_admission import admit_teacher_rows
from test_b025 import model_fixture,example
from test_b045 import value_fixture


def launcher():
    spec=importlib.util.spec_from_file_location('shadow_verification_launcher',ROOT/'scripts/continue-rust-shadow.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        model=model_fixture().eval();_,job,_=value_fixture(model)
        row,_=example(model,job=job,round=job['reservations']['0']['canal_early'])
        history=[dict(generation=10,model_id=model_id(model))]
        verifier=F.TurnVerifier(model,history,[job],threading.Event())
        meta=dict(job=job['expected'],timing=TIMING,index=1,profile='full',
            remaining_ns=900_000_000,native_post_ns=10_000_000,search_ns=800_000_000,
            rows=[copy.deepcopy(row.row)])
        return model,job,history,verifier,meta

    def test_exact_row_feature_and_probability_parity_with_cold_admission(self):
        model,job,history,v,meta=self.fixture()
        expected,dropped=admit_teacher_rows(meta['rows'],job=job,model=model,history=history)
        rng=torch.get_rng_state().clone();mode=model.training
        self.assertEqual(v.verify(meta,time.monotonic()-5)['status'],'ok')
        record=v.cache[job['expected']['job_id']]
        self.assertEqual(digest([r.row for r in record['accepted']]),digest([r.row for r in expected]))
        self.assertEqual(dict(record['rejected']),dropped)
        for actual,reference in zip(record['accepted'],expected,strict=True):
            np.testing.assert_array_equal(actual.encoded.state,reference.encoded.state)
            np.testing.assert_array_equal(actual.encoded.actions,reference.encoded.actions)
        self.assertIs(record['accepted'][0].row,meta['rows'][0])
        self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(model.training,mode)

    def test_full_model_hash_and_source_scan_are_not_repeated_per_turn(self):
        _,job,_,v,meta=self.fixture()
        with patch.object(F,'model_id',side_effect=AssertionError('Repeated weight copy/hash')), \
             patch('brass_ai.rust_backend.lab_teacher_overlay.source_fingerprint',side_effect=AssertionError('Repeated source scan')):
            self.assertEqual(v.verify(meta,time.monotonic())['status'],'ok')
        self.assertEqual(v.cache[job['expected']['job_id']]['last'],1)

    def test_both_profiles_accept_late_native_handoff_with_zero_remaining(self):
        for profile in ('fast','full'):
            _,job,_,v,meta=self.fixture()
            meta.update(profile=profile,remaining_ns=0,native_post_ns=2_000_000_000,
                        search_ns=800_000_000 if profile=='fast' else 4_000_000_000)
            if profile=='fast':meta['rows']=[]
            with self.subTest(profile=profile):
                self.assertEqual(v.verify(meta,time.monotonic()-3)['status'],'ok')
                self.assertIsNone(v.cache[job['expected']['job_id']]['timings'][0]['verification_limit_seconds'])

    def test_handoff_timing_metadata_and_search_limits_still_checked(self):
        for profile in ('fast','full'):
            for change in ({'native_post_ns':-1},{'native_post_ns':True},{'native_post_ns':2**64},
                           {'remaining_ns':-1},{'native_post_ns':2_000_000_000,'remaining_ns':1},
                           {'search_ns':800_000_001 if profile=='fast' else 4_000_000_001}):
                _,_,_,v,meta=self.fixture();meta.update(profile=profile,**change)
                if profile=='fast':meta['rows']=[]
                with self.subTest(profile=profile,change=change),self.assertRaises(ValueError):v.verify(meta,time.monotonic())

    def test_completed_table_handoffs_hash_frozen_model_once_and_detect_later_mutation(self):
        for mutate in (False,True):
            model,job,history,_,meta=self.fixture()
            second=copy.deepcopy(job);second['expected'].update(job_id='other-table',game_id='other-game')
            v=F.TurnVerifier(model,history,[job,second],threading.Event());results=[]
            for frozen_job in (job,second):
                raw=copy.deepcopy(meta['rows'][0]);raw['teacher'].update(frozen_job['expected'])
                rows=[]
                for index in range(1,65):
                    current=dict(meta,job=frozen_job['expected'],index=index,rows=[raw] if index==1 else [])
                    v.verify(current,time.monotonic());rows.extend(current['rows'])
                results.append(dict(job_id=frozen_job['expected']['job_id'],rows=rows,
                    metrics=dict(searched_turns=64,reserved_turns=16)))
            with patch.object(F,'model_id',wraps=model_id) as hashed:
                v.take(results[0],job,model)
                if mutate:
                    with torch.no_grad():next(model.parameters()).add_(.1)
                    with self.assertRaisesRegex(ValueError,'model changed'):v.take(results[1],second,model)
                else:v.take(results[1],second,model)
                self.assertEqual(hashed.call_count,1)

    def test_changed_weights_or_config_cannot_reuse_frozen_binding(self):
        model,_,_,v,meta=self.fixture()
        with torch.no_grad():next(model.parameters()).add_(.1)
        with self.assertRaisesRegex(ValueError,'model changed'):v.verify(meta,time.monotonic())
        self.assertTrue(all(not r['accepted'] for r in v.cache.values()))

    def test_teacher_job_batch_reuses_one_source_fingerprint(self):
        from brass_ai.rust_backend import lab_teacher as T
        model,job,history,_,_=self.fixture()
        second=copy.deepcopy(job);second['expected'].update(job_id='other-table',game_id='other-game')
        producer=T.source_fingerprint();expected=job['expected'];service=Mock();service.device='cpu'
        class StopBeforeWorkers(Exception):pass
        service.channels.side_effect=StopBeforeWorkers()
        calibration=dict(contract=T.CONTRACT,source=producer,core_source='core',
            model_id=expected['model_id'],export_hash=expected['export_hash'],reward_id=expected['reward_id'],
            device='cpu',profile=digest(job['search']),samples=[1],estimate_ns=1000)
        collector=T.TeacherCollector('unused','core',[],service,calibration,model=model,history=history)
        with patch.object(T,'source_fingerprint',return_value=producer) as source,patch.object(F,'TurnVerifier'):
            with self.assertRaises(StopBeforeWorkers):collector.collect([job,second])
        source.assert_called_once()

    def test_row_preparation_runs_outside_shared_model_lock(self):
        _,_,_,v,meta=self.fixture();ready=threading.Event();original=F.prepare_rows
        def prepare(*args,**kwargs):
            result=original(*args,**kwargs);ready.set();return result
        with ThreadPoolExecutor(max_workers=1) as pool,patch.object(F,'prepare_rows',side_effect=prepare):
            v.lock.acquire()
            try:
                future=pool.submit(v.verify,meta,time.monotonic())
                self.assertTrue(ready.wait(5),'Row preparation waited for model lock')
                self.assertFalse(future.done())
            finally:v.lock.release()
            self.assertEqual(future.result(timeout=5)['status'],'ok')

    def test_both_profiles_allow_slow_verification_and_sample_few_turns(self):
        _,job,_,v,meta=self.fixture();meta['rows']=[]
        with patch.object(F.time,'monotonic',return_value=5.) as clock:
            for index in range(1,65):
                profile='fast' if index<3 else 'full'
                current=dict(meta,index=index,profile=profile,remaining_ns=400_000_000)
                self.assertEqual(v.verify(current,0.)['status'],'ok')
        samples=v.cache[job['expected']['job_id']]['timings']
        self.assertEqual([r['index'] for r in samples],[1,3,64])
        self.assertLessEqual(clock.call_count,8)
        self.assertEqual({r['profile'] for r in samples},{'fast','full'})
        self.assertTrue(all(r['verification_limit_seconds'] is None and r['receive_to_ready_seconds']==5. for r in samples))

    def test_cancellation_retires_a_lane_waiting_for_policy_lock(self):
        _,_,_,v,meta=self.fixture();ready=threading.Event();original=F.prepare_rows
        def prepare(*args,**kwargs):
            result=original(*args,**kwargs);ready.set();return result
        with ThreadPoolExecutor(max_workers=1) as pool,patch.object(F,'prepare_rows',side_effect=prepare):
            v.lock.acquire()
            try:
                future=pool.submit(v.verify,meta,time.monotonic())
                self.assertTrue(ready.wait(5));v.cancel.set()
                with self.assertRaisesRegex(ValueError,'cancelled'):future.result(timeout=5)
            finally:v.lock.release()

    def test_slow_verification_still_rejects_cancel_stale_and_bad_logits(self):
        for error in ('cancel','stale','logits'):
            _,_,_,v,meta=self.fixture()
            if error=='cancel':v.cancel.set()
            elif error=='stale':meta['job']=dict(meta['job'],epoch=999)
            else:meta['rows'][0]['teacher']['logits'][0]+=100
            with self.subTest(error=error),self.assertRaises(ValueError):v.verify(meta,time.monotonic()-5)
            self.assertTrue(all(not r['accepted'] for r in v.cache.values()))

    def test_sparse_recorded_timings_keep_offline_row_coverage_replay(self):
        from rust_lab_b06_teacher import replay_stream,summarize_turns
        model,job,history,_,meta=self.fixture()
        turns=[dict(index=i,label_rows=1 if i==1 else 0,profile='full',native_post_ns=2_000_000_000,
                    search_ns=800_000_000,post_search_ns=2_000_000_000,selection_ns=0,labels_ns=0,
                    commit_ns=0,elapsed_ns=3_000_000_000,**({'verification_ipc_ns':2_000_000_000} if i==1 else {}))
               for i in range(1,65)]
        table=dict(job_id=job['expected']['job_id'],rows=meta['rows'],metrics=dict(turns=turns,
            searched_turns=64,reserved_turns=16,labels=1,searches=[],verification=[dict(index=1,rows=1),dict(index=64,rows=0)]))
        rows,_,_=replay_stream(model,history,job,table)
        self.assertEqual(len(rows),1);self.assertEqual(summarize_turns([table])['turns'],64)


class FailureLogs(unittest.TestCase):
    def test_child_stdout_and_actual_traceback_survive_failure(self):
        module=launcher()
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp,contextlib.redirect_stdout(io.StringIO()):
            log=Path(temp)/'training.log'
            argv=[sys.executable,'-u','-c','print("started",flush=True); raise ValueError("root validation fault")']
            with self.assertRaises(subprocess.CalledProcessError):module.run_training(argv,log)
            self.assertIn('started',log.read_text());self.assertIn('ValueError: root validation fault',log.read_text())

    def test_main_exports_child_log_and_bounded_error_tail(self):
        module=launcher()
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);config=root/'config.json';config.write_text('{}')
            state=dict(iteration=1257,signature={'imitation_source':{'bc_run':'artifacts/il'}})
            def fail(argv,log):
                log.write_text('start\n'+'x'*20000+'\nValueError: actual child error\n',encoding='utf-8')
                raise subprocess.CalledProcessError(1,argv)
            with patch.object(module,'ROOT',root),patch.object(module,'prepare',return_value=(state,config,2)), \
                patch.object(module,'run_training',side_effect=fail),patch('rust_lab_results.export',return_value=root/'results') as export, \
                patch.object(sys,'argv',['continue-rust-shadow.py','--expected-generation','1257']):
                with self.assertRaises(subprocess.CalledProcessError):module.main()
            report=read(next((root/'artifacts/rust-backend').glob('*/qualification.json')))
            self.assertFalse(report['training_passed']);self.assertIn('actual child error',report['training_error_tail'])
            self.assertLessEqual(len(report['training_error_tail'].encode()),16384)
            self.assertTrue(Path(report['training_log']['path']).is_file());export.assert_called_once()

    def test_slow_verification_is_excluded_from_active_search_watchdog(self):
        from brass_ai.rust_backend.lab_teacher import Worker
        from brass_ai.rust_backend.lab_teacher_overlay import CONTRACT
        worker=Worker.__new__(Worker);worker.cancel=threading.Event();worker.events=queue.Queue()
        worker.send=Mock();worker.close=Mock();worker.job={};clock=[0.]
        expected=dict(job_id='frozen')
        worker.events.put((dict(event='hello',job=expected,contract=CONTRACT,core={'source_sha256':'core'},
            method='search_v1',v205_enabled=False),b'',0.))
        worker.events.put((dict(event='turn',rows=[]),b'',0.))
        worker.events.put((dict(event='finished',job=expected,ordinary_training=False,
            metrics=dict(labels=0,reserved_turns=16),result={}),b'',1000.))
        verifier=Mock()
        def verify(meta,received):clock[0]=1000.;return dict(status='ok')
        verifier.verify.side_effect=verify
        with patch('brass_ai.rust_backend.lab_teacher.time.monotonic',side_effect=lambda:clock[0]):
            result=worker.collect(expected,'core',verifier=verifier)
        self.assertEqual(result['job_id'],'frozen');worker.close.assert_called_once()


class RuntimePreparation(unittest.TestCase):
    def setUp(self):
        from brass_ai.rust_backend.lab_contract import pin
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        from brass_ai.rust_backend.lab_teacher_runtime import fingerprint,POLICY,SELECTOR
        from brass_ai.step5.league import implementation_id,write_json
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.selector=self.root/SELECTOR
        self.run=self.root/'artifacts/run';self.run.mkdir(parents=True)
        checkpoint=self.run/'generation-1257.pt';checkpoint.write_bytes(b'untouched checkpoint')
        tools=self.root/'tools';tools.mkdir()
        for name in ('binary','tests','build-log','bootstrap','qualified'):(tools/name).write_bytes(name.encode())
        self.original=dict(contract='brass-lab-b07-teacher-build/1.0.0',core_source='fixed-core',
            bootstrap=dict(binary=pin(tools/'bootstrap'),dependencies=[]),receipt=pin(tools/'qualified'))
        self.built=dict(contract=self.original['contract'],core_source=self.original['core_source'],
            bootstrap=copy.deepcopy(self.original['bootstrap']),producer=source_fingerprint(),implementation=implementation_id(),
            binary=pin(tools/'binary'),tests=pin(tools/'tests'),build_log=pin(tools/'build-log'),wire_regressions_passed=True)
        self.selected=dict(self.built,verification_policy=POLICY,verification_source=fingerprint(),
                           original_teacher_receipt=self.original['receipt'])
        self.state=dict(iteration=1257,history=[dict(generation=1257,path='artifacts/run/generation-1257.pt',
            sha256=pin(checkpoint)['sha256'])])
        write_json(self.run/'state.json',self.state)
        spec=importlib.util.spec_from_file_location('prepare_verification',ROOT/'scripts/prepare-rust-teacher-verification.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module);self.module.ROOT=self.root
        self.write_json=write_json

    def test_missing_or_tampered_runtime_fails_before_training(self):
        from brass_ai.rust_backend.lab_teacher_runtime import resolve
        with self.assertRaisesRegex(ValueError,'prepare-rust-teacher'):resolve(self.root,self.original)
        self.write_json(self.selector,self.selected);resolve(self.root,self.original)
        wrong=dict(self.selected,verification_source='unreviewed');self.write_json(self.selector,wrong)
        with self.assertRaisesRegex(ValueError,'source/original'):resolve(self.root,self.original)
        self.write_json(self.selector,self.selected);Path(self.selected['binary']['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'runtime/evidence changed'):resolve(self.root,self.original)

    def test_prepare_then_reuse_preserves_checkpoint_state_and_qualification(self):
        evidence=self.root/'artifacts/evidence';evidence.mkdir()
        args=SimpleNamespace(run='artifacts/run',expected_generation=1257,offline=True)
        with patch('brass_ai.rust_backend.lab_backend.validate_state',return_value={'mode':'rust'}), \
             patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value=self.original) as qualified, \
             patch('rust_lab_b07_retry.build_teacher',return_value=self.built) as build:
            first=self.module.prepare(args,evidence);second=self.module.prepare(args,evidence)
        self.assertFalse(first['reused']);self.assertTrue(second['reused']);build.assert_called_once()
        self.assertTrue(all(c.kwargs=={'prepared':False} for c in qualified.call_args_list))
        self.assertTrue(build.call_args.kwargs['offline'])
        self.assertEqual(read(self.run/'state.json'),self.state)
        self.assertEqual((self.run/'generation-1257.pt').read_bytes(),b'untouched checkpoint')

    def test_checkpoint_change_during_build_prevents_selector_publication(self):
        evidence=self.root/'artifacts/evidence';evidence.mkdir()
        args=SimpleNamespace(run='artifacts/run',expected_generation=1257,offline=True)
        def build(*args,**kwargs):
            (self.run/'generation-1257.pt').write_bytes(b'changed');return self.built
        with patch('brass_ai.rust_backend.lab_backend.validate_state',return_value={'mode':'rust'}), \
             patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value=self.original), \
             patch('rust_lab_b07_retry.build_teacher',side_effect=build):
            with self.assertRaisesRegex(ValueError,'changed during teacher build'):self.module.prepare(args,evidence)
        self.assertFalse(self.selector.exists());self.assertEqual(read(self.run/'state.json'),self.state)

    def test_old_handoff_limited_adapter_is_rebuilt_and_selector_retained(self):
        evidence=self.root/'artifacts/evidence';evidence.mkdir()
        old=dict(self.selected,verification_policy='sampled-unbounded-verification/1',verification_source='old-source')
        self.write_json(self.selector,old);original_bytes=self.selector.read_bytes()
        args=SimpleNamespace(run='artifacts/run',expected_generation=1257,offline=True)
        with patch('brass_ai.rust_backend.lab_backend.validate_state',return_value={'mode':'rust'}), \
             patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value=self.original), \
             patch('rust_lab_b07_retry.build_teacher',return_value=self.built) as build:
            result=self.module.prepare(args,evidence)
        self.assertFalse(result['reused']);build.assert_called_once()
        self.assertEqual((evidence/'previous-teacher-runtime.json').read_bytes(),original_bytes)
        self.assertEqual(read(self.selector)['verification_policy'],'sampled-unbounded-verification/2')
        self.assertEqual(read(self.run/'state.json'),self.state)


if __name__=='__main__':unittest.main()
