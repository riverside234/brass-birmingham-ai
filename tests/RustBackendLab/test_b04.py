"""Short B04 dispatch, native-wire, GPU-service and integrity smoke tests."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from brass_ai.learning.batching import model_id, predict_encoded
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.teacher_phase import TeacherPhase,plan_jobs,readiness
from brass_ai.recovery.teacher_labels import validate_improved_policy
from brass_ai.recovery.teacher_term import policy_logits
from brass_ai.rust_backend import lab_teacher as T
from brass_ai.rust_backend.lab_teacher_inference import Client,predict,decode
from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint,apply
from brass_ai.rust_backend.lab_contract import extract_new,SOURCE,native_fingerprint
from test_b025 import config,model_fixture,example


def job(model):
    c=config();c['teacher_phase']['mode']='shadow'
    state=dict(iteration=10)
    jobs,_=plan_jobs(c,state,model_id=model_id(model),export_hash='a'*64,epoch=1,ready=True,
                     native_source=source_fingerprint())
    j=jobs[0];j['readiness']=dict(ready=True,reward_id=j['expected']['reward_id']);return j


def wire(model,sequence=1,epoch=1):
    row,_=example(model);e=row.encoded;j=job(model)
    b=dict(job=j['expected'],sequence=sequence,root_epoch=epoch,revision=row.row['view']['context']['revision'],
        seat=0,legal_ids=list(e.choice_ids),context=dict(model_id=model_id(model),reward_id=j['expected']['reward_id'],
        style=[0.]*4,intent=[0.]*8,stable_descriptors=None,remaining=[.075]*4),information='fixture')
    data=e.state.astype('<f4').tobytes()+e.actions.astype('<f4').tobytes()+e.stable_to_relative.astype('<i8').tobytes()
    return dict(binding=b,candidates=len(e.choice_ids),state_len=len(e.state),action_len=e.actions.size,
                remaining_ns=10_000_000_000),data,row


class B04(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.model=model_fixture().eval()

    def test_exact_job_profile_and_dedicated_schedule(self):
        j=job(self.model);T.validate_job(j)
        for key,value in [('method','search_v2'),('v205_enabled',True),('k',8),('sessions',2),('allowance_seconds',8.),('pending_paths',2)]:
            bad=copy.deepcopy(j);bad['search'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):T.validate_job(bad)
        for mutate in [lambda b:b['reservations']['0'].update(canal_early=8),
                       lambda b:b['expected'].update(source_fingerprint=SOURCE),
                       lambda b:b.update(ordinary_training=True),lambda b:b['seats'].__setitem__(2,'foreign'),
                       lambda b:b['readiness'].update(ready=False),lambda b:b.pop('readiness')]:
            bad=copy.deepcopy(j);mutate(bad)
            with self.assertRaises(ValueError):T.validate_job(bad)

    def test_initial_write_failure_retires_native_worker(self):
        process=Mock()
        with patch.object(T.subprocess,'Popen',return_value=process), \
             patch.object(T.threading,'Thread'), \
             patch.object(T.Worker,'send',side_effect=BrokenPipeError('closed on startup')), \
             patch.object(T.Worker,'close') as close:
            with self.assertRaisesRegex(BrokenPipeError,'startup'):
                T.Worker('fixture',{},Mock(),threading.Event())
            close.assert_called_once()

    def test_finished_frame_racing_cancellation_cannot_publish(self):
        worker=T.Worker.__new__(T.Worker)
        worker.cancel=threading.Event();worker.close=Mock()
        expected={'job_id':'fixture'}
        hello=dict(event='hello',contract=T.CONTRACT,job=expected,
                   core={'source_sha256':'core'},method='search_v1',v205_enabled=False)
        finished=dict(event='finished',job=expected,ordinary_training=False,
                      metrics=dict(labels=0,reserved_turns=16),result={})
        frames=iter([(hello,b''),(finished,b'')])
        def receive(**kwargs):
            frame=next(frames)
            if frame[0]['event']=='finished':worker.cancel.set()
            return frame
        worker.events=Mock();worker.events.get.side_effect=receive
        with self.assertRaisesRegex(ValueError,'cancelled before publication'):
            worker.collect(expected,'core')
        worker.close.assert_called_once()

    def test_off_and_unready_never_dispatch_and_keep_ordinary_cursor(self):
        c=config();state=dict(iteration=10,history=[dict(generation=10,model_id=model_id(self.model))],next_seed=937)
        old=copy.deepcopy(state)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            for mode,receipt in [('off',dict(ready=True,reward_id=digest(self.model.config.utility))),('on',dict(ready=False,reward_id=digest(self.model.config.utility)))]:
                c['teacher_phase']['mode']=mode
                with patch.object(self.model,'utility_calibration',receipt):
                    phase=TeacherPhase(SOURCE_ROOT,Path(temp),c,state,self.model)
                    backend=Mock();normal=Mock(return_value=['ordinary'])
                    records,teachers=T.collect_barrier(phase,state,self.model,export_hash='a'*64,epoch=1,
                        backend=backend,ordinary_jobs=['ordinary'],ordinary_collect=normal)
                    backend.collect.assert_not_called();self.assertEqual(records,['ordinary']);self.assertEqual(teachers,[])
                    self.assertIsNone(phase.prepare(self.model))
        self.assertEqual(state,old)
        from brass_ai.recovery.teacher_collection import due_collections
        self.assertEqual(due_collections(100000,1,c),0)

    def test_shadow_and_on_same_frozen_jobs_no_legacy_work_fields(self):
        c=config();s=dict(iteration=10,next_seed=31)
        c['teacher_phase']['mode']='shadow'
        a,cursor=plan_jobs(c,s,model_id='same',export_hash='a'*64,epoch=1,ready=True)
        c['teacher_phase']['mode']='on';c['teacher_work_budget']={'invalid':'ignored'};c['search_wall_fraction']=0
        b,next_cursor=plan_jobs(c,s,model_id='same',export_hash='a'*64,epoch=1,ready=True)
        self.assertEqual(a,b);self.assertEqual(cursor,next_cursor);self.assertEqual(len(a),32)
        self.assertEqual(sum(j['split']=='heldout' for j in a),4);self.assertEqual(s['next_seed'],31)

    def test_readiness_null_retains_latest_but_false_or_reward_change_defers(self):
        ready=dict(ready=True,reward_id='current')
        self.assertEqual(readiness(ready,None,'current'),(ready,True))
        self.assertFalse(readiness(ready,dict(ready=False,reward_id='current'),'current')[1])
        self.assertFalse(readiness(ready,None,'changed')[1])

    def test_raw_logits_match_policy_and_preserve_normal_outputs_including_chunking(self):
        row,_=example(self.model);e=row.encoded
        with torch.no_grad():
            for limit in [2,8192]:
                got=predict(self.model,[e],'cpu',limit,torch.tensor([[.075]*4]))
                expected=predict_encoded(self.model,[e],'cpu',limit,torch.tensor([[.075]*4]))
                for a,b in zip(got[:-1],expected):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
                np.testing.assert_allclose(got[-1][0],policy_logits(self.model,[row])[0],rtol=1e-5,atol=2e-5)

    def test_shared_service_roundtrip_and_replayed_epoch_rejection(self):
        meta,payload,row=wire(self.model)
        with InferenceService(config()['execution'],'cpu') as service:
            service.register(self.model);e=service.channels(1).acquire();client=Client(e,meta['binding']['job'],threading.Event())
            try:
                reply=client.evaluate(meta,payload)
                self.assertEqual(reply['status'],'ok');self.assertIsNone(e.pending)
                np.testing.assert_allclose(reply['logits'],row.label['logits'],atol=2e-5,rtol=1e-5)
                with self.assertRaisesRegex(ValueError,'Stale'):client.evaluate(meta,payload)
                self.assertGreater(service.metrics['shared_requests'],0)
            finally:e.close()

    def test_gpu_wait_cancellation_drains_slot_and_does_not_expose_logits(self):
        meta,payload,_=wire(self.model);started=threading.Event();release=threading.Event();cancel=threading.Event()
        class Slow(InferenceService):
            def _evaluate(s,items):
                started.set();release.wait(3);return super()._evaluate(items)
        with Slow(config()['execution'],'cpu') as service:
            service.register(self.model);e=service.channels(1).acquire();client=Client(e,meta['binding']['job'],cancel);out=[]
            try:
                t=threading.Thread(target=lambda:out.append(client.evaluate(meta,payload)));t.start()
                self.assertTrue(started.wait(3));cancel.set();release.set();t.join(5)
                self.assertFalse(t.is_alive());self.assertEqual(out[0]['status'],'cancelled')
                self.assertNotIn('logits',out[0]);self.assertIsNone(e.pending)
            finally:release.set();e.close()

    def test_wrong_model_reply_retires_owner(self):
        meta,payload,_=wire(self.model)
        class Wrong(InferenceService):
            def _evaluate(s,items):
                rows=super()._evaluate(items)
                for row in rows.values():row['model_id']='foreign'
                return rows
        with Wrong(config()['execution'],'cpu') as service:
            service.register(self.model);e=service.channels(1).acquire();client=Client(e,meta['binding']['job'],threading.Event())
            try:
                with self.assertRaisesRegex(ValueError,'wrong-model'):client.evaluate(meta,payload)
                self.assertTrue(client.retired)
            finally:e.close()

    def test_stuck_gpu_watchdog_retires_owner_without_replenishing_deadline(self):
        meta,payload,_=wire(self.model)
        e=SimpleNamespace(lane=0,lock=threading.RLock(),pending=None,slot=lambda:None,requests=Mock(),connection=Mock())
        e.connection.poll.return_value=False
        client=Client(e,meta['binding']['job'],threading.Event())
        with patch('brass_ai.rust_backend.lab_teacher_inference.time.monotonic',side_effect=[0.,0.,0.,31.,31.]):
            with self.assertRaisesRegex(ValueError,'watchdog'):client.evaluate(meta,payload)
        self.assertTrue(client.retired);self.assertIsNotNone(e.pending)
        self.assertEqual(e.requests.put.call_count,1)

    def test_latest_committed_readiness_stops_at_false_reward_boundary_or_gap(self):
        from brass_ai.rust_backend.lab_b04_smoke import latest_readiness
        from brass_ai.rust_backend.lab_contract import sha
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as d:
            root=Path(d);p=root/'old.pt'
            old=dict(step=9,model_id='m9',model_config=dict(utility={'reward':'current'}),
                     training=dict(utility_calibration=dict(ready=False,reward_id='current')))
            torch.save(old,p)
            state=dict(iteration=10,history=[dict(path='old.pt',sha256=sha(p),generation=9,model_id='m9'),dict(generation=10)])
            raw=dict(model_config=old['model_config'],training=dict(utility_calibration=None))
            receipt,ready,evidence=latest_readiness(root,state,raw,'current')
            self.assertFalse(ready);self.assertFalse(receipt['ready'])
            old['training']['utility_calibration']['ready']=True;torch.save(old,p);state['history'][0]['sha256']=sha(p)
            self.assertTrue(latest_readiness(root,state,raw,'current')[1])
            self.assertFalse(latest_readiness(root,state,raw,'changed')[1])
            p.unlink();self.assertFalse(latest_readiness(root,state,raw,'current')[1])

    def test_cancel_between_native_actions_discards_unpublished_labels(self):
        import sys
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as d:
            # A tiny protocol fixture sends a label but never a completed table.
            path=Path(d)/'fixture.py';expected=job(self.model)['expected'];core='f'*64
            path.write_text('''import sys,json,struct,time
def emit(x):
 b=json.dumps(x).encode();p=struct.pack('<I',len(b))+b
 sys.stdout.buffer.write(struct.pack('<I',len(p))+p);sys.stdout.buffer.flush()
n=struct.unpack('<I',sys.stdin.buffer.read(4))[0];r=json.loads(sys.stdin.buffer.read(n))
emit(dict(event='hello',contract=r['contract'],job=r['expected'],method='search_v1',v205_enabled=False,core=dict(source_sha256=r['core'])))
emit(dict(event='label',row={'unpublished':True}))
time.sleep(5)
''')
            cancel=threading.Event()
            # Worker accepts an executable path in production; patch only Popen's
            # argv in this fixture to run the same framing/cancellation code.
            import subprocess
            popen=subprocess.Popen
            with patch('brass_ai.rust_backend.lab_teacher.subprocess.Popen',side_effect=lambda argv,**kw:popen([sys.executable,str(path)],**kw)):
                worker=T.Worker('fixture',dict(contract=T.CONTRACT,expected=expected,core=core),None,cancel)
            timer=threading.Timer(.3,cancel.set);timer.start()
            try:
                with self.assertRaisesRegex(ValueError,'cancelled'):worker.collect(expected,core)
                self.assertIsNotNone(worker.process.poll());self.assertFalse(worker.reader.is_alive())
            finally:timer.cancel();worker.close()

    def test_wire_rejects_truncation_mapping_critic_context_and_nonfinite(self):
        meta,data,_=wire(self.model);decode(meta,data)
        with self.assertRaises(ValueError):decode(meta,data[:-1])
        bad=copy.deepcopy(meta);bad['binding']['context']['stable_descriptors']=[1]*8
        with self.assertRaises(ValueError):decode(bad,data)
        altered=bytearray(data);altered[:4]=np.float32(np.nan).tobytes()
        with self.assertRaises(ValueError):decode(meta,altered)
        altered=bytearray(data);altered[-32:]=np.array([1,0,2,3],'<i8').tobytes()
        with self.assertRaises(ValueError):decode(meta,altered)

    def test_continuation_reports_original_search_k_without_inventing_a_tournament(self):
        row,_=example(self.model,kind='continuation');l=row.label
        l.update(effective_k=16,root_legal_count=40)
        self.assertIsNotNone(validate_improved_policy(l,row.row['choices']))
        l['root_legal_count']=3
        with self.assertRaises(ValueError):validate_improved_policy(l,row.row['choices'])

    def test_overlay_only_accepts_exact_detached_core_and_changes_fingerprint(self):
        import zipfile
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            tmp=Path(tmp)
            with zipfile.ZipFile(SOURCE_ROOT/'distribution/rust-lab-b01/b01-search-v1-inputs.zip') as z:
                (tmp/'source.zip').write_bytes(z.read('source.zip'))
            extract_new(tmp/'source.zip',tmp/'checkout');core=tmp/'checkout/native'
            self.assertEqual(native_fingerprint(core),SOURCE);result=apply(core)
            self.assertNotEqual(result['core_source'],SOURCE);self.assertEqual(native_fingerprint(core),result['core_source'])
            with self.assertRaises(ValueError):apply(core)
            with self.assertRaises(ValueError):apply(SOURCE_ROOT/'native')

    @unittest.skipUnless(os.environ.get('BRASS_B04_BINARY'),'Native count probe requires the built export-overlay binary')
    def test_native_count_probe_targets_and_actual_model_logits(self):
        j=job(self.model);core=os.environ['BRASS_B04_CORE'];native=Path(os.environ['BRASS_B04_NATIVE'])
        state=json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        with InferenceService(config()['execution'],'cpu') as service:
            service.register(self.model)
            c=T.calibrate(os.environ['BRASS_B04_BINARY'],core,[dict(state=state,rail_keys=list(range(64)))],j,service)
        row=c['samples'][0]['probe'];self.assertTrue(row['diagnostic'])
        from brass_ai.recovery.teacher_labels import admit_rows
        admitted,_=admit_rows([row],model=self.model,expected=j['expected'],
            history=[dict(generation=10,model_id=model_id(self.model))],admitted_sources={source_fingerprint()})
        self.assertEqual(len(admitted),1);self.assertEqual(sum(row['teacher']['counts']),8)
        from brass_ai.rust_backend.lab_admission import admit_teacher_rows
        with self.assertRaisesRegex(ValueError,'probes'):
            admit_teacher_rows([row],job=j,model=self.model,history=[dict(generation=10,model_id=model_id(self.model))])


if __name__=='__main__':unittest.main()
