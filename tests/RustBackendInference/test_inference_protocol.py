"""Real process lifetime and binding regressions; no Python inference service."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from brass_ai.native.protocol import NativeWorker,NativeError,WorkerFailure
from brass_ai.native.inference import NativeEvaluator
from brass_ai.native.simulation import NativeSpielGame

ROOT=Path(__file__).resolve().parents[2]
WORKER=Path(os.environ.get('BRASS_NATIVE_WORKER',ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-worker.exe'))
MODEL=Path(os.environ.get('BRASS_ONNX_MANIFEST',ROOT/'artifacts/rust-backend/20261025-l09-export-v2/manifest.json'))


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(ROOT/'artifacts/rust-backend/20261025-l06-capture/roots.jsonl.gz','rt',encoding='utf-8') as f:r=json.loads(next(f))
        cls.state=r['state'];cls.view=r['views'][r['state']['context']['actor']]
        with gzip.open(ROOT/'artifacts/rust-backend/20261025-l05-capture-v2/roots.jsonl.gz','rt',encoding='utf-8') as f:cls.choices=json.loads(next(f))['expected']['choices']
        cls.meta=json.loads(MODEL.read_text())
        cls.ctx=dict(model_id=cls.meta['model_id'],reward_id=cls.meta['reward_id'],profile_id='neutral',style=[0.]*4,intent=[0.]*8,
                     remaining=[0.,.01,.05,.075],max_cost=.075,stable_descriptors=None)
    def worker(self):
        w=NativeWorker(WORKER,timeout=60);self.addCleanup(w.close);return w
    def request(self,e,**kw):
        return e.request(self.view,self.choices,self.ctx,request_id='fixture',**kw)
    def test_two_owners_concurrent_and_state_handle_path(self):
        w=self.worker();a,b=[NativeEvaluator(w,s,MODEL) for s in (0,1)]
        for e in (a,b):e.submit([self.request(e)],budget_ms=30000)
        x,y=a.wait(),b.wait();self.assertEqual(x['state'],'complete');self.assertEqual(y['state'],'complete')
        self.assertNotEqual(x['owner'],y['owner']);np.testing.assert_array_equal(x['evaluations'][0]['logits'],y['evaluations'][0]['logits'])
        for e in (a,b):
            with NativeSpielGame(worker=w,session=e.session) as game:
                state=game.import_state(self.state)
                e.submit([self.request(e,state=state)],budget_ms=30000)
                r=e.wait();self.assertEqual(r['state'],'complete')
                np.testing.assert_array_equal(r['evaluations'][0]['logits'],x['evaluations'][0]['logits'])
                request=self.request(e,state=state);request['source']['revision']+=1
                e.submit([request],budget_ms=30000)
                with self.assertRaises(NativeError):e.wait()
    def test_cancellation_keeps_owner_busy_until_drain_and_other_owner_works(self):
        w=self.worker();a=NativeEvaluator(w,0,MODEL)
        t=a.submit([self.request(a)],budget_ms=30000);a.cancel()
        with self.assertRaises(NativeError):w.snapshot(0)
        with self.assertRaises(NativeError):w.call('inference',session=1,command=dict(kind='poll',ticket=t.ticket))
        w.new_game(1,34)
        r=a.wait();self.assertEqual(r['state'],'cancelled');self.assertEqual(r['evaluations'],[])
        with self.assertRaises(NativeError):w.call('inference',session=0,command=dict(kind='poll',ticket=t.ticket))
        a.submit([self.request(a)],budget_ms=30000);self.assertEqual(a.wait()['state'],'complete')
    def test_expiry_before_work_and_after_completed_delivery(self):
        w=self.worker();e=NativeEvaluator(w,0,MODEL)
        e.submit([self.request(e)],budget_ms=0);r=e.wait();self.assertEqual(r['state'],'expired');self.assertEqual(r['evaluations'],[])
        e.submit([self.request(e)],budget_ms=20);time.sleep(.08)
        r=e.wait();self.assertEqual(r['state'],'expired');self.assertEqual(r['evaluations'],[])
    def test_wrong_model_reward_seat_revision_and_duplicate_requests(self):
        w=self.worker();e=NativeEvaluator(w,0,MODEL)
        for field,value in [('model_id','0'*64),('export_sha256','0'*64),('reward_id','0'*64),('observer',(self.view['seat']+1)%4),('revision',self.view['context']['revision']+1),('profile_id','foreign')]:
            r=self.request(e);r['id'][field]=value
            e.submit([r],budget_ms=30000)
            with self.subTest(field=field),self.assertRaises(NativeError):e.wait()
        r=self.request(e)
        with self.assertRaises(NativeError):e.submit([r,copy.deepcopy(r)],cap=2,budget_ms=30000)
        e.submit([self.request(e)],budget_ms=30000);self.assertEqual(e.wait()['state'],'complete')
    def test_frozen_model_rejects_reload_and_manifest_corruption(self):
        w=self.worker()
        with self.assertRaises(NativeError):w.call('inference',session=0,command=dict(kind='load',manifest=str(MODEL.resolve()),manifest_sha256='0'*64))
        e=NativeEvaluator(w,0,MODEL)
        with self.assertRaises(NativeError):w.call('inference',session=0,command=dict(kind='load',manifest=str(MODEL.resolve()),manifest_sha256='0'*64))
        e.submit([self.request(e)],budget_ms=30000);self.assertEqual(e.wait()['state'],'complete')
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/rust-backend') as folder:
            p=Path(folder)/'manifest.json';bad=dict(self.meta,mechanical_only=False);p.write_text(json.dumps(bad))
            with self.assertRaises(NativeError):w.call('inference',session=1,command=dict(kind='load',manifest=str(p),manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    def test_invalid_bound_and_generation_do_not_create_orphan(self):
        w=self.worker();e=NativeEvaluator(w,0,MODEL);e.submit([self.request(e)],budget_ms=30000);e.wait()
        base=dict(kind='start',generation=1,batch_id='old',budget_ms=4000,requests=[self.request(e)],policy=dict(cap=1,pending_limit=1,max_candidate_cells=8192,flush_us=0))
        with self.assertRaises(NativeError):w.call('inference',session=0,command=base)
        base['generation']=100;base['requests'][0]['id']['generation']=100;base['policy']['pending_limit']=5
        with self.assertRaises(NativeError):w.call('inference',session=0,command=base)
        e.submit([self.request(e)],budget_ms=30000);self.assertEqual(e.wait()['state'],'complete')
    def test_mutated_response_retires_client(self):
        w=self.worker();e=NativeEvaluator(w,0,MODEL);e.submit([self.request(e)],budget_ms=30000)
        original=w.call
        def corrupt(*args,**kwargs):
            r=original(*args,**kwargs)
            if r['status']=='inference' and r['data']['state']=='complete':r['data']['evaluations'][0]['choice_ids'].reverse()
            return r
        with patch.object(w,'call',side_effect=corrupt),self.assertRaises(WorkerFailure):e.wait()
        self.assertTrue(w._closed)
    def test_shutdown_during_work_joins_and_closes_cleanly(self):
        w=self.worker();e=NativeEvaluator(w,0,MODEL);e.submit([self.request(e)],budget_ms=30000)
        w.close();self.assertIsNotNone(w.process.poll());self.assertFalse(w._io.is_alive())

    def test_nonretiring_runtime_hits_drain_watchdog(self):
        w=self.worker();e=NativeEvaluator(w,0,MODEL);e.submit([self.request(e)],budget_ms=30000)
        # A responsive control connection can otherwise keep returning 'running'
        # forever. Advance the caller clock without sleeping or launching a hang.
        expired=e.active.deadline+5.01
        with patch('brass_ai.native.inference.time.monotonic',return_value=expired),patch.object(e,'poll',return_value=None),self.assertRaises(WorkerFailure):
            e.wait()
        self.assertTrue(w._closed);self.assertIsNone(e.active)


if __name__=='__main__':unittest.main()
