"""Short reply-resource regressions; no simulator, benchmark or training run."""
import gc
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import psutil
import torch

from brass_ai.learning.features import Encoded,feature_schema
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery import inference,shared_transport


def handles():
    process=psutil.Process()
    return process.num_fds() if hasattr(process,'num_fds') else process.num_handles()


def envelope(endpoint,token=None,*,encoded=None):
    token=token or 'a'*32
    rows=[dict(request_id=token,model='fixture',revision=7,seat=2,digest='binding',
        deadline=time.monotonic()+30,submitted=time.monotonic(),cancelled=True,encoded=encoded)]
    slot=endpoint.slot();packed=slot.pack_frames(rows,token) if slot else None
    return dict(lane=endpoint.lane,token=token,frames=packed if packed is not None else rows,shared=packed is not None)


class BlockedService(inference.InferenceService):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.entered=threading.Event();self.resume=threading.Event()
    def _evaluate(self,items):
        self.entered.set()
        if not self.resume.wait(5):raise TimeoutError('debug inference barrier')
        return {key:self._response(frame,'cancelled') for key,frame in items}


def spawned_endpoints(lease,out):
    endpoints=[lease.acquire() for _ in range(2)]
    try:out.put([e.lane for e in endpoints])
    finally:
        for endpoint in endpoints:endpoint.slot();endpoint.close()


class ReplyLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def service(self,kind=inference.InferenceService,**config):
        settings=dict(EXECUTION,processes=1,hosts_per_process=2,shared_slot_bytes=1024);settings.update(config)
        service=kind(settings,'cpu')
        self.addCleanup(service.__exit__,None,None,None)
        return service

    def empty(self,service):
        deadline=time.monotonic()+3
        while service.reply_channels.report()['retiring'] and time.monotonic()<deadline:time.sleep(.001)
        self.assertEqual(service.leases,[]);self.assertEqual(service._lease_channels,{})
        self.assertEqual(service.connections,{});self.assertEqual(service.children,[]);self.assertEqual(service.slots,{})
        report=service.reply_channels.report()
        self.assertEqual(report['created'],report['released']);self.assertEqual(report['active'],0)

    def test_repeated_48_lane_rounds_release_claimed_and_unclaimed_handles(self):
        service=self.service()
        def roundtrip():
            lease=service.channels(48)
            endpoints=[lease.acquire() for _ in range(3)]
            names=[e.shared[0] for e in endpoints]
            for endpoint in endpoints:endpoint.slot();endpoint.close()
            service.release_channels(lease);service.release_channels(lease)
            self.empty(service)
            with self.assertRaisesRegex(ValueError,'closed'):lease.acquire()
            for name in names:
                with self.assertRaises(FileNotFoundError):SharedMemory(name=name)
        roundtrip();gc.collect();baseline=handles()
        for _ in range(12):roundtrip()
        gc.collect()
        self.assertLessEqual(handles(),baseline+4)
        self.assertEqual(service.reply_channels.report()['created'],624)

    def test_partial_pipe_and_slot_and_queue_allocation_are_transactional(self):
        service=self.service();original=service.mp.Pipe
        calls=0
        def pipe(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==3:raise OSError(24,'debug descriptor limit')
            return original(*args,**kwargs)
        with patch.object(service.mp,'Pipe',side_effect=pipe):
            with self.assertRaisesRegex(OSError,'descriptor limit'):service.channels(4)
        self.empty(service)
        with patch.object(shared_transport,'SharedSlot',side_effect=OSError(24,'debug slot failure')):
            with self.assertRaisesRegex(OSError,'slot failure'):service.channels(1)
        self.empty(service)
        with patch.object(service.mp,'Queue',side_effect=OSError(24,'debug queue failure')):
            with self.assertRaisesRegex(OSError,'queue failure'):service.channels(4)
        self.empty(service)
        lease=service.channels(2);service.release_channels(lease);self.empty(service)

    def test_retired_lanes_never_reused_and_stale_queued_envelopes_are_discarded(self):
        with self.service() as service:
            lease=service.channels(2);old=lease.acquire();unused=lease.acquire()
            late=envelope(old);old_lane=old.lane
            old.close();unused.close();service.release_channels(lease)
            lease=service.channels(1);current=lease.acquire();self.assertGreater(current.lane,old_lane)
            service.requests.put(late)
            service.requests.put(envelope(current,'b'*32))
            self.assertTrue(current.connection.poll(3));reply=current.connection.recv()
            self.assertEqual(reply['token'],'b'*32);self.assertEqual(reply['responses'][0]['revision'],7)
            self.assertFalse(current.connection.poll(.01));self.assertIsNone(service.error)
            current.close();service.release_channels(lease);self.empty(service)

    def test_cancelled_inflight_shared_views_stay_pinned_until_forward_retires(self):
        with self.service(BlockedService,shared_slot_bytes=1024*1024) as service:
            lease=service.channels(1);endpoint=lease.acquire()
            schema=feature_schema()
            encoded=Encoded(np.zeros(schema['state_dim'],dtype='<f4'),np.zeros((1,schema['action_dim']),dtype='<f4'),
                np.array([2,3,0,1],dtype='<i8'),('action',))
            service.requests.put(envelope(endpoint,encoded=encoded))
            self.assertTrue(service.entered.wait(3))
            name=endpoint.shared[0];endpoint.close();service.release_channels(lease)
            self.assertEqual(service.reply_channels.report()['retiring'],1)
            self.assertEqual(len(service.slots),1)
            service.resume.set();deadline=time.monotonic()+3
            while service.connections and time.monotonic()<deadline:time.sleep(.01)
            self.empty(service);self.assertIsNone(service.error)
            with self.assertRaises(FileNotFoundError):SharedMemory(name=name)

    def test_spawned_consumers_close_before_parent_releases_their_lease(self):
        service=self.service();lease=service.channels(3);out=service.mp.Queue()
        process=service.mp.Process(target=spawned_endpoints,args=(lease,out))
        try:
            process.start();lanes=out.get(timeout=30)
            process.join(timeout=10);self.assertEqual(process.exitcode,0);self.assertEqual(len(set(lanes)),2)
        finally:
            if process.is_alive():process.terminate();process.join(timeout=5)
            process.close();out.close();out.join_thread()
            service.release_channels(lease)
        self.empty(service)

    def test_shutdown_also_drains_leases_that_were_never_claimed(self):
        service=self.service();service.channels(48)
        service.__exit__(None,None,None);service.__exit__(None,None,None)
        self.empty(service);self.assertTrue(service.closed)

    def test_another_service_cannot_retire_a_live_lease(self):
        first=self.service();second=self.service();lease=first.channels(1)
        with self.assertRaisesRegex(ValueError,'another service'):second.release_channels(lease)
        self.assertEqual(len(first.connections),1)
        first.release_channels(lease);self.empty(first)

    def test_native_actor_start_failure_releases_lease_pipes_and_unstarted_process(self):
        from brass_ai.rust_backend import lab_pool as P
        from test_b03 import family_fixture
        service=self.service();f,_,config=family_fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            job=f['frozen_job'];job['path']=str(Path(temp)/'job.jsonl.gz')
            build=dict(_receipt_sha256=f['producer']['build_sha256'],b01={'sha256':f['producer']['b01_sha256']})
            pool=P.NativePool(ROOT,service,dict(EXECUTION,processes=1,hosts_per_process=2),
                build,config,f['producer']['signature'])
            created=[]
            class Process:
                pid=None
                closed=False
                def __init__(self,*,target,args):self.target=target;created.append(self)
                def start(self):
                    if self.target is P._actors:raise OSError(24,'original actor startup failure')
                    self.pid=123
                def join(self,timeout=None):pass
                def is_alive(self):return False
                def close(self):self.closed=True
            with patch.object(pool.mp,'Process',Process),patch.object(pool,'receive',return_value={'ready':True}):
                with self.assertRaisesRegex(OSError,'original actor startup failure'):pool.collect([job])
            self.assertEqual(len(created),2);self.assertTrue(all(p.closed for p in created))
            self.assertEqual(pool.processes,[]);self.assertEqual(pool.connections,[]);self.assertEqual(pool.channel_leases,[])
            self.empty(service);pool.close()

    def test_reference_pool_closes_process_handles_and_lease_once_after_join(self):
        from brass_ai.recovery.execution import ActorPool
        service=self.service();pool=ActorPool(ROOT,'unused-host',service,dict(EXECUTION,processes=1))
        lease=service.channels(2);pool.channel_leases.append(lease)
        process=pool.mp.Process(target=spawned_endpoints);pool.processes.append(process)
        pool.close();pool.close()
        self.assertTrue(process._closed);self.assertTrue(pool.closed);self.assertTrue(lease.closed)
        self.empty(service)

    def test_round_cleanup_preserves_predictions_batching_and_request_count(self):
        from brass_ai.learning.batching import Evaluator,Request
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from review_fixture import current_position
        model=FourSeatNet(ModelConfig.preset('small'));local=Evaluator(model);self.addCleanup(local.close)
        with self.service(shared_slot_bytes=1024*1024) as service:
            identity=service.register(model)
            with patch.object(service.requests,'put',wraps=service.requests.put) as sent:
                for i in range(3):
                    lease=service.channels(2);endpoint=lease.acquire()
                    request=Request('round-'+str(i),identity,*current_position(i),time.monotonic()+30)
                    expected=local.predict([request])[0]
                    actual=inference.RemoteEvaluator(endpoint,identity).predict([request])[0]
                    for key in ('request_id','model_id','observer','revision','input_digest','choice_ids'):
                        self.assertEqual(actual[key],expected[key])
                    for key in ('probabilities','win_share','placement','vp'):
                        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-5,atol=2e-5)
                    endpoint.close();service.release_channels(lease);self.empty(service)
                self.assertEqual(sent.call_count,3)  # No new inference IPC for retirement.
            self.assertEqual(service.metrics['positions'],3);self.assertEqual(service.metrics['batches'],3)
            self.assertEqual(service.metrics['model_loads'],1);self.assertIsNone(service.error)


class MultiReplyLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def pool(self,routing):
        config=dict(EXECUTION,processes=1,hosts_per_process=2,shared_slot_bytes=1024,inference_routing=routing)
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'device_count',return_value=2),\
                patch.object(inference,'_single_service',side_effect=lambda c,d,p,s:inference.InferenceService(c,'cpu')):
            service=inference.InferenceService(config,'cpu',inference_devices=['cuda:0','cuda:1'])
        self.addCleanup(service.__exit__,None,None,None);return service

    def test_both_replica_routes_release_resources_without_routing_underflow(self):
        for routing in ('demand','round_robin'):
            with self.subTest(routing=routing),self.pool(routing) as service:
                for i in range(4):
                    lease=service.channels(4);endpoints=[lease.acquire() for _ in range(4)]
                    for endpoint in endpoints:endpoint.requests.put(envelope(endpoint))
                    for endpoint in endpoints:
                        self.assertTrue(endpoint.connection.poll(3));self.assertNotIn('error',endpoint.connection.recv())
                        endpoint.close()
                    service.release_channels(lease)
                    deadline=time.monotonic()+3
                    while service.report()['reply_channels']['retiring'] and time.monotonic()<deadline:time.sleep(.001)
                    report=service.report()['reply_channels']
                    self.assertEqual(report['active'],0);self.assertEqual(report['leases'],0)
                    self.assertEqual(report['created'],report['released'])
                if routing=='demand':self.assertEqual(service.loads,[0,0])
                self.assertIsNone(service.error)


if __name__=='__main__':unittest.main()
