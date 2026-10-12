"""Multi-device routing/lifecycle exercised with real CPU-backed replicas.

Hardware validation is mocked only where a selected CUDA replica is mapped to
CPU; the actual queues, private pipes, batching, model predictions and spawned
actor transport are used. This is not a multi-GPU performance benchmark.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixtures import position
from brass_ai.learning.batching import Evaluator,Request
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery import inference


def spawned_predict(lease,identity,seat,results):
    try:
        torch.set_num_threads(1)
        evaluator=inference.RemoteEvaluator(lease.acquire(),identity)
        view,choices=position(seat)
        response=evaluator.predict([Request('spawn-'+str(seat),identity,view,choices,time.monotonic()+60)])[0]
        results.put(dict(response=response))
    except BaseException as error:results.put(dict(error=repr(error)))


class MultiGPUInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(292)
        self.model=FourSeatNet(ModelConfig.preset('small'))
        self.config=dict(EXECUTION,processes=2,batch_size=4,max_models=1,flush_ms=2.,cpu_budget=8,cpu_reserve=2,inference_routing='round_robin')

    def pool(self,devices=None):
        selected=devices or ['cuda:0','cuda:1']
        # Explicit selected GPU names remain in pool receipts; only the replica
        # implementation is CPU-backed to test the full IPC path locally.
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'device_count',return_value=6),\
             patch.object(inference,'_single_service',side_effect=lambda config,device,profile,steps:inference.InferenceService(config,'cpu')):
            service=inference.InferenceService(self.config,'cuda:0',inference_devices=selected)
        self.addCleanup(service.__exit__,None,None,None)
        return service

    def test_validation_requires_explicit_visible_devices_or_opted_in_auto(self):
        with patch.object(torch.cuda,'is_available',return_value=True) as available,patch.object(torch.cuda,'device_count',return_value=6):
            self.assertIsNone(inference.validate_inference_devices(None,'cpu'));available.assert_not_called()
            self.assertEqual(inference.validate_inference_devices(['cuda:1','cuda:5']),['cuda:1','cuda:5'])
            self.assertEqual(inference.validate_inference_devices(['auto']),['cuda:'+str(i) for i in range(6)])
            for devices in ([],['cpu'],['cuda'],['cuda:6'],['cuda:1','cuda:1'],['auto','cuda:0']):
                with self.assertRaises(ValueError):inference.validate_inference_devices(devices)
        with patch.object(torch.cuda,'is_available',return_value=False),patch.object(inference,'_single_service') as create:
            with self.assertRaisesRegex(ValueError,'unavailable'):
                inference.InferenceService(self.config,'cpu',inference_devices=['cuda:0'])
            create.assert_not_called()

    def test_private_round_robin_channels_preserve_exact_model_and_observer_outputs(self):
        control=Evaluator(self.model)
        with self.pool() as service:
            identity=service.register(self.model);leases=service.channels(4)
            endpoints=[leases.acquire() for _ in range(4)]
            self.assertEqual([e.requests is service.services[0].requests for e in endpoints],[True,False,True,False])
            self.assertTrue(all(child.cpu is service.cpu for child in service.services))
            clients=[inference.RemoteEvaluator(endpoint,identity) for endpoint in endpoints]
            requests=[Request('observer-'+str(s),identity,*position(s),time.monotonic()+60) for s in range(4)]
            expected=control.predict(requests)
            with ThreadPoolExecutor(max_workers=4) as threads:
                actual=list(threads.map(lambda pair:pair[0].predict([pair[1]])[0],zip(clients,requests)))
            for a,e in zip(actual,expected):
                for key in ('request_id','observer','input_digest','choice_ids','model_id'):self.assertEqual(a[key],e[key])
                for key in ('probabilities','win_share','placement','vp'):np.testing.assert_allclose(a[key],e[key],rtol=1e-5,atol=2e-5)
            report=service.report()
            self.assertEqual(report['positions'],4)
            self.assertEqual([r['positions'] for r in report['per_device']],[2,2])
            self.assertEqual(report['inference_devices'],['cuda:0','cuda:1'])
            self.assertEqual(report['cpu_admission']['capacity'],6)
            self.assertIsNone(report['error'])

    def test_spawned_actors_receive_the_queue_matching_their_private_reply_pipe(self):
        with self.pool() as service:
            identity=service.register(self.model);leases=service.channels(2);results=service.mp.Queue()
            processes=[service.mp.Process(target=spawned_predict,args=(leases,identity,seat,results)) for seat in range(2)]
            try:
                for process in processes:process.start()
                messages=[results.get(timeout=90) for _ in processes]
                for message in messages:self.assertNotIn('error',message)
                self.assertEqual({m['response']['observer'] for m in messages},{0,1})
                self.assertEqual({m['response']['status'] for m in messages},{'ok'})
                for process in processes:
                    process.join(timeout=30);self.assertEqual(process.exitcode,0)
                self.assertEqual([r['positions'] for r in service.report()['per_device']],[1,1])
            finally:
                for process in processes:
                    if process.is_alive():process.terminate();process.join(timeout=5)
                results.close();results.join_thread()

    def test_model_registration_cache_bound_and_trim_apply_to_every_replica(self):
        with self.pool() as service:
            one=service.register(self.model);other=copy.deepcopy(self.model)
            with torch.no_grad():next(other.parameters()).add_(.001)
            two=service.register(other)
            self.assertNotEqual(one,two)
            for child in service.services:
                child._model(one);child._model(two)
                self.assertEqual(set(child.models),{one,two})
                self.assertEqual(list(child.cache),[two])
            service.trim([one])
            self.assertEqual(set(service.models),{one})
            for child in service.services:
                self.assertEqual(set(child.models),{one});self.assertEqual(len(child.cache),0)

    def test_aggregate_receipts_sum_counts_maxima_and_overlapping_times_honestly(self):
        service=self.pool()
        for child,positions,maximum,seconds in zip(service.services,[3,4],[3,2],[1.,2.]):
            child.metrics['positions']=positions;child.metrics['max_batch_positions']=maximum
            child.timings.observe('inference.forward',seconds)
        report=service.report()
        self.assertEqual((report['positions'],report['max_batch_positions']),(7,3))
        self.assertEqual(report['timings']['inference.forward']['seconds'],3.)
        self.assertEqual(report['timings']['inference.forward']['calls'],2)
        self.assertEqual(report['replica_count'],2)
        error=ValueError('replica failure');service.services[1].error=error
        self.assertIs(service.error,error);self.assertEqual(service.report()['error'],'replica failure')

    def test_explicit_single_replica_and_legacy_default_preserve_lifecycle(self):
        selected=self.pool(['cuda:1'])
        self.assertEqual(selected.device,'cuda:1');self.assertEqual(selected.learner_device,'cuda:0')
        self.assertIsInstance(selected,inference.InferenceService)
        self.assertFalse(hasattr(selected,'services'));self.assertFalse(hasattr(selected,'router'))
        selected.__enter__();selected.__exit__(None,None,None);selected.__exit__(None,None,None)
        self.assertTrue(selected.closed)
        legacy=inference.InferenceService(self.config,'cpu')
        self.assertIsInstance(legacy,inference.InferenceService)
        legacy.__exit__(None,None,None)
        self.assertTrue(legacy.closed)

    def test_demand_routing_consolidates_sparse_traffic_and_preserves_outputs(self):
        self.config['inference_routing']='demand'
        self.config['replica_backlog']=32
        with self.pool(['cuda:'+str(i) for i in range(6)]) as service:
            identity=service.register(self.model);lease=service.channels(4)
            for seat in range(4):
                endpoint=lease.acquire();client=inference.RemoteEvaluator(endpoint,identity)
                response=client.predict([Request('demand-'+str(seat),identity,*position(seat),time.monotonic()+60)])[0]
                self.assertEqual(response['observer'],seat);self.assertEqual(response['model_id'],identity)
                endpoint.close()
            report=service.report()
            self.assertEqual([r['positions'] for r in report['per_device']],[4,0,0,0,0,0])
            self.assertEqual(report['shared_requests'],4);self.assertEqual(report['shared_replies'],4)
            self.assertEqual(report['routing']['mode'],'demand')

    def test_demand_routing_uses_additional_replica_with_backlog(self):
        self.config.update(inference_routing='demand',replica_backlog=1)
        with self.pool() as service:
            identity=service.register(self.model);lease=service.channels(4)
            clients=[inference.RemoteEvaluator(lease.acquire(),identity) for _ in range(4)]
            tickets=[c.submit([Request('backlog-'+str(i),identity,*position(i),time.monotonic()+60)]) for i,c in enumerate(clients)]
            for c,ticket in zip(clients,tickets):self.assertEqual(c.poll(ticket,wait=True)[0]['status'],'ok');c.endpoint.close()
            counts=service.report()['routing']['requests_per_replica']
            self.assertTrue(all(n>0 for n in counts));self.assertEqual(sum(counts),4)


if __name__=='__main__':unittest.main()
