"""Fixed-input reuse, exact PPO updates, and fair streaming model batches."""
import copy
import gc
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.batching import collate
from brass_ai.learning.features import Encoded
from brass_ai.learning.local_entities import entity_layout
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.recovery import batch_cache, updates
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery.history import compact
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.inference_batches import ModelBatches
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.review_metrics import CategoricalProbe
from brass_ai.recovery.telemetry import Timings
import test_categorical_kl_guard as guard_fixtures
from test_categorical_kl_guard import probe_fixture
from test_mlp_v11 import inputs


class Caches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_probe_reuses_all_packed_inputs_without_repacking_or_rng(self):
        model,ds,_,_,probe=probe_fixture((3,2,1,3),2)
        expected=probe.measure_step(model);rng=torch.get_rng_state().clone()
        with patch.object(ds,'batch',side_effect=AssertionError('repacked fixed probe')):
            self.assertEqual(expected,probe.measure_step(model))
            self.assertAlmostEqual(probe.measure(model)['categorical_kl'],0.,places=7)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertGreater(probe.cache_report()['hits'],0)
        self.assertLessEqual(probe.cache_report()['resident_bytes'],probe.cache_report()['limit_bytes'])

    def test_probe_zero_budget_fallback_matches_cached_with_identical_masks(self):
        model,ds,behavior,cfg,cached=probe_fixture((3,2,1,3),2)
        with patch.object(batch_cache,'PROBE_CACHE_BYTES',0):
            fallback=CategoricalProbe(model,ds,[0,1,3],behavior,cfg)
        with torch.no_grad():model.logits.add_(torch.tensor([.2,-.3,.1]))
        self.assertEqual(cached.measure_step(model),fallback.measure_step(model))
        self.assertEqual(cached.measure(model),fallback.measure(model))
        self.assertEqual(fallback.cache_report()['resident_bytes'],0)
        self.assertEqual(fallback.cache_report()['cached_parts'],0)

    def test_probe_preserves_v3_exploration_and_per_row_opening_temperatures(self):
        model,ds,_,cfg,_=probe_fixture((3,2,1,3),2)
        for i,row in enumerate(ds.rows):
            row['behavior'].update(exploration_version=3,exploration=.02,
                temperature=2. if i<2 else 1.,action_decision=i)
        ds=updates.Dataset(ds.data,'cpu')
        behavior=SimpleNamespace(exploration=.02,temperature=1.,exploration_version=3,
                                 opening=dict(temperature=2.,action_decisions_per_seat=2))
        for part in ds.microbatches(range(4),2,cfg['max_candidate_cells']):
            ix,inputs,q=ds.batch(part,.02)
            with torch.no_grad():ds.old[ix]=updates.row_probabilities(ds,ix,model(*inputs)[0],q,behavior)[:,0]
        probe=CategoricalProbe(model,ds,[0,1,3],behavior,cfg)
        with torch.no_grad():model.logits.add_(torch.tensor([.1,-.2,.3]))
        self.assertAlmostEqual(probe.measure_step(model)['sampled_kl'],
            updates.policy_measure(model,ds,[0,1,3],behavior,cfg,Timings(),verify=False),places=7)

    def test_probe_changed_state_q_layout_and_selectors_are_rejected(self):
        for field in ('state','q','layout','selectors'):
            model,ds,_,_,probe=probe_fixture()
            if field=='state':ds.encoded[0].state[0]+=1
            elif field=='q':ds.q[0][0]+=.1
            elif field=='layout':ds.encoded[0].feature_version='stale'
            else:
                offset=entity_layout()['action_fields']['target']['start']
                ds.encoded[0].actions[0,offset]=.5
            with self.subTest(field=field),self.assertRaises(ValueError):probe.measure_step(model)

    def test_content_audit_revalidates_mutations_and_does_not_retain_slices(self):
        _,ds,_,_,_=probe_fixture()
        row=ds.encoded[0];audit=ds.input_audit
        calls=audit.checks
        collate([row],audit=audit);self.assertEqual(audit.checks,calls)
        offset=entity_layout()['action_fields']['target']['start']
        changed=copy.copy(row);changed.actions=row.actions.copy();changed.actions[0,offset]=.5
        with self.assertRaisesRegex(ValueError,'selector'):collate([changed],audit=audit)
        original=len(audit.proofs)
        for _ in range(5):
            temporary=Encoded(row.state,row.actions[:1],row.stable_to_relative,row.choice_ids[:1])
            collate([temporary],audit=audit)
        del temporary;gc.collect()
        self.assertEqual(len(audit.proofs),original)
        # The public/untrusted path always performs full selector admission.
        with self.assertRaisesRegex(ValueError,'selector'):collate([changed])

    def test_full_scan_reuses_owned_buffers_and_rejects_changed_inputs(self):
        model,ds,behavior,cfg,_=probe_fixture((3,2,1,3),2)
        indices=[0,1,3];timings=Timings()
        updates.policy_measure(model,ds,range(4),behavior,cfg,timings,verify=True)
        self.assertEqual(ds.scan_cache.bytes,0)
        expected=updates.policy_measure(model,ds,indices,behavior,cfg,timings)
        with patch.object(ds,'batch_cpu',side_effect=AssertionError('repacked cached scan')):
            self.assertEqual(expected,updates.policy_measure(model,ds,indices,behavior,cfg,timings))
        self.assertGreater(ds.scan_cache.report()['hits'],0)
        ds.old[0]*=.5
        with self.assertRaisesRegex(ValueError,'Wrong/stale'):
            updates.policy_measure(model,ds,indices,behavior,cfg,timings,verify=True)
        ds.encoded[0].state[0]+=1
        with self.assertRaisesRegex(ValueError,'inputs changed'):
            updates.policy_measure(model,ds,indices,behavior,cfg,timings)

    def test_bounded_scan_cache_pinned_prefetch_preserves_order(self):
        _,ds,_,_,_=probe_fixture()
        ds.scan_cache=batch_cache.BatchCache(limit=100000,max_entries=1)
        parts=[[2,0],[1,3]]
        with patch.object(torch.Tensor,'pin_memory',lambda t:t),ThreadPoolExecutor(max_workers=2) as pool:
            first=list(ds.policy_batches(parts,.08,pool,cache=True))
            again=list(ds.policy_batches(parts,.08,pool,cache=True))
        for expected,actual in zip(first,again):
            self.assertEqual(expected[0],actual[0])
            for a,b in zip((expected[1],*expected[2],expected[3]),(actual[1],*actual[2],actual[3])):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(ds.scan_cache.report()['entries'],1)
        self.assertLessEqual(ds.scan_cache.bytes,ds.scan_cache.limit)
        self.assertGreater(ds.scan_cache.refused,0)

    def test_complete_ppo_cache_on_off_preserves_weights_adam_rng_and_guards(self):
        original,optimizer,behavior,data,cfg=guard_fixtures.GuardTests().fixture()
        initial_adam=copy.deepcopy(optimizer.state_dict());results=[]
        for enabled in (False,True):
            model=copy.deepcopy(original);opt=torch.optim.AdamW(model.parameters())
            opt.load_state_dict(copy.deepcopy(initial_adam));torch.manual_seed(617)
            with patch.object(batch_cache,'SCAN_CACHE_BYTES',128*1024*1024 if enabled else 0), \
                 patch.object(batch_cache,'PROBE_CACHE_BYTES',256*1024*1024 if enabled else 0):
                report=updates.ppo_update(model,opt,[dict(split='train')],behavior,cfg,learner_data=data)
            self.assertEqual(report['kl_guard']['probe_checks'],report['optimizer_steps'])
            self.assertEqual(compact(dict(actor=report))['actor']['input_cache'],report['input_cache'])
            semantic={k:v for k,v in report.items() if k not in ('timings','memory','input_cache')}
            results.append((copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),
                            torch.get_rng_state().clone(),semantic))
        self.assertTrue(_same_payload(results[0],results[1]))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_probe_residency_and_zero_device_budget_host_fallback(self):
        model,ds,behavior,cfg,_=probe_fixture()
        model=model.cuda();gpu=updates.Dataset(ds.data,'cuda')
        resident=CategoricalProbe(model,gpu,[0,1,2,3],behavior,cfg)
        with patch.object(batch_cache,'probe_device_limit',return_value=0):
            host=CategoricalProbe(model,gpu,[0,1,2,3],behavior,cfg)
        self.assertEqual(resident.measure_step(model),host.measure_step(model))
        self.assertGreater(host.cache_report()['host_fallback_bytes'],0)
        self.assertTrue(all(t.device.type=='cuda' for t in resident.batches[0]))


class ThreadQueue(queue.Queue):
    def cancel_join_thread(self):pass
    def close(self):pass


class StreamingBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def frame(self,model='a',width=3,submitted=None,deadline=None):
        state,actions,_,mapping=inputs(1,width)
        now=time.monotonic()
        return dict(model=model,submitted=now if submitted is None else submitted,
                    deadline=now+30 if deadline is None else deadline,cancelled=False,
                    request_id='request',revision=1,seat=0,digest='source',
                    encoded=Encoded(state[0].numpy(),actions[0].numpy(),mapping[0].numpy(),
                                    tuple(str(i) for i in range(width))))

    def test_two_separate_models_outlier_padding_limits_and_no_starvation(self):
        schedule=ModelBatches(8,10000)
        frames=[self.frame('a',n,submitted=1+i*.001,deadline=100) for i,n in enumerate((1,2,4,772))]
        frames+=[self.frame('b',3,submitted=.5,deadline=100)]
        for i,frame in enumerate(frames):schedule.add((i,0),frame)
        seen=[]
        while schedule:
            batch=schedule.take(now=10);seen.extend(key for key,_ in batch)
            self.assertEqual(len({frame['model'] for _,frame in batch}),1)
            counts=[len(frame['encoded'].choice_ids) for _,frame in batch]
            self.assertLessEqual(len(batch)*max(counts),10000)
            self.assertLessEqual(len(batch)*max(counts),2*sum(counts))
        self.assertEqual(seen[0],(4,0));self.assertEqual(len(set(seen)),5)
        self.assertGreater(schedule.padding_splits,0)

    def test_oversized_position_is_alone_and_keeps_every_candidate(self):
        schedule=ModelBatches(8,64)
        for i,n in enumerate((772,2)):schedule.add((i,0),self.frame(width=n,submitted=i))
        batch=schedule.take()
        self.assertEqual(len(batch),1);self.assertEqual(len(batch[0][1]['encoded'].choice_ids),772)
        self.assertEqual(len(schedule.take()),1)

    def test_ready_model_batch_is_preferred_until_an_older_queue_needs_service(self):
        schedule=ModelBatches(8,10000)
        schedule.add((0,0),self.frame('thin',submitted=99.999,deadline=200))
        for i in range(4):schedule.add((i+1,0),self.frame('busy',submitted=99.9999,deadline=200))
        self.assertEqual(len(schedule.take(now=100)),4)
        for i in range(4):schedule.add((i+5,0),self.frame('busy',submitted=100.029,deadline=200))
        self.assertEqual(schedule.take(now=100.030)[0][1]['model'],'thin')

    def test_young_expiring_request_cannot_hide_behind_its_models_older_head(self):
        schedule=ModelBatches(8,10000)
        schedule.add((0,0),self.frame('a',submitted=99.985,deadline=200))
        schedule.add((1,0),self.frame('a',submitted=99.999,deadline=100.001))
        for i in range(4):schedule.add((i+2,0),self.frame('b',submitted=99.999,deadline=200))
        self.assertEqual(schedule.take(now=100)[0][0],(1,0))

    def test_stopped_unencoded_requests_retire_without_loading_a_model(self):
        for cancelled in (False,True):
            with self.subTest(cancelled=cancelled),InferenceService(
                    dict(EXECUTION,shared_slot_bytes=0),'cpu') as service:
                endpoint=service.channels(1).acquire()
                frame=self.frame('not-registered');frame['encoded']=None
                frame['cancelled']=cancelled
                frame['deadline']=time.monotonic()+30 if cancelled else time.monotonic()-1
                try:
                    service.requests.put(dict(lane=endpoint.lane,token='stopped',frames=[frame]))
                    self.assertTrue(endpoint.connection.poll(5));response=endpoint.connection.recv()
                    self.assertNotIn('error',response)
                    self.assertEqual(response['responses'][0]['status'],'cancelled' if cancelled else 'expired')
                    self.assertEqual(service.metrics['batches'],0);self.assertIsNone(service.error)
                finally:endpoint.close()

    def test_arrivals_during_forward_join_pending_model_and_reply_once(self):
        service=InferenceService(dict(EXECUTION,shared_slot_bytes=0,flush_ms=1.),'cpu')
        original_queue=service.requests;service.requests=ThreadQueue()
        original_queue.cancel_join_thread();original_queue.close()
        lease=service.channels(3);endpoints=[lease.acquire() for _ in range(3)]
        frames=[self.frame('a',submitted=1),self.frame('b',submitted=2),self.frame('b',submitted=3)]
        messages=[dict(lane=endpoints[i].lane,token=str(i),frames=[frame]) for i,frame in enumerate(frames)]
        service.requests.put(messages[0]);service.requests.put(messages[1]);batches=[];completed=[]
        all_completed=threading.Event()
        def evaluate(items):
            batches.append([frame['model'] for _,frame in items])
            if len(batches)==1:service.requests.put(messages[2])
            return {key:dict(status='ok',request_id=frame['request_id'],model_id=frame['model']) for key,frame in items}
        def complete(messages):
            completed.extend(m['token'] for m in messages)
            if len(completed)==3:all_completed.set()
        service._evaluate=evaluate;service.on_complete=complete
        try:
            with service:
                for i,endpoint in enumerate(endpoints):
                    self.assertTrue(endpoint.connection.poll(5))
                    result=endpoint.connection.recv();self.assertEqual(result['token'],str(i))
                    self.assertEqual(result['responses'][0]['model_id'],frames[i]['model'])
                self.assertEqual(batches,[['a'],['b','b']])
                self.assertTrue(all_completed.wait(2))
                self.assertEqual(completed,['0','1','2'])
                for endpoint in endpoints:self.assertFalse(endpoint.connection.poll(.01))
                self.assertIsNone(service.error)
        finally:
            for endpoint in endpoints:endpoint.close()
            service.__exit__(None,None,None)

    def test_shared_mixed_model_envelope_keeps_order_and_waits_for_all_frames(self):
        for slot_bytes in (0,65536):
            with self.subTest(slot_bytes=slot_bytes),InferenceService(
                    dict(EXECUTION,shared_slot_bytes=slot_bytes,flush_ms=1.),'cpu') as service:
                torch.manual_seed(88);model=FourSeatNet(ModelConfig.preset('small')).eval()
                a=service.register(model)
                with torch.no_grad():next(model.parameters()).add_(.01)
                b=service.register(model);endpoint=service.channels(1).acquire()
                frames=[self.frame(b,2),self.frame(a,3),self.frame(b,1)]
                frames[1]['cancelled']=True
                frames[2]['deadline']=time.monotonic()-1
                token='a'*32;slot=endpoint.slot();packed=slot.pack_frames(frames,token) if slot else None
                try:
                    service.requests.put(dict(lane=endpoint.lane,token=token,shared=packed is not None,
                                              frames=packed if packed is not None else frames))
                    self.assertTrue(endpoint.connection.poll(10));reply=endpoint.connection.recv()
                    self.assertNotIn('error',reply)
                    results=slot.unpack_responses(reply['responses'],token) if reply['shared'] else reply['responses']
                    self.assertEqual([r['status'] for r in results],['ok','cancelled','expired'])
                    self.assertEqual([r['model_id'] for r in results],[b,a,b])
                    self.assertEqual(len(results[0]['probabilities']),2)
                    for result in results[1:]:self.assertNotIn('probabilities',result)
                    self.assertFalse(endpoint.connection.poll(.01));self.assertIsNone(service.error)
                finally:endpoint.close()


if __name__=='__main__':unittest.main()
