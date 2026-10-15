"""Actor-lifetime feature reuse without cross-model prediction reuse."""
import copy
from concurrent.futures import ThreadPoolExecutor
import gc
from threading import Barrier, Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import torch

from brass_ai.learning.batching import Evaluator, Request
from brass_ai.learning.encoding_budget import EncodingStopped
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery.encoding import EncodingCache, PreparedEncoder
from brass_ai.recovery.execution import policy_encoder, run_game
from brass_ai.recovery.inference import Endpoint, InferenceService, RemoteEvaluator, share_encoder_cache
from brass_ai.recovery.telemetry import Timings
from fe_fixtures import Factory
from fixtures import rotate


def endpoints(count):
    return [Endpoint(Mock(),Mock(),i) for i in range(count)]


def equal_encoded(test,actual,expected):
    test.assertEqual(actual.choice_ids,expected.choice_ids)
    for name in ('state','actions','stable_to_relative'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
        test.assertFalse(getattr(actual,name).flags.writeable)


class EncoderReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        factory=Factory()
        try:cls.view,cls.choices=factory.position()
        finally:factory.close()

    def test_one_encoder_per_endpoint_survives_frozen_model_replacement(self):
        lanes=endpoints(9);shared=share_encoder_cache(lanes)
        with patch('brass_ai.recovery.inference.PreparedEncoder',wraps=PreparedEncoder) as constructor:
            first=RemoteEvaluator(lanes[0],'a'*64)
            encoded=first.encoder.encode(self.view,self.choices)
            reference=weakref.ref(first);del first;gc.collect()
            second=RemoteEvaluator(lanes[0],'b'*64)
            self.assertIsNone(reference())
            self.assertIs(second.encoder.encode(self.view,self.choices),encoded)
            other=RemoteEvaluator(lanes[1],'a'*64)
            self.assertIsNot(second.encoder,other.encoder)
            self.assertIsNot(second.encoder.action_cache,other.encoder.action_cache)
            self.assertIsNot(second.encoder.derived,other.encoder.derived)
            self.assertIs(other.encoder.encode(self.view,self.choices),encoded)
            self.assertEqual(constructor.call_count,2)
        self.assertEqual(shared.report(),dict(warm_hit=2,miss=1,evictions=0,retained=1,capacity=2304))
        with self.assertRaisesRegex(ValueError,'before creating'):share_encoder_cache(lanes)
        lanes[0].close()
        self.assertIsNone(lanes[0]._encoder)
        self.assertIs(other.encoder.encode(self.view,self.choices),encoded)

    def test_inputs_context_choices_history_and_stable_seats_are_content_bound(self):
        shared=EncodingCache();a=PreparedEncoder(context_cache=shared);b=PreparedEncoder(context_cache=shared)
        oracle=Encoder();view,choices=self.view,self.choices
        cases=[(view,choices,[0.]*4,[0.]*8)]
        for shift in range(1,4):cases.append((rotate(view,shift),choices,[0.]*4,[0.]*8))
        changed=copy.deepcopy(view);changed['players'][view['seat']]['cash']+=1
        cases.append((changed,choices,[0.]*4,[0.]*8))
        changed=copy.deepcopy(view);changed['context']['revision']+=1
        cases.append((changed,choices,[0.]*4,[0.]*8))
        cases.extend([(view,list(reversed(choices)),[0.]*4,[0.]*8),
                      (view,choices,[.2,-.1,0.,.3],[0.]*8),
                      (view,choices,[0.]*4,[1.]+[0.]*7)])
        # A public history edit must miss even when all current board facts match.
        changed=copy.deepcopy(view)
        # Use a genuine wire history entry, including its required event fields.
        factory=Factory()
        try:
            factory.choose('develop',industry='CottonMill')
            changed['history']=factory.view()['history']
        finally:factory.close()
        self.assertTrue(changed['history']);cases.append((changed,choices,[0.]*4,[0.]*8))
        for v,c,style,intent in cases:
            with self.subTest(seat=v['seat'],style=style,intent=intent,revision=v['context']['revision']):
                expected=oracle.encode(v,c,style,intent)
                got=a.encode(v,c,style,intent)
                equal_encoded(self,got,expected)
                self.assertIs(b.encode(copy.deepcopy(v),copy.deepcopy(c),tuple(style),tuple(intent)),got)
        self.assertEqual(shared.report()['miss'],len(cases))
        self.assertEqual(shared.report()['warm_hit'],len(cases))
        invalid=copy.deepcopy(view);invalid['unknown']=1
        with self.assertRaises(Exception):b.encode(invalid,choices)
        duplicate=choices+[choices[0]]
        with self.assertRaisesRegex(ValueError,'duplicate'):b.encode(view,duplicate)

    def test_parallel_misses_do_not_hold_the_shared_lock_and_lru_stays_bounded(self):
        shared=EncodingCache(3);encoders=[PreparedEncoder(context_cache=shared) for _ in range(8)]
        barrier=Barrier(8);original=PreparedEncoder.append_derived
        def build(owner,*args,**kwargs):
            barrier.wait(timeout=10)
            return original(owner,*args,**kwargs)
        def run(i):
            view=copy.deepcopy(self.view);view['players'][view['seat']]['cash']+=i
            return encoders[i].encode(view,self.choices),view
        with patch.object(PreparedEncoder,'append_derived',build),ThreadPoolExecutor(max_workers=8) as pool:
            rows=list(pool.map(run,range(8)))
        oracle=Encoder()
        for got,view in rows:equal_encoded(self,got,oracle.encode(view,self.choices))
        self.assertEqual((len(shared),shared.report()['evictions']),(3,5))
        # Atomic lookup retains its local reference even if another lane evicts it.
        item=rows[0][0];shared.clear();shared.put('a',item);shared.put('b',item);shared.put('c',item)
        self.assertIs(shared.get('a'),item);shared.put('d',item)
        self.assertIsNone(shared.get('b'));self.assertIs(shared.get('a'),item)
        disabled=PreparedEncoder(cache_size=0);disabled.encode(self.view,self.choices)
        self.assertEqual(len(disabled.cache),0)

    def test_shared_hits_and_failed_misses_still_obey_deadlines_and_cancellation(self):
        shared=EncodingCache();a=PreparedEncoder(context_cache=shared);b=PreparedEncoder(context_cache=shared)
        saved=a.encode(self.view,self.choices);event=Event();event.set()
        with self.assertRaises(EncodingStopped):b.encode(self.view,self.choices,cancelled=event)
        with self.assertRaises(EncodingStopped):b.encode(self.view,self.choices,deadline=time.monotonic()-1)
        event.clear();view=copy.deepcopy(self.view);view['players'][view['seat']]['cash']+=3
        original=b.append_derived
        def stopped(*args,**kwargs):
            value=original(*args,**kwargs);event.set();return value
        with patch.object(b,'append_derived',stopped):
            with self.assertRaises(EncodingStopped):b.encode(view,self.choices,cancelled=event)
        self.assertEqual(len(shared),1);event.clear()
        self.assertIs(b.encode(self.view,self.choices),saved)

    def test_teacher_replay_uses_its_base_encoder_without_reconstructing(self):
        lane=endpoints(1)[0];client=RemoteEvaluator(lane,'a'*64)
        policy=SimpleNamespace(base=SimpleNamespace(evaluator=client),policy_id='test')
        self.assertIs(policy_encoder(policy),client.encoder)
        self.assertIs(policy_encoder(SimpleNamespace(evaluator=client)),client.encoder)
        result=SimpleNamespace();host=SimpleNamespace(reset=lambda seed:{})
        with patch('brass_ai.recovery.execution._run_turns',return_value=result) as turns, \
             patch('brass_ai.recovery.execution.PreparedEncoder',side_effect=AssertionError('rebuilt')):
            actual=run_game(host,dict(seed=1,sampling_seed=2,provenance={}),[policy]*4,Timings(),10)
        self.assertIs(actual,result);self.assertIs(turns.call_args.args[9],client.encoder)

    def test_actual_remote_predictions_keep_model_and_remaining_budget_separate(self):
        reward=specification(dict(win=1.,placement=.1,vp=.001),dict(event_cost=.005,max_cost=.075))
        models=[]
        for seed in (7481,7482):
            torch.manual_seed(seed);models.append(UtilityNet(utility_config(ModelConfig.preset('small'),reward)))
        with InferenceService(dict(EXECUTION,processes=1,hosts_per_process=1),'cpu') as service:
            identities=[service.register(model) for model in models]
            lane=service.channels(1).acquire();self.addCleanup(lane.close)
            clients=[RemoteEvaluator(lane,identity) for identity in identities]
            self.assertIs(clients[0].encoder,clients[1].encoder)
            replies=[]
            for i,(model,client) in enumerate(zip(models,clients)):
                local=Evaluator(model);self.addCleanup(local.close)
                requests=[Request(f'{i}-{j}',client.model_id,self.view,self.choices,time.monotonic()+60,
                          remaining=remaining) for j,remaining in enumerate((None,(0.,)*4))]
                expected=local.predict(requests);actual=client.predict(requests)
                for got,want in zip(actual,expected):
                    self.assertEqual(got['model_id'],want['model_id'])
                    self.assertEqual(got['choice_ids'],want['choice_ids'])
                    for field in ('probabilities','win_share','placement','vp','utility'):
                        np.testing.assert_allclose(got[field],want[field],rtol=1e-5,atol=2e-5)
                replies.append(actual)
            self.assertFalse(np.array_equal(replies[0][0]['probabilities'],replies[1][0]['probabilities']))
            cache=clients[0].encoder.cache.report()
            self.assertEqual((cache['miss'],cache['warm_hit']),(1,3))
            # Request mutation is rejected even when its original features are warm.
            request=Request('mutated',identities[0],self.view,self.choices,time.monotonic()+60)
            request.view['players'][self.view['seat']]['cash']+=1
            with self.assertRaisesRegex(ValueError,'Mutated'):clients[0].submit([request])
            self.assertIsNone(lane.pending);self.assertIsNone(service.error)


if __name__=='__main__':unittest.main()
