"""Focused transport, replay overlap and simulator fast-path checks."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch

from fixtures import position
from brass_ai.bank import sha
from brass_ai.host import RulesHost,HostError
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.batching import Request,Evaluator
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.shared_transport import SharedSlot
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery.inference import InferenceService,RemoteEvaluator
from brass_ai.recovery.ingress import StreamingCollection,load_collection
from brass_ai.recovery.replay import SerializedGame,write_shard,cache_receipt
from brass_ai.recovery.host_runtime import training_host
from brass_ai.step5.policy import NeuralPolicy,predict_groups

class ThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_shared_roundtrip_bounds_token_and_oversize(self):
        e=PreparedEncoder().encode(*position());token='a'*32
        slot=SharedSlot(size=1024*1024)
        try:
            packed=slot.pack_frames([dict(encoded=e)],token)
            decoded=slot.unpack_frames(packed,token)[0]['encoded']
            for name in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(e,name),getattr(decoded,name))
            self.assertEqual(e.choice_ids,decoded.choice_ids)
            with self.assertRaisesRegex(ValueError,'Stale'):slot.unpack_frames(packed,'b'*32)
            bad=copy.deepcopy(packed);bad[0]['encoded']['shared'][0]['offset']=slot.size
            with self.assertRaisesRegex(ValueError,'exceeds'):slot.unpack_frames(bad,token)
            del decoded
            reply=[dict(probabilities=[.1,.2,.7],win_share=[.25]*4,placement=[.5]*4,vp=[300.]*4,utility=[1.234]*4)]
            self.assertEqual(reply,slot.unpack_responses(slot.pack_responses(reply,token),token))
        finally:slot.close()
        small=SharedSlot(size=128)
        try:self.assertIsNone(small.pack_frames([dict(encoded=e)],token))
        finally:small.close()

    def test_mixed_model_ready_leaves_keep_model_and_seat_attribution(self):
        models=[FourSeatNet(ModelConfig.preset('small')) for _ in range(2)]
        with InferenceService(dict(EXECUTION,processes=1,hosts_per_process=1),'cpu') as service:
            ids=[service.register(m) for m in models];endpoint=service.channels(1).acquire()
            policies=[NeuralPolicy(None,focus='neutral',evaluator=RemoteEvaluator(endpoint,i)) for i in ids]
            inputs=[(*position(i),random.Random(44+i)) for i in range(2)]
            before=[p.predict([x])[0] for p,x in zip(policies,inputs)]
            after=predict_groups([(p,[x]) for p,x in zip(policies,inputs)])
            for i,((prob,response),got) in enumerate(zip(before,after)):
                np.testing.assert_allclose(prob,got[0][0]);self.assertEqual(got[0][1]['model_id'],ids[i]);self.assertEqual(got[0][1]['observer'],i)
            self.assertEqual(service.report()['shared_requests'],3)
            endpoint.close()

    def test_inline_overflow_keeps_all_candidates(self):
        model=FourSeatNet(ModelConfig.preset('small'))
        with InferenceService(dict(EXECUTION,shared_slot_bytes=128),'cpu') as service:
            identity=service.register(model);endpoint=service.channels(1).acquire()
            request=Request('oversize',identity,*position(),time.monotonic()+60)
            result=RemoteEvaluator(endpoint,identity).predict([request])[0]
            expected=Evaluator(model).predict([request])[0]
            self.assertEqual(result['choice_ids'],expected['choice_ids']);np.testing.assert_array_equal(result['probabilities'],expected['probabilities'])
            self.assertEqual(service.report()['inline_requests'],1);endpoint.close()

    def test_shared_service_failure_releases_mappings(self):
        service=InferenceService(dict(EXECUTION,processes=1,hosts_per_process=1),'cpu')
        with service:
            endpoint=service.channels(1).acquire()
            try:
                request=Request('missing','unknown-model',*position(),time.monotonic()+60)
                with self.assertRaisesRegex(ValueError,'Unregistered frozen model'):
                    RemoteEvaluator(endpoint,'unknown-model').predict([request])
                self.assertIsNotNone(service.error)
            finally:endpoint.close()
        self.assertTrue(service.closed)

    def test_streaming_preparation_matches_barrier_and_rejects_receipt_change(self):
        import test_revision as revision
        model,policy,game=revision.RevisionTests().onpolicy({1})
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp,ThreadPoolExecutor(2) as pool:
            records=[];kinds={game['setup_group']:'selfplay'}
            with StreamingCollection(pool,2,policy.policy_id,kinds) as stream:
                for i in range(2):
                    episode=copy.deepcopy(game);episode['game_id']='stream-'+str(i)
                    serialized=SerializedGame(episode);path=Path(temp)/f'{i}.jsonl.gz'
                    with patch('brass_ai.recovery.replay.SerializedGame.__init__',side_effect=AssertionError('serialized twice')):
                        write_shard(path,[serialized],dict(purpose='training'))
                    record=dict(path=str(path),job=i,game_id=episode['game_id'],sha256=sha(path),features=cache_receipt(path))
                    records.append(record);stream.submit([record])
                metrics={};headers,data,diagnostics=stream.finish(records,metrics)
                expected=load_collection(records,policy.policy_id,kinds)
                self.assertEqual(headers,expected[0]);self.assertEqual(diagnostics,expected[2]);self.assertTrue(metrics['streaming'])
                for a,b in zip(data,expected[1]):
                    self.assertEqual(a[:3],b[:3]);np.testing.assert_array_equal(a[3].state,b[3].state);np.testing.assert_array_equal(a[3].actions,b[3].actions)
                with self.assertRaisesRegex(ValueError,'Changed streaming'):
                    stream.result(([dict(records[0],sha256='wrong')],policy.policy_id,kinds,None,None))

    def test_native_fused_forced_trace_and_revision_cache(self):
        directory=training_host(SOURCE_ROOT)
        with RulesHost() as old,RulesHost(directory) as fast:
            a=old.reset(153);b=fast.reset(153);self.assertTrue(fast.enable_fastpath())
            for _ in range(20):
                if a['terminal']:break
                view=old.observe(a['actor']);choices=old.choices()
                decision=fast.decision();self.assertEqual(view,decision['view']);self.assertEqual(choices,decision['choices'])
                chosen=next((c for c in choices if c['kind']=='loan'),choices[0])
                a=old.apply(a,chosen)
                reply=fast.step_decision(b,chosen,16)
                for entry in reply['forced']:
                    self.assertEqual(old.observe(a['actor']),entry['before']['view']);self.assertEqual(old.choices(),entry['before']['choices'])
                    self.assertEqual(len(entry['before']['choices']),1)
                    a=old.apply(a,entry['before']['choices'][0])
                b=reply['decision']['context']
                self.assertEqual(old.call('canonical_hash'),fast.call('canonical_hash'))
            fast.choices();fast.choices();self.assertGreater(fast.last_native_timing['choice_cache_hits'],0)
            saved=fast.call('checkpoint');before=fast.call('canonical_hash')
            with self.assertRaises(HostError):fast.apply(dict(b,revision=-1),chosen)
            self.assertEqual(before,fast.call('canonical_hash'))
            fast.call('restore',checkpoint=saved);self.assertEqual(before,fast.call('canonical_hash'))

    def test_native_search_fused_snapshots_preserve_chance_and_views(self):
        from brass_ai.searchlab.spiel import BrassSpielGame,action_id
        with RulesHost() as host:
            context=host.reset(153);view=host.observe(context['actor'])
        old=BrassSpielGame(root=SOURCE_ROOT);fast=BrassSpielGame(root=SOURCE_ROOT,fast=True)
        try:
            a=old.from_observation(view,842);b=fast.from_observation(view,842)
            for _ in range(16):
                self.assertEqual(a.views,b.views);self.assertEqual(a.pending,b.pending);self.assertEqual(a.legal,b.legal)
                if a.is_terminal():break
                action=a.chance_outcomes()[0][0] if a.pending else action_id(next((c for c in a.legal if c['kind']=='loan'),a.legal[0])['id'])
                a.apply_action(action);b.apply_action(action)
                self.assertEqual(a.checkpoint['hash'],b.checkpoint['hash'])
            c=b.clone();self.assertEqual(c.views,b.views)
        finally:old.close();fast.close()

if __name__=='__main__':unittest.main()
