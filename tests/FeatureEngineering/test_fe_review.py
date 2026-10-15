"""Review regressions: cooperative encoding stops and explicit raw re-audits."""
from collections import defaultdict
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace
import copy
import gzip
import json
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import torch

from brass_ai.bank import sha
from brass_ai.feature_engineering.context import DerivedContextBuilder
from brass_ai.feature_engineering.data import raw_compatible, convert
from brass_ai.learning.batching import Evaluator, Request
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, assert_compatible, assert_heuristic_data_compatible
from brass_ai.learning.encoding_budget import EncodingStopped
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.personality.contracts import context
from brass_ai.personality.search.inference import PersonalityRemote, PersonalityRequest, SharedInference
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.inference import InferenceService, RemoteEvaluator
from brass_ai.recovery.telemetry import Timings
from fe_fixtures import Factory
from fixtures import game


class EncodingBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        f=Factory()
        try:cls.view,cls.choices=f.position()
        finally:f.close()

    def test_cancel_during_dynamic_effects_stops_without_committing_context(self):
        from brass_ai.feature_engineering.effects import action_effects
        builder=DerivedContextBuilder();event=Event();calls=[]
        def effect(*args):
            calls.append(args[1]['id']);event.set();return action_effects(*args)
        with patch('brass_ai.feature_engineering.context.action_effects',side_effect=effect):
            with self.assertRaises(EncodingStopped) as stopped:
                builder.build(self.view,self.choices,cancelled=event)
        self.assertEqual(stopped.exception.reason,'cancelled')
        self.assertEqual(len(calls),1);self.assertFalse(builder.cache)
        event.clear()
        result=builder.build(self.view,self.choices,cancelled=event)
        self.assertEqual(len(result.actions),len(self.choices))

    def test_cancel_during_tensorization_or_cached_lookup_cannot_publish_arrays(self):
        from brass_ai.feature_engineering.context import DerivedContext
        encoder=PreparedEncoder();event=Event();original=DerivedContext.action_tensors
        def tensors(value):
            result=original(value);event.set();return result
        with patch.object(DerivedContext,'action_tensors',tensors):
            with self.assertRaises(EncodingStopped):encoder.encode(self.view,self.choices,cancelled=event)
        self.assertFalse(encoder.cache)
        event.clear();expected=encoder.encode(self.view,self.choices)
        event.set()
        with self.assertRaises(EncodingStopped):encoder.encode(self.view,self.choices,cancelled=event)
        event.clear()
        self.assertIs(encoder.encode(self.view,self.choices,cancelled=event),expected)

    def test_cached_lookup_rechecks_clock_after_content_hashing(self):
        encoder=PreparedEncoder();encoder.encode(self.view,self.choices)
        from brass_ai.recovery.encoding import digest
        clock=Mock(return_value=10.)
        def hashed(value):
            result=digest(value);clock.return_value=21.;return result
        with patch('brass_ai.learning.encoding_budget.time.monotonic',clock), \
             patch('brass_ai.recovery.encoding.digest',side_effect=hashed):
            with self.assertRaises(EncodingStopped) as stopped:
                encoder.encode(self.view,self.choices,deadline=20.)
        self.assertEqual(stopped.exception.reason,'expired')

    def test_nonfinite_clocks_do_not_disable_per_turn_admission(self):
        encoder=PreparedEncoder()
        for deadline in (float('nan'),float('inf'),float('-inf')):
            with self.subTest(deadline=deadline),self.assertRaisesRegex(ValueError,'Invalid feature decision deadline'):
                encoder.encode(self.view,self.choices,deadline=deadline)

    def test_local_evaluator_retires_cancelled_encoding_and_keeps_live_request(self):
        evaluator=Evaluator(FourSeatNet(ModelConfig.preset('small')))
        self.addCleanup(evaluator.close)
        event=Event();deadline=time.monotonic()+120
        cancelled=Request('cancel',evaluator.model_id,self.view,self.choices,deadline,cancelled=event)
        live=Request('live',evaluator.model_id,self.view,self.choices,deadline)
        encode=evaluator.encoder.encode;calls=[]
        def encode_request(*args,**kwargs):
            calls.append(kwargs)
            if len(calls)==1:event.set()
            return encode(*args,**kwargs)
        with patch.object(evaluator.encoder,'encode',side_effect=encode_request):
            replies=evaluator.predict([cancelled,live])
        self.assertEqual([r['status'] for r in replies],['cancelled','ok'])
        self.assertEqual(calls[0],dict(deadline=deadline,cancelled=event))
        self.assertAlmostEqual(sum(replies[1]['probabilities']),1.,places=6)
        self.assertEqual(evaluator.telemetry['positions'],1)

    def remote(self,personality):
        endpoint=SimpleNamespace(pending=None,requests=Mock(),lock=RLock(),lane=0,slot=lambda:None)
        owner=(PersonalityRemote if personality else RemoteEvaluator)(endpoint,'a'*64)
        server=SimpleNamespace(timings=Timings(),metrics=defaultdict(int),_response=InferenceService._response)
        evaluate=SharedInference._evaluate if personality else InferenceService._evaluate
        return owner,endpoint,server,evaluate

    def test_remote_and_personality_forward_the_same_clock_and_retire_encoding_stops(self):
        for personality in (False,True):
            for reason in ('expired','cancelled'):
                with self.subTest(personality=personality,reason=reason):
                    owner,endpoint,server,evaluate=self.remote(personality)
                    event=Event();deadline=20.;clock=Mock(return_value=10.)
                    cls=PersonalityRequest if personality else Request
                    extra=dict(personality=context()) if personality else {}
                    request=cls('stop',owner.model_id,self.view,self.choices,deadline,cancelled=event,**extra)
                    encode=owner.encoder.encode
                    def interrupt(*args,**kwargs):
                        if reason=='expired':clock.return_value=21.
                        else:event.set()
                        return encode(*args,**kwargs)
                    with patch('brass_ai.learning.encoding_budget.time.monotonic',clock), \
                         patch.object(owner.encoder,'encode',side_effect=interrupt) as encoded:
                        owner.submit([request])
                        frames=endpoint.requests.put.call_args.args[0]['frames']
                        replies=evaluate(server,[(0,frames[0])])
                    self.assertIsNone(frames[0]['encoded']);self.assertEqual(replies[0]['status'],reason)
                    self.assertEqual(encoded.call_args.kwargs,dict(deadline=deadline,cancelled=event))

    def test_unrelated_encoding_errors_still_abort_instead_of_becoming_timeouts(self):
        for personality in (False,True):
            owner,endpoint,_,_=self.remote(personality)
            cls=PersonalityRequest if personality else Request
            extra=dict(personality=context()) if personality else {}
            request=cls('bad',owner.model_id,self.view,self.choices,time.monotonic()+120,**extra)
            with patch.object(owner.encoder,'encode',side_effect=ValueError('corrupt permitted facts')):
                with self.assertRaisesRegex(ValueError,'corrupt permitted facts'):owner.submit([request])
            endpoint.requests.put.assert_not_called();self.assertIsNone(endpoint.pending)


class RawReauditTests(unittest.TestCase):
    def previous(self):
        old=compatibility()
        old.update(encoder_sha256='a'*64,architecture_sha256='b'*64,derived_source_sha256='c'*64)
        return old

    def test_previous_features2_sources_can_only_enter_explicit_raw_conversion(self):
        old=self.previous();raw_compatible(old)
        with self.assertRaises(ValueError):assert_compatible(old)
        with self.assertRaises(ValueError):assert_heuristic_data_compatible(old)
        for field in ('rules','features_sha256','feature_version','wire_sha256','loss','objective'):
            with self.subTest(field=field),self.assertRaises(ValueError):raw_compatible(dict(old,**{field:'unknown'}))
        for field in ('encoder_sha256','architecture_sha256','derived_source_sha256'):
            with self.subTest(field=field),self.assertRaises(ValueError):raw_compatible(dict(old,**{field:'bad'}))

    def test_previous_features2_neural_or_search_labels_cannot_be_relabelled(self):
        for teacher in (False,True):
            with self.subTest(teacher=teacher),tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
                directory=Path(directory);source=directory/'old.jsonl.gz';episode=copy.deepcopy(game())
                for row in episode['records']:row['behavior']['kind']='external'
                if teacher:episode['records'][0]['teacher']={}
                else:episode['records'][0]['behavior']['kind']='model'
                with gzip.open(source,'wt',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(format='four-seat-replay/1.0.0',compatibility=self.previous(),provenance=dict(purpose='training')))+'\n')
                    stream.write(json.dumps(episode)+'\n')
                token=sha(source);manifest=directory/'manifest.json'
                manifest.write_text(json.dumps(dict(format='bc-dataset/1.0.0',label_kind='qualified-v3-choice/1.0.0',
                    context='neutral-predecision/1.0.0',compatibility=self.previous(),teachers=['fixture'],entries=[
                    dict(path=source.relative_to(SOURCE_ROOT).as_posix(),sha256=token,game_id=episode['game_id'],
                         split=episode['split'],setup_group=episode['setup_group'],transitions_verified=True)])))
                with self.assertRaisesRegex(ValueError,'Neural behavior/search teachers'):convert(SOURCE_ROOT,manifest,directory/'out')
                self.assertEqual(sha(source),token);self.assertFalse((directory/'out/manifest.json').exists())


if __name__=='__main__':unittest.main()
