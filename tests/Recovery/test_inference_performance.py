"""Exact feature/output preservation for inference transport optimizations."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixtures import position
from brass_ai.learning.batching import predict_encoded
from brass_ai.learning.contracts import digest
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery import encoding,inference
from brass_ai.recovery.inputs import InputBoundary
from brass_ai.recovery.config import EXECUTION


class InferencePerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_boundary_content_identity_reuses_one_full_input_hash(self):
        boundary=InputBoundary();encoder=encoding.PreparedEncoder();view,choices=position()
        with patch.object(encoding,'BOUNDARY',boundary),patch.object(encoding,'digest',wraps=digest) as hashed:
            first=encoder.encode(view,choices)
            self.assertEqual(hashed.call_count,1)
            self.assertEqual(set(hashed.call_args.args[0]),{'input','style','intent'})
            second=encoder.encode(view,choices,validated=True)
            self.assertIs(first,second)
            self.assertEqual(hashed.call_count,3)  # input + context on the prevalidated path.
        self.assertEqual(boundary.checked_frames,1)
        for array in (first.state,first.actions,first.stable_to_relative):self.assertFalse(array.flags.writeable)

    def test_mutations_and_context_cannot_hit_stale_feature_cache(self):
        encoder=encoding.PreparedEncoder();view,choices=position()
        initial=encoder.encode(view,choices)
        view['players'][0]['cash']+=1
        changed=encoder.encode(view,choices)
        self.assertIsNot(initial,changed)
        self.assertFalse(np.array_equal(initial.state,changed.state))
        style=[1.,0.,0.,0.];context=encoder.encode(view,choices,style,[0.]*8)
        self.assertIsNot(changed,context)
        np.testing.assert_array_equal(context.state[-12:],style+[0.]*8)
        view['unpermitted_opponent_hand']=['industry:Pottery']
        with self.assertRaises(Exception):encoder.encode(view,choices)

    def test_changed_cache_key_preserves_pinned_layout_all_seats(self):
        fast,reference=encoding.PreparedEncoder(),Encoder()
        for seat in range(4):
            view,choices=position(seat);style=[.5,-.25,0.,1.];intent=[.25]*8
            a=fast.encode(view,choices,style,intent);b=reference.encode(view,choices,style,intent)
            self.assertEqual(a.choice_ids,b.choice_ids)
            for name in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(a,name),getattr(b,name))

    def models(self):
        config=ModelConfig.preset('small')
        reward=specification(dict(win=1.,placement=.1,vp=.0005),dict(event_cost=.005,max_cost=.025))
        return [FourSeatNet(config),UtilityNet(utility_config(config,reward))]

    def test_packed_transfer_preserves_four_and_five_head_predictions_exactly(self):
        encoder=encoding.PreparedEncoder();encoded=[encoder.encode(*position(seat)) for seat in range(4)]
        for model in self.models():
            model.eval();remaining=torch.tensor([[.025,.020,.010,0.]]*4) if getattr(model.config,'utility',None) else None
            with torch.inference_mode():
                before=predict_encoded(model,encoded,'cpu',1024,remaining)
                after=inference.predict_packed(model,encoded,'cpu',1024,remaining)
            self.assertEqual(len(before),len(after))
            for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_normal_batch_uses_one_host_transfer(self):
        model=self.models()[1].eval();encoder=encoding.PreparedEncoder();encoded=[encoder.encode(*position())]
        original=torch.Tensor.cpu;calls=[]
        def tracked(tensor,*args,**kwargs):
            calls.append(tuple(tensor.shape));return original(tensor,*args,**kwargs)
        with torch.inference_mode(),patch.object(torch.Tensor,'cpu',tracked):
            actual=inference.predict_packed(model,encoded,'cpu',1024)
        self.assertEqual(len(actual),5);self.assertEqual(calls,[(1,19)])

    def test_oversized_position_keeps_existing_candidate_chunking(self):
        model=self.models()[1].eval();encoder=encoding.PreparedEncoder();encoded=[encoder.encode(*position())]
        remaining=torch.tensor([[.025,.020,.010,0.]])
        with torch.inference_mode(),patch.object(inference,'predict_encoded',wraps=predict_encoded) as chunked:
            actual=inference.predict_packed(model,encoded,'cpu',2,remaining)
            expected=predict_encoded(model,encoded,'cpu',2,remaining)
        self.assertEqual(chunked.call_count,1)
        self.assertIs(chunked.call_args.args[-1],remaining)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)

    def test_cpu_service_keeps_zero_copy_head_output_path(self):
        with inference.InferenceService(EXECUTION,'cpu') as service:
            self.assertFalse(service.pack_outputs)


if __name__=='__main__':unittest.main()
