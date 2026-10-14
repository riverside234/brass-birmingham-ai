"""Run the same numerical and buffer-ownership checks on CPU and lab CUDA."""
import copy
import time
import unittest

import numpy as np
import torch

from fixtures import position
from brass_ai.learning.batching import Evaluator,Request,accept_response,predict_encoded
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.inference import InferenceService,RemoteEvaluator,predict_packed


class PerformanceDeviceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def models(self):
        torch.manual_seed(573)
        config=ModelConfig.preset('small')
        reward=specification(dict(win=1.,placement=.1,vp=.0005),dict(event_cost=.005,max_cost=.025))
        return [FourSeatNet(config).eval(),UtilityNet(utility_config(config,reward)).eval()]

    def packed_outputs(self,device):
        encoder=PreparedEncoder();encoded=[]
        for seat in range(4):
            view,choices=position(seat)
            encoded.append(encoder.encode(view,choices[:1+seat%3],[.5,-.25,0.,1.],[.125]*8))
        for model in self.models():
            model.to(device)
            # Full padded batch plus every observer through the chunked path.
            for rows,limit in [(encoded,128)]+[([e],1) for e in encoded]:
                remaining=(torch.tensor([[.025,.020,.010,0.]]*len(rows),device=device)
                           if getattr(model.config,'utility',None) else None)
                with torch.inference_mode():
                    expected=predict_encoded(model,rows,device,limit,remaining)
                    actual=predict_packed(model,rows,device,limit,remaining)
                for left,right in zip(expected,actual):
                    self.assertEqual(left.shape,right.shape);self.assertEqual(right.dtype,np.float32)
                    self.assertTrue(np.isfinite(right).all())
                    np.testing.assert_allclose(left,right,rtol=1e-5,atol=1e-6)
                for row,prob in zip(rows,actual[0]):
                    self.assertAlmostEqual(float(prob.sum()),1.,places=6)
                    self.assertTrue(np.all(prob[len(row.choice_ids):]==0))

    def mixed_channel(self,device):
        models=self.models()
        with InferenceService(dict(EXECUTION,processes=1,hosts_per_process=1),device) as service:
            ids=[service.register(model) for model in models];endpoint=service.channels(1).acquire()
            try:
                clients=[RemoteEvaluator(endpoint,identity) for identity in ids]
                requests=[Request(str(i),identity,*position(3*i),time.monotonic()+120,
                    style=(.5,-.25,0.,1.),intent=(.125,)*8,remaining=(.025,.020,.010,0.) if i else None)
                    for i,identity in enumerate(ids)]
                groups=[(client,[request]) for client,request in zip(clients,requests)]
                received=clients[0].predict_mixed(groups);saved=copy.deepcopy(received)
                for model,request,replies in zip(models,requests,received):
                    expected=Evaluator(model,device).predict([request])[0]
                    accept_response(request,replies[0],request.view['context']['revision'],request.expected_model)
                    for field in ('probabilities','win_share','placement','vp','utility'):
                        if field in expected:np.testing.assert_allclose(replies[0][field],expected[field],rtol=1e-5,atol=1e-6)
                later=clients[0].predict_mixed(list(reversed(groups)))
                for request,replies in zip(reversed(requests),later):
                    accept_response(request,replies[0],request.view['context']['revision'],request.expected_model)
                # Reusing the shared slot must not alter already delivered data.
                self.assertEqual(received,saved)
                self.assertIsNone(endpoint.pending)
            finally:endpoint.close()

    def test_cpu_padded_and_chunked_four_seat_heads(self):self.packed_outputs('cpu')
    def test_cpu_mixed_models_and_reply_slot_reuse(self):self.mixed_channel('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
    def test_cuda_padded_and_chunked_four_seat_heads(self):self.packed_outputs('cuda')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
    def test_cuda_mixed_models_and_reply_slot_reuse(self):self.mixed_channel('cuda')


if __name__=='__main__':unittest.main()
