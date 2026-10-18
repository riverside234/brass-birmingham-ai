"""ONNX v11.2 input/default, active critic conditioning and actor invariance."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.learning.reward import specification
from brass_ai.learning.seat_descriptors import unknown_numpy
from brass_ai.personality.laptop.export import export,Graph,samples,BASE_INPUTS,BASE_OUTPUTS
from brass_ai.learning.export import export_onnx,load_export,INPUTS,OUTPUTS
from brass_ai.learning.features import Encoded
from test_mlp_v11 import inputs


class Exports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_trained_descriptor_heads_export_parity_unknown_default_and_masking(self):
        model=UtilityNet(utility_config(ModelConfig.preset('small'),specification())).eval()
        with torch.no_grad():
            model.outcomes[0].weight[:,-32:].normal_(std=.01)
            model.utility_head[0].weight[:,-32:].normal_(std=.01);model.utility_head[2].weight.normal_(std=.01)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'utility.onnx';meta=export(model,path);runtime=Graph(path)
            self.assertTrue(meta['parity']['passed']);self.assertEqual(meta['optimization']['execution']['unfused_erf'],0)
            for batch in samples(model.config.to_dict(),False)[:2]:
                explicit=dict(batch,descriptors=np.stack([unknown_numpy() for _ in range(len(batch['states']))]))
                missing={k:v for k,v in batch.items() if k!='descriptors'}
                a=runtime.run(explicit);b=runtime.run(missing);raw=runtime.engine.run(BASE_OUTPUTS,missing)
                for k,v in zip(BASE_OUTPUTS,raw):
                    np.testing.assert_allclose(v,a[k],rtol=2e-4,atol=2e-4)
                    np.testing.assert_array_equal(b[k],a[k])
                with torch.no_grad():out=model(*(torch.from_numpy(batch[k]) for k in BASE_INPUTS))
                for k,v in zip(BASE_OUTPUTS,out):np.testing.assert_allclose(runtime.run(batch)[k],v.numpy(),rtol=2e-4,atol=2e-4)
            bad=dict(batch,descriptors=np.ones((2,4,34),dtype='float32'))
            with self.assertRaisesRegex(ValueError,'one-hot'):runtime.run(bad)

    def test_base_export_and_unknown_initializer(self):
        model=FourSeatNet(ModelConfig.preset('small')).eval();corpus=[]
        with torch.no_grad():model.outcomes[0].weight[:,-32:].normal_(std=.01)
        for batch,count in ((1,1),(2,7)):
            x,a,m,mapping=inputs(batch,count)
            corpus.append([Encoded(x[i].numpy(),a[i].numpy(),mapping[i].numpy(),tuple(map(str,range(count)))) for i in range(batch)])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'base.onnx';meta=export_onnx(model,path,corpus);engine,_=load_export(path)
            from brass_ai.learning.batching import collate
            x=collate(corpus[-1]);data={k:v.numpy() for k,v in zip(INPUTS,x)}
            actual=engine.run(OUTPUTS,data)
            with torch.no_grad():expected=model(*x)
            for a,b in zip(actual,expected):np.testing.assert_allclose(a,b.numpy(),rtol=2e-4,atol=2e-4)


if __name__=='__main__':unittest.main()
