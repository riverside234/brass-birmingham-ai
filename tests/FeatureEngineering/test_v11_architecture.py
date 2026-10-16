"""V11 neighborhood semantics, initialization and dataset/static admission."""
import copy
import unittest
from unittest.mock import patch
import torch

from brass_ai.learning.board_topology import topology,build_topology,admit_topology,TOPOLOGY_FILE
from brass_ai.learning.contracts import SOURCE_ROOT,read_json,compatibility,assert_heuristic_data_compatible
from brass_ai.learning.features import Encoder
from brass_ai.learning.local_entities import entity_layout,WIDTHS
from brass_ai.learning.model import FourSeatNet,ModelConfig,ResidualBlock
from brass_ai.learning.state_layout import state_layout,model_layout,MODEL_LAYOUT_VERSION,MODEL_LAYOUT_FILE
from brass_ai.recovery.encoding import PreparedEncoder
from test_mlp_v11 import inputs
from fe_fixtures import benchmark_positions


class V11ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_medium_dimensions_and_new_identities(self):
        model=FourSeatNet();c=model.config
        self.assertEqual(c.architecture,'brass-four-seat-mlp-resnet/11.0.0')
        self.assertEqual(c.projection_widths,(32,64,48,320,352,48))
        self.assertEqual((model.state_encoder.fusion[0].in_features,c.hidden),(876,512))
        self.assertEqual(sum(isinstance(m,ResidualBlock) for m in model.modules()),8)
        self.assertEqual((model.action_encoder[0].in_features,c.action_hidden),(327,160))
        self.assertEqual((model.policy_scorer[0].in_features,model.policy_scorer[0].out_features),(672,192))
        self.assertEqual((model.outcomes[0].in_features,model.outcomes[0].out_features,model.outcomes[-1].out_features),(576,128,3))
        self.assertEqual((model.parameter_count,c.parameter_limit),(5904644,6500000))
        self.assertEqual(entity_layout()['format'],'grouped-mlp-local-entities/2.0.0')
        self.assertEqual((WIDTHS['slot'],WIDTHS['link'],WIDTHS['industry']),(72,148,219))
        record=model_layout().specification(version=MODEL_LAYOUT_VERSION)
        self.assertEqual(model_layout().specification(),record)
        self.assertEqual(record,read_json(SOURCE_ROOT/'configs/learning'/MODEL_LAYOUT_FILE))
        self.assertEqual(model_layout().permutation,state_layout().permutation)
        self.assertEqual(record['segments'],state_layout().specification()['segments'])
        self.assertNotEqual(model_layout().identity,state_layout().identity)

    def test_every_neighborhood_order_and_exact_padding(self):
        model=FourSeatNet(ModelConfig.preset('small'));layout=entity_layout();board=topology()
        segments=state_layout().segments
        row=lambda name:[i for s in segments if s.name.startswith('slot.'+name+'.') for i in range(s.start,s.stop)]
        states=torch.arange(1,3129,dtype=torch.float32)[None]
        for i,name in enumerate(board['slot_order']):
            names=[name]+[s for s in board['location_slots'][name.split(':')[0]] if s!=name]
            wanted=sum((row(s) for s in names),[])
            self.assertEqual(layout['groups']['slot'][i],wanted+[-1]*(72-len(wanted)))
            actual=model.entity_encoder.gather(states,'slot')[0,i]
            torch.testing.assert_close(actual[:len(wanted)],states[0,wanted],rtol=0,atol=0)
            self.assertTrue((actual[len(wanted):]==0).all())
        for i,name in enumerate(board['link_order']):
            names=[s for region in board['link_endpoints'][name] for s in board['location_slots'].get(region,[])]
            wanted=sum((row(s) for s in names),[])
            self.assertEqual(layout['groups']['link'][i][4:],wanted+[-1]*(144-len(wanted)))
        junction=board['link_order'].index('RailWorcesterSouthWest')
        self.assertEqual(board['link_endpoints']['RailWorcesterSouthWest'],['Worcester','SouthWest','Kidderminster'])
        self.assertEqual(sum(i>=0 for i in layout['groups']['link'][junction]),4+5*18)
        self.assertLessEqual(max(sum(len(board['location_slots'].get(r,[])) for r in regions)
                                 for regions in board['link_endpoints'].values()),8)

    def test_zero_role_init_preserves_unconditioned_policy_then_learns(self):
        torch.manual_seed(31);model=FourSeatNet(ModelConfig.preset('small'))
        x,a,mask,mapping=inputs();encoded=model.encode_state(x);prepared=model.prepare_candidates(encoded)
        for layer in model.entity_projections.values():
            self.assertIsNone(layer.bias);self.assertEqual(int(torch.count_nonzero(layer.weight)),0)
        expected=model.score_candidates(model.action_encoder(a),model.project_policy_context(encoded.context))
        torch.testing.assert_close(model.score_prepared(a,prepared),expected,rtol=0,atol=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=6e-5)
        (-model(x,a,mask,mapping)[0].log_softmax(-1)[:,0].mean()).backward()
        for layer in model.entity_projections.values():self.assertGreater(float(layer.weight.grad.abs().sum()),0)
        self.assertEqual(float(model.entity_encoder.encoders['slot'][0].weight.grad.abs().sum()),0)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
        (-model(x,a,mask,mapping)[0].log_softmax(-1)[:,0].mean()).backward()
        self.assertGreater(float(model.entity_encoder.encoders['slot'][0].weight.grad.abs().sum()),0)

    def test_topology_admission_and_unchanged_feature_values(self):
        sample=benchmark_positions()[0];view=copy.deepcopy(sample['view']);choices=sample['choices']
        admit_topology(view)
        ref=Encoder().encode(view,choices);fast=PreparedEncoder().encode(view,choices)
        torch.testing.assert_close(torch.from_numpy(ref.state.copy()),torch.from_numpy(fast.state.copy()),rtol=0,atol=0)
        bad=copy.deepcopy(view);link=next(l for l in bad['links'] if l['id']=='RailWorcesterSouthWest')
        link['regions']=link['regions'][:2]
        for encoder in (Encoder(),PreparedEncoder()):
            with self.assertRaisesRegex(ValueError,'V11 link topology'):encoder.encode(bad,choices)
        # Even an ingress-validation flag must not bypass the new omitted-static checks.
        with self.assertRaisesRegex(ValueError,'V11 link topology'):PreparedEncoder().encode(bad,choices,validated=True)
        current=compatibility()
        self.assertEqual(current['features_sha256'],'59facdd72a386b6046abbb875daf3e1051b92e580ff55d0ab37250dd0c1dfb6a')
        self.assertEqual(current['encoder_sha256'],'7d21b3052bf65c2d272a59859eebc6b3be1117b77e906a677402784b5ab90f06')
        self.assertEqual(current['derived_source_sha256'],'47afe02b5e98549081f5ab119a2dca2c90448c74fba8020fc7e04cb6bc16f260')
        old=dict(current,architecture_sha256='7328ca65ec087183a4c92c7606e48b982d127003121ede84da01388dabf8181e')
        assert_heuristic_data_compatible(old)

    def test_frozen_topology_tampering_rejected(self):
        current=build_topology();wrong=copy.deepcopy(current)
        wrong['link_endpoints']['RailWorcesterSouthWest'].pop()
        topology.cache_clear()
        try:
            with patch('brass_ai.learning.board_topology.read_json',side_effect=lambda p:wrong if p.name==TOPOLOGY_FILE else read_json(p)):
                with self.assertRaisesRegex(ValueError,'frozen local topology'):topology()
        finally:topology.cache_clear()


if __name__=='__main__':unittest.main()
