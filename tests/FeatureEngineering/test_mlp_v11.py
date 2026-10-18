"""V11 entity references, shared player heads, and strict model/data boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.batching import predict_encoded
from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint, resume_optimizer
from brass_ai.learning.contracts import (SOURCE_ROOT, compatibility, assert_heuristic_data_compatible,
                                        assert_compatible)
from brass_ai.learning.features import Encoded
from brass_ai.learning.local_entities import entity_layout, COUNTS, WIDTHS
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.state_layout import state_layout
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.personality.model import import_parent, base_outputs


def inputs(batch=2, candidates=17):
    generator=torch.Generator().manual_seed(412)
    states=torch.randn(batch,3128,generator=generator)*.2
    actions=torch.randn(batch,candidates,327,generator=generator)*.2
    fields=entity_layout()['action_fields']
    for field in fields.values():
        actions[:,:,field['start']:field['stop']]=0
        for i in range(candidates):
            if i % 5: actions[:,i,field['start']+i%len(field['identities'])]=1
    mask=torch.ones(batch,candidates,dtype=torch.bool)
    if candidates>1: mask[-1,1:]=False
    mapping=torch.tensor([[(s-p)%4 for s in range(4)] for p in range(batch)])
    return states,actions,mask,mapping


class V11ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(631)
        self.model=FourSeatNet(ModelConfig.preset('small'))

    def enable_roles(self):
        for layer in self.model.entity_projections.values():torch.nn.init.normal_(layer.weight,std=.02)

    def test_same_feature_tensor_contract_and_named_entity_coverage(self):
        model=self.model; layout=entity_layout(); segments={s.name:s for s in state_layout().segments}
        self.assertEqual((model.config.state_dim,model.config.action_dim),(3128,327))
        self.assertEqual(model.config.architecture,'brass-four-seat-mlp-resnet/11.2.0')
        self.assertEqual(model.config.entity_layout,layout['identity'])
        states=torch.arange(3128,dtype=torch.float32)[None]
        for name in COUNTS:
            rows=layout['groups'][name]
            actual=model.entity_encoder.gather(states,name)
            self.assertEqual(tuple(actual.shape),(1,COUNTS[name],WIDTHS[name]))
            torch.testing.assert_close(actual,states[:,torch.tensor(rows).clamp_min(0)].masked_fill(torch.tensor(rows)[None]<0,0))
        # An industry is keyed by its feature name, not the exploration-context order.
        for i,name in enumerate(layout['action_fields']['industry']['identities']):
            actual=layout['groups']['industry'][i]
            expected=[j for p in range(4) for s in state_layout().segments if s.name.startswith(f'mat.{p}.{name}.') for j in range(s.start,s.stop)]
            expected += [j for s in state_layout().segments if s.name.startswith(f'development.{name}.') for j in range(s.start,s.stop)]
            self.assertEqual(actual,expected)
        hand=set(range(segments['hand.counts'].start,segments['hand.counts'].stop))
        for row in layout['groups']['other_player']:self.assertFalse(hand.intersection(row))

    def test_all_target_merchant_industry_references_and_absence(self):
        self.enable_roles()
        model=self.model; states=inputs(1)[0]
        encoded=model.encode_state(states); prepared=model.prepare_candidates(encoded)
        for name,bank in (('target',prepared.targets),('merchant',prepared.merchants),('industry',prepared.industries)):
            field=entity_layout()['action_fields'][name]; count=len(field['identities'])
            actions=torch.zeros(1,count+1,327)
            actions[0,:count,field['start']:field['stop']]=torch.eye(count)
            actual=model.reference_context(actions,prepared)
            torch.testing.assert_close(actual[:,:count],bank[:,1:])
            torch.testing.assert_close(bank[:,0],torch.zeros_like(bank[:,0]),rtol=0,atol=0)
            torch.testing.assert_close(actual[:,count],torch.zeros_like(actual[:,count]),rtol=0,atol=0)
        self.assertEqual(encoded.targets.shape[1],128)
        torch.testing.assert_close(encoded.targets[:,118:127],encoded.merchants)

    def test_selected_entity_local_information_reaches_scorer(self):
        self.enable_roles()
        model=self.model; x,a,_,_=inputs(1,3)
        a.zero_(); field=entity_layout()['action_fields']['target']
        a[0,0,field['start']]=1; a[0,1,field['start']+1]=1
        other=next(i for i,name in enumerate(field['identities'][:49]) if name.split(':')[0]!=field['identities'][0].split(':')[0])
        a[0,2,field['start']+other]=1
        # Freeze global context to isolate the new branch from existing features.
        g=torch.zeros(1,model.config.hidden)
        before=model.score_prepared(a,model.prepare_candidates(model.encode_state(x,context=g)))
        changed=x.clone(); changed[:,entity_layout()['groups']['slot'][0][0]]+=2
        after=model.score_prepared(a,model.prepare_candidates(model.encode_state(changed,context=g)))
        self.assertGreater(float((after[:,0]-before[:,0]).abs().max().detach()),1e-7)
        # A sibling slot now sees the changed location; a different location does not.
        self.assertGreater(float((after[:,1]-before[:,1]).abs().max().detach()),1e-7)
        torch.testing.assert_close(after[:,2],before[:,2],rtol=0,atol=0)

    def test_shared_player_head_reference_mapping_and_candidate_independence(self):
        model=self.model; x,a,mask,mapping=inputs(2)
        e=model.encode_state(x)
        embeddings=model.descriptor_embeddings(e)
        raw=model.outcomes(torch.cat((e.context[:,None].expand(-1,4,-1),e.players,embeddings),-1))
        expected=(raw[:,:,0].softmax(-1).gather(1,mapping),
                  (1+3*raw[:,:,1].sigmoid()).gather(1,mapping),
                  (100*raw[:,:,2]).gather(1,mapping))
        for value,reference in zip(model.official_heads(e,mapping,embeddings),expected):
            torch.testing.assert_close(value,reference,rtol=2e-5,atol=2e-6)
        before=model(x,a,mask,mapping)
        after=model(x,a[:,:1]*3,mask[:,:1],mapping)
        for v,w in zip(before[1:],after[1:]):torch.testing.assert_close(v,w,rtol=0,atol=0)
        # Each public player's facts enter their own local vector directly.
        changed=x.clone(); changed[:,entity_layout()['groups']['other_player'][1][0]]+=2
        changed_players=model.encode_state(changed,context=e.context).players
        for p in (0,1,3):torch.testing.assert_close(changed_players[:,p],e.players[:,p],rtol=0,atol=0)
        self.assertFalse(torch.equal(changed_players[:,2],e.players[:,2]))

    def test_new_branches_get_gradients_and_masks_hold(self):
        self.enable_roles()
        model=self.model; x,a,mask,mapping=inputs()
        out=model(x,a,mask,mapping)
        self.assertTrue(torch.isneginf(out[0][~mask]).all())
        loss=-out[0].log_softmax(-1)[:,0].mean()+sum(t.square().mean() for t in out[1:])
        loss.backward()
        for name,module in model.entity_encoder.encoders.items():
            grad=module[0].weight.grad
            self.assertIsNotNone(grad,name); self.assertTrue(torch.isfinite(grad).all(),name)
        for name in ('slot','merchant','industry','own_player','other_player'):
            self.assertGreater(float(model.entity_encoder.encoders[name][0].weight.grad.abs().sum()),0,name)
        for name,layer in model.entity_projections.items():
            self.assertGreater(float(layer.weight.grad.abs().sum()),0,name)

    def test_chunking_prepares_entities_and_banks_once(self):
        x,a,_,mapping=inputs(1,33)
        row=Encoded(x[0].numpy(),a[0].numpy(),mapping[0].numpy(),tuple(map(str,range(33))))
        for model in (self.model,UtilityNet(utility_config(self.model.config,specification()))):
            with torch.inference_mode():
                expected=predict_encoded(model,[row],'cpu',100)
                with patch.object(model,'encode_state',wraps=model.encode_state) as encode, \
                     patch.object(model,'prepare_candidates',wraps=model.prepare_candidates) as prepare:
                    actual=predict_encoded(model,[row],'cpu',3)
                self.assertEqual(encode.call_count,1);self.assertEqual(prepare.call_count,1)
                for v,w in zip(actual,expected):np.testing.assert_allclose(v,w,rtol=2e-5,atol=2e-6)

    def test_base_to_utility_personality_and_restart(self):
        model=self.model; optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        out=model(*inputs()); (out[1].square().mean()+out[2].mean()+out[3].mean()).backward();optimizer.step()
        from brass_ai.recovery.utility import migrate
        before={n:t.clone() for n,t in model.state_dict().items()}; rng=torch.get_rng_state().clone()
        migrate(model,optimizer,specification())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,t in before.items():self.assertTrue(torch.equal(t,model.state_dict()[name]),name)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'v11.pt'
            save_checkpoint(path,model,optimizer,step=19,training=dict(games=320,lineage='v11-test'))
            restored,raw=load_checkpoint(path); opt=torch.optim.AdamW(restored.parameters(),lr=1e-4)
            resume_optimizer(raw,opt)
            self.assertEqual(raw['step'],19);self.assertEqual(raw['training']['games'],320)
            self.assertEqual(raw['training']['lineage'],'v11-test')
            self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
            for name,t in model.state_dict().items():self.assertTrue(torch.equal(t,restored.state_dict()[name]),name)
            for key,state in optimizer.state_dict()['state'].items():
                for name,t in state.items():torch.testing.assert_close(t,opt.state_dict()['state'][key][name],rtol=0,atol=0)
            actor,_=import_parent(restored); data=inputs(); remaining=torch.full((2,4),.075)
            from brass_ai.personality.contracts import CONTEXT_DIM
            with torch.inference_mode():
                expected=restored(*data,remaining=remaining)
                actual=base_outputs(actor(*data,personality=torch.zeros(2,CONTEXT_DIM),remaining=remaining))
            for v,w in zip(actual,expected):torch.testing.assert_close(v,w,rtol=0,atol=0)

    def test_v10_rejected_and_static_gather_tampering_detected(self):
        raw=self.model.config.to_dict();raw['architecture']='brass-four-seat-mlp-resnet/10.0.0'
        with self.assertRaisesRegex(ValueError,'11.2'):ModelConfig.from_dict(raw)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'tampered.pt';save_checkpoint(path,self.model)
            payload=torch.load(path,weights_only=True)
            index=payload['state_dict']['entity_encoder.slot_indices'];index[0,0]=index[0,1]
            torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'entity index buffer'):load_checkpoint(path)
        old=copy.deepcopy(compatibility());old['architecture_sha256']='f'*64
        assert_heuristic_data_compatible(old)
        with self.assertRaises(ValueError):assert_compatible(old)
        wrong=dict(old,features_sha256='f'*64)
        with self.assertRaisesRegex(ValueError,'Incompatible heuristic'):assert_heuristic_data_compatible(wrong)


if __name__=='__main__':unittest.main()
