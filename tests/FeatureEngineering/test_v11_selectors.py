"""Indexed categorical references preserve trained weights, gradients and batching."""
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.batching import collate, model_id, predict_encoded
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, architecture_fingerprint
from brass_ai.learning.features import Encoded, Encoder
from brass_ai.learning.local_entities import LocalCandidates, entity_layout, check_action_selectors, COUNTS, WIDTHS
from brass_ai.learning.model import FourSeatNet, ModelConfig, dense_multiply_adds
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.learning.reward import specification
from brass_ai.personality.model import import_parent
from brass_ai.personality.contracts import CONTEXT_DIM
from brass_ai.recovery.encoding import PreparedEncoder
from test_mlp_v11 import inputs


def dense_prepare(self, encoded):
    return LocalCandidates(self.project_policy_context(encoded.context),
        self.entity_projections['target'](encoded.targets),
        self.entity_projections['merchant'](encoded.merchants),
        self.entity_projections['industry'](encoded.industries))


def dense_reference(self, actions, prepared):
    fields=entity_layout()['action_fields']
    return sum(actions[...,fields[name]['start']:fields[name]['stop']] @ bank for name,bank in
               (('target',prepared.targets),('merchant',prepared.merchants),('industry',prepared.industries)))


def dense_copy(model):
    """Dense selector reference for the current pre-GELU action computation."""
    old=copy.deepcopy(model);core=old.core if hasattr(old,'core') else old
    core.prepare_candidates=types.MethodType(dense_prepare,core)
    core.reference_context=types.MethodType(dense_reference,core)
    return old


def trained_branches(model):
    core=model.core if hasattr(model,'core') else model
    for layer in core.entity_projections.values():torch.nn.init.normal_(layer.weight,std=.03)
    for block in list(core.state_encoder.backbone)[:-1]:torch.nn.init.normal_(block.branch[-1].weight,std=.01)
    if hasattr(core,'utility_head'):torch.nn.init.normal_(core.utility_head[-1].weight,std=.02)


class V11SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native_encoder_selectors_are_categorical_and_cached_inputs_reject_corruption(self):
        from fe_fixtures import benchmark_positions
        for row in benchmark_positions():
            reference=Encoder().encode(row['view'],row['choices'])
            encoded=PreparedEncoder().encode(row['view'],row['choices'])
            check_action_selectors(reference.actions);check_action_selectors(encoded.actions)
            np.testing.assert_array_equal(reference.actions,encoded.actions)
        x,a,mask,mapping=inputs(1,4)
        good=Encoded(x[0].numpy(),a[0].numpy(),mapping[0].numpy(),tuple(map(str,range(4))))
        for field in entity_layout()['action_fields'].values():
            for values in ((1,1),(.5,0),(-1,0),(float('nan'),0)):
                bad=copy.deepcopy(good);bad.actions[0,field['start']:field['stop']]=0
                bad.actions[0,field['start']:field['start']+2]=values
                with self.assertRaisesRegex(ValueError,'one-hot or absent'):collate([bad])
                with self.assertRaisesRegex(ValueError,'one-hot or absent'):
                    predict_encoded(FourSeatNet(ModelConfig.preset('small')),[bad],'cpu',1)

    def test_all_rows_absence_and_batch_offsets(self):
        torch.manual_seed(18);model=FourSeatNet(ModelConfig.preset('small'));trained_branches(model)
        x=inputs(3)[0];prepared=model.prepare_candidates(model.encode_state(x))
        for role,bank in (('target',prepared.targets),('merchant',prepared.merchants),('industry',prepared.industries)):
            field=entity_layout()['action_fields'][role];n=len(field['identities'])
            actions=torch.zeros(3,n+1,327)
            for b in range(3):
                order=(torch.arange(n)+b)%n
                actions[b,torch.arange(n),field['start']+order]=1
            actual=model.reference_context(actions,prepared)
            for b in range(3):torch.testing.assert_close(actual[b,:n],bank[b,(torch.arange(n)+b)%n+1],rtol=0,atol=0)
            torch.testing.assert_close(actual[:,-1],torch.zeros_like(actual[:,-1]),rtol=0,atol=0)
            self.assertEqual(bank.shape[1],n+1)

    def test_nonzero_model_outputs_parameter_state_gradients_and_optimizer_step(self):
        for size,dtype,tolerance in (('small',torch.float64,2e-10),('medium',torch.float32,3e-5)):
            torch.manual_seed(27);model=UtilityNet(utility_config(ModelConfig.preset(size),specification())).to(dtype)
            trained_branches(model);old=dense_copy(model)
            x,a,mask,mapping=inputs(2,31);x=x.to(dtype).requires_grad_();a=a.to(dtype)
            previous=x.detach().clone().requires_grad_()
            out=model(x,a,mask,mapping);ref=old(previous,a,mask,mapping)
            for u,v in zip(out,ref):torch.testing.assert_close(u,v,rtol=tolerance,atol=tolerance)
            def loss(outputs):return outputs[0][mask].square().mean()+sum(v.square().mean()*.001 for v in outputs[1:])
            loss(out).backward();loss(ref).backward()
            torch.testing.assert_close(x.grad,previous.grad,rtol=tolerance,atol=tolerance)
            for (name,p),(other,q) in zip(model.named_parameters(),old.named_parameters()):
                self.assertEqual(name,other);torch.testing.assert_close(p.grad,q.grad,rtol=tolerance,atol=tolerance)
            # SGD avoids Adam amplifying tiny roundoff gradients near zero.
            torch.optim.SGD(model.parameters(),lr=1e-4).step();torch.optim.SGD(old.parameters(),lr=1e-4).step()
            for key,value in model.state_dict().items():torch.testing.assert_close(value,old.state_dict()[key],rtol=tolerance,atol=tolerance)

    def test_bfloat16_and_personality_parity_with_nonzero_roles(self):
        torch.manual_seed(84);utility=UtilityNet(utility_config(ModelConfig.preset('small'),specification()))
        trained_branches(utility);actor,_=import_parent(utility)
        torch.nn.init.normal_(actor.context_projection.weight,std=.02);old=dense_copy(actor)
        x,a,mask,mapping=inputs(2,17);kwargs=dict(remaining=torch.full((2,4),.075),personality=torch.full((2,CONTEXT_DIM),.2))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual=actor(x,a,mask,mapping,**kwargs);expected=old(x,a,mask,mapping,**kwargs)
        torch.testing.assert_close(actual['policy_logits'],expected['policy_logits'],rtol=.02,atol=.01)
        for name in ('win_share_4','placement_4','vp_4','remaining_utility_4'):
            torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
        actual['policy_logits'][mask].float().square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_family_and_four_seat_batching(self):
        model=FourSeatNet(ModelConfig.preset('small'));seen={};hooks=[]
        for name,layer in model.entity_encoder.encoders.items():
            hooks.append(layer.register_forward_pre_hook(lambda module,args,n=name:seen.setdefault(n,[]).append(tuple(args[0].shape))))
        hooks.append(model.outcomes[2].register_forward_pre_hook(lambda module,args:seen.setdefault('outcomes',[]).append(tuple(args[0].shape))))
        x,a,mask,mapping=inputs(3,5)
        with patch.object(model.entity_encoder,'gather',wraps=model.entity_encoder.gather) as gather:
            model(x,a,mask,mapping);self.assertEqual(gather.call_count,len(COUNTS))
        for hook in hooks:hook.remove()
        for name in COUNTS:self.assertEqual(seen[name],[(3,COUNTS[name],WIDTHS[name])])
        self.assertEqual(seen['outcomes'],[(3,4,128)])

    def test_current_identity_checkpoint_adam_and_rng_roundtrip(self):
        self.assertNotEqual(architecture_fingerprint(),'c7df5bf97fbe02f605aabeea89c9cc24d74981c5a4355a15f050f6b97f0cae39')
        model=FourSeatNet(ModelConfig.preset('small'));trained_branches(model);old=dense_copy(model)
        self.assertEqual(model_id(model),model_id(old));self.assertEqual(model.config,old.config)
        opt=torch.optim.AdamW(old.parameters(),lr=6e-6)
        sum(p.square().mean() for p in old.parameters()).backward();opt.step()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'old-dense.pt';save_checkpoint(path,old,opt,step=12)
            before=path.read_bytes();restored,raw=load_checkpoint(path)
            other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,other)
            self.assertEqual(raw['step'],12);self.assertEqual(opt.state_dict()['param_groups'],other.state_dict()['param_groups'])
            self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
            for key,value in old.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            for key,state in opt.state_dict()['state'].items():
                for name,value in state.items():torch.testing.assert_close(value,other.state_dict()['state'][key][name],rtol=0,atol=0)
            self.assertEqual(path.read_bytes(),before)
        self.assertEqual(dense_multiply_adds(model)['per_padded_candidate'],83232)


if __name__=='__main__':unittest.main()
