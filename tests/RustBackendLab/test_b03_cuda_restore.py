"""Exact CUDA/CPU payload comparisons and narrowly reviewed prerequisite reuse."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.rust_backend import lab_continuation as C
from brass_ai.rust_backend import lab_prerequisite_reuse as R


class Payload(unittest.TestCase):
    def payload(self):
        return dict(state={0:dict(step=torch.tensor(12.),exp_avg=torch.tensor([.125,.25]),
                                 exp_avg_sq=torch.tensor([.03125,.0625]))},param_groups=[dict(lr=.001,params=[0])])

    def test_exact_values_structure_dtype_and_shape_are_required(self):
        left=self.payload();right=copy.deepcopy(left)
        self.assertTrue(C._same_payload(left,right))
        for mutate in (lambda x:x['state'][0]['exp_avg'].add_(1),
                       lambda x:x['state'][0].update(exp_avg=x['state'][0]['exp_avg'].double()),
                       lambda x:x['state'][0].update(exp_avg=x['state'][0]['exp_avg'].reshape(1,2)),
                       lambda x:x.update(param_groups=tuple(x['param_groups'])),
                       lambda x:x['state'][0].update(step=torch.tensor(13.))):
            right=copy.deepcopy(left);mutate(right);self.assertFalse(C._same_payload(left,right))
        self.assertEqual(left['state'][0]['step'].item(),12.)

    def test_cross_device_comparison_moves_only_detached_copies(self):
        class Tensor:
            def __init__(self,device,value):self.device=device;self.value=value;self.dtype='float32';self.shape=(1,)
            def detach(self):return Tensor(self.device,self.value)
            def cpu(self):self.device='cpu';return self
        def equal(a,b):
            if a.device!=b.device:raise RuntimeError('Expected all tensors to be on the same device')
            return a.value==b.value
        left,right=Tensor('cuda:0',.5),Tensor('cpu',.5)
        with self.assertRaises(RuntimeError):equal(left,right)
        with patch.object(C,'torch',SimpleNamespace(Tensor=Tensor,equal=equal)):
            self.assertTrue(C._same_payload({'moment':left},{'moment':right}))
            self.assertFalse(C._same_payload(left,Tensor('cpu',.75)))
        self.assertEqual(left.device,'cuda:0');self.assertEqual(right.device,'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA restoration runs on the lab')
    def test_real_cuda_adam_moments_match_cpu_checkpoint_without_mutation(self):
        parameter=torch.nn.Parameter(torch.tensor([1.,2.],device='cuda:0'))
        optimizer=torch.optim.AdamW([parameter]);parameter.sum().backward();optimizer.step()
        saved=copy.deepcopy(optimizer.state_dict())
        for state in saved['state'].values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):state[key]=value.cpu()
        target=torch.optim.AdamW([parameter]);target.load_state_dict(saved)
        self.assertEqual(target.state_dict()['state'][0]['exp_avg'].device.type,'cuda')
        self.assertTrue(C._same_payload(target.state_dict(),saved))
        saved['state'][0]['exp_avg'][0]+=1
        self.assertFalse(C._same_payload(target.state_dict(),saved))


class Reuse(unittest.TestCase):
    def inventories(self):
        previous={name:'a'*64 for name in R.ALLOWED if not name.endswith('lab_prerequisite_reuse.py')}
        previous['src/brass_ai/recovery/inference.py']='c'*64
        current=dict(previous,**{name:'b'*64 for name in R.ALLOWED})
        changes={name:dict(before=previous.get(name),after=current[name]) for name in R.ALLOWED}
        return previous,current,dict(changes=changes)

    def test_exact_review_preserves_every_other_executable_input(self):
        before,after,review=self.inventories();R.verify_delta(before,after,review)
        for name in ('src/brass_ai/recovery/inference.py','src/extra.py','training/runtime/ordinary/BrassHost.exe'):
            changed=dict(after);changed[name]='d'*64
            with self.assertRaisesRegex(ValueError,'outside'):R.verify_delta(before,changed,review)
        changed=dict(after);changed.pop('src/brass_ai/recovery/inference.py')
        with self.assertRaisesRegex(ValueError,'outside'):R.verify_delta(before,changed,review)

    def test_wrong_before_after_and_expanded_review_are_rejected(self):
        before,after,review=self.inventories();name=next(iter(R.ALLOWED))
        for field in ('before','after'):
            changed=copy.deepcopy(review);changed['changes'][name][field]='d'*64
            with self.assertRaisesRegex(ValueError,'exact reviewed'):R.verify_delta(before,after,changed)
        review['changes']['src/brass_ai/recovery/inference.py']=dict(before='c'*64,after='c'*64)
        with self.assertRaisesRegex(ValueError,'outside'):R.verify_delta(before,after,review)

    def test_b02_requires_the_original_pinned_receipt_and_source(self):
        proof=dict(from_implementation='old',to_implementation='new',receipts={'b02':dict(path='/original',sha256='a'*64,bytes=7)})
        with patch.object(R,'reviewed',return_value=proof),patch.object(R,'pin',return_value=proof['receipts']['b02']):
            self.assertEqual(R.verify_b02(Path('/original'),'old')['from_implementation'],'old')
            with self.assertRaises(ValueError):R.verify_b02(Path('/original'),'foreign')
        with patch.object(R,'reviewed',return_value=proof),patch.object(R,'pin',return_value=dict(path='/other',sha256='a'*64,bytes=7)):
            with self.assertRaises(ValueError):R.verify_b02(Path('/other'),'old')

    def test_read_review_requires_manifest_seal_and_current_implementation(self):
        proof=dict(contract='brass-lab-prerequisite-reuse/1.0.0',to_implementation='current',producer='native',
                   scope=['b01','b02','b025'],changes={n:{} for n in R.ALLOWED})
        raw=json.dumps(proof).encode();manifest=dict(implementation='current',producer='native',
            files={R.REVIEW:R.hashlib.sha256(raw).hexdigest()})
        with patch.object(Path,'read_bytes',return_value=raw),patch.object(R,'read_json',side_effect=[manifest,proof]),\
             patch('brass_ai.step5.league.implementation_id',return_value='current'),\
             patch('brass_ai.rust_backend.lab_teacher_overlay.source_fingerprint',return_value='native'):
            self.assertEqual(R.reviewed(),proof)
        with patch.object(Path,'read_bytes',return_value=raw+b' '),patch.object(R,'read_json',return_value=manifest):
            with self.assertRaisesRegex(ValueError,'not sealed'):R.reviewed()

    def test_resume_at_b03_runs_only_remaining_stages(self):
        import sys
        with patch.object(sys,'path',[str(SOURCE_ROOT/'scripts'),*sys.path]):
            spec=importlib.util.spec_from_file_location('resume_runner',SOURCE_ROOT/'scripts/run-rust-lab-qualification.py')
            runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        args=SimpleNamespace(stage='all',start_at='b03')
        self.assertEqual(runner.selected_stages(args,{'b01':None,'b02':Path('original'),'b03':None}),('b03','b04'))
