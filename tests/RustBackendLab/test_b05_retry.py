"""B05 diagnostic GPU topology and first-error propagation, without CUDA hardware."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.rust_backend.lab_snapshot import tree_digest


def launcher():
    spec=importlib.util.spec_from_file_location('b05_retry',SOURCE_ROOT/'scripts/qualify-rust-lab-b05.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Topology(unittest.TestCase):
    def test_default_explicit_and_strict_policy_survive_child_launch(self):
        module=launcher()
        for flags,enabled in (([],True),(['--allow-cuda-topology-change'],True),(['--strict-cuda-topology'],False)):
            with self.subTest(flags=flags):
                args=module.arguments(flags)
                child=module.arguments(module.child_command(args,Path('out'))[2:])
                self.assertEqual(child.allow_cuda_topology_change,enabled)
                self.assertTrue(child.lab_child);self.assertEqual(child.device,'cuda:0')
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            module.arguments(['--device','cuda:1'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            module.arguments(['--allow-cuda-topology-change','--strict-cuda-topology'])

    def test_default_restores_saved_streams_and_adam_with_more_or_fewer_devices(self):
        module=launcher();args=module.arguments([])
        with torch.random.fork_rng(devices=[]):
            parameter=torch.nn.Parameter(torch.tensor([1.,2.]));optimizer=torch.optim.AdamW([parameter])
            parameter.sum().backward();optimizer.step()
            for saved,visible in ((1,6),(6,1),(2,2)):
                with self.subTest(saved=saved,visible=visible):
                    raw=dict(optimizer=copy.deepcopy(optimizer.state_dict()),rng=torch.get_rng_state().clone(),
                             cuda_rng=[torch.tensor([i,19],dtype=torch.uint8) for i in range(saved)])
                    original=tree_digest(raw)
                    streams={i:torch.tensor([i,29],dtype=torch.uint8) for i in range(visible)}
                    before=copy.deepcopy(streams);target=torch.optim.AdamW([parameter],lr=.5)
                    with patch.object(torch.cuda,'is_available',return_value=True), \
                         patch.object(torch.cuda,'device_count',return_value=visible), \
                         patch.object(torch.cuda,'set_rng_state',side_effect=lambda value,index:streams.__setitem__(index,value.clone())):
                        receipt=resume_optimizer(raw,target,allow_cuda_topology_change=args.allow_cuda_topology_change)
                    self.assertEqual(receipt,dict(saved_devices=saved,visible_devices=visible,restored_devices=min(saved,visible)))
                    self.assertEqual(tree_digest(target.state_dict()),tree_digest(raw['optimizer']))
                    self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
                    self.assertEqual(tree_digest(raw),original)
                    for i in range(visible):
                        self.assertTrue(torch.equal(streams[i],raw['cuda_rng'][i] if i<saved else before[i]))

    def test_strict_option_and_library_default_still_reject_mismatched_count(self):
        module=launcher();args=module.arguments(['--strict-cuda-topology'])
        parameter=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([parameter])
        raw=dict(optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=[torch.tensor([1])])
        for options in ({},dict(allow_cuda_topology_change=args.allow_cuda_topology_change)):
            with patch.object(torch.cuda,'is_available',return_value=True), \
                 patch.object(torch.cuda,'device_count',return_value=6), \
                 patch.object(torch.cuda,'set_rng_state') as restored, self.assertRaisesRegex(ValueError,'topology differs'):
                resume_optimizer(raw,optimizer,**options)
            restored.assert_not_called()

    def test_child_writes_original_failure_for_parent_export(self):
        module=launcher()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            args=module.arguments(['--out',temporary])
            with patch('brass_ai.rust_backend.lab_b05_smoke.run',side_effect=ValueError('original CUDA failure')):
                with self.assertRaisesRegex(ValueError,'original CUDA'):module.run_child(args,{})
            report=json.loads((Path(temporary)/'lab-checks.json').read_text())
            self.assertFalse(report['passed']);self.assertEqual(report['error'],'original CUDA failure')
            self.assertEqual(report['error_type'],'ValueError');self.assertIn('original CUDA',report['traceback'])


if __name__=='__main__':unittest.main()
