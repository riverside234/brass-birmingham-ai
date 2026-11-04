"""B03 logical-GPU compatibility, launcher propagation and exact RNG checks."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.rust_backend.lab_continuation import _same_payload


def load(name):
    with patch.object(sys, 'path', [str(SOURCE_ROOT/'scripts'), *sys.path]):
        spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SOURCE_ROOT/'scripts'/name)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module


class Launcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load('run-rust-lab-qualification.py')
        cls.b03 = load('qualify-rust-lab-b03.py')

    def args(self, **values):
        return SimpleNamespace(**dict(dict(device='cuda:0', fixtures_only=False, run=Path('run'),
            expected_generation=570, checkpoint_sha256='a'*64, b02_workers=60,
            allow_cuda_topology_change=False, b04_build=None), **values))

    def test_flag_is_opt_in_and_forwarded_only_to_real_b03(self):
        receipts = {s: Path(s+'.json') for s in ('b01', 'b02', 'b03')}
        for enabled in (False, True):
            for stage in self.runner.STAGES:
                cmd = self.runner.command(stage, self.args(allow_cuda_topology_change=enabled), Path('out'), receipts)
                self.assertEqual('--allow-cuda-topology-change' in cmd, enabled and stage == 'b03')
        cmd = self.runner.command('b03', self.args(fixtures_only=True, allow_cuda_topology_change=True), Path('out'), {})
        self.assertNotIn('--allow-cuda-topology-change', cmd)

    def test_cli_resumes_only_b03_b04_and_records_the_choice(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root = Path(temp); run = root/'run'; run.mkdir()
            state = dict(iteration=570, history=[dict(generation=570, sha256='a'*64)],
                         signature=dict(implementation='fixture', config={'recipe': 'unchanged'}))
            (run/'state.json').write_text(json.dumps(state), encoding='utf-8')
            config = root/'config.json'; config.write_text(json.dumps(state['signature']['config']), encoding='utf-8')
            original = root/'b02.json'; original.write_text('{"passed": true}', encoding='utf-8')
            before = (run/'state.json').read_bytes(), original.read_bytes()
            commands = []
            def execute(cmd, log):
                commands.append(cmd)
                out = Path(cmd[cmd.index('--out')+1]); out.mkdir()
                (out/'qualification.json').write_text('{"passed": true}', encoding='utf-8')
                return 0
            with patch.object(self.runner, 'ROOT', root), \
                 patch.object(self.runner, 'verify_package', return_value={'training_reference': 'fixture'}), \
                 patch.object(self.runner, 'execute', side_effect=execute), \
                 patch.object(self.runner, 'export', return_value=root/'results'), contextlib.redirect_stdout(io.StringIO()):
                code = self.runner.main(['--stage', 'all', '--start-at', 'b03', '--device', 'cuda:0',
                    '--allow-cuda-topology-change', '--run', str(run), '--config', str(config),
                    '--b02', str(original), '--expected-generation', '570', '--checkpoint-sha256', 'a'*64])
            self.assertEqual(code, 0)
            self.assertEqual([Path(cmd[1]).name for cmd in commands], ['qualify-rust-lab-b03.py', 'qualify-rust-lab-b04.py'])
            self.assertIn('--allow-cuda-topology-change', commands[0])
            self.assertNotIn('--allow-cuda-topology-change', commands[1])
            report = json.loads(next((root/'artifacts/rust-backend').glob('*/runner.json')).read_bytes())
            self.assertTrue(report['allow_cuda_topology_change'])
            self.assertEqual(report['selected_stages'], ['b03', 'b04'])
            self.assertEqual(report['reused_receipts']['b02']['path'], str(original.resolve()))
            self.assertEqual(before, ((run/'state.json').read_bytes(), original.read_bytes()))

    def test_cli_rejects_fixture_nonlearner_and_irrelevant_stage_use(self):
        cases = [('--stage', 'b04'), ('--stage', 'all', '--start-at', 'b04'),
                 ('--stage', 'b03', '--device', 'cuda:1'),
                 ('--stage', 'b03', '--fixtures-only', '--device', 'cpu')]
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(self.runner, 'verify_package') as verify, self.assertRaises(SystemExit):
                self.runner.main([*args, '--allow-cuda-topology-change'])
            verify.assert_not_called()
        for args in (['--device', 'cuda:1'], ['--fixtures-only']):
            with patch.object(sys, 'argv', ['qualify-rust-lab-b03.py', *args, '--allow-cuda-topology-change']), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.b03.main()


class LogicalRNG(unittest.TestCase):
    def test_existing_compatibility_route_restores_exact_streams_and_preserves_new_devices(self):
        # Exercise real Adam loading and RNG restoration; mock only CUDA hardware.
        with torch.random.fork_rng(devices=[]):
            parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
            optimizer = torch.optim.AdamW([parameter]); parameter.sum().backward(); optimizer.step()
            for saved, visible in ((1, 6), (6, 1), (2, 2)):
                with self.subTest(saved=saved, visible=visible):
                    raw = dict(optimizer=copy.deepcopy(optimizer.state_dict()), rng=torch.get_rng_state().clone(),
                               cuda_rng=[torch.tensor([i, 17], dtype=torch.uint8) for i in range(saved)])
                    original = copy.deepcopy(raw)
                    streams = {i: torch.tensor([i, 99], dtype=torch.uint8) for i in range(visible)}
                    before = copy.deepcopy(streams)
                    target = torch.optim.AdamW([parameter], lr=.5)
                    def set_state(value, index): streams[index] = value.clone()
                    with patch.object(torch.cuda, 'is_available', return_value=True), \
                         patch.object(torch.cuda, 'device_count', return_value=visible), \
                         patch.object(torch.cuda, 'set_rng_state', side_effect=set_state):
                        report = resume_optimizer(raw, target, allow_cuda_topology_change=True)
                    self.assertEqual(report, dict(saved_devices=saved, visible_devices=visible, restored_devices=min(saved, visible)))
                    for i in range(visible):
                        self.assertTrue(torch.equal(streams[i], raw['cuda_rng'][i] if i < saved else before[i]))
                    self.assertTrue(torch.equal(torch.get_rng_state(), raw['rng']))
                    self.assertTrue(_same_payload(target.state_dict(), raw['optimizer']))
                    self.assertTrue(_same_payload(raw, original))

    def test_strict_default_still_rejects_a_different_device_count(self):
        parameter = torch.nn.Parameter(torch.ones(1)); optimizer = torch.optim.AdamW([parameter])
        raw = dict(optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), cuda_rng=[torch.tensor([0])])
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=6), \
             patch.object(torch.cuda, 'set_rng_state') as restore, self.assertRaisesRegex(ValueError, 'topology differs'):
            resume_optimizer(raw, optimizer)
        restore.assert_not_called()


if __name__ == '__main__': unittest.main()
