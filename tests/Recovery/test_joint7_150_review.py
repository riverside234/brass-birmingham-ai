"""Generation-150 continuation and evidence needed for a controlled comparison."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import save_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.config import UPDATE_V21, time_budget_config, validate_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.resume import check_signature, load_resume, apply_optimizer_config, commit_transition
from brass_ai.recovery.review_metrics import teacher_drift_probe
from brass_ai.recovery.updates import prepared
from brass_ai.step5.league import implementation_id, write_json


EVIDENCE = SOURCE_ROOT/'results/v4-selfplay-joint-g0150-20261013T173011108713Z'


class Review150Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_presets_adopt_current_runtime_and_change_one_learning_factor(self):
        state = read(EVIDENCE/'state.json')
        control = read(SOURCE_ROOT/'configs/step5/joint7-g150-control.json')
        candidate = read(SOURCE_ROOT/'configs/step5/joint7-g150-distill4.json')
        self.assertEqual({k: v for k, v in candidate.items() if k != 'distill_steps'},
                         {k: v for k, v in control.items() if k != 'distill_steps'})
        self.assertEqual((control['distill_steps'], candidate['distill_steps']), (8, 4))
        self.assertEqual(control['execution'], read(SOURCE_ROOT/'configs/step5/lab.json')['execution'])
        for cfg in (control, candidate):
            validate_config(cfg)
            self.assertEqual(time_budget_config(cfg), cfg)
            self.assertEqual(cfg['reward'], state['signature']['config']['reward'])
            signature = dict(state['signature'], config=cfg, implementation=implementation_id())
            self.assertTrue(check_signature(state, signature, True))
            with self.assertRaisesRegex(ValueError, 'resume-reviewed-source'):check_signature(state, signature)
            changed = copy.deepcopy(signature);changed['config']['learning_rate'] *= 2
            with self.assertRaisesRegex(ValueError, 'unchanged'):check_signature(state, changed, True)

    def test_generation150_adam_rng_lineage_and_restart_are_preserved(self):
        state = read(EVIDENCE/'state.json');state['signature']['config']['model_size'] = 'small'
        cfg = read(SOURCE_ROOT/'configs/step5/joint7-g150-control.json');cfg['model_size'] = 'small'
        model = UtilityNet(utility_config(ModelConfig.preset('small'), cfg['search']['reward']))
        model.utility_calibration = dict(ready=True, reward_id=digest(cfg['search']['reward']))
        original = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=1e-4)
        sum(p.square().mean() for p in model.parameters()).backward();original.step()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run = Path(tmp).resolve();path = run/'generation-0150.pt'
            identity = save_checkpoint(path, model, original, step=150, training=dict(
                fresh_run=run.relative_to(SOURCE_ROOT).as_posix(), signature=digest(state['signature']), mode='step5-population-ppo'))
            entry = dict(path=path.relative_to(SOURCE_ROOT).as_posix(), sha256=sha(path), model_id=identity, generation=150)
            state.update(history=[entry], population=[entry], resume=[entry], evaluation_anchor=entry, champion=entry)
            write_json(run/'state.json', state);before = path.read_bytes()
            signature = dict(state['signature'], config=cfg, implementation=implementation_id())
            restored, raw, changed = load_resume(SOURCE_ROOT, run, state, signature, reviewed=True)
            self.assertTrue(changed);self.assertFalse(restored.utility_calibration['ready'])
            for k, v in model.state_dict().items():torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
            optimizer = torch.optim.AdamW(restored.parameters(), lr=cfg['learning_rate'])
            resume_optimizer(raw, optimizer);apply_optimizer_config(optimizer, cfg)
            self.assertEqual(optimizer.param_groups[0]['lr'], 1e-5)
            self.assertTrue(torch.equal(torch.get_rng_state(), raw['rng']))
            for k, values in original.state_dict()['state'].items():
                for name, value in values.items():torch.testing.assert_close(value, optimizer.state_dict()['state'][k][name], rtol=0, atol=0)
            module_spec = importlib.util.spec_from_file_location('joint7_preflight', SOURCE_ROOT/'scripts/joint7_preflight.py')
            preflight = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(preflight)
            with patch.object(preflight, 'completed_imitation', return_value=(None, state['signature']['imitation_source'])):
                receipt = preflight.inspect(SOURCE_ROOT, run, cfg, run/'state.json')
            self.assertEqual((receipt['iteration'], receipt['games_done']), (150, 48000))
            self.assertEqual(read(run/'state.json'), state)
            write_json(run/'evidence.json', state)
            write_json(run/'state.json', dict(state, context_counter=state['context_counter'] + 1))
            with self.assertRaisesRegex(ValueError, 'context_counter differs'):
                preflight.inspect(SOURCE_ROOT, run, cfg, run/'evidence.json')
            write_json(run/'state.json', state)
            updated = commit_transition(run, state, signature)
            for key, value in state.items():
                if key not in ('signature', 'source_transitions'):self.assertEqual(updated[key], value, key)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(load_resume(SOURCE_ROOT, run, updated, signature)[2])

    def test_teacher_only_probe_is_bounded_exact_and_rng_neutral(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.logits = torch.nn.Parameter(torch.log(torch.tensor([.2, .3, .5])))
            def forward(self, states, actions, legal, mapping):
                return (self.logits.expand(len(states), -1).masked_fill(~legal, -torch.inf),)
        model = Model();data = prepared([game()]);rng = torch.get_rng_state().clone()
        probe = teacher_drift_probe(model, data, SimpleNamespace(exploration=0., temperature=1.),
                                    dict(UPDATE_V21, kl_probe_rows=2))
        self.assertEqual(probe.positions, 2)
        self.assertEqual(len(probe.dataset.data), 2)
        self.assertAlmostEqual(probe.measure(model)['categorical_kl'], 0., places=6)
        with torch.no_grad():model.logits.copy_(torch.log(torch.tensor([.4, .2, .4])))
        expected = sum(p * math.log(p/q) for p, q in zip((.2, .3, .5), (.4, .2, .4)))
        self.assertAlmostEqual(probe.measure(model)['categorical_kl'], expected, places=6)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(teacher_drift_probe(model, [], SimpleNamespace(), UPDATE_V21))

    def test_retention_preserves_setup_identity_calibration_and_missingness(self):
        source = read(EVIDENCE/'iteration-0149/report.json');before = copy.deepcopy(source)
        retained = compact(source)
        self.assertEqual(source, before)
        self.assertEqual(retained['summary_version'], 5)
        self.assertEqual(retained['learner_by_composition_seat']['legacy'], source['by_composition']['legacy']['learner_by_seat'])
        self.assertEqual(len(retained['search']['collection_summaries']), 2)
        for row, old in zip(retained['search']['collection_summaries'], source['search']['collections']):
            self.assertEqual(row['teacher_model_id'], old['teacher_model_id'])
            self.assertEqual(row['post_distill_calibration']['receipt']['ready'], True)
        self.assertLess(len(json.dumps(retained)), 60000)
        old = dict(summary_version=3, actor={})
        self.assertEqual(compact(old), old)
        self.assertNotIn('utility_calibration', compact(old)['actor'])
        report = read(EVIDENCE/'evaluations/monitor-0007/report.json')
        out = compact(report)['evaluation_evidence']
        self.assertEqual(out['spec']['entries'], report['spec']['entries'])
        self.assertEqual(out['spec']['config'], report['spec']['config'])
        for key, row in report['results'].items():
            self.assertEqual(out['panels'][key]['outcomes'], row['outcomes'])
            self.assertFalse(out['panels'][key]['outcomes_truncated'])
            row['outcomes'] *= 30
            bounded = compact(report)['evaluation_evidence']['panels'][key]
            self.assertEqual(len(bounded['outcomes']), 1024)
            self.assertTrue(bounded['outcomes_truncated'])


if __name__ == '__main__':unittest.main()
