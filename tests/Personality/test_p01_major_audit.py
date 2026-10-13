"""Authored P0/P1 audit checks; no real model or qualification evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from personality_fixtures import episode, parent, qualified_run
from brass_ai.learning.contracts import SOURCE_ROOT, read_json
from brass_ai.learning.reward import specification
from brass_ai.personality import checkpoint, handoff, training
from brass_ai.personality.collection import distribution
from brass_ai.personality.model import import_parent
from brass_ai.personality.traces import examples
from brass_ai.step5.league import write_json


AUDIT = SOURCE_ROOT/'artifacts/p0p6-audit-20261013'


class P01MajorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        (AUDIT/'tmp').mkdir(parents=True, exist_ok=True)

    def test_warmup_to_joint_uses_saved_rng_with_a_fresh_optimizer(self):
        model, _ = import_parent(parent())
        data = examples([episode(rounds=4)], specification(), roots_only=True)
        optimizer, _ = training.fit(model, data, 'cpu', epochs=1)
        rng = torch.get_rng_state().clone()
        data = [d for d in data if d['row']['seat']==0]
        for item in data:
            item['on_policy'] = True
        with torch.no_grad():
            outputs = training.forward(model, data, list(range(len(data))), 'cpu')
            probs = distribution(outputs['policy_logits'], [d['row']['choices'] for d in data])
            for i, item in enumerate(data):
                item['row']['behavior']['probability'] = float(probs[i, item['row']['chosen']])
        raw = dict(training=dict(phase='warmup'), optimizer=optimizer.state_dict(), rng=rng, cuda_rng=[])
        manifest = dict(reward=specification(), utility_calibration=model.core.utility_calibration,
                        parent={}, signature=dict(config=dict(host='unused-host')))
        cfg = dict(training.DEFAULTS, batch_size=2, critic_batch_size=2, microbatch_size=2, target_kl=100.)
        results = []
        with tempfile.TemporaryDirectory(dir=AUDIT/'tmp') as tmp:
            base = Path(tmp)
            for index, unrelated_seed in enumerate((19, 8801)):
                run = base/str(index); restored = copy.deepcopy(model)
                state = dict(warmup_runs=1, joint_runs=0, validation_runs=0, reports=[])
                torch.manual_seed(unrelated_seed)
                with patch.object(training, 'initialize', return_value=(SOURCE_ROOT,run,manifest,None,restored,state,raw)), \
                     patch.object(training, 'collect'), \
                     patch.object(training, 'read_collection', return_value=(copy.deepcopy(data), None, [])), \
                     patch.object(training, 'resolved_config', return_value=cfg):
                    training.execute(SOURCE_ROOT, base/'bundle', run, mode='joint', games=1, epochs=1)
                _, saved = checkpoint.load(checkpoint.committed_path(run, read_json(run/'state.json')))
                # Warm-up and joint parameter sets differ: retain RNG, never old Adam groups.
                self.assertEqual(max(float(s['step']) for s in saved['optimizer']['state'].values()), 2.)
                results.append(saved)
        self.assertTrue(all(torch.equal(t, results[1]['state_dict'][name])
                            for name, t in results[0]['state_dict'].items()),
                        'Same committed warm-up and fresh games changed with process RNG')
        self.assertTrue(torch.equal(results[0]['rng'], results[1]['rng']))

    def test_handoff_claims_are_bound_to_copied_source_evidence(self):
        with tempfile.TemporaryDirectory(dir=AUDIT/'tmp') as tmp:
            base = Path(tmp); source = qualified_run(base/'source', traces=False)
            bundle = base/'bundle'; original = handoff.create(SOURCE_ROOT, source, bundle, trace_shards=0)
            self.assertEqual(handoff.load(bundle)[0], original)
            with patch.object(handoff, 'project_path', side_effect=AssertionError('Source run reopened')):
                self.assertEqual(handoff.load(bundle)[0], original)
            cases = [('signature', lambda m: m['signature']['config'].update(learning_rate=.05)),
                     ('origin', lambda m: m.update(origin=dict(fixture=False))),
                     ('generation', lambda m: m['parent'].update(generation=999)),
                     ('selection', lambda m: m['selection']['gate'].update(accepted=False)),
                     ('source run', lambda m: m.update(source_run='artifacts/unrelated-run')),
                     ('unlisted parent', lambda m: m.update(files=[x for x in m['files'] if x['path']!='parent.pt'])),
                     ('unlisted reference', lambda m: m.update(files=[x for x in m['files'] if not x['path'].startswith('references/')]))]
            for label, change in cases:
                mutated = copy.deepcopy(original); change(mutated)
                write_json(bundle/'manifest.json', mutated)
                with self.subTest(field=label), self.assertRaisesRegex(ValueError, 'evidence|differs'):
                    handoff.load(bundle)


if __name__ == '__main__':
    unittest.main()
