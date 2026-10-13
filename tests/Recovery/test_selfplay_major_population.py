"""Population cache integrity across refresh and interrupted publication."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import retention


class PopulationIntegrityTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run = root / 'artifacts/run'
        run.mkdir(parents=True)
        source = run / 'generation-0001.pt'
        raw = dict(model_id='a' * 64, state_dict={'weight': torch.tensor([1., 2.])},
                   optimizer={'state': 'training only'}, step=1, rng=torch.get_rng_state(),
                   training=dict(fresh_run='artifacts/run', utility_calibration={'ready': True},
                                 evidence={'large': 'report'}, imitation={'large': 'report'}))
        torch.save(raw, source)
        entry = dict(path=source.relative_to(root).as_posix(), sha256=sha(source),
                     model_id=raw['model_id'], generation=1)
        state = dict(history=[entry], signature={'config': {'retention': {'resume_checkpoints': 2}}})
        retention.update_population(root, run, state, dict(recent=2, anchors=1))
        return root, run, source, raw, entry, state

    def test_refresh_cannot_bless_changed_committed_cache(self):
        root, run, _, _, _, state = self.fixture()
        cached = root / state['population'][0]['path']
        cached.write_bytes(b'corrupted checkpoint')
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, 'Population.*changed'):
            retention.update_population(root, run, state, dict(recent=2, anchors=1))
        self.assertEqual(state, before)
        self.assertEqual(cached.read_bytes(), b'corrupted checkpoint')

    def test_orphan_cache_requires_exact_slim_content(self):
        root, run, _, _, entry, state = self.fixture()
        cached = root / state['population'][0]['path']
        valid = torch.load(cached, weights_only=True)
        for field in ('weights', 'calibration', 'lineage', 'dtype'):
            with self.subTest(field=field):
                changed = copy.deepcopy(valid)
                if field == 'weights': changed['state_dict']['weight'][0] += 1
                elif field == 'calibration': changed['training']['utility_calibration']['ready'] = False
                elif field == 'lineage': changed['training']['fresh_run'] = 'artifacts/foreign'
                else: changed['state_dict']['weight'] = changed['state_dict']['weight'].double()
                torch.save(changed, cached)
                before = cached.read_bytes()
                with self.assertRaisesRegex(ValueError, 'Population.*changed'):
                    retention.slim_checkpoint(root, run, entry)
                self.assertEqual(cached.read_bytes(), before)

    def test_valid_orphan_and_committed_reuse_preserve_bytes_and_rng(self):
        root, run, source, _, entry, state = self.fixture()
        cached = root / state['population'][0]['path']
        before, full_before, rng = cached.read_bytes(), source.read_bytes(), torch.get_rng_state()
        self.assertEqual(retention.slim_checkpoint(root, run, entry), state['population'][0])
        with patch.object(retention.torch, 'load', side_effect=AssertionError('committed cache needs only its receipt')):
            retention.update_population(root, run, state, dict(recent=2, anchors=1))
        self.assertEqual(cached.read_bytes(), before)
        self.assertEqual(source.read_bytes(), full_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_champion_receipt_survives_reentry_of_unchanged_weights(self):
        root, run, source, raw, entry, state = self.fixture()
        state['champion'] = state['population'].pop()
        cached = root / state['champion']['path']
        before = cached.read_bytes()
        # A no-op learner generation has the same model ID and different
        # training metadata; its older frozen champion remains valid.
        raw['step'] = 2
        torch.save(raw, source)
        entry.update(sha256=sha(source), generation=2)
        with patch.object(retention.torch, 'load', side_effect=AssertionError('use the champion receipt')):
            retention.update_population(root, run, state, dict(recent=2, anchors=1))
        self.assertEqual(state['population'][0]['generation'], 2)
        self.assertEqual(state['population'][0]['sha256'], state['champion']['sha256'])
        self.assertEqual(cached.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
