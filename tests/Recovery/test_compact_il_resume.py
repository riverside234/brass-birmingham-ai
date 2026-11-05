"""Pinned g921 continuation after the IL storage-only source update."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read, sha
from brass_ai.learning import contracts
from brass_ai.learning.checkpoint import save_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, digest, compatibility
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.recovery import entropy_schedule
from brass_ai.recovery.interrupted import archive_uncommitted_iteration
from brass_ai.recovery.resume import check_signature, load_resume, commit_transition, apply_optimizer_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import migrate, reward_spec
from brass_ai.step5.league import implementation_id, write_json


BUNDLE = SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0921-20261105T213224468369Z'
PREVIOUS = '4c514df4931721b1f8a96b6639ef9f5ee3b7cc1c98010e24536276c712f80fa6'


class CompactILResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = read(BUNDLE/'state.json')
        cls.current = implementation_id()

    def test_actual_export_accepts_only_explicit_same_config_review(self):
        state = copy.deepcopy(self.original)
        self.assertEqual(state['signature']['implementation'], PREVIOUS)
        self.assertEqual(state['signature']['compatibility'], compatibility())
        requested = dict(state['signature'], implementation=self.current)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, 'resume-reviewed-source'):
            check_signature(state, requested)
        self.assertTrue(check_signature(state, requested, reviewed=True))
        self.assertEqual(state, before)
        for field in ('config', 'compatibility', 'implementation', 'imitation_source'):
            bad = copy.deepcopy(requested)
            if field == 'config':bad[field]['learning_rate'] *= 2
            elif field == 'implementation':bad[field] = '0'*64
            else:bad[field]['unreviewed'] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_signature(state, bad, reviewed=True)

    def test_original_launch_config_follows_committed_rewards_then_reviews_source(self):
        launch = copy.deepcopy(self.original['signature']['config'])
        # The g760 file retains its first automatically applied reward stage.
        launch['insolvency'] = dict(event_cost=.005, max_cost=.05)
        launch['search']['reward']['insolvency'] = dict(launch['insolvency'])
        if 'teacher_phase' in launch:
            launch['teacher_phase']['search']['reward']['insolvency'] = dict(launch['insolvency'])
        effective = entropy_schedule.follow_committed(launch, self.original)
        self.assertEqual(effective, self.original['signature']['config'])
        self.assertTrue(check_signature(self.original,
            dict(self.original['signature'], config=effective, implementation=self.current), reviewed=True))

    def test_derived_equivalence_is_exact_and_unknown_edits_keep_their_hash(self):
        receipt = read(contracts.CONFIG_DIR/'derived-source-equivalence.json')
        self.assertEqual(contracts.derived_source_fingerprint(), receipt['original_sha256'])
        original_read = Path.read_text
        source = SOURCE_ROOT/'src/brass_ai/feature_engineering/data.py'
        def edited(path, *args, **kwargs):
            text = original_read(path, *args, **kwargs)
            return text+'\n# unreviewed edit\n' if path == source else text
        with patch.object(Path, 'read_text', edited):
            expected = digest({p.name:p.read_text(encoding='utf-8').replace('\r\n','\n')
                for p in sorted(source.parent.glob('*.py'))})
            self.assertEqual(contracts.derived_source_fingerprint(), expected)
            self.assertNotEqual(expected, receipt['original_sha256'])

    def fixture(self):
        temp = tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(temp.cleanup)
        run = Path(temp.name)
        state = copy.deepcopy(self.original)
        config = state['signature']['config']
        config['model_size'] = 'small'
        model = FourSeatNet(ModelConfig.preset('small'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        migrate(model, optimizer, reward_spec(config))
        apply_optimizer_config(optimizer, config)
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.utility_calibration = dict(ready=True, reward_id=digest(reward_spec(config)))
        path = run/'generation-0921.pt'
        identity = save_checkpoint(path, model, optimizer, step=921, training=dict(
            fresh_run=run.relative_to(SOURCE_ROOT).as_posix(), signature=digest(state['signature']),
            exploration=state['exploration'], fixture_auxiliary_tensor=torch.arange(8),
            fixture_value_optimizer=copy.deepcopy(optimizer.state_dict())))
        entry = dict(generation=921, path=path.relative_to(SOURCE_ROOT).as_posix(), sha256=sha(path), model_id=identity)
        state['history'] = [entry]
        state['population'] = [entry]
        state['evaluation_anchor'] = entry
        write_json(run/'state.json', state)
        requested = dict(state['signature'], implementation=self.current)
        return run, state, requested

    def test_real_checkpoint_commit_and_retry_preserve_weights_adam_rng_and_calibration(self):
        run, state, requested = self.fixture()
        path = SOURCE_ROOT/state['history'][-1]['path']
        original_hash = sha(path)
        expected = torch.load(path, map_location='cpu', weights_only=True)
        original_state = copy.deepcopy(state)
        for retry in range(2):
            model, raw, changed = load_resume(SOURCE_ROOT, run, state, requested, reviewed=retry == 0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=requested['config']['learning_rate'])
            resume_optimizer(raw, optimizer)
            self.assertTrue(_same_payload(expected['state_dict'], dict(model.state_dict())))
            self.assertTrue(_same_payload(expected['optimizer'], optimizer.state_dict()))
            self.assertTrue(_same_payload(expected['training'], raw['training']))
            self.assertTrue(torch.equal(expected['rng'], torch.get_rng_state()))
            self.assertEqual(model.utility_calibration, expected['training']['utility_calibration'])
            self.assertEqual(changed, retry == 0)
            self.assertEqual(sha(path), original_hash)
            if retry == 0:state = commit_transition(run, state, requested)
        self.assertEqual(state['source_transitions'][-1]['config_changes'], {})
        for key in original_state.keys()-{'signature', 'source_transitions'}:
            self.assertEqual(state[key], original_state[key], key)

    def test_partial_collection_is_archived_before_adoption_and_retry_is_safe(self):
        run, state, requested = self.fixture()
        pending = run/'iteration-0921'
        pending.mkdir()
        (pending/'partial.jsonl.gz').write_bytes(b'incomplete old-source replay')
        before = (run/'state.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Uncommitted iteration'):
            load_resume(SOURCE_ROOT, run, state, requested, reviewed=True)
        result = archive_uncommitted_iteration(SOURCE_ROOT, run, expected_iteration=921, apply=True)
        self.assertEqual(result['status'], 'archived')
        self.assertEqual((run/result['archive']/'iteration-0921/partial.jsonl.gz').read_bytes(),
                         b'incomplete old-source replay')
        self.assertEqual((run/'state.json').read_bytes(), before)
        self.assertEqual(archive_uncommitted_iteration(SOURCE_ROOT, run, expected_iteration=921, apply=True)['status'],
                         'nothing-to-archive')
        self.assertTrue(load_resume(SOURCE_ROOT, run, state, requested, reviewed=True)[2])


if __name__ == '__main__':unittest.main()
