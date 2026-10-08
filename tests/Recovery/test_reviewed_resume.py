"""Same-run v2 continuation preserves training state without disabling guards."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, digest
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.recovery.resume import check_signature, load_resume, commit_transition
from brass_ai.step5.cli import parser
from brass_ai.step5.league import implementation_id, write_json


class ReviewedResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.evidence = read(SOURCE_ROOT / 'results/selfplay-v2/state.json')
        # Keep exercising the historical 50% -> 70% transition itself. The
        # generation-70 continuation has its own current-source coverage.
        cls.target = read(SOURCE_ROOT/'results/selfplay-v2-70/state.json')['signature']['implementation']

    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix='resume-', dir=SOURCE_ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name).resolve()
        state = copy.deepcopy(self.evidence)
        state['signature']['config']['model_size'] = 'small'
        model = FourSeatNet(ModelConfig.preset('small'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        sum(p.square().mean() for p in model.parameters()).backward(); optimizer.step()
        path = run / 'generation-0002.pt'
        identity = save_checkpoint(path, model, optimizer, step=2, training=dict(
            fresh_run=run.relative_to(SOURCE_ROOT).as_posix(), signature=digest(state['signature']),
            mode='step5-population-ppo'))
        entry = dict(path=path.relative_to(SOURCE_ROOT).as_posix(), sha256=sha(path), model_id=identity, generation=2)
        state.update(history=[entry], population=[entry], evaluation_anchor=entry, resume=[entry])
        write_json(run / 'state.json', state)
        signature = dict(state['signature'], implementation=self.target)
        return run, state, signature, model, optimizer

    def test_real_v2_signature_has_exact_reviewed_active_lab_continuation(self):
        old = self.evidence['signature']
        self.assertEqual(old['compatibility'], compatibility())
        current = read(SOURCE_ROOT / 'configs/step5/lab.json')
        current['rehearsal_manifest'] = old['imitation_source']['manifest']
        for key in ('learning_rate','update','curriculum','search','imitation'):
            self.assertEqual(current[key],old['config'][key])
        target = dict(old, config=current, implementation=implementation_id())
        self.assertTrue(check_signature(self.evidence, target, True))
        with self.assertRaisesRegex(ValueError, '--resume-reviewed-source'):
            check_signature(self.evidence, target)
        args = parser().parse_args(['train', '--run', 'artifacts/step5/selfplay-v2', '--resume-reviewed-source'])
        self.assertTrue(args.resume_reviewed_source)

    def test_config_contract_origin_and_unreviewed_source_changes_rejected(self):
        old = self.evidence['signature']
        for field in ('config', 'compatibility', 'legacy_pool', 'teachers', 'imitation_source'):
            target = copy.deepcopy(old); target['implementation'] = self.target
            target[field] = dict(target[field], changed=True)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'unchanged'):
                check_signature(self.evidence, target, True)
        with self.assertRaisesRegex(ValueError, 'not been reviewed'):
            check_signature(self.evidence, dict(old, implementation='unreviewed'), True)
        state = copy.deepcopy(self.evidence); state['signature']['implementation'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'not been reviewed'):
            check_signature(state, dict(old, implementation=self.target), True)

    def test_weights_adam_rng_counters_and_repeat_resume_preserved(self):
        run, state, signature, original, optimizer = self.fixture()
        old_bytes = (run / 'generation-0002.pt').read_bytes()
        model, raw, changed = load_resume(SOURCE_ROOT, run, state, signature, reviewed=True)
        self.assertTrue(changed)
        restored = torch.optim.AdamW(model.parameters(), lr=1e-5)
        resume_optimizer(raw, restored)
        self.assertTrue(torch.equal(torch.get_rng_state(), raw['rng']))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items()))
        expected, actual = optimizer.state_dict(), restored.state_dict()
        self.assertEqual(expected['param_groups'], actual['param_groups'])
        for key, values in expected['state'].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, actual['state'][key][name]))
        updated = commit_transition(run, state, signature)
        for k, value in state.items():
            if k != 'signature': self.assertEqual(updated[k], value, k)
        self.assertEqual(read(run / updated['source_transitions'][0]['backup']), state)
        self.assertEqual((run / 'generation-0002.pt').read_bytes(), old_bytes)
        for flag in (False, True):
            _, again, changed = load_resume(SOURCE_ROOT, run, updated, signature, reviewed=flag)
            self.assertFalse(changed)
            self.assertEqual(again['model_id'], raw['model_id'])
        broken = copy.deepcopy(updated); broken['source_transitions'][0]['checkpoint']['sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'not bound'):
            load_resume(SOURCE_ROOT, run, broken, signature)

    def test_only_pinned_50_to_70_budget_change_preserves_optimizer_and_progress(self):
        run,state,signature,_,_=self.fixture()
        signature=copy.deepcopy(signature);signature['config']['search_wall_fraction']=.70
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        resume_optimizer(raw,optimizer)
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-5)
        updated=commit_transition(run,state,signature)
        self.assertEqual(updated['source_transitions'][-1]['config_changes'],
                         dict(search_wall_fraction=dict(before=.5,after=.7)))
        self.assertEqual((updated['games_done'],updated['next_search_games']),
                         (state['games_done'],state['next_search_games']))
        self.assertFalse(load_resume(SOURCE_ROOT,run,updated,signature)[2])
        for fraction in (.6,.8):
            other=copy.deepcopy(signature);other['config']['search_wall_fraction']=fraction
            with self.assertRaisesRegex(ValueError,'unchanged'):
                check_signature(state,other,True)
        other=copy.deepcopy(signature);other['config']['learning_rate']=2e-5
        with self.assertRaisesRegex(ValueError,'unchanged'):
            check_signature(state,other,True)

    def test_partial_iteration_and_changed_checkpoint_are_not_overwritten(self):
        run, state, signature, _, _ = self.fixture()
        pending = run / 'iteration-0002'; pending.mkdir()
        shard = pending / 'actor-000000.jsonl.gz'; shard.write_bytes(b'old partial collection')
        with self.assertRaisesRegex(ValueError, 'Uncommitted iteration'):
            load_resume(SOURCE_ROOT, run, state, signature, reviewed=True)
        self.assertEqual(read(run / 'state.json'), state)
        self.assertEqual(shard.read_bytes(), b'old partial collection')
        (run / 'generation-0002.pt').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
            load_resume(SOURCE_ROOT, run, state, signature, reviewed=True)

    def test_checkpoint_lineage_and_live_population_still_checked(self):
        run, state, signature, _, _ = self.fixture()
        bad = copy.deepcopy(state); bad['history'][-1]['generation'] = 1
        with self.assertRaisesRegex(ValueError, 'generation/model'):
            load_resume(SOURCE_ROOT, run, bad, signature, reviewed=True)
        bad = copy.deepcopy(state); bad['population'] = [dict(state['history'][-1], path='artifacts/outside.pt')]
        with self.assertRaisesRegex(ValueError, 'escapes owned run'):
            load_resume(SOURCE_ROOT, run, bad, signature, reviewed=True)
        bad = copy.deepcopy(state)
        path = run / 'frozen.pt'; path.write_bytes(b'changed')
        bad['population'] = [dict(state['history'][-1], path=path.relative_to(SOURCE_ROOT).as_posix())]
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            load_resume(SOURCE_ROOT, run, bad, signature, reviewed=True)

    def test_atomic_state_failure_can_retry_from_untouched_backup(self):
        run, state, signature, _, _ = self.fixture()
        def fail_state(path, value):
            if path.name == 'state.json': raise OSError('simulated interruption')
            return write_json(path, value)
        with patch('brass_ai.recovery.resume.write_json', side_effect=fail_state):
            with self.assertRaisesRegex(OSError, 'simulated'):
                commit_transition(run, state, signature)
        self.assertEqual(read(run / 'state.json'), state)
        updated = commit_transition(run, state, signature)
        self.assertEqual(len(updated['source_transitions']), 1)
        self.assertEqual(len(list(run.glob('state-before-source-*.json'))), 1)
        self.assertFalse(load_resume(SOURCE_ROOT, run, updated, signature)[2])


if __name__ == '__main__': unittest.main()
