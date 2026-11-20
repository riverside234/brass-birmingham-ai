"""g4138 teacher search revision: 44+4 tables, short search 128 at C_scale .6, long labels kept, and the
search-value head trained only on games collected from adoption."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brass_ai.recovery import training_revision_g4138 as R


def before():
    from test_teacher_revision_g3270 import before as g3170
    from brass_ai.recovery import training_revision_g3270 as g3270
    return g3270.configure(g3170(), 3270)


def signature(config):
    from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
    return dict(config=config, implementation=QUALIFIED)


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before(); cls.after = R.configure(cls.before, 4138)

    def test_it_overlays_the_tables_and_short_search_and_keeps_the_labels(self):
        from brass_ai.recovery import training_revision_g1491 as g1491, training_revision_g3270 as g3270
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual(after['teacher_phase'], before['teacher_phase'])  # the qualified recipe is untouched
        self.assertEqual(R.previous(after), before)
        self.assertTrue(is_on(before) and is_on(after))
        self.assertEqual(buffer_identity(after), buffer_identity(before))  # long labels and beta carry over
        self.assertEqual(R.configure(after, 4139), after)  # adopted once
        self.assertEqual((g3270.tables(after), g3270.tables(before)), (dict(train=44, held_out=4), dict(train=36, held_out=4)))
        self.assertEqual(g3270.phase_options(after, after['teacher_phase'])['tables'], dict(train=44, held_out=4))
        self.assertEqual(g1491.count_search(after), g1491.G2600_SEARCH)
        self.assertEqual(R.search_profile(after), dict(full_k=8, c_scale=.2, fast_c_scale=.6, fast=128, fast_k=8))
        self.assertEqual(R.search_profile(before), g1491.G2600_SEARCH)
        self.assertEqual((R.rebuild(after), R.rebuild(before)), (dict(origin_generation=4138, required_ages=8), None))
        bad = copy.deepcopy(after); bad[R.KEY]['short_search']['simulations'] = 96
        with self.assertRaisesRegex(ValueError, 'g4138 teacher search revision'):validate_config(bad)
        other = copy.deepcopy(before); other['teacher_phase'][g1491.KEY]['count_search'] = dict(g1491.UPDATED_SEARCH)
        with self.assertRaisesRegex(ValueError, 'g2600 search profile'):R.configure(other, 4138)

    def test_it_is_adopted_once_and_changes_the_collection(self):
        from brass_ai.rust_backend.lab_shadow_config import collection_signature, signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        changed = copy.deepcopy(after); changed[R.KEY]['from_generation'] = 4140
        self.assertFalse(signature_transition(signature(after), signature(changed)))
        self.assertNotEqual(collection_signature(signature(before)), collection_signature(signature(after)))

    def test_forty_eight_tables_with_four_validating(self):
        from collections import Counter
        from brass_ai.recovery.teacher_phase import plan_jobs
        for generation in (4138, 4139):
            state = dict(iteration=generation, history=[], signature=signature(self.after))
            jobs, _ = plan_jobs(self.after, state, model_id='m'*64, export_hash='a'*64, epoch=generation, ready=True)
            self.assertEqual(Counter(j['split'] for j in jobs), Counter(train=44, heldout=4))

    def test_the_cpu_budget_must_carry_the_short_search(self):
        from brass_ai.recovery.collector_runtime import cpu_block, validate_cpu, validate_teacher_budget
        good = cpu_block(teacher_budget=dict(R.search_profile(self.after), mode='count', full=640))
        validate_cpu(good); validate_teacher_budget(self.after, good)
        for change in (dict(fast=96), dict(fast_c_scale=.7), dict(fast_k=4)):
            bad = cpu_block(teacher_budget=dict(good['teacher_budget'], **change))
            with self.assertRaisesRegex(ValueError, 'committed teacher search profile'):validate_teacher_budget(self.after, bad)
        with self.assertRaisesRegex(ValueError, 'teacher search budget'):
            validate_cpu(cpu_block(teacher_budget=dict(good['teacher_budget'], fast_c_scale=.65)))


class SearchValue(unittest.TestCase):
    def test_only_games_from_adoption_train_after_eight_ages(self):
        from brass_ai.recovery import training_revision_g3270 as g3270
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        from brass_ai.recovery.search_value_training import CONTRACT
        after = R.configure(before(), 4138)
        options = g3270.search_value_options(after, after['teacher_phase']['search_value'])
        stage = dict(contract=CONTRACT, stage='C', w=1., scale=1., generation=4137, lam=.95, startup='immediate',
                     g0_passed=True, history=[], ordinary=[], baseline=None, full_scale_generation=None, h2h=None)
        row = lambda generation, game, split='train': dict(generation=generation, game_id=f'teacher/{game}', split=split)
        seen = {}
        with tempfile.TemporaryDirectory() as directory:
            phase = SearchValuePhase(Path(directory)/'values', object(), options, 'r', saved=dict(stage=stage), trainable=False,
                                     revision=g3270.value_settings(after), rebuild=R.rebuild(after))
            for generation in range(4138, 4147):
                train = [row(g, f'{g}-{k}') for g in range(generation - 15, generation + 1) for k in range(2)]
                retired = [row(generation - 9, 'held-old'), row(4138, 'held-new')] if generation >= 4146 else []
                rows = lambda g, split='train', **_: train if split == 'train' else []
                captured = []
                with patch.object(phase.buffer, 'rows', side_effect=rows), \
                        patch('brass_ai.recovery.search_value_retention.rotating_holdout',
                              return_value=([row(generation, 'validation', 'heldout')], retired, dict(ready=True))), \
                        patch.object(phase, '_targets', side_effect=lambda rows: captured.append(rows) or rows):
                    report = phase.after_ppg(generation, mode='on')
                seen[generation] = (report['buffer_rebuild']['fresh_ages'], report['buffer_rebuild']['paused'],
                                    sorted({r['game_id'] for r in captured[0]}) if captured else None)
                self.assertEqual(report['buffer_rebuild']['excluded_rows'],
                                 sum(r['generation'] < 4138 for r in train + retired))
        self.assertEqual(seen[4138], (1, True, None))
        self.assertEqual(seen[4144], (7, True, None))  # seven fresh ages: still rebuilding
        fresh = seen[4145]
        self.assertEqual(fresh[:2], (8, False))
        self.assertEqual(fresh[2], sorted(f'teacher/{g}-{k}' for g in range(4138, 4146) for k in range(2)))
        self.assertIn('teacher/held-new', seen[4146][2]); self.assertNotIn('teacher/held-old', seen[4146][2])


if __name__ == '__main__':
    unittest.main()
