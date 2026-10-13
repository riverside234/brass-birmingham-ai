"""Exact committed continuation through the remaining self-play bug review."""
import copy
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, digest
from brass_ai.learning.replay import validate_teacher
from brass_ai.recovery.resume import check_signature, load_resume
from brass_ai.recovery.updates import distill_update
from brass_ai.step5.league import implementation_id
import test_audit_resume as audit_checks


PREVIOUS = 'aa41d2a962ded63fb83f89ec4b31fddc47a550ecab0ef8be8654d56262911f46'


class SelfplayMajorResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def signatures(self, name='joint7-g150-gate72.json', *, small=True):
        config = read(SOURCE_ROOT / 'configs/step5' / name)
        if small:
            config['model_size'] = 'small'
        new = dict(config=config, compatibility=compatibility(),
                   implementation=implementation_id(), legacy_pool={}, teachers={})
        old = copy.deepcopy(new)
        old['implementation'] = PREVIOUS
        return old, new

    def fixture(self):
        from test_teacher_work_resume import TeacherWorkResumeTests
        fixture = TeacherWorkResumeTests()
        fixture.signatures = self.signatures
        self.addCleanup(fixture.doCleanups)
        return fixture.fixture()

    # The shared fixture has real nonempty Adam moments, a committed utility
    # checkpoint, champion/population references, counters and search debt.
    test_committed_weights_optimizer_rng_population_and_progress_survive = (
        audit_checks.AuditResumeTests.test_committed_weights_optimizer_rng_population_and_progress_survive)

    def test_all_active_presets_require_exact_settings_and_reviewed_source(self):
        presets = [path for path in sorted((SOURCE_ROOT / 'configs/step5').rglob('*.json'))
                   if read(path).get('format') == 'step5-training/2.2.0']
        self.assertTrue(presets)
        for path in presets:
            with self.subTest(preset=path.name):
                old, new = self.signatures(path.relative_to(SOURCE_ROOT / 'configs/step5'), small=False)
                state = dict(format='step5-recovery-league/2.2.0', iteration=70, signature=old)
                self.assertEqual(old['config'], read(path))
                self.assertEqual(old['config'], new['config'])
                self.assertTrue(check_signature(state, new, reviewed=True))
                with self.assertRaisesRegex(ValueError, 'resume-reviewed-source'):
                    check_signature(state, new)
                changes = [
                    lambda s: s['config'].__setitem__('learning_rate', s['config']['learning_rate'] * 2),
                    lambda s: s['config']['search'].__setitem__('seconds', s['config']['search']['seconds'] * 2),
                    lambda s: s['config']['teacher_work_budget']['seconds_per_phase'].__setitem__('rail_early', 17.),
                    lambda s: s['config']['legacy_opponent_work_budget'].__setitem__('seconds_per_turn', 2.),
                    lambda s: s['config']['execution'].__setitem__('search_workers_per_turn', 2),
                    lambda s: s['compatibility'].__setitem__('unknown-change', True),
                    lambda s: s.__setitem__('implementation', '0' * 64),
                ]
                for index, change in enumerate(changes):
                    altered = copy.deepcopy(new)
                    change(altered)
                    with self.subTest(change=index), self.assertRaises(ValueError):
                        check_signature(state, altered, reviewed=True)
                unknown = copy.deepcopy(state)
                unknown['signature']['implementation'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'not been reviewed'):
                    check_signature(unknown, new, reviewed=True)

    def test_prior_partial_iteration_and_teacher_labels_cannot_be_reused(self):
        run, state, new, _, _ = self.fixture()
        pending = run / 'iteration-0070'
        pending.mkdir()
        partial = pending / 'search-000000.jsonl.gz'
        partial.write_bytes(b'prior source partial collection')
        before = (run / 'state.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Uncommitted iteration'):
            load_resume(SOURCE_ROOT, run, state, new, reviewed=True)
        self.assertEqual((run / 'state.json').read_bytes(), before)
        self.assertEqual(partial.read_bytes(), b'prior source partial collection')

        from test_teacher_readiness_review import TeacherReadinessReviewTests
        model, optimizer, episode, config, quality = TeacherReadinessReviewTests().fixture()
        for row in episode['records']:
            teacher = row['teacher']
            teacher['source_fingerprint'] = PREVIOUS
            teacher['fingerprint'] = digest(dict(model=teacher['model_id'], config=teacher['search'],
                contract='masked-ismcts/' + teacher['source'].rsplit('/', 1)[1], source=PREVIOUS))
            validate_teacher(teacher, len(row['choices']))
        with self.assertRaisesRegex(ValueError, 'fresh current-model/current-source'):
            distill_update(model, optimizer, [episode], config, steps=1, quality=quality)


if __name__ == '__main__':
    unittest.main()
