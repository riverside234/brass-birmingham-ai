"""Production g570 adoption must remain reviewed after Rust lab source updates."""
import copy
import unittest
from unittest.mock import patch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery import entropy_schedule as S, resume
from brass_ai.recovery.source_reviews import EXPANDED_FORMAT, expand_source_reviews
from brass_ai.recovery.training_revision_g570 import MIGRATION, configure
from brass_ai.step5.league import implementation_id


class G570Reviews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = read(SOURCE_ROOT / 'results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z/state.json')
        cls.current = implementation_id()
        cls.updated = S.configure(configure(cls.state['signature']['config']), cls.state, 'on', start='next')

    def test_current_source_admits_exact_pending_helper_config(self):
        # The lab's unchanged-config receipt alone passed its qualification
        # tests, but the user's production helper failed on this same source.
        before = copy.deepcopy(self.state)
        for origin in (self.state['signature']['implementation'], self.current):
            state = dict(self.state, signature=dict(self.state['signature'], implementation=origin))
            for config in (self.state['signature']['config'], configure(self.state['signature']['config']), self.updated):
                with self.subTest(origin=origin, scheduled=config['update'].get(S.KEY, {}).get('mode')):
                    target = dict(state['signature'], implementation=self.current, config=config)
                    resume.check_signature(state, target, reviewed=True)
        self.assertEqual(self.state, before)

    def test_lab_source_only_approval_cannot_implicitly_approve_new_recipe(self):
        approvals = expand_source_reviews(read(SOURCE_ROOT / 'configs/learning/selfplay-resume-sources.json'))
        rows = [row for row in approvals['transitions']
                if row['from_implementation'] == self.state['signature']['implementation']
                and row['to_implementation'] == self.current
                and row['learning_compatibility'] == digest(self.state['signature']['compatibility'])
                and MIGRATION not in row.get('config_migrations', [])
                and S.MIGRATION not in row.get('config_migrations', [])]
        self.assertTrue(rows)
        target = dict(self.state['signature'], implementation=self.current)
        with patch.object(resume, 'read', return_value=dict(format=EXPANDED_FORMAT, transitions=rows)):
            self.assertTrue(resume.check_signature(self.state, target, reviewed=True))
            with self.assertRaisesRegex(ValueError, 'explicitly pinned review changes'):
                resume.check_signature(self.state, dict(target, config=self.updated), reviewed=True)

    def test_production_review_still_rejects_unrelated_edits_and_unknown_source(self):
        target = dict(self.state['signature'], implementation=self.current, config=self.updated)
        with self.assertRaisesRegex(ValueError, 'resume-reviewed-source'):
            resume.check_signature(self.state, target)
        for field, value in (('learning_rate', .001), ('exploration', .03), ('teacher_enabled', True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.check_signature(self.state, dict(target, config=dict(self.updated, **{field: value})), reviewed=True)
        with self.assertRaisesRegex(ValueError, 'has not been reviewed'):
            resume.check_signature(self.state, dict(target, implementation='unreviewed-source'), reviewed=True)
        with self.assertRaises(ValueError):
            resume.check_signature(self.state, dict(target, compatibility={'model': 'changed'}), reviewed=True)


if __name__ == '__main__':
    unittest.main()
