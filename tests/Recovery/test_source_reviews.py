"""Compact source receipts retain exact approvals and fail closed on bad references."""
import copy
import json
import unittest
from unittest.mock import patch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery import resume
from brass_ai.recovery.source_reviews import (COMPACT_FORMAT, EXPANDED_FORMAT,
                                              compact_source_reviews, expand_source_reviews)
import test_audit_resume as audit_checks


class SourceReviewTests(unittest.TestCase):
    def receipt(self):
        entry = dict(from_implementation='old', to_implementation='new',
                     learning_compatibility=digest({'model': 'unchanged'}),
                     reason='An exact reviewed configuration change.',
                     config_changes={'learning_rate': {'from': .1, 'to': .2}},
                     config_migrations=['first', 'second'], config_migration='archived',
                     review_report='docs/original-review.md', reviewed_evidence=['a', 'b'])
        return dict(format=EXPANDED_FORMAT, transitions=[entry, copy.deepcopy(entry)])

    def test_roundtrip_order_metadata_and_independent_values(self):
        original = self.receipt()
        before = copy.deepcopy(original)
        packed = compact_source_reviews(original)
        self.assertEqual(packed['format'], COMPACT_FORMAT)
        self.assertEqual(len(packed['definitions']['reason']), 1)
        self.assertEqual(len(packed['definitions']['config_changes']), 1)
        unpacked = expand_source_reviews(packed)
        self.assertEqual(unpacked, before)
        self.assertEqual(original, before)
        self.assertEqual(compact_source_reviews(packed), packed)
        unpacked['transitions'][0]['config_changes']['learning_rate']['to'] = .5
        self.assertEqual(unpacked['transitions'][1], before['transitions'][1])
        self.assertEqual(expand_source_reviews(packed), before)

    def test_numbers_and_absent_optional_fields_are_not_merged(self):
        original = self.receipt()
        original['transitions'][0]['config_changes']['learning_rate']['to'] = 1
        original['transitions'][1]['config_changes']['learning_rate']['to'] = 1.0
        original['transitions'][1].pop('config_migrations')
        packed = compact_source_reviews(original)
        self.assertEqual(len(packed['definitions']['config_changes']), 2)
        self.assertEqual(json.dumps(expand_source_reviews(packed)), json.dumps(original))

    def test_missing_unknown_ambiguous_and_invalid_references_are_rejected(self):
        packed = compact_source_reviews(self.receipt())
        mutations = [
            lambda r: r.__setitem__('format', 'selfplay-resume-sources/99'),
            lambda r: r['transitions'][0].__setitem__('config_changes_ref', 'missing'),
            lambda r: r['transitions'][0].__setitem__('config_changes_ref', 0),
            lambda r: r['transitions'][0].__setitem__('config_changes', {}),
            lambda r: r['transitions'][0].__setitem__('unknown_ref', '0'),
            lambda r: r['definitions']['config_changes'].__setitem__('0', None),
            lambda r: r['definitions'].__setitem__('unknown', {}),
            lambda r: r['transitions'][0].pop('learning_compatibility_ref'),
            lambda r: r.__setitem__('transitions', {}),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(packed)
            mutate(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                expand_source_reviews(bad)
        legacy = self.receipt()
        legacy['transitions'][0]['config_changes_ref'] = '0'
        with self.assertRaises(ValueError):
            expand_source_reviews(legacy)

    def test_both_versions_enforce_the_same_exact_resume_guards(self):
        receipt = self.receipt()
        for row in receipt['transitions']:
            row.pop('config_migrations')
            row.pop('config_migration')
        old = dict(implementation='old', config={'learning_rate': .1, 'seconds': 16},
                   compatibility={'model': 'unchanged'}, teachers={'identity': 'fixed'})
        state = dict(format='step5-recovery-league/2.2.0', iteration=2, signature=old)
        target = dict(old, implementation='new', config=dict(old['config'], learning_rate=.2))
        for approvals in (receipt, compact_source_reviews(receipt)):
            with self.subTest(format=approvals['format']), patch.object(resume, 'read', return_value=approvals):
                self.assertTrue(resume.check_signature(state, target, True))
                self.assertFalse(resume.check_signature(state, old))
                with self.assertRaisesRegex(ValueError, 'resume-reviewed-source'):
                    resume.check_signature(state, target)
                for changed in (dict(target, config=dict(target['config'], seconds=17)),
                                dict(target, config=dict(target['config'], learning_rate=.3)),
                                dict(target, implementation='unknown'),
                                dict(target, compatibility={'model': 'changed'})):
                    with self.assertRaises(ValueError):
                        resume.check_signature(state, changed, True)

    def test_repository_registry_is_canonical_and_keeps_pinned_history(self):
        registry = read(SOURCE_ROOT / 'configs/learning/selfplay-resume-sources.json')
        expanded = expand_source_reviews(registry)
        self.assertEqual(compact_source_reviews(registry), registry)
        # Original v1 history is immutable; later source reviews may append.
        original = expanded['transitions'][:934]
        self.assertEqual(digest(original), 'e3568b780f0bd3b8815918f0349758508d6610479c61c728db7cf74f8bc29fa4')


class CompactResumeTests(audit_checks.AuditResumeTests):
    def signatures(self):
        old, new = super().signatures()
        # Use a reviewed source with the current model compatibility. The old
        # fixture hash predates architecture revisions and is not an admission
        # route for checkpoints authored with today's model tensors.
        old['implementation'] = '603d1d236ca3da4511467956a9b656b41a4f03017d814b1abd8f060467a4c05c'
        return old, new


if __name__ == '__main__':
    unittest.main()
