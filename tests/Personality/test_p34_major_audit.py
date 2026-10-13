"""P3/P4 audit regressions; authored data exercises contracts, never qualification."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.reward import specification
from brass_ai.personality.deployment import evaluation, panels, roster
from brass_ai.personality.deployment.contracts import validate_roster
from brass_ai.personality.repertoire import archive, discovery
from brass_ai.personality.repertoire.contracts import identity as repertoire_identity
from brass_ai.personality.repertoire.traces import FIELDS, MAX_TURNS
from brass_ai.step5.league import write_json
from personality_fixtures import episode
from test_deployment import candidate, entries


class P34MajorAuditTests(unittest.TestCase):
    def select_fixture(self, out, *, origin_fixture=False, scope='real readiness'):
        es, reps = entries()
        ready = out/'ready'; ready.mkdir()
        write_json(ready/'ready.json', dict(fixture=origin_fixture, scope=scope))
        (ready/'model.pt').write_bytes(b'authored placeholder; never learned weights')
        ready_value = dict(id='ready', model=dict(model_id='fixture', path='model.pt'),
                           reward=specification(), scope=scope)
        handoff = dict(origin=dict(fixture=origin_fixture))
        proposal = out/'proposal'; proposal.mkdir()
        p = dict(format=discovery.VERSION, contract=repertoire_identity(),
                 candidates=[e['controller'] for e in es], control_id=es[0]['candidate_id'],
                 ready=ready.relative_to(SOURCE_ROOT).as_posix(), ready_id='ready',
                 ready_sha256=sha(ready/'ready.json'), pool_id='pool')
        p['id'] = digest(p); write_json(proposal/'manifest.json', p)
        enc = out/'encoder'; enc.mkdir()
        write_json(enc/'manifest.json', dict(id='encoder'))
        (enc/'encoder.pt').write_bytes(b'authored descriptor placeholder')
        a = dict(format=archive.VERSION, contract=repertoire_identity(), entries=es, elites=reps,
                 proposal_id=p['id'], proposal_path=proposal.relative_to(SOURCE_ROOT).as_posix(),
                 pool_id='pool', sources=[], encoder=dict(path=enc.relative_to(SOURCE_ROOT).as_posix(),
                 id='encoder', sha256=sha(enc/'manifest.json')))
        a['id'] = digest(a); source = out/'archive'; source.mkdir(); write_json(source/'manifest.json', a)
        enc_value = dict(id='encoder', report=dict(collapsed=False), training_groups=[],
                         weights=dict(path='encoder.pt'))
        loaded = (None, [object()], ready_value, handoff, 'pool')
        with patch.object(roster, 'load_ready', return_value=loaded), \
                patch.object(roster, 'model_id', return_value='parent'), \
                patch.object(roster.descriptor, 'load', return_value=(enc_value, None)):
            value = roster.select(SOURCE_ROOT, source, out/'roster')
        return value, loaded

    def test_select_preserves_fixture_origin_and_scope(self):
        for origin_fixture, scope, expected in (
                (True, 'calibrated readiness', True),
                (False, 'AUTHORED FIXTURE ONLY', True),
                (False, 'calibrated readiness', False)):
            with self.subTest(origin_fixture=origin_fixture, scope=scope), \
                    tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                value, _ = self.select_fixture(Path(tmp), origin_fixture=origin_fixture, scope=scope)
                self.assertIs(value.get('fixture'), expected)

    def test_fixture_provenance_cannot_be_cleared_on_a_roster(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            value, loaded = self.select_fixture(Path(tmp), origin_fixture=True)
            with patch.object(roster, 'load_ready', return_value=loaded), \
                    patch.object(roster, 'model_id', return_value='parent'):
                self.assertEqual(roster.validate_sources(SOURCE_ROOT, value), 1)
                value['fixture'] = False
                with self.assertRaisesRegex(ValueError, 'fixture|Fixture'):
                    roster.validate_sources(SOURCE_ROOT, value)

    def check_with_passing_statistical_gates(self, out, fixture):
        # Isolate the artifact-origin gate: every previous P4 gate is true.
        # This is a contract fixture, not simulated or measured playing strength.
        value = candidate(16, 8); value['encoder'] = dict(path='authored-encoder', id='encoder')
        if fixture is not None: value['fixture'] = fixture
        else: value.pop('fixture', None)
        jobs = [dict(index=i, seed=3, neural_seats=[0],
                     p4=dict(role='member', group=m['archetype_id'], member=m['id']))
                for i, m in enumerate(value['members'])]
        panel = dict(id='panel', jobs=jobs, labels=None)
        path = out/'panel'; path.mkdir(); write_json(path/'manifest.json', dict(authored=True))
        game = episode(4)
        pairs = [(game, dict(job=j, seconds=1., telemetry=dict(completed=1, seconds=.1))) for j in jobs]
        features = np.zeros((MAX_TURNS, len(FIELDS)), np.float32)
        features[0, FIELDS.index('flips_Pottery')] = 1.
        mask = np.arange(MAX_TURNS) == 0
        behavior = dict(both_eras=True, pottery_opportunities=1, pottery_built=1,
                        public_turns={}, public_boards={})
        with patch.object(panels, 'read_chunks', return_value=(panel, value, iter(pairs), set(range(len(jobs))), [])), \
                patch.object(evaluation.descriptor, 'load', return_value=(dict(id='encoder'), None)), \
                patch.object(evaluation.descriptor, 'transform', return_value=np.zeros((len(jobs), 2))), \
                patch.object(evaluation, 'trajectory', return_value=(features, mask)), \
                patch.object(evaluation, 'public_behavior', return_value=behavior), \
                patch.object(evaluation, 'quality', return_value=dict(passed=True)), \
                patch.object(evaluation, 'variety', return_value=dict(passed=True)), \
                patch.object(evaluation, 'geometry', return_value=dict(passed=True)), \
                patch.object(evaluation, 'human_review', return_value=dict(passed=True)):
            return evaluation.check(SOURCE_ROOT, path, [], out/'checked')

    def test_full_fixture_cannot_qualify_even_when_all_other_gates_pass(self):
        for fixture in (True, None, False):
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                result = self.check_with_passing_statistical_gates(Path(tmp), fixture)
                self.assertTrue(all(v for k, v in result['checks'].items() if k != 'real_artifacts'))
                self.assertIs(result['p4_qualified'], fixture is False)
                self.assertIs(result['checks']['real_artifacts'], fixture is False)
                self.assertIs(result['fixture'], fixture is not False)

    def test_qualified_roster_requires_explicit_nonfixture_provenance(self):
        value = candidate(16, 8)
        value.update(p4_qualified=True, qualification=dict(report='authored-gate-fixture'))
        with self.assertRaisesRegex(ValueError, 'Fixture'):
            validate_roster(value)
        value.pop('fixture')
        with self.assertRaisesRegex(ValueError, 'Fixture'):
            validate_roster(value)
        value['fixture'] = False
        validate_roster(value)


if __name__ == '__main__':
    unittest.main()
