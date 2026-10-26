import json
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[2]
REF = runpy.run_path(str(ROOT/'scripts/reproduce-gumbel-coordinator.py'))


class CoordinatorReferenceTests(unittest.TestCase):
    def test_recorded_contract(self):
        self.assertEqual(REF['document'](), json.loads(
            (ROOT/'tests/GumbelISMCTS/fixtures/coordinator-v1.json').read_text(encoding='utf-8')))

    def test_partial_work_is_absent_from_pool(self):
        case = REF['document']()['cases'][1]
        self.assertEqual(case['expected']['count'], [3]*4)
        self.assertEqual(case['expected']['eligible'], [0])

    def test_union_recomputed_instead_of_visit_vote(self):
        case = REF['document']()['cases'][3]
        self.assertEqual(case['expected']['count'], [7]*4)
        self.assertEqual(set(case['expected']['ranking']), {0,1,2,3})
        self.assertNotEqual(case['expected']['ranking'][0], 0)

    def test_reservations_never_complete_or_eliminate(self):
        trace = REF['document']()['reservations'][0]['expected']
        self.assertEqual(trace[4]['pending'], [0,1,0,1])
        self.assertEqual(trace[4]['completed'], [])
        self.assertFalse(trace[-2]['balanced'])
        self.assertTrue(trace[-1]['balanced'])
        end = REF['document']()['reservations'][1]['expected'][-1]
        self.assertEqual(end['retired'], 1)
        self.assertFalse(end['balanced'])
