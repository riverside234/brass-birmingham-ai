import json
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
REF=runpy.run_path(str(ROOT/'scripts/reproduce-gumbel-sessions.py'))

class SessionReferenceTests(unittest.TestCase):
    def test_recorded_contract(self):
        expected=json.loads((ROOT/'tests/GumbelISMCTS/fixtures/session-v1.json').read_text(encoding='utf-8'))
        self.assertEqual(REF['document'](),expected)

    def test_incomplete_round_never_changes_recommendation(self):
        cases={c['name']:c for c in REF['document']()['cases']}
        a=cases['exact-round-boundary']['expected'];b=cases['incomplete-second-round']['expected']
        self.assertEqual(a['snapshot'],b['snapshot']);self.assertEqual(a['ranking'],b['ranking'])
        self.assertEqual(b['partial_completed'],3)

    def test_remainder_and_zero_work(self):
        cases={c['name']:c for c in REF['document']()['cases']}
        partial=cases['during-remainder']['expected'];finished=cases['remainder-complete']['expected']
        self.assertEqual(partial['snapshot']['support_watermark'],18)
        self.assertEqual(partial['partial_completed'],2)
        self.assertEqual(finished['snapshot']['support_watermark'],23)
        self.assertEqual(finished['snapshot']['remainder_simulations'],5)
        self.assertIsNone(cases['before-first-round']['expected']['snapshot'])
        self.assertIsNone(cases['k1-no-backup']['expected']['snapshot'])

    def test_startup_and_identity_matched_completed_costs(self):
        q=REF['quota'];self.assertEqual(q(3600,[],10),360);self.assertEqual(q(3600,[],None),0)
        self.assertEqual(q(3600,[5,6,7],10),360);self.assertEqual(q(3600,[5,6,7,8],10),720)
        self.assertEqual(q(3600,list(range(1,11)),10),900)

if __name__=='__main__':unittest.main()
