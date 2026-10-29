"""Short deterministic policy checks; integration evidence is kept separately."""
import copy
import json
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = runpy.run_path(str(ROOT / 'tests/RustBackend/gumbel_session_reference.py'))
FakeClock, FocusGate = (REFERENCE[n] for n in ('FakeClock', 'FocusGate'))


class ReferenceTests(unittest.TestCase):
    def test_complete_paths_and_two_advancing_owners(self):
        gate = FocusGate()
        a = [['build|card-A'], ['coal|mine-A'], ['finish']]
        b = copy.deepcopy(a); b[1] = ['coal|mine-B']
        self.assertFalse(gate.observe(a, [16, 16]))
        self.assertFalse(gate.observe(a, [16, 16]))
        self.assertFalse(gate.observe(a, [32, 16]))
        self.assertTrue(gate.observe(a, [32, 32]))
        self.assertFalse(gate.observe(b, [48, 48]))
        self.assertFalse(gate.observe(None, [64, 64]))
        self.assertFalse(gate.observe(b, [80, 80]))

    def test_shared_clock_and_conservative_cost(self):
        clock = FakeClock(4_000_000_000)
        before = FocusGate.affordable(clock, 7_000_000, [1_000_000, 30_000_000], 8, 100)
        self.assertEqual(before['cost_ns'], 30_000_000)
        clock.advance(3_580_000_000)
        self.assertFalse(FocusGate.affordable(clock, 7_000_000, [], 8, 100)['useful'])
        self.assertEqual(clock.deadline, 4_000_000_000)
        clock.cancelled = True
        self.assertFalse(FocusGate.affordable(clock, 7_000_000, [], 8, 100)['useful'])

    def test_effective_k_reduces_without_changing_requested_k(self):
        for requested in (8, 16):
            clock = FakeClock(4_000_000_000)
            clock.advance(clock.work_deadline - 30_000_000)
            row = FocusGate.affordable(clock, 5_000_000, [], requested, 40)
            self.assertEqual(row['quota'], 2)
            self.assertEqual(row['effective_k'], 2)
            self.assertTrue(row['useful'])


if __name__ == '__main__': unittest.main()
