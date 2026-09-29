import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from brass_ai.host import RulesHost
from brass_ai.runtime_contracts import policy_inputs, validate
from jsonschema import ValidationError


class SchemaTests(unittest.TestCase):
    def test_live_allowlists_and_reject_private_fields(self):
        with RulesHost() as host:
            host.reset(20)
            view, choices = policy_inputs(host)
            validate("checkpoint", host.call("checkpoint"))
            self.assertGreater(len(choices), 1)
            for path, key, value in (((), "seed", 20), (("players", 1), "hand", ["location:Derby"]),
                                     (("context",), "rng", [9, 10])):
                altered = copy.deepcopy(view)
                node = altered
                for part in path: node = node[part]
                node[key] = value
                with self.assertRaises(ValidationError): validate("observation", altered)

    def test_all_recorded_phase_checkpoints(self):
        # Populated by check-r1-traces.py. No proprietary snapshots are versioned.
        traces = ROOT / "artifacts/r1-host/traces.json"
        self.assertTrue(traces.exists(), "Run scripts/check-r1-traces.py first")
        rows = json.loads(traces.read_text())
        with RulesHost() as host:
            for row in rows:
                validate("checkpoint", row["checkpoint"])
                context = validate("context", host.call("restore", checkpoint=row["checkpoint"]))
                for seat in range(4): validate("observation", host.observe(seat))
                validate("choices", host.choices())
                if context["terminal"]: validate("result", host.call("result"))


if __name__ == "__main__":
    unittest.main()
