"""Run the R1 unit/contract suite and retain an auditable local result."""
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
binary = ROOT / "artifacts/r1-host/guarded/BrassHost.exe"
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
start = time.monotonic()
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests/R1"), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {"passed": result.wasSuccessful(), "tests": result.testsRun, "failures": len(result.failures),
          "errors": len(result.errors), "host_sha256": digest, "seconds": round(time.monotonic() - start, 3)}
if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
    raise RuntimeError("Host changed during validation")
(ROOT / "artifacts/r1-host/unit-checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
sys.exit(0 if report["passed"] else 1)
