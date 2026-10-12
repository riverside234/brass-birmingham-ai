"""Rules-client failures use isolated workers, never game binaries or saves."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from brass_ai.host import HostError, RulesHost
from brass_ai.runtime_contracts import validate


class ClientFailureTests(unittest.TestCase):
    def worker(self, code):
        launch = subprocess.Popen
        with patch("brass_ai.host.subprocess.Popen", side_effect=lambda _, **kw:
                   launch([sys.executable, "-u", "-c", code], **kw)):
            return RulesHost(verify=False)

    def test_stderr_backpressure_does_not_block_replies(self):
        host = self.worker("""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    sys.stderr.write('diagnostic' * 100000)
    sys.stderr.flush()
    print(json.dumps(dict(id=request['id'], ok=True, data=request['op'])), flush=True)
""")
        outcomes = []
        def call():
            try:
                outcomes.append(host.call("first"))
                outcomes.append(host.call("second"))
            except Exception as error:
                outcomes.append(error)
        thread = threading.Thread(target=call, daemon=True)
        try:
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "Worker blocked on its unread stderr pipe")
            self.assertEqual(outcomes, ["first", "second"])
        finally:
            if thread.is_alive():
                host.process.kill()
                host.process.wait(timeout=5)
                thread.join(timeout=5)
            host.close()

    def test_close_releases_stdin_after_worker_has_exited(self):
        host = self.worker("pass")
        host.process.wait(timeout=5)
        try:
            host.close()
            self.assertTrue(host.process.stdin.closed)
            self.assertTrue(host.process.stdout.closed)
            self.assertTrue(host.process.stderr.closed)
            host.close()
        finally:
            host.process.stdin.close()

    def test_broken_pipe_and_invalid_replies_are_host_errors(self):
        from types import SimpleNamespace
        for reply in ("not json\n", "[]\n", '{"id":"1"}\n',
                      '{"id":"1","ok":"false"}\n'):
            host = RulesHost.__new__(RulesHost)
            host.sequence = 0
            host.process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(reply))
            with self.subTest(reply=reply), self.assertRaises(HostError):
                host.call("context")
        class BrokenWriter(io.StringIO):
            def write(self, value):
                raise BrokenPipeError("worker exited")
        host.process.stdin = BrokenWriter()
        with self.assertRaises(HostError):
            host.call("context")

    def test_missing_or_duplicate_executable_pin_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            root = Path(directory)
            config = root / "configs/rules"
            config.mkdir(parents=True)
            dependency = dict(name="dependency.dll", sha256=hashlib.sha256(b"dep").hexdigest())
            executable = dict(name="BrassHost.exe", sha256=hashlib.sha256(b"exe").hexdigest())
            (root / dependency["name"]).write_bytes(b"dep")
            (root / executable["name"]).write_bytes(b"exe")
            (config / "runtime-dependencies-1.1.0.json").write_text(json.dumps(dict(files=[dependency])))
            for files in ([dependency], [dependency, executable, executable]):
                (root / "runtime.json").write_text(json.dumps(dict(rules="steam-brass-patch-1.1.0", files=files)))
                with self.subTest(files=len(files)), patch("brass_ai.host.ROOT", root), \
                        patch("brass_ai.host.subprocess.Popen") as launch:
                    with self.assertRaises(HostError):
                        RulesHost(root)
                    launch.assert_not_called()


class ResultBoundaryTests(unittest.TestCase):
    def result(self):
        return dict(objective="win-share/1.0.0", seats=[
            dict(seat=i, vp=40-i, income=0, cash=0, rank=i+1, utility=1.0 if i == 0 else 0.0)
            for i in range(4)])

    def test_nonfinite_or_inconsistent_official_results_are_rejected(self):
        for field, value in (("utility", float("nan")), ("rank", 2)):
            result = self.result()
            result["seats"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate("result", result)
        result = self.result()
        result["seats"][0]["utility"], result["seats"][1]["utility"] = 0., 1.
        with self.assertRaises(ValueError):
            validate("result", result)

    def test_competition_ties_and_income_cash_tiebreaks(self):
        result = self.result()
        result["seats"][1].update(vp=40, rank=1, utility=.5)
        result["seats"][0]["utility"] = .5
        validate("result", result)
        result["seats"][1].update(income=1, utility=1.)
        result["seats"][0].update(rank=2, utility=0.)
        validate("result", result)
        result["seats"][1].update(income=0, cash=1)
        validate("result", result)


if __name__ == "__main__":
    unittest.main()
