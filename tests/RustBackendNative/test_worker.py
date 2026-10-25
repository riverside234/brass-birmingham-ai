"""Real-process L03 tests, separate from the frozen L02 oracle suite."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from brass_ai.native import NativeError, NativeWorker, WorkerFailure, select_backend
from brass_ai.native.protocol import CAPABILITIES, DIAGNOSTIC_RULES, MAX_FRAME_BYTES, PROTOCOL, STATE_FORMAT

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(os.environ.get("BRASS_NATIVE_WORKER", str(ROOT / "artifacts/rust-native-target" /
              "x86_64-pc-windows-gnu/release/brass-ai-worker.exe")))


def frame(value):
    data = json.dumps(value).encode()
    return struct.pack("<I", len(data)) + data


def request(op, **fields):
    return dict(protocol=PROTOCOL, request_id="test", request=dict(op=op, **fields))


class WorkerTests(unittest.TestCase):
    def worker(self, **kwargs):
        worker = NativeWorker(WORKER, **kwargs)
        self.addCleanup(worker.close)
        return worker

    def test_handshake_build_and_independent_owners(self):
        worker = self.worker()
        hello = worker.hello
        self.assertEqual((hello["sessions"], hello["threads_per_session"]), (2, 1))
        self.assertFalse(hello["target_rules_qualified"])
        self.assertEqual(hello["build"]["target"], "x86_64-pc-windows-gnu")
        for key in ("source_sha256", "cargo_lock_sha256"):
            self.assertRegex(hello["build"][key], r"^[a-f0-9]{64}$")
        first = worker.new_game(0, 890)
        second = worker.new_game(1, 891)
        self.assertNotEqual(first["owner"], second["owner"])
        for seed in range(100, 110):
            fresh = worker.new_game(1, seed)
            self.assertEqual(second["owner"], fresh["owner"])
            self.assertEqual(first, worker.snapshot(0))
        for session in (0, 1):
            self.assertTrue(worker.clone_check(session)["clone_isolated"])
        self.assertEqual(first, worker.snapshot(0))

    def test_rejected_operations_preserve_live_state(self):
        worker = self.worker()
        initial = worker.new_game(0, 9)
        for operation, fields, code in [
            ("search", {"contract": "masked-ismcts/1"}, "unavailable_capability"),
            ("import_reference", {"contract": "rust-reference-state/1.0.0"}, "unavailable_capability"),
            ("diagnostic_snapshot", {"session": 2}, "invalid_request"),
            ("diagnostic_snapshot", {"session": 1}, "no_game"),
            ("diagnostic_new", {"session": 0, "players": 3, "seed": 12}, "invalid_request"),
        ]:
            with self.subTest(operation=operation, fields=fields):
                with self.assertRaises(NativeError) as caught:
                    worker.call(operation, **fields)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(initial, worker.snapshot(0))

    def test_clean_shutdown_and_closed_use(self):
        worker = self.worker()
        worker.close()
        self.assertEqual(worker.process.returncode, 0)
        self.assertFalse(worker._io.is_alive())
        with self.assertRaises(WorkerFailure):
            worker.snapshot(0)

    def raw(self, wire):
        result = subprocess.run([str(WORKER)], input=wire, capture_output=True, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        replies = []
        raw = result.stdout
        while raw:
            self.assertGreaterEqual(len(raw), 4)
            length, = struct.unpack("<I", raw[:4])
            self.assertLessEqual(length, MAX_FRAME_BYTES)
            self.assertGreaterEqual(len(raw), length + 4)
            replies.append(json.loads(raw[4:4+length]))
            raw = raw[4+length:]
        return result, replies

    def test_unknown_contract_fields_and_capabilities_fail_closed(self):
        hello = request("hello", sessions=2, threads_per_session=1, required_capabilities=[])
        wrong = dict(hello, protocol="brass-native-control/999")
        outer = dict(hello, hidden_state={})
        nested = request("hello", sessions=2, threads_per_session=1, required_capabilities=[], hidden_state={})
        unknown = request("hello", sessions=2, threads_per_session=1, required_capabilities=["search"])
        wrong_threads = request("hello", sessions=2, threads_per_session=2, required_capabilities=[])
        result, replies = self.raw(b"".join(frame(x) for x in [wrong, outer, nested, unknown, wrong_threads, hello]))
        self.assertEqual(result.returncode, 0)
        self.assertEqual([r["response"]["status"] for r in replies], ["error"]*5 + ["hello"])
        self.assertEqual(replies[0]["response"]["code"], "unsupported_contract")

    def test_bad_frame_lengths_truncation_and_json(self):
        for wire in [struct.pack("<I", 0), struct.pack("<I", MAX_FRAME_BYTES + 1), b"\x01\x00", struct.pack("<I", 9)+b"{"]:
            with self.subTest(wire=wire):
                result, replies = self.raw(wire)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(replies, [])
        result, replies = self.raw(struct.pack("<I", 1) + b"{" + frame(request("shutdown")))
        self.assertEqual(result.returncode, 0)
        self.assertEqual([r["response"]["status"] for r in replies], ["error", "error"])

    def test_dead_worker_is_retired(self):
        worker = self.worker(timeout=1)
        worker.process.kill()
        worker.process.wait(timeout=5)
        with self.assertRaises(WorkerFailure):
            worker.snapshot(0)
        self.assertFalse(worker._io.is_alive())
        with self.assertRaises(WorkerFailure):
            worker.snapshot(0)

    def test_handshake_timeout_and_bad_reply_are_bounded(self):
        started = time.monotonic()
        with self.assertRaises(WorkerFailure):
            NativeWorker(command=[sys.executable, "-c", "import time; time.sleep(30)"], timeout=.2)
        self.assertLess(time.monotonic() - started, 5)
        with self.assertRaises(WorkerFailure):
            NativeWorker(command=[sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff'*4); sys.stdout.flush()"], timeout=2)

    def fake_worker(self, request_id="1"):
        reply = frame(dict(protocol=PROTOCOL, request_id=request_id, response=dict(
            status="hello", sessions=2, threads_per_session=1, diagnostic_rules=DIAGNOSTIC_RULES,
            target_rules_qualified=False, max_frame_bytes=MAX_FRAME_BYTES,
            state_format=STATE_FORMAT,
            capabilities=sorted(CAPABILITIES), build={})))
        script = ("import sys, struct, time; "
                  "n = struct.unpack('<I', sys.stdin.buffer.read(4))[0]; sys.stdin.buffer.read(n); "
                  f"sys.stdout.buffer.write({reply!r}); sys.stdout.buffer.flush(); time.sleep(30)")
        return [sys.executable, "-c", script]

    def test_reply_identity_mismatch_retires_worker(self):
        with self.assertRaises(WorkerFailure):
            NativeWorker(command=self.fake_worker("wrong-id"), timeout=2)

    def test_pipe_write_timeout_is_also_bounded(self):
        worker = NativeWorker(command=self.fake_worker(), timeout=.5)
        self.addCleanup(worker.close)
        started = time.monotonic()
        with self.assertRaises(WorkerFailure):
            # Fill the pipe after hello; the fake peer no longer reads stdin.
            worker.call("search", contract="x" * (MAX_FRAME_BYTES // 2))
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNotNone(worker.process.poll())
        self.assertFalse(worker._io.is_alive())

    def test_reference_remains_default_and_rust_is_explicit(self):
        with patch("brass_ai.searchlab.spiel.BrassSpielGame") as game:
            self.assertIs(select_backend(root=ROOT), game.return_value)
            game.assert_called_once_with(root=ROOT, fast=True)
        with self.assertRaises(ValueError):
            select_backend(backend="rust")
        with select_backend(backend="rust-diagnostic", root=ROOT, executable=WORKER) as worker:
            self.assertEqual(worker.new_game(0, 42)["state"]["seed"], 42)

    def test_reference_host_still_initializes_and_has_chance_choices(self):
        game = select_backend(root=ROOT)
        try:
            state = game.new_initial_state()
            self.assertTrue(state.is_chance_node())
            self.assertTrue(state.chance_outcomes())
        finally:
            game.close()


if __name__ == "__main__":
    unittest.main()
