"""Bounded diagnostic worker transport. No model, feature or search translation.

One Python I/O thread covers both pipe writes and reads with a caller deadline.
It does no game computation. A broken/late transport is killed and never reused.
"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import struct
import subprocess
import tempfile
import threading

PROTOCOL = "brass-native-control/0.1.0"
DIAGNOSTIC_RULES = "daiyu-upstream-diagnostic/0.1.0"
MAX_FRAME_BYTES = 1024 * 1024
STATE_FORMAT = "brass-native-state/1.0.0"
CAPABILITIES = {"inference", "features", "simulation", "diagnostic_new", "diagnostic_snapshot", "diagnostic_clone_check",
                "state_import", "state_export", "state_save", "state_restore", "state_clone_check",
                "rules_snapshot", "rules_apply", "rules_chance",
                "observation_snapshot", "belief_prepare", "belief_sample"}


class WorkerFailure(RuntimeError):
    """Transport/identity failure; this worker is no longer usable."""


class NativeError(RuntimeError):
    """Structured rejected request; a healthy connection can still be used."""
    def __init__(self, response):
        self.code = response["code"]
        super().__init__(response["message"])


def _read_exact(stream, count):
    parts = []
    while count:
        part = stream.read(count)
        if not part:
            raise WorkerFailure("worker closed its output")
        parts.append(part)
        count -= len(part)
    return b"".join(parts)


def _reject_constant(value):
    raise WorkerFailure(f"non-finite JSON constant: {value}")


class NativeWorker:
    def __init__(self, executable=None, *, timeout=10.0, command=None):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if command is not None and executable is not None:
            raise ValueError("choose executable or command")
        if command is None:
            executable = executable or (Path(__file__).resolve().parents[3] /
                "artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-worker.exe")
            command = [str(executable)]
        self.timeout = timeout
        self._closed = False
        self._sequence = 0
        self._lock = threading.RLock()
        self._jobs = queue.Queue(maxsize=1)
        self._replies = queue.Queue(maxsize=1)
        self._stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except BaseException:
            self._stderr.close()
            raise
        self._io = threading.Thread(target=self._exchange_loop, name="brass-native-io", daemon=True)
        self._io.start()
        try:
            self.hello = self.call("hello", sessions=2, threads_per_session=1,
                                   required_capabilities=sorted(CAPABILITIES))
            if (self.hello.get("status") != "hello" or self.hello.get("sessions") != 2
                or self.hello.get("threads_per_session") != 1
                or self.hello.get("diagnostic_rules") != DIAGNOSTIC_RULES
                or self.hello.get("target_rules_qualified") is not True
                or self.hello.get("state_format") != STATE_FORMAT
                or self.hello.get("max_frame_bytes") != MAX_FRAME_BYTES
                or set(self.hello.get("capabilities", [])) != CAPABILITIES
                or not isinstance(self.hello.get("build"), dict)):
                raise WorkerFailure("worker handshake does not match the native diagnostic/state contract")
        except BaseException:
            self._retire()
            raise

    def _exchange_loop(self):
        try:
            while True:
                payload = self._jobs.get()
                if payload is None:
                    return
                # FileIO may short-write even though the frames are bounded.
                wire = memoryview(struct.pack("<I", len(payload)) + payload)
                while wire:
                    written = self.process.stdin.write(wire)
                    if not written:
                        raise WorkerFailure("worker input closed")
                    wire = wire[written:]
                self.process.stdin.flush()
                length, = struct.unpack("<I", _read_exact(self.process.stdout, 4))
                if not 1 <= length <= MAX_FRAME_BYTES:
                    raise WorkerFailure("worker reply length exceeds protocol bounds")
                value = json.loads(_read_exact(self.process.stdout, length), parse_constant=_reject_constant)
                self._replies.put_nowait(value)
        except BaseException as error:
            try:
                self._replies.put_nowait(error)
            except queue.Full:
                pass

    def call(self, op, **fields):
        with self._lock:
            if self._closed:
                raise WorkerFailure("worker is retired")
            self._sequence += 1
            request_id = str(self._sequence)
            payload = json.dumps(dict(protocol=PROTOCOL, request_id=request_id,
                request=dict(op=op, **fields)), allow_nan=False, separators=(",", ":")).encode("utf-8")
            if len(payload) > MAX_FRAME_BYTES:
                raise ValueError("request exceeds protocol bounds")
            try:
                self._jobs.put_nowait(payload)
                value = self._replies.get(timeout=self.timeout)
                if isinstance(value, BaseException):
                    raise value
                if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL
                    or value.get("request_id") != request_id or not isinstance(value.get("response"), dict)):
                    raise WorkerFailure("reply identity or envelope mismatch")
                response = value["response"]
                if response.get("status") == "error":
                    if not isinstance(response.get("code"), str) or not isinstance(response.get("message"), str):
                        raise WorkerFailure("malformed error reply")
                    raise NativeError(response)
                expected = {"inference": "inference", "features": "features", "simulation": "simulation", "hello": "hello", "shutdown": "shutdown", "observation_snapshot": "observation",
                            "belief_prepare": "prepared", "belief_sample": "sampled"}.get(op,
                            "rules" if op.startswith("rules_") else "state" if op.startswith("state_") else "diagnostic")
                if response.get("status") != expected:
                    raise WorkerFailure("reply status does not match request")
                return response
            except NativeError:
                raise
            except BaseException as error:
                self._retire()
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise WorkerFailure(f"native transport failed: {type(error).__name__}: {error}") from error

    def new_game(self, session, seed):
        return self.call("diagnostic_new", session=session, players=4, seed=seed)

    def simulation(self, session, kind, **fields):
        return self.call("simulation", session=session, command=dict(kind=kind, **fields))["data"]

    def snapshot(self, session):
        return self.call("diagnostic_snapshot", session=session)

    def clone_check(self, session):
        return self.call("diagnostic_clone_check", session=session)

    def import_state(self, session, state):
        """Trusted fixtures only; this is not a policy or belief-sampling endpoint."""
        return self.call("state_import", session=session, state=state)

    def export_state(self, session):
        return self.call("state_export", session=session)

    def save_state(self, session):
        """Replace this owner's single bounded save slot."""
        return self.call("state_save", session=session)

    def restore_state(self, session):
        return self.call("state_restore", session=session)

    def state_clone_check(self, session):
        return self.call("state_clone_check", session=session)

    def rules_snapshot(self, session):
        return self.call("rules_snapshot", session=session)

    def apply_choice(self, session, revision, choice):
        return self.call("rules_apply", session=session, revision=revision, choice=choice)

    def resolve_chance(self, session, revision, card):
        return self.call("rules_chance", session=session, revision=revision, card=card)

    def observation(self, session, seat):
        """One seat's permitted information; no complete cards or chance picks."""
        return self.call("observation_snapshot", session=session, seat=seat)

    def prepare_belief(self, session, view, recall):
        """Detached permitted view only. Recall affects identity, not conditioning."""
        return self.call("belief_prepare", session=session, view=view, recall=recall)

    def sample_belief(self, session, random):
        """Trusted diagnostic result contains a sampled world's hidden state."""
        return self.call("belief_sample", session=session, random=random)

    def _retire(self):
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._io.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self._stderr.close()

    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    self.call("shutdown")
                    self.process.wait(timeout=self.timeout)
                except (NativeError, WorkerFailure, subprocess.TimeoutExpired):
                    pass
                finally:
                    self._retire()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
