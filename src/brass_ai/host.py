"""R1 trusted environment client. A policy gets only observe()/choices() results.

Workers own one game each; checkpoints must stay in the trusted coordinator.
The fingerprint manifest is checked before launching any proprietary assembly.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from pathlib import Path
import subprocess
import threading

PROTOCOL = "0.2.0"
ROOT = Path(__file__).resolve().parents[2]


class HostError(RuntimeError):
    pass


class RulesHost:
    def __init__(self, directory=None, *, verify=True, mono=None):
        directory = Path(directory or ROOT / "artifacts/r1-host/guarded").resolve()
        if verify:
            manifest = json.loads((directory / "runtime.json").read_text(encoding="utf-8-sig"))
            if manifest["rules"] != "steam-brass-patch-1.1.0":
                raise HostError("Unknown rules manifest")
            names = [entry["name"] for entry in manifest["files"]]
            if names.count("BrassHost.exe") != 1 or len(names) != len(set(names)):
                raise HostError("Runtime manifest must pin each dependency and BrassHost.exe exactly once")
            pins = json.loads((ROOT / "configs/rules/runtime-dependencies-1.1.0.json").read_text(encoding="utf-8-sig"))
            expected = {f["name"]: f["sha256"] for f in pins["files"]}
            found = {f["name"]: f["sha256"] for f in manifest["files"] if f["name"] != "BrassHost.exe"}
            if expected != found:
                raise HostError("Runtime manifest does not match the declared dependency pins")
            for entry in manifest["files"]:
                path = directory / entry["name"]
                if path.parent != directory or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise HostError(f"Runtime fingerprint mismatch: {entry['name']}")
        executable = directory / "BrassHost.exe"
        command = [str(executable)] if os.name == "nt" else [mono or "mono", str(executable)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.sequence = 0
        self.last_native_timing = None
        self.fast = False
        self.fast_checked = False
        # A worker can emit diagnostics while Python waits for its stdout reply.
        # Drain fixed-size blocks so both pipe backpressure and retained output
        # stay bounded even when diagnostics contain no newline.
        self.errors = deque(maxlen=16)
        def drain_errors():
            while chunk := self.process.stderr.buffer.read(2048):
                self.errors.append(chunk.decode("utf-8", errors="replace"))
        self._stderr_thread = threading.Thread(target=drain_errors, daemon=True)
        self._stderr_thread.start()

    def enable_fastpath(self):
        try:self.fast=self.call('capabilities').get('fast')=='training-host/1.0.0'
        except HostError as error:
            if str(error)!='unknown_operation':raise
        self.fast_checked = True
        return self.fast

    def call(self, op, **arguments):
        self.sequence += 1
        request = {"id": str(self.sequence), "protocol": PROTOCOL, "op": op, **arguments}
        payload = json.dumps(request, separators=(",", ":")) + "\n"
        self.last_native_timing = None
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (OSError, ValueError) as error:
            raise HostError("Worker transport failed") from error
        if not line:
            raise HostError(f"Worker exited: {self.process.poll()}")
        try:
            reply = json.loads(line)
        except ValueError as error:
            raise HostError("Worker returned invalid JSON") from error
        if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
            raise HostError("Worker returned an invalid response envelope")
        self.last_native_timing = reply.get('native')
        if reply.get("id") != request["id"]:
            raise HostError("Worker correlation mismatch")
        if not reply["ok"]:
            raise HostError(reply.get("message", "Worker error"))
        return reply.get("data")

    def reset(self, seed=0, setup=None):
        return self.call("reset", seed=seed, setup=setup)

    def observe(self, seat):
        return self.call("observe", seat=seat)

    def choices(self):
        return self.call("choices")

    def apply(self, context, choice):
        return self.call("apply", game_id=context["game_id"], revision=context["revision"], choice=choice["id"])

    def decision(self):
        return self.call('decision')

    def step_decision(self,context,choice,forced_limit=0):
        return self.call('step_decision',game_id=context['game_id'],revision=context['revision'],choice=choice['id'],forced_limit=forced_limit)

    def close(self):
        # Closing stdin is required even after an unexpected worker exit.
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._stderr_thread.join()
        self.process.stdout.close()
        self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
