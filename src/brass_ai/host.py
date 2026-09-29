"""R1 trusted environment client. A policy gets only observe()/choices() results.

Workers own one game each; checkpoints must stay in the trusted coordinator.
The fingerprint manifest is checked before launching any proprietary assembly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

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

    def call(self, op, **arguments):
        self.sequence += 1
        request = {"id": str(self.sequence), "protocol": PROTOCOL, "op": op, **arguments}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise HostError(f"Worker exited: {self.process.poll()}")
        reply = json.loads(line)
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

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
