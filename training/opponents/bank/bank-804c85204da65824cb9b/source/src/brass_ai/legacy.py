"""Trusted client for the CPU legacy bank. No proprietary runtime is distributed.

Lab paths default to the user's project root. A local root must be explicit.
Policies consume observations, never this client or its trusted checkpoints.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

LAB_ROOT = Path('/data/not_backed_up/yxu209/ai-training/')


class LegacyWorker:
    def __init__(self, root=LAB_ROOT, *, mono=None, timeout=120, runtime_directory=None):
        self.root = Path(root).resolve()
        directory = (self.root / (runtime_directory or 'artifacts/legacy-ai/runtime')).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError('Runtime must remain below the project root')
        self.manifest = json.loads((directory / 'runtime.json').read_text(encoding='utf-8-sig'))
        pins = json.loads((self.root / 'configs/rules/runtime-dependencies-1.1.0.json').read_text())
        expected = {p['name']: p['sha256'] for p in pins['files']}
        actual = {p['name']: p['sha256'] for p in self.manifest['files'] if p['name'] != 'BrassLegacyAI.exe'}
        if expected != actual:
            raise ValueError('Legacy dependencies differ from the R1 rules contract')
        for p in self.manifest['files']:
            file = directory / p['name']
            if file.parent != directory or hashlib.sha256(file.read_bytes()).hexdigest() != p['sha256']:
                raise ValueError(f'Runtime fingerprint mismatch: {p["name"]}')
        for p in self.manifest['sources']:
            file = (self.root / p['path']).resolve()
            if not file.is_relative_to(self.root) or hashlib.sha256(file.read_bytes()).hexdigest() != p['sha256']:
                raise ValueError(f'Rebuild legacy runtime after source change: {p["path"]}')
        exe = str(directory / 'BrassLegacyAI.exe')
        command = [exe, str(self.root)] if os.name == 'nt' else [mono or 'mono', exe, str(self.root)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.timeout, self.sequence = timeout, 0
        self.responses = queue.Queue()
        self.errors = []
        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        def errors():
            for line in self.process.stderr:
                self.errors.append(line[:2000])
                del self.errors[:-20]
        threading.Thread(target=read, daemon=True).start()
        threading.Thread(target=errors, daemon=True).start()

    def call(self, op, **arguments):
        self.sequence += 1
        request = dict(protocol='legacy/0.1.0', id=str(self.sequence), op=op, **arguments)
        self.process.stdin.write(json.dumps(request, separators=(',', ':')) + '\n')
        self.process.stdin.flush()
        try:
            line = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.process.kill()
            raise TimeoutError('Legacy worker exceeded the request watchdog; no live action was applied')
        if line is None:
            raise RuntimeError('Legacy worker exited: ' + ''.join(self.errors))
        result = json.loads(line)
        if result.get('id') != request['id'] or not result.get('ok'):
            raise RuntimeError(str(result))
        return result['data']

    def profile(self, name='balanced-v1'):
        return json.loads((self.root / 'configs/legacy-ai' / (name + '.json')).read_text())

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
