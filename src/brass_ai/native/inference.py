"""Diagnostic control only. Native owners encode and evaluate; no Python NN RPC.

Submit both owners before polling to exercise independent computation. A ticket
remains busy through cancellation until a final reply is drained. Generations
are monotonic per owner. The caller's clock also rejects late delivery.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .protocol import NativeError, WorkerFailure

FORMAT = 'brass-native-evaluation/1.0.0'


@dataclass
class Ticket:
    ticket: int
    generation: int
    batch_id: str
    ids: list
    choices: list
    deadline: float


class NativeEvaluator:
    def __init__(self, worker, session, manifest):
        if type(session) is not int or session not in (0, 1):
            raise ValueError('session must be 0 or 1')
        self.worker, self.session = worker, session
        self.path = Path(manifest).resolve()
        data = self.path.read_bytes()
        self.manifest = json.loads(data)
        self.generation = 0
        self.active = None
        self.loaded = self._call('load', manifest=str(self.path), manifest_sha256=hashlib.sha256(data).hexdigest())
        m = self.loaded.get('model', {})
        self._require(self.loaded['state'] == 'loaded' and m['manifest_sha256'] == hashlib.sha256(data).hexdigest()
                      and m['model_id'] == self.manifest['model_id'] and m['export_sha256'] == self.manifest['export_sha256']
                      and m['intra_threads'] == m['inter_threads'] == 1 and m['sequential'] is True
                      and m['live_ready'] is False and m['cached_models'] == 1)

    def _require(self, ok):
        if not ok:
            self.worker._retire()
            raise WorkerFailure('native evaluation identity/shape mismatch; worker retired')

    def _call(self, kind, **fields):
        try:
            r = self.worker.call('inference', session=self.session, command=dict(kind=kind, **fields))
            self._require(r['session'] == self.session and r['data']['format'] == FORMAT)
            return r['data']
        except (KeyError, TypeError, ValueError) as error:
            self.worker._retire()
            raise WorkerFailure('malformed native evaluation response') from error

    def request(self, view, choices, context, *, request_id, state=None):
        """Construct a permitted observation or same-owner native-state request."""
        source = dict(kind='permitted', view=view, choices=choices)
        if state is not None:
            if state.game.worker is not self.worker or state.game.session != self.session:
                raise ValueError('foreign native state')
            source = dict(kind='state', handle=state.handle, revision=state.context['revision'])
        identity = dict(request_id=request_id, generation=self.generation+1, model_id=self.manifest['model_id'],
                        export_sha256=self.manifest['export_sha256'], reward_id=self.manifest['reward_id'],
                        profile_id=context['profile_id'], revision=view['context']['revision'], observer=view['seat'])
        return dict(id=identity, context=context, source=source)

    def submit(self, requests, *, budget_ms=4000, cap=1, pending_limit=None, flush_us=0, max_candidate_cells=32768):
        if self.active is not None:
            raise ValueError('drain the active ticket first')
        self.generation += 1
        batch_id = f'owner-{self.session}-batch-{self.generation}'
        for r in requests:
            r['id']['generation'] = self.generation
        # The local caller includes transport and accepts no delivery after its
        # deadline. Deployment adds its own coordinator clock in L11-L13.
        deadline = time.monotonic()+budget_ms/1000
        result = self._call('start', generation=self.generation, batch_id=batch_id, budget_ms=budget_ms,
                            policy=dict(cap=cap, pending_limit=pending_limit or max(cap,len(requests)),
                                        max_candidate_cells=max_candidate_cells, flush_us=flush_us), requests=requests)
        self._require(result['state'] == 'running' and result['generation'] == self.generation
                      and result['batch_id'] == batch_id and type(result['ticket']) is int
                      and result['ticket'] > 0 and result['ticket'] % 2 == self.session)
        self.active = Ticket(result['ticket'], self.generation, batch_id, [r['id'].copy() for r in requests],
                             [[c['id'] for c in r['source']['choices']] if r['source']['kind']=='permitted' else None for r in requests], deadline)
        return self.active

    def poll(self):
        if self.active is None:
            raise ValueError('no active evaluation')
        ticket = self.active
        try:
            result = self._call('poll', ticket=ticket.ticket)
        except NativeError:
            self.active = None
            raise
        try:
            self._require(result['ticket'] == ticket.ticket and result['generation'] == ticket.generation
                          and result['batch_id'] == ticket.batch_id)
            if result['state'] == 'running':
                self._require(result['evaluations'] == [])
                return None
            self.active = None
            self._require(result['state'] in ('complete','cancelled','expired'))
            if result['state'] != 'complete':
                self._require(result['evaluations'] == [])
                return result
            self._require(len(result['evaluations']) == len(ticket.ids))
            for expected, choices, row in zip(ticket.ids, ticket.choices, result['evaluations']):
                self._require(row['id'] == expected)
                self._require(choices is None or row['choice_ids'] == choices)
                n = len(row['choice_ids'])
                self._require(0<n<=8192 and len(set(row['choice_ids'])) == n)
                for field in ('logits','probabilities','log_probabilities','win_share','placement','vp','utility'):
                    values = np.asarray(row[field], dtype=np.float64)
                    self._require(values.shape == (n if field in ('logits','probabilities','log_probabilities') else 4,)
                                  and np.isfinite(values).all())
                probs = np.asarray(row['probabilities'])
                self._require((probs >= 0).all() and (probs <= 1).all() and abs(probs.sum()-1)<1e-10)
                self._require(np.allclose(np.exp(row['log_probabilities']), probs, atol=0, rtol=1e-12))
            if time.monotonic() >= ticket.deadline:
                result = dict(result, state='expired', evaluations=[])
            return result
        except (KeyError, TypeError, ValueError) as error:
            self.worker._retire()
            raise WorkerFailure('malformed evaluation tensors') from error

    def cancel(self):
        if self.active is None:
            raise ValueError('no active evaluation')
        result = self._call('cancel', ticket=self.active.ticket)
        self._require(result['state'] == 'cancelling' and result['ticket'] == self.active.ticket
                      and result['generation'] == self.active.generation and result['batch_id'] == self.active.batch_id)

    def wait(self):
        if self.active is None:
            raise ValueError('no active evaluation')
        # A synchronous runtime call is not preempted by cancellation. Bound the
        # diagnostic drain as well: a nonreturning runtime retires the process,
        # never releases live buffers for another request. Transport calls retain
        # their independent NativeWorker timeout.
        deadline = self.active.deadline
        retire_after = deadline + min(5., self.worker.timeout)
        while True:
            now = time.monotonic()
            if now >= retire_after:
                self.worker._retire()
                self.active = None
                raise WorkerFailure('native runtime did not retire before drain watchdog; worker killed')
            result = self.poll()
            if result is not None:
                return result
            time.sleep(.001)


class ReferenceEvaluator:
    """Parity-service adapter; never selected by the native production path."""
    def __init__(self, model_path):
        from brass_ai.personality.laptop.export import Graph
        self.graph = Graph(model_path, 'ort-cpu', 1)

    def evaluate(self, inputs):
        return self.graph.run(inputs)
