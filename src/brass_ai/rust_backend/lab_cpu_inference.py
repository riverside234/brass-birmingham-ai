"""Worker-side CPU evaluators with the same interface as RemoteEvaluator.

Ordinary lanes ask their Rust collector for raw float32 outputs. Rating lanes
keep the C# host and evaluate here with ONNX Runtime's Python binding. Sampling,
RNG and every response check downstream stay exactly as on the GPU path.
"""
from collections import OrderedDict
import json
from pathlib import Path
import threading
import time

import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import require
from brass_ai.recovery.admitted_inputs import request as admitted_request

CONTRACT = 'brass-lab-cpu-inference/1.0.0'
PROTOCOL = 'four-seat-inference/1.0.0'
MODEL_CACHE = 8


def softmax32(logits):
    """float32 ``torch.softmax``: subtract the maximum, exponentiate, normalize."""
    values = np.asarray(logits, dtype=np.float32)
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all(), 'Invalid CPU policy logits')
    exponent = np.exp(values-values.max())
    return exponent/exponent.sum()


def _identity(request, model_id):
    return dict(protocol=PROTOCOL, request_id=request.request_id, model_id=model_id,
                revision=request.view['context']['revision'], observer=request.view['seat'],
                input_digest=request.input_digest)


def _telemetry():
    return dict(batches=0, positions=0, max_batch_positions=0, max_padded_candidate_cells=0, predict_seconds=0.,
                validation_seconds=0., encoding_seconds=0.)


def _admit(request, model_id):
    require(request.expected_model == model_id and request.digest() == request.input_digest,
            'Mutated/mismatched inference request')
    require(np.isfinite(request.deadline), 'Invalid inference deadline')


class NoEncoder:
    """Features stay inside the Rust collector; nothing is re-encoded here."""
    @staticmethod
    def encode(*_args, **_kwargs):
        return None


class NativeCpuEvaluator:
    request_factory = staticmethod(admitted_request)

    def __init__(self, lane, model_id, bundle):
        self.lane, self.model_id, self.bundle = lane, model_id, bundle
        self.encoder = NoEncoder()
        self.telemetry = _telemetry()
        self.reward_id = None

    def begin_game(self, lane):
        """Idempotent; the collector keeps a bounded cache of model sessions."""
        self.lane = lane
        reply = lane.call('load_model', model_id=self.model_id, bundle=self.bundle)
        if reply.get('loaded'):
            self.reward_id = reply['receipt']['reward_id']
        elif self.reward_id is None:
            self.reward_id = json.loads(Path(self.bundle['manifest']).read_text(encoding='utf-8'))['reward_id']

    def predict(self, requests):
        if not requests:
            return []
        require(len(requests) == 1, 'Ordinary CPU inference evaluates one decision at a time')
        request = requests[0]
        began = time.monotonic()
        _admit(request, self.model_id)
        base = _identity(request, self.model_id)
        stopped = request.stopped()
        if stopped:
            return [dict(base, status=stopped)]
        from .lab_transport import feature_context
        c = self.lane
        current = c.current['decision']
        require((current['view'] is request.view or current['view'] == request.view) and
                (current['choices'] is request.choices or current['choices'] == request.choices), 'Foreign native input')
        revision = request.view['context']['revision']
        context = feature_context(self.model_id, request.style, request.intent)
        c.context = context
        page = c.call('infer', token=c.token, revision=revision, context=context)
        count = len(request.choices)
        raw = page['_arrays']
        require(page['wire'] == 'ordinary-infer/1' and page['token'] == c.token and page['revision'] == revision
                and page['context'] == context and page['total_candidates'] == count
                and page['choice_ids'] == [x['id'] for x in request.choices]
                and page['bytes'] == len(raw) == 4*(count+16), 'CPU inference binding/shape differs')
        values = np.frombuffer(raw, dtype='<f4')
        logits, heads = values[:count], values[count:]
        require(np.isfinite(values).all(), 'Nonfinite CPU inference output')
        c.metrics['cpu_forward_ns'] += page['forward_ns']
        self.telemetry['batches'] += 1
        self.telemetry['positions'] += 1
        self.telemetry['max_batch_positions'] = 1
        self.telemetry['predict_seconds'] += time.monotonic()-began
        return [dict(base, status='ok', choice_ids=[x['id'] for x in request.choices],
                     probabilities=softmax32(logits).tolist(), win_share=heads[:4].tolist(),
                     placement=heads[4:8].tolist(), vp=heads[8:12].tolist(), utility=heads[12:16].tolist(),
                     reward_id=self.reward_id)]


class _Sessions:
    """Per-process LRU of ONNX Runtime sessions; each load re-checks its hashes."""
    def __init__(self):
        self.values, self.lock = OrderedDict(), threading.Lock()

    def get(self, identity, bundle):
        with self.lock:
            value = self.values.get(identity)
            if value is not None:
                self.values.move_to_end(identity)
                return value
            value = load_session(identity, bundle)
            self.values[identity] = value
            while len(self.values) > MODEL_CACHE:
                self.values.popitem(last=False)
            return value


def load_session(identity, bundle):
    import onnxruntime as ort
    from brass_ai.learning.onnx_tools import cpu_options
    from .lab_cpu_models import CONTRACT as BUNDLE, RUNTIME_VERSION
    require(ort.__version__ == RUNTIME_VERSION, 'CPU rating requires onnxruntime '+RUNTIME_VERSION)
    path = Path(bundle['manifest'])
    require(sha(path) == bundle['sha256'], 'CPU bundle manifest differs from its pin')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    require(manifest['contract'] == BUNDLE and manifest['model_id'] == identity, 'Foreign CPU bundle')
    graph = path.with_name(manifest['export_file'])
    require(sha(graph) == manifest['export_sha256'], 'CPU bundle graph differs from its manifest')
    session = ort.InferenceSession(str(graph), cpu_options(1), providers=['CPUExecutionProvider'])
    # The rating path never blends, so it fetches the five base outputs only.
    return dict(session=session, reward_id=manifest['reward_id'], max_cost=manifest['max_cost'],
                outputs=['policy_logits', 'win_share_4', 'placement_4', 'vp_4', 'remaining_utility_4'])


SESSIONS = _Sessions()


class OrtEvaluator:
    """Rating lanes: same encoder and responses, without the GPU service."""
    request_factory = staticmethod(admitted_request)

    def __init__(self, endpoint, identity, bundles):
        from brass_ai.recovery.inference import endpoint_encoder
        require(identity in bundles, 'Rated model has no CPU bundle')
        self.endpoint, self.model_id, self.bundle = endpoint, identity, bundles[identity]
        self.encoder = endpoint_encoder(endpoint)
        self.telemetry = _telemetry()

    def predict(self, requests):
        if not requests:
            return []
        began = time.monotonic()
        replies = [self._one(request) for request in requests]
        self.telemetry['batches'] += 1
        self.telemetry['positions'] += len(requests)
        self.telemetry['max_batch_positions'] = max(self.telemetry['max_batch_positions'], len(requests))
        self.telemetry['predict_seconds'] += time.monotonic()-began
        return replies

    def predict_mixed(self, groups):
        return [owner.predict(rows) for owner, rows in groups]

    def _one(self, request):
        from brass_ai.learning.encoding_budget import EncodingStopped
        from brass_ai.learning.seat_descriptors import unknown_numpy
        _admit(request, self.model_id)
        base = _identity(request, self.model_id)
        stopped = request.stopped()
        if stopped:
            return dict(base, status=stopped)
        phase = time.monotonic()
        try:
            e = self.encoder.encode(request.view, request.choices, request.style, request.intent,
                                    deadline=request.deadline, cancelled=request.cancelled)
        except EncodingStopped as error:
            return dict(base, status=error.reason)
        self.telemetry['encoding_seconds'] += time.monotonic()-phase
        entry = SESSIONS.get(self.model_id, self.bundle)
        cap = entry['max_cost']
        remaining = [cap]*4 if request.remaining is None else request.remaining
        budget = np.asarray(remaining, dtype=np.float32)[None]
        require(budget.shape == (1, 4) and np.isfinite(budget).all() and (budget >= 0).all() and
                (budget <= cap+1e-8).all(), 'Invalid inference utility budget')
        logits, win, placement, vp, utility = entry['session'].run(entry['outputs'], dict(
            states=e.state[None], actions=e.actions[None], legal_mask=np.ones((1, len(e.choice_ids)), dtype=bool),
            stable_to_relative=np.asarray(e.stable_to_relative, dtype=np.int64)[None], remaining=budget,
            descriptors=unknown_numpy()[None]))
        return dict(base, status='ok', choice_ids=list(e.choice_ids), probabilities=softmax32(logits[0]).tolist(),
                    win_share=win[0].tolist(), placement=placement[0].tolist(), vp=vp[0].tolist(),
                    utility=utility[0].tolist(), reward_id=entry['reward_id'])
