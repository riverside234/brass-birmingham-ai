"""Request/evaluate/submit boundary inspired by Daiyu's neural_search.py.

R2 evaluates legal choices, not search trees. R3 can submit leaf requests with
the SAME absolute whole-turn deadline. Expired/cancelled results are discarded;
an already running native kernel is not preempted by this synchronous worker.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import time
from threading import Event

import numpy as np
import torch

from .contracts import compatibility, digest, finite_vector, require
from .features import Encoder


def model_id(model):
    value = hashlib.sha256(digest({'compatibility': compatibility(), 'model': model.config.to_dict()}).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def collate(encoded, device='cpu'):
    require(bool(encoded), 'Cannot batch zero positions')
    count, maximum = len(encoded), max(len(x.choice_ids) for x in encoded)
    require(maximum > 0, 'No legal choices')
    actions = np.zeros((count, maximum, encoded[0].actions.shape[-1]), dtype=np.float32)
    mask = np.zeros((count, maximum), dtype=np.bool_)
    for i, row in enumerate(encoded):
        size = len(row.choice_ids)
        actions[i, :size] = row.actions
        mask[i, :size] = True
    return tuple(torch.from_numpy(a).to(device) for a in (
        np.stack([x.state for x in encoded]), actions, mask,
        np.stack([x.stable_to_relative for x in encoded])))


@dataclass(frozen=True)
class Request:
    request_id: str
    expected_model: str
    view: dict
    choices: list
    deadline: float
    style: tuple = (0.0,) * 4
    intent: tuple = (0.0,) * 8
    cancelled: Event | None = None
    input_digest: str = field(init=False)

    def __post_init__(self):
        # The coordinator may advance/reuse its buffers while a request waits.
        # Snapshot the permitted inputs and bind the response to this snapshot.
        object.__setattr__(self, 'view', copy.deepcopy(self.view))
        object.__setattr__(self, 'choices', copy.deepcopy(self.choices))
        object.__setattr__(self, 'style', tuple(self.style))
        object.__setattr__(self, 'intent', tuple(self.intent))
        object.__setattr__(self, 'input_digest', self.digest())

    def digest(self):
        return digest({'view': self.view, 'choices': self.choices, 'style': self.style,
                       'intent': self.intent, 'model': self.expected_model})

    def stopped(self):
        if self.cancelled is not None and self.cancelled.is_set():
            return 'cancelled'
        if time.monotonic() >= self.deadline:
            return 'expired'
        return None


class Evaluator:
    def __init__(self, model, device='cpu', batch_size=32, max_candidate_cells=8192):
        require(batch_size > 0 and max_candidate_cells > 0, 'Invalid inference limits')
        self.model = copy.deepcopy(model).to(device).eval().requires_grad_(False)
        self.model_id = model_id(self.model)
        self.device, self.batch_size = device, batch_size
        self.max_candidate_cells = max_candidate_cells
        self.encoder = Encoder()

    def predict(self, requests):
        require(len({r.request_id for r in requests}) == len(requests), 'Duplicate request ID')
        responses, active = {}, []
        for request in requests:
            require(request.expected_model == self.model_id, 'Request/model version mismatch')
            require(request.digest() == request.input_digest, 'Request inputs changed after submission')
            require(isinstance(request.request_id, str) and request.request_id, 'Missing request ID')
            require(np.isfinite(request.deadline), 'Deadline must be finite')
            stopped = request.stopped()
            if stopped:
                responses[request.request_id] = self._response(request, stopped)
            else:
                encoded = self.encoder.encode(request.view, request.choices, request.style, request.intent)
                require(len(encoded.choice_ids) <= self.max_candidate_cells, 'Single position exceeds candidate capacity')
                active.append((request, encoded))
        while active:
            group, maximum = [], 0
            while active and len(group) < self.batch_size:
                maximum = max(maximum, len(active[0][1].choice_ids))
                if group and (len(group) + 1) * maximum > self.max_candidate_cells:
                    break
                group.append(active.pop(0))
            live = []
            for request, encoded in group:
                if request.stopped():
                    responses[request.request_id] = self._response(request, request.stopped())
                else:
                    live.append((request, encoded))
            if not live:
                continue
            with torch.inference_mode():
                logits, win, placement, vp = self.model(*collate([e for _, e in live], self.device))
                probabilities = torch.softmax(logits, -1).cpu().numpy()
                predictions = [a.cpu().numpy() for a in (win, placement, vp)]
            for i, (request, encoded) in enumerate(live):
                status = request.stopped() or 'ok'
                response = self._response(request, status)
                if status == 'ok':
                    response.update(choice_ids=list(encoded.choice_ids),
                        probabilities=probabilities[i, :len(encoded.choice_ids)].tolist(),
                        win_share=predictions[0][i].tolist(), placement=predictions[1][i].tolist(),
                        vp=predictions[2][i].tolist())
                    require(all(np.isfinite(response[k]).all() for k in ('probabilities', 'win_share', 'placement', 'vp')),
                            'Non-finite network output')
                responses[request.request_id] = response
        return [responses[r.request_id] for r in requests]

    def _response(self, request, status):
        return {'protocol': 'four-seat-inference/1.0.0', 'request_id': request.request_id,
                'model_id': self.model_id, 'revision': request.view['context']['revision'],
                'observer': request.view['seat'], 'input_digest': request.input_digest, 'status': status}


def accept_response(request, response, current_revision, current_model):
    """Coordinator gate before using a prediction; identities never enter tensors."""
    require(not request.stopped(), 'Request no longer live')
    require(response['status'] == 'ok' and response['protocol'] == 'four-seat-inference/1.0.0', 'No usable prediction')
    require(response['request_id'] == request.request_id and
            response['model_id'] == current_model == request.expected_model and
            response['revision'] == current_revision == request.view['context']['revision'] and
            response['observer'] == request.view['seat'], 'Stale or misrouted inference result')
    require(response['choice_ids'] == [c['id'] for c in request.choices], 'Candidate ordering mismatch')
    require(response['input_digest'] == request.input_digest == request.digest(), 'Permitted input/style snapshot mismatch')
    finite_vector(response['probabilities'], len(request.choices), 0, 1)
    finite_vector(response['win_share'], 4, 0, 1)
    finite_vector(response['placement'], 4, 1, 4)
    finite_vector(response['vp'], 4)
    require(abs(sum(response['probabilities']) - 1) < 1e-5 and
            abs(sum(response['win_share']) - 1) < 1e-5, 'Prediction probability sum mismatch')
    return response
