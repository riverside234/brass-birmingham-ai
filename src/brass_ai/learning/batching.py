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


def predict_encoded(model, encoded, device, candidate_limit, remaining=None):
    """Score ALL legal candidates, chunking a large position before softmax.

    The oversized path encodes its state once and never changes legality/order.
    Smaller positions retain the normal padded, masked CPU/GPU batch path.
    """
    if max(len(e.choice_ids) for e in encoded)<=candidate_limit:
        outputs=(model(*collate(encoded,device),remaining=remaining) if getattr(model.config,'utility',None) is not None else model(*collate(encoded,device)))
        return [torch.softmax(outputs[0].float(),-1).cpu().numpy()]+[a.cpu().numpy() for a in outputs[1:]]
    require(len(encoded)==1,'Oversized position must have its own candidate batch')
    e=encoded[0];state=torch.from_numpy(e.state[None].copy()).to(device)
    context=model.state_encoder(state);parts=[]
    for start in range(0,len(e.choice_ids),candidate_limit):
        actions=torch.from_numpy(e.actions[None,start:start+candidate_limit].copy()).to(device)
        candidates=model.action_encoder(actions)
        parts.append(model.policy_scorer(torch.cat((context[:,None].expand(-1,actions.shape[1],-1),candidates),-1)).squeeze(-1))
    logits=torch.cat(parts,1);outcomes=model.outcomes(context)
    mapping=torch.from_numpy(e.stable_to_relative[None].copy()).to(device)
    heads=(torch.softmax(outcomes[:,:4],-1).gather(1,mapping),
           (1+3*torch.sigmoid(outcomes[:,4:8])).gather(1,mapping),
           (100*outcomes[:,8:12]).gather(1,mapping))
    if getattr(model.config,'utility',None) is not None:heads=heads+(model.predict_utility(context,mapping,heads,remaining),)
    return [torch.softmax(logits.float(),-1).cpu().numpy()]+[a.cpu().numpy() for a in heads]


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
    remaining: tuple | None = None
    input_digest: str = field(init=False)

    def __post_init__(self):
        # The coordinator may advance/reuse its buffers while a request waits.
        # Snapshot the permitted inputs and bind the response to this snapshot.
        object.__setattr__(self, 'view', copy.deepcopy(self.view))
        object.__setattr__(self, 'choices', copy.deepcopy(self.choices))
        object.__setattr__(self, 'style', tuple(self.style))
        object.__setattr__(self, 'intent', tuple(self.intent))
        if self.remaining is not None:
            finite_vector(self.remaining,4,0,.1)
            object.__setattr__(self,'remaining',tuple(self.remaining))
        object.__setattr__(self, 'input_digest', self.digest())

    def digest(self):
        value={'view': self.view, 'choices': self.choices, 'style': self.style,
               'intent': self.intent, 'model': self.expected_model}
        if self.remaining is not None:value['remaining']=self.remaining
        return digest(value)

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
        from brass_ai.recovery.encoding import PreparedEncoder
        self.encoder = PreparedEncoder()
        self.telemetry = dict(batches=0, positions=0, max_batch_positions=0,
                              max_padded_candidate_cells=0, predict_seconds=0.0)
        self._executor=None
        self._pending=None

    def submit(self,requests):
        """Start one detached batch without blocking the planner's caller.

        A cancelled Request suppresses its result, but the future remains live
        until the actual tensor/device work retires. Only one batch is in flight
        per evaluator; search path/position limits are independently bounded.
        """
        from concurrent.futures import ThreadPoolExecutor
        require(self._pending is None or self._pending.done(),'Evaluator batch is still running')
        if self._executor is None:self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='brass-cpu-inference')
        self._pending=self._executor.submit(self._predict,list(requests))
        return self._pending

    def poll(self,ticket,*,wait=False):
        require(ticket is self._pending,'Unknown evaluator ticket')
        return ticket.result() if wait or ticket.done() else None

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor=None
        self._pending=None

    def predict(self, requests):
        return self.poll(self.submit(requests),wait=True)

    def _predict(self, requests):
        began = time.monotonic()
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
                remaining=None
                if getattr(self.model.config,'utility',None) is not None:
                    remaining=torch.tensor([r.remaining if r.remaining is not None else [self.model.config.utility['insolvency']['max_cost']]*4 for r,_ in live],device=self.device)
                probabilities,*predictions=predict_encoded(self.model,[e for _,e in live],self.device,self.max_candidate_cells,remaining)
            self.telemetry['batches'] += 1
            self.telemetry['positions'] += len(live)
            self.telemetry['max_batch_positions'] = max(self.telemetry['max_batch_positions'], len(live))
            cells = min(self.max_candidate_cells,len(live) * max(len(e.choice_ids) for _, e in live))
            self.telemetry['max_padded_candidate_cells'] = max(self.telemetry['max_padded_candidate_cells'], cells)
            for i, (request, encoded) in enumerate(live):
                status = request.stopped() or 'ok'
                response = self._response(request, status)
                if status == 'ok':
                    response.update(choice_ids=list(encoded.choice_ids),
                        probabilities=probabilities[i, :len(encoded.choice_ids)].tolist(),
                        win_share=predictions[0][i].tolist(), placement=predictions[1][i].tolist(),
                        vp=predictions[2][i].tolist())
                    if len(predictions)>3:
                        response.update(utility=predictions[3][i].tolist(),reward_id=digest(self.model.config.utility))
                    require(all(np.isfinite(response[k]).all() for k in ('probabilities', 'win_share', 'placement', 'vp')),
                            'Non-finite network output')
                responses[request.request_id] = response
        self.telemetry['predict_seconds'] += time.monotonic() - began
        return [responses[r.request_id] for r in requests]

    def _response(self, request, status):
        return {'protocol': 'four-seat-inference/1.0.0', 'request_id': request.request_id,
                'model_id': self.model_id, 'revision': request.view['context']['revision'],
                'observer': request.view['seat'], 'input_digest': request.input_digest, 'status': status}


class RequestStopped(ValueError):
    """The request expired/cancelled at the final response acceptance gate."""

    def __init__(self, reason):
        super().__init__('Request no longer live')
        self.reason = reason


def accept_response(request, response, current_revision, current_model):
    """Coordinator gate before using a prediction; identities never enter tensors."""
    stopped = request.stopped()
    if stopped:
        raise RequestStopped(stopped)
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
    if 'utility' in response:finite_vector(response['utility'],4)
    require(abs(sum(response['probabilities']) - 1) < 1e-5 and
            abs(sum(response['win_share']) - 1) < 1e-5, 'Prediction probability sum mismatch')
    return response
