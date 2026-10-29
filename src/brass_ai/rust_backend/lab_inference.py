"""B01 native-input adapter to the existing frozen-model inference service.

Uses its scheduler, private reply pipe and shared slot unchanged. This is the
qualification transport, not B04's search coroutine/logit/teacher integration.
Search/actor requests have unknown critic descriptors; explicit descriptors are
checked separately as critic-only diagnostics by the B01 runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import time
import uuid

import numpy as np

from brass_ai.learning.contracts import digest, require
from brass_ai.learning.features import Encoded, check_encoded
from brass_ai.learning.seat_descriptors import unknown_numpy
from .lab_contract import admit


@dataclass(frozen=True)
class Scope:
    owner: int
    request: str
    epoch: int


@dataclass
class Ticket:
    token: str
    scope: Scope
    rows: list
    cancelled: object


class NativeInputClient:
    def __init__(self, endpoint, encoder, profile, scope):
        admit(profile)
        require(type(scope.owner) is int and scope.owner >= 0 and scope.request and scope.epoch == 0,
                'Invalid initial native input scope')
        self.endpoint, self.encoder, self.scope = endpoint, encoder, scope
        self.profile_id = admit(profile)
        self.retired = False

    def advance(self, scope):
        with self.endpoint.lock:
            require(not self.retired and self.endpoint.pending is None, 'Drain/retire old epoch before advancing')
            require(scope.owner == self.scope.owner and scope.request == self.scope.request
                    and scope.epoch == self.scope.epoch + 1, 'Invalid native root epoch')
            self.scope = scope

    def submit(self, rows, *, deadline, cancelled=None):
        """Rows: permitted view, full ordered choices, frozen model/reward context.

        The absolute caller deadline is never regenerated at an epoch change.
        Encoding and service wait both consume it. An endpoint owns one envelope
        until the reply is drained, including when that work has been canceled.
        """
        with self.endpoint.lock:
            require(not self.retired and self.endpoint.pending is None, 'Native input lane is busy/retired')
            require(rows and np.isfinite(deadline), 'Invalid native input batch/deadline')
            # Detach permitted wire containers before hashing/encoding. The
            # caller may advance its environment while this queue is in flight.
            rows = copy.deepcopy(rows)
            if hasattr(self, 'deadline'):
                require(deadline == self.deadline, 'Cannot replenish the original inference allowance')
            else:
                self.deadline = deadline
            semantics = {}
            for _, _, context in rows:
                require(context['stable_descriptors'] is None, 'Actor/search descriptors must be unknown')
                key = context['model_id']
                identity = digest({k: v for k, v in context.items() if k != 'remaining'})
                require(key not in semantics or semantics[key] == identity, 'Conflicting frozen model context')
                semantics[key] = identity
            if hasattr(self, 'frozen_contexts'):
                require(all(self.frozen_contexts.get(k) == v for k, v in semantics.items()),
                        'Model/reward/profile changed inside the original request')
            else:
                self.frozen_contexts = semantics
            frames, records = [], []
            token = uuid.uuid4().hex
            for index, (view, choices, context) in enumerate(rows):
                require(context['stable_descriptors'] is None, 'Actor/search descriptors must be unknown')
                binding = dict(scope=dict(vars(self.scope)), model=context['model_id'], reward=context['reward_id'],
                               profile=self.profile_id, context=digest(context), information=digest(view),
                               legal=digest(choices), revision=view['context']['revision'], observer=view['seat'])
                identity = digest(binding)
                request_id = token + ':' + str(index)
                stopped = cancelled is not None and cancelled.is_set()
                native = None
                if not stopped and time.monotonic() < deadline:
                    native = self.encoder.encode(view, choices, context)
                    require(native.revision == view['context']['revision'] and native.observer == view['seat'],
                            'Native encoding attribution differs')
                    require(tuple(native.choice_ids) == tuple(c['id'] for c in choices), 'Native legal order differs')
                    require(np.array_equal(native.descriptors, unknown_numpy()), 'Explicit critic metadata in actor input')
                encoded = None if native is None else Encoded(native.state, native.actions,
                    native.stable_to_relative, native.choice_ids, native.feature_version)
                if encoded is not None:
                    check_encoded(encoded)
                stopped = cancelled is not None and cancelled.is_set()
                frame = dict(request_id=request_id, model=context['model_id'], revision=view['context']['revision'],
                             seat=view['seat'], digest=identity, deadline=deadline, submitted=time.monotonic(),
                             cancelled=bool(stopped), encoded=encoded, remaining=list(context['remaining']))
                frames.append(frame)
                records.append(dict(binding=binding, request_id=request_id, digest=identity,
                                    choice_ids=[c['id'] for c in choices], deadline=deadline))
            slot = self.endpoint.slot()
            packed = slot.pack_frames(frames, token) if slot is not None else None
            ticket = Ticket(token, self.scope, records, cancelled)
            try:
                self.endpoint.requests.put(dict(lane=self.endpoint.lane, token=token,
                    frames=frames if packed is None else packed, shared=packed is not None),
                    timeout=self.endpoint.timeout)
                self.endpoint.pending = ticket
            except BaseException:
                self.retired = True
                raise
            return ticket

    def poll(self, ticket, *, wait=False):
        with self.endpoint.lock:
            require(not self.retired and self.endpoint.pending is ticket and ticket.scope == self.scope,
                    'Stale/foreign native inference ticket')
            if not self.endpoint.connection.poll(self.endpoint.timeout if wait else 0):
                if not wait:
                    return None
                self.retired = True
                raise TimeoutError('Native inference service timed out; lane must be retired')
            try:
                reply = self.endpoint.connection.recv()
                require(reply['token'] == ticket.token and 'error' not in reply,
                        'Inference service failure or misrouted reply: ' + str(reply.get('error', 'token')))
                if reply.get('shared'):
                    require(self.endpoint.slot() is not None, 'Missing native reply slot')
                    reply['responses'] = self.endpoint.slot().unpack_responses(reply['responses'], ticket.token)
                responses = reply['responses']
                require(len(responses) == len(ticket.rows), 'Incomplete inference batch')
                for row, result in zip(ticket.rows, responses):
                    b = row['binding']
                    expected = dict(protocol='four-seat-inference/1.0.0', request_id=row['request_id'],
                                    model_id=b['model'], revision=b['revision'], observer=b['observer'],
                                    input_digest=row['digest'])
                    require(all(result.get(k) == v for k, v in expected.items()), 'Native response identity differs')
                    require(result.get('status') in ('ok', 'cancelled', 'expired'), 'Invalid inference result status')
                    if ticket.cancelled is not None and ticket.cancelled.is_set():
                        result['status'] = 'cancelled'
                    elif time.monotonic() >= row['deadline']:
                        result['status'] = 'expired'
                    if result['status'] == 'ok':
                        require(result.get('choice_ids') == row['choice_ids'], 'Response legal order differs')
                        for name, size in [('probabilities', len(row['choice_ids'])), ('win_share', 4),
                                           ('placement', 4), ('vp', 4), ('utility', 4)]:
                            if name == 'utility' and name not in result:
                                continue
                            values = np.asarray(result.get(name), dtype=float)
                            require(values.shape == (size,) and np.isfinite(values).all(), 'Invalid native output ' + name)
                        p = np.asarray(result['probabilities'])
                        require((p >= 0).all() and abs(p.sum() - 1) < 1e-5, 'Invalid policy normalization')
                        if 'utility' in result:
                            require(result.get('reward_id') == b['reward'], 'Utility reward identity differs')
                    else:
                        # Do not expose stale numerical predictions to downstream code.
                        for key in ('choice_ids', 'probabilities', 'win_share', 'placement', 'vp', 'utility', 'reward_id'):
                            result.pop(key, None)
                    result['native_binding'] = b
                self.endpoint.pending = None
                return responses
            except BaseException:
                self.retired = True
                raise
