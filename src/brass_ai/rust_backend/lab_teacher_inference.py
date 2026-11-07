"""Optional raw-logit output through the existing frozen GPU service/queues."""
import time
import uuid

import numpy as np
import torch
from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import require, digest
from brass_ai.learning.features import Encoded, check_encoded


def predict(model, encoded, device, candidate_limit, remaining, *, weights=None):
    """Same heads and all-candidate softmax; retain unnormalised float32 logits."""
    entities = None
    if weights is not None and any(weights):
        from brass_ai.learning.search_value import correction
        require(len(weights) == len(encoded), 'Search blend batch differs')
        states, actions, mask, mapping = collate(encoded, device)
        entities = model.encode_state(states)
        prepared = model.prepare_candidates(entities)
        logits = torch.cat([model.score_prepared(actions[:, start:start+candidate_limit], prepared)
                            for start in range(0, actions.shape[1], candidate_limit)], 1).masked_fill(~mask, -torch.inf)
        outputs = (logits, *model.heads_from_encoded(entities, mapping, remaining))
    elif max(len(e.choice_ids) for e in encoded) <= candidate_limit:
        outputs = model(*collate(encoded, device), remaining=remaining)
    else:
        require(len(encoded) == 1, 'Oversized teacher position needs its own batch')
        e = encoded[0]
        state = torch.from_numpy(e.state[None].copy()).to(device)
        entities = model.encode_state(state); prepared = model.prepare_candidates(entities)
        logits = torch.cat([model.score_prepared(torch.from_numpy(e.actions[None, start:start+candidate_limit].copy()).to(device), prepared)
                            for start in range(0, len(e.choice_ids), candidate_limit)], 1)
        mapping = torch.from_numpy(e.stable_to_relative[None].copy()).to(device)
        outputs = (logits, *model.heads_from_encoded(entities, mapping, remaining))
    require(len(outputs) == 5, 'Teacher GPU inference requires the current utility head')
    raw = outputs[0].float().clone()
    for i,e in enumerate(encoded): raw[i,len(e.choice_ids):] = 0.
    parts = [torch.softmax(outputs[0].float(), -1), *[v.float() for v in outputs[1:]]]
    if weights is not None:
        # Zero blend uses the original path and exact utility bytes; no extra encoder pass.
        search = outputs[4].float()
        if any(weights):
            delta = correction(model, entities, mapping, remaining)
            w = search.new_tensor(weights)[:, None]
            search = torch.where(w == 0, search, search + w * delta.float())
        parts.append(search)
    parts.append(raw)
    widths = [v.shape[1] for v in parts]; offsets = np.cumsum([0, *widths])
    packed = torch.cat(parts, 1).cpu().numpy()
    return [packed[:, offsets[i]:offsets[i+1]] for i in range(len(parts))]


def decode(meta, payload):
    binding = meta['binding']; n = meta['candidates']
    require(type(n) is int and 1 <= n <= 100000 and meta['action_len'] == 327*n,
            'Invalid native teacher tensor dimensions')
    state_n = meta['state_len']
    require(type(state_n) is int and 0 < state_n <= 65536 and
            len(payload) == 4*(state_n+327*n)+32, 'Invalid native teacher tensor bytes')
    values = np.frombuffer(payload, '<f4', count=state_n+327*n)
    require(np.isfinite(values).all(), 'Non-finite native teacher tensors')
    mapping = np.frombuffer(payload, '<i8', offset=4*(state_n+327*n), count=4)
    encoded = Encoded(values[:state_n], values[state_n:].reshape(n,327), mapping, tuple(binding['legal_ids']))
    check_encoded(encoded)
    require(np.array_equal(mapping, [(s-binding['seat'])%4 for s in range(4)]), 'Teacher seat mapping differs')
    require(binding['context']['stable_descriptors'] is None and binding['context']['style'] == [0.]*4
            and binding['context']['intent'] == [0.]*8, 'Teacher context is not neutral/unknown')
    return encoded


class Client:
    """One pending envelope per owner. Drain cancelled/late GPU work before reuse."""
    def __init__(self, endpoint, expected, cancel, search_value=None, seat_values=None):
        self.endpoint, self.expected, self.cancel = endpoint, expected, cancel
        self.search_value = search_value
        self.seat_values = seat_values
        self.sequence = 0; self.epoch = 0; self.retired = False
        self.wait_seconds = 0.; self.requests = 0

    def evaluate(self, meta, payload):
        b = meta['binding']; e = self.endpoint
        search_value = self.search_value
        if self.seat_values is not None:
            require(type(b.get('value_seat')) is int and 0 <= b['value_seat'] < 4, 'Missing value owner')
            search_value = self.seat_values[b['value_seat']]
        require(b.get('search_value') == search_value, 'Search value source/weight differs')
        require(not self.retired and b['job'] == self.expected and b['sequence'] == self.sequence+1
                and type(b['root_epoch']) is int and b['root_epoch'] >= self.epoch,
                'Stale/wrong model/job/epoch native request')
        require(b['context']['model_id'] == self.expected['model_id'] and
                b['context']['reward_id'] == self.expected['reward_id'], 'Teacher frozen context differs')
        self.sequence, self.epoch = b['sequence'], b['root_epoch']
        require(type(meta['remaining_ns']) is int and 0 <= meta['remaining_ns'] <= 10_000_000_000,
                'Invalid native inference deadline')
        deadline = time.monotonic() + meta['remaining_ns']/1e9
        encoded = decode(meta, payload)
        frame = dict(request_id=str(self.sequence), model=self.expected['model_id'], revision=b['revision'],
            seat=b['seat'], digest=digest(b), deadline=deadline, submitted=time.monotonic(),
            cancelled=self.cancel.is_set(), encoded=encoded, remaining=b['context']['remaining'], want_logits=True)
        if search_value is not None: frame['search_value'] = search_value
        token = uuid.uuid4().hex; began = time.monotonic()
        with e.lock:
            require(e.pending is None, 'Teacher GPU owner is still pending')
            slot = e.slot(); packed = slot.pack_frames([frame],token) if slot else None
            try:
                e.requests.put(dict(lane=e.lane,token=token,frames=packed or [frame],shared=packed is not None),timeout=1)
                e.pending = token
                # The search deadline suppresses numerical use, but the slot stays
                # owned until the real GPU response retires. Thirty seconds is a
                # stuck-device watchdog, never a second search allowance.
                while not e.connection.poll(.05):
                    require(time.monotonic()-began < 30, 'Teacher GPU watchdog; retire failed table and owner')
                reply = e.connection.recv()
                require(reply.get('token') == token and 'error' not in reply, 'Misrouted/failed teacher GPU envelope')
                results = slot.unpack_responses(reply['responses'],token) if reply.get('shared') else reply['responses']
                require(len(results) == 1, 'Teacher GPU reply count differs')
                r = results[0]
                require(all(r.get(k) == v for k,v in dict(protocol='four-seat-inference/1.0.0',request_id=frame['request_id'],
                    model_id=frame['model'],revision=frame['revision'],observer=frame['seat'],input_digest=frame['digest']).items()),
                    'Stale/wrong-model/wrong-epoch teacher GPU reply')
                require(r['status'] in ('ok','expired','cancelled'), 'Invalid teacher GPU status')
                result = dict(binding=b,status='cancelled' if self.cancel.is_set() else 'expired' if time.monotonic() >= deadline else r['status'])
                if result['status'] == 'ok':
                    if search_value is not None:
                        require(r.get('search_value') == search_value, 'Search value reply differs')
                        a = np.asarray(r.get('search_utility'), dtype=float)
                        require(a.shape == (4,) and np.isfinite(a).all(), 'Invalid search utility')
                        result.update(search_utility=a.tolist(), search_value=search_value)
                    require(r['choice_ids'] == b['legal_ids'] and r['reward_id'] == self.expected['reward_id'], 'Teacher GPU legal/reward differs')
                    for key,n in (('logits',len(b['legal_ids'])),('utility',4)):
                        a = np.asarray(r.get(key),dtype=float)
                        require(a.shape == (n,) and np.isfinite(a).all(), 'Invalid teacher GPU '+key)
                        result[key] = a.tolist()
                    result['reward_id'] = r['reward_id']
                e.pending = None
                return result
            except BaseException:
                self.retired = True
                raise
            finally:
                self.wait_seconds += time.monotonic()-began; self.requests += 1
