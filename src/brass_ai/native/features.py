"""Diagnostic access to native Features-4; production inference borrows Rust buffers.

Pages transport every legal candidate without exceeding the control frame bound.
The lock spans all pages so another diagnostic request cannot replace the snapshot.
This module never computes model features or admits a model for gameplay.
"""
from dataclasses import dataclass
import numpy as np

from .protocol import WorkerFailure

FORMAT = 'brass-native-features/1.0.0'
VERSION = 'permitted-features/4.0.0'
SCHEMA = '59facdd72a386b6046abbb875daf3e1051b92e580ff55d0ab37250dd0c1dfb6a'
GROUP = '9dcb18c8bfc04ca9b020b43ddcfa20a27785d3baf08d19679c8751747a18d53d'
ENTITY = 'a4e1ca6a7588f8839275ff527918a3080f324db6d300215e7328410fb430a4e9'


@dataclass
class NativeFeatures:
    state: np.ndarray
    actions: np.ndarray
    legal_mask: np.ndarray
    stable_to_relative: np.ndarray
    remaining: np.ndarray
    descriptors: np.ndarray
    choice_ids: tuple
    encoding_id: int
    revision: int
    observer: int
    stats: dict
    feature_version: str = VERSION


class NativeFeatureEncoder:
    def __init__(self, worker, session=0):
        if type(session) is not int or session not in (0, 1):
            raise ValueError('session must be 0 or 1')
        self.worker, self.session = worker, session

    def encode(self, view, choices, context):
        return self._read(dict(kind='encode', view=view, choices=choices, context=context),
                          context, view['context']['revision'], view['seat'], tuple(c['id'] for c in choices))

    def from_state(self, state, context):
        if state.game.worker is not self.worker or state.game.session != self.session:
            raise ValueError('feature state belongs to another worker/session')
        with state.game.lock:
            return self._read(dict(kind='from_state', handle=state.handle,
                                   revision=state.context['revision'], context=context),
                              context, state.context['revision'], state.context['actor'],
                              tuple(c['id'] for c in state.legal))

    def _read(self, command, context, revision, observer, ids):
        try:
            return self._collect(command, context, revision, observer, ids)
        except (WorkerFailure, KeyError, TypeError, IndexError, ValueError) as error:
            self.worker._retire()
            raise WorkerFailure('invalid native feature reply; worker retired') from error

    def _collect(self, command, context, revision, observer, ids):
        def require(ok):
            if not ok:
                raise WorkerFailure('native feature identity/shape/order mismatch')
        def call(cmd):
            reply = self.worker.call('features', session=self.session, command=cmd)
            require(reply['session'] == self.session)
            return reply['data']
        with self.worker._lock:
            first = call(command)
            sequence = first['encoding_id']
            require(type(sequence) is int and sequence > 0 and sequence % 2 == self.session)
            pages, masks, returned, start = [], [], [], 0
            page = first
            while True:
                t = page['tensors']
                require(page['encoding_id'] == sequence and page['revision'] == revision and page['observer'] == observer)
                require((t['format'], t['feature_version'], t['schema_sha256'], t['group_layout'], t['entity_layout']) ==
                        (FORMAT, VERSION, SCHEMA, GROUP, ENTITY))
                require(t['state_dim'] == 3128 and t['action_dim'] == 327 and t['total_candidates'] == len(ids))
                end = min(start + 64, len(ids))
                require(t['candidate_offsets'] == [start, end] and tuple(t['choice_ids']) == ids[start:end])
                # Explicit descriptors are f32 on the wire; compare that field after conversion.
                require(all(t['context'][k] == v for k, v in context.items() if k != 'stable_descriptors'))
                actual_desc, expected_desc = t['context']['stable_descriptors'], context.get('stable_descriptors')
                require((actual_desc is None and expected_desc is None) or
                        np.array_equal(np.asarray(actual_desc, dtype=np.float32), np.asarray(expected_desc, dtype=np.float32)))
                for key in ('state', 'remaining', 'descriptors', 'stable_to_relative', 'stats'):
                    require(t[key] == first['tensors'][key])
                a = np.asarray(t['actions'], dtype=np.float32)
                require(a.shape == ((end-start)*327,) and np.isfinite(a).all())
                require(t['legal_mask'] == [True]*(end-start) and all(type(v) is bool for v in t['legal_mask']))
                pages.append(a.reshape(end-start, 327)); masks.extend(t['legal_mask']); returned.extend(t['choice_ids'])
                start = end
                if start == len(ids):
                    break
                page = call(dict(kind='page', encoding_id=sequence, start=start, limit=64))
            t = first['tensors']
            state = np.asarray(t['state'], dtype=np.float32)
            remaining = np.asarray(t['remaining'], dtype=np.float32)
            descriptors = np.asarray(t['descriptors'], dtype=np.float32)
            mapping = np.asarray(t['stable_to_relative'], dtype=np.int64)
            require(state.shape == (3128,) and remaining.shape == (4,) and descriptors.shape == (136,))
            require(np.isfinite(state).all() and np.isfinite(descriptors).all())
            require(np.array_equal(remaining, np.asarray(context['remaining'], dtype=np.float32)))
            require(mapping.tolist() == [(s-observer) % 4 for s in range(4)])
            return NativeFeatures(state, np.concatenate(pages), np.asarray(masks, dtype=bool), mapping,
                                  remaining, descriptors.reshape(4, 34), tuple(returned), sequence,
                                  revision, observer, t['stats'])
