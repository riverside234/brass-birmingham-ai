"""Opt-in Variant B: the existing Python search over native simulator objects.

Only diagnostic control crosses JSON. Clones/restore use owner-local native
handles; portable serialization is an explicit trusted checkpoint operation.
The Python RNG supplies recorded shuffle indices for exact reference sampling.
"""
import copy
import json
from pathlib import Path
import random
import threading
import weakref

import pyspiel

from brass_ai.learning.contracts import LAB_ROOT, require
from brass_ai.searchlab.spiel import TYPE, INFO, BrassSpielState, action_id
from .protocol import NativeWorker, WorkerFailure

FORMAT = "brass-native-spiel/1.0.0"


class NativeSpielGame(pyspiel.Game):
    """One diagnostic game owns one of a worker's two single-thread sessions."""
    def __init__(self, *, root=None, executable=None, worker=None, session=0, timeout=30):
        super().__init__(TYPE, INFO, {})
        if session not in (0, 1):
            raise ValueError("session must be 0 or 1")
        if worker is not None and executable is not None:
            raise ValueError("choose an existing worker or executable")
        self.root = Path(root or LAB_ROOT).resolve()
        self._owns_worker = worker is None
        self.worker = worker or NativeWorker(executable or self.root / "artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-worker.exe", timeout=timeout)
        self.session = session
        self.lock = threading.RLock()
        self._closed = False
        self._handles = set()
        self._prepared = None
        # Exact RNG consumption depends only on unseen pool size, validated by Rust.
        # One game per session prevents another adapter replacing its prepared root.
        owners = getattr(self.worker, '_simulation_owners', None)
        if owners is None:
            owners = self.worker._simulation_owners = weakref.WeakValueDictionary()
        if session in owners:
            if self._owns_worker:
                self.worker.close()
            raise ValueError("simulation session already owned by a game")
        owners[session] = self

    def _call(self, kind, **fields):
        if self._closed:
            raise WorkerFailure("native game is closed")
        return self.worker.simulation(self.session, kind, **fields)

    def _state(self, reply, actions=()):
        state = NativeSpielState(self, reply, actions)
        self._handles.add(state.handle)
        return state

    def new_initial_state(self):
        with self.lock:
            return self._state(self._call('new'))

    def import_state(self, state):
        """Trusted full-state fixtures only; chance imports have no invented recall."""
        with self.lock:
            return self._state(self._call('import', state=state))

    def from_observation(self, view, seed, recall=None):
        # Match random.Random(seed).shuffle exactly without Python world construction.
        with self.lock:
            recall = ([dict(revision=view['context']['revision'], hand=list(view['own_hand']))]
                      if recall is None else recall)
            key = (view, recall)
            if self._prepared is None or key != self._prepared[0]:
                prepared = self._call('prepare', view=view, recall=recall)
                self._prepared = (copy.deepcopy(key), prepared['draw_count'])
            rng = random.Random(seed)
            indices = [rng.randrange(bound) for bound in range(self._prepared[1]+1, 1, -1)]
            return self._state(self._call('sample', random=dict(kind='recorded', indices=indices)))

    def deserialize_state(self, data):
        value = json.loads(data)
        require(set(value) == {'format', 'checkpoint', 'actions'} and value['format'] == FORMAT,
                'Unknown native simulator checkpoint')
        require(isinstance(value['actions'], list) and len(value['actions']) <= 5000 and
                all(type(x) is int and 0 <= x < 2**31-1 for x in value['actions']), 'Invalid action history')
        with self.lock:
            return self._state(self._call('deserialize', checkpoint=value['checkpoint']), value['actions'])

    def _release(self, handle):
        with self.lock:
            if handle in self._handles:
                self._handles.remove(handle)
                if not self._closed:
                    self._call('release', handle=handle)

    def close(self):
        with self.lock:
            if self._closed:
                return
            try:
                for handle in list(self._handles):
                    self._release(handle)
            finally:
                self._closed = True
                self.worker._simulation_owners.pop(self.session, None)
                if self._owns_worker:
                    self.worker.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def _release(game, handle):
    try:
        game._release(handle)
    except (WorkerFailure, RuntimeError):
        # Transport retirement already drops every native state. Explicit close
        # surfaces failures; garbage collection must not raise unraisable errors.
        pass


class NativeSpielState(BrassSpielState):
    """Same semantic IDs/observation identity and public search API as the oracle."""
    def __init__(self, game, reply, actions=()):
        pyspiel.State.__init__(self, game)
        self.game, self.handle = game, reply['handle']
        self.actions = list(actions)
        self._finalizer = weakref.finalize(self, _release, game, self.handle)
        self._update(reply['snapshot'])

    def _update(self, snapshot):
        self._snapshot = snapshot
        self.context = snapshot['context']
        self.setup_stage = snapshot['setup_stage']
        self._setup_outcomes = snapshot['setup_outcomes']
        self.pending = snapshot['chance']
        self.legal = snapshot['choices']
        self.final_result = snapshot['result']
        self.final_returns = [s['utility'] for s in self.final_result['seats']] if self.final_result else None
        self.views = [i['observation']['view'] for i in snapshot['information']]
        self.recall = [i['recall'] for i in snapshot['information']]

    def _live(self):
        require(self._finalizer.alive and not self.game._closed, 'Native state is closed')

    def semantic_map(self):
        if self.setup_stage:
            result = {action_id(name): name for name, _ in self._setup_outcomes}
            require(len(result) == len(self._setup_outcomes), 'Semantic action hash collision')
            return result
        return super().semantic_map()

    def chance_outcomes(self):
        if self.setup_stage:
            return sorted((action_id(name), p) for name, p in self._setup_outcomes)
        return super().chance_outcomes()

    def _apply_action(self, action):
        self._live()
        names = self.semantic_map()
        require(action in names, 'Illegal semantic ID')
        with self.game.lock:
            reply = self.game._call('step', handle=self.handle,
                revision=self.context['revision'] if self.context else None, semantic=names[action])
            self._update(reply['snapshot'])
            self.actions.append(int(action))

    def clone(self):
        self._live()
        with self.game.lock:
            reply = self.game._call('clone', handle=self.handle)
            reply['snapshot'] = copy.deepcopy(self._snapshot)
            return self.game._state(reply, self.actions)

    def restore(self, source):
        self._live(); source._live()
        require(source.game is self.game, 'Restore requires the same game owner')
        with self.game.lock:
            self.game._call('restore', handle=self.handle, source=source.handle)
            self._update(copy.deepcopy(source._snapshot))
            self.actions = source.actions.copy()

    def trusted_checkpoint(self):
        self._live()
        with self.game.lock:
            return self.game._call('export', handle=self.handle)['checkpoint']

    def serialize(self):
        return json.dumps(dict(format=FORMAT, checkpoint=self.trusted_checkpoint(), actions=self.actions), sort_keys=True)

    def close(self):
        if self._finalizer.alive:
            self.game._release(self.handle)
            self._finalizer.detach()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
