"""Canonical decision-input JSON without re-encoding what an earlier row already encoded.

An ordinary game's rows repeat most of the board and the whole public history, so
encoding every row's view again made publication quadratic in game length. These
functions return exactly the plain formulas' results:

    dumps(value)                == json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    input_digest(view, choices) == contracts.digest(dict(view=view, choices=choices))
    dumps_family(family)        == dumps(family)

Two things are reused, both only for immutable inputs. A frozen view that carries a
CanonicalInput (attached by the publisher's decoder, lab_delta.SharedRows, right after
encoding it exactly) supplies its bytes and digest. Otherwise a long history reuses the
encoding of each deeply frozen event (FrozenDict/FrozenList down to scalars); any other
event is encoded afresh, so a row edited after its seal still fails that seal exactly as
before. When a splice point cannot be proven unique, the plain formula runs.
"""
from dataclasses import dataclass
import hashlib
import json
import re

from brass_ai.learning.contracts import digest
from .admitted_inputs import FrozenDict, FrozenList

_SCALARS = frozenset((str, int, float, bool, type(None)))
_MARK = '\x00canonical-history\x00'
SMALL = 8  # shorter histories take the plain formula


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


_ENCODED_MARK = dumps(_MARK)
_INSERT = re.compile(rb'"\\u0000canonical-insert:(\d+)\\u0000"')


def _deep_frozen(value):
    kind = type(value)
    if kind in _SCALARS: return True
    if kind is FrozenDict: return all(type(k) is str and _deep_frozen(v) for k, v in dict.items(value))
    if kind is FrozenList: return all(_deep_frozen(v) for v in list.__iter__(value))
    return False


class Events:
    """Process-local identity cache of event encodings.

    Each entry keeps its event alive, so an id cannot be reused while it is cached.
    """
    def __init__(self, limit=1 << 17):
        self.saved = {}; self.limit = limit; self.hits = self.misses = 0

    def encode(self, event):
        if type(event) is not FrozenDict: return dumps(event)
        saved = self.saved.get(id(event))
        if saved is not None and saved[0] is event:
            self.hits += 1; return saved[1]
        raw = dumps(event); self.misses += 1
        if _deep_frozen(event):
            if len(self.saved) >= self.limit: self.saved.clear()
            self.saved[id(event)] = (event, raw)
        return raw


EVENTS = Events()


def history_bytes(history):
    return b'[' + b','.join([EVENTS.encode(e) for e in history]) + b']'


def _history(view):
    history = view.get('history') if isinstance(view, dict) else None
    return history if isinstance(history, list) and len(history) >= SMALL else None


@dataclass(frozen=True)
class CanonicalInput:
    """The exact canonical bytes and digest of one immutable (view, choices) pair.

    Process-local, held by the frozen view itself (a pickled FrozenDict drops it). It is
    not an actor Admission: the input boundary still validates such views in full, since
    admitted_inputs.admitted_digest ignores this type.
    """
    digest: str
    view_id: int
    choices_id: int
    view: bytes
    choices: bytes


def attach(view, choices, view_bytes, choices_bytes):
    """Record the canonical bytes of a deeply frozen pair that the caller has just encoded exactly."""
    if type(view) is not FrozenDict or type(choices) is not FrozenList:
        raise TypeError('Canonical receipts need frozen inputs')
    hashed = hashlib.sha256(b'{"choices":'); hashed.update(choices_bytes); hashed.update(b',"view":')
    hashed.update(view_bytes); hashed.update(b'}')
    saved = CanonicalInput(hashed.hexdigest(), id(view), id(choices), view_bytes, choices_bytes)
    object.__setattr__(view, '_admission', saved)
    return saved


def receipt(view, choices):
    if type(view) is not FrozenDict or type(choices) is not FrozenList: return None
    saved = getattr(view, '_admission', None)
    if type(saved) is CanonicalInput and saved.view_id == id(view) and saved.choices_id == id(choices):
        return saved
    return None


def input_digest(view, choices):
    saved = receipt(view, choices)
    if saved is not None: return saved.digest
    history = _history(view)
    if history is None: return digest(dict(view=view, choices=choices))
    shell = dict(view); shell['history'] = _MARK
    encoded = dumps(dict(view=shell, choices=choices))
    if encoded.count(_ENCODED_MARK) != 1: return digest(dict(view=view, choices=choices))
    head, _, tail = encoded.partition(_ENCODED_MARK)
    hashed = hashlib.sha256(head); hashed.update(history_bytes(history)); hashed.update(tail)
    return hashed.hexdigest()


def _mark(inserts, raw):
    inserts.append(raw)
    return '\x00canonical-insert:%d\x00' % (len(inserts)-1)


def _shell_game(game, inserts):
    records = []
    for row in game['records']:
        view = row.get('view') if isinstance(row, dict) else None
        saved = receipt(view, row.get('choices')) if view is not None else None
        if saved is not None:
            records.append(dict(row, view=_mark(inserts, saved.view), choices=_mark(inserts, saved.choices)))
            continue
        history = _history(view) if view is not None else None
        if history is None:
            records.append(row); continue
        records.append(dict(row, view=dict(view, history=_mark(inserts, history_bytes(history)))))
    return dict(game, records=records)


def dumps_family(family):
    """dumps(family) for a replay family {..., parent: {game: {records: [...]}}, child: ...}."""
    inserts = []; shell = dict(family)
    for role in ('parent', 'child'):
        part = shell.get(role)
        if isinstance(part, dict) and isinstance(part.get('game'), dict) and isinstance(part['game'].get('records'), list):
            shell[role] = dict(part, game=_shell_game(part['game'], inserts))
    if not inserts: return dumps(family)
    pieces = _INSERT.split(dumps(shell))
    order = [int(x) for x in pieces[1::2]]
    if sorted(order) != list(range(len(inserts))): return dumps(family)  # each mark exactly once
    out = [pieces[0]]
    for index, text in zip(order, pieces[2::2], strict=True):
        out.append(inserts[index]); out.append(text)
    return b''.join(out)
