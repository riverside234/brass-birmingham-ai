"""Lossless disk transport for new heuristic IL games, not tensor packing.

Each game carries one history prefix table and deltas for four repeated public
view fields. Full observations are restored before ordinary replay admission.
Choices, labels, hands, behavior and provenance retain their original values.
The decoded rows own independent mutable trees, just like ordinary JSON loads.
"""
import json
import math

from brass_ai.learning.contracts import keys, require


FORMAT = 'four-seat-il-compact-replay/1.0.0'
LEGACY_FORMAT = 'four-seat-replay/1.0.0'
FIELDS = ('players', 'slots', 'links', 'merchants')
_SAME = object()
_MISSING = object()


def storage_format(provenance):
    """Explicitly pinned per-job opt-in; never compact ordinary PPO replay."""
    require(type(provenance) is dict, 'Invalid IL replay provenance')
    jobs = provenance.get('jobs')
    require(jobs is None or (type(jobs) is list and all(type(j) is dict and
            type(j.get('provenance')) is dict for j in jobs)), 'Invalid IL replay jobs')
    proofs = [j['provenance'] for j in jobs] if jobs is not None else [provenance]
    formats = {p.get('replay_format', LEGACY_FORMAT) for p in proofs}
    require(len(formats) == 1, 'Mixed IL replay storage formats')
    selected = next(iter(formats))
    require(selected in (FORMAT, LEGACY_FORMAT), 'Unsupported IL replay storage format')
    if selected == FORMAT:
        require(provenance.get('purpose') == 'training' and all(
            p.get('purpose') == 'training' and p.get('kind') == 'v3-demonstration'
            for p in proofs), 'Compact replay is restricted to IL demonstrations')
    return selected


def _copy(value):
    # JSON trees have no cycles or non-JSON objects. Avoid deepcopy's memo table
    # while preserving independent rows and exact scalar types/signed zeros.
    if type(value) is dict:
        return {k: _copy(v) for k, v in value.items()}
    if type(value) is list:
        return [_copy(v) for v in value]
    return value


def _delta(before, after):
    if type(before) is type(after):
        if type(after) is dict and list(before) == list(after):
            changes = {}
            for key, value in after.items():
                edit = _delta(before[key], value)
                if edit is not _SAME:
                    changes[key] = edit
            return [1, changes] if changes else _SAME
        if type(after) is list and len(before) == len(after):
            changes = []
            for index, (old, new) in enumerate(zip(before, after)):
                edit = _delta(old, new)
                if edit is not _SAME:
                    changes.append([index, edit])
            return [2, changes] if changes else _SAME
        if type(after) not in (dict, list) and before == after:
            if type(after) is not float or after != 0.0 or math.copysign(1, before) == math.copysign(1, after):
                return _SAME
    return [0, after]


def _apply(before, edit):
    if edit is None:
        require(before is not _MISSING, 'IL delta reuses a missing field')
        return before
    require(type(edit) is list and len(edit) == 2 and type(edit[0]) is int,
            'Malformed IL view delta')
    op, changes = edit
    if op == 0:
        return changes
    require(op in (1, 2), 'Unknown IL view delta operation')
    if op == 1:
        require(type(before) is dict and type(changes) is dict and changes.keys() <= before.keys(),
                'Invalid IL object delta')
        after = before.copy()
        for key, child in changes.items():
            after[key] = _apply(before[key], child)
        return after
    require(type(before) is list and type(changes) is list, 'Invalid IL list delta')
    after = before.copy()
    previous = -1
    for pair in changes:
        require(type(pair) is list and len(pair) == 2 and type(pair[0]) is int
                and previous < pair[0] < len(before), 'Invalid IL delta index/order')
        previous, child = pair
        after[previous] = _apply(before[previous], child)
    return after


def _heuristic_game(game):
    require(isinstance(game, dict), 'Invalid compact IL game')
    require(not any(k in game for k in ('branch', 'branches', 'branch_statistics')),
            'Compact IL replay cannot contain branches')
    rows = game.get('records')
    require(type(rows) is list and rows, 'Empty compact IL game')
    require(all(type(r) is dict and type(r.get('behavior')) is dict and
                r['behavior'].get('kind') == 'external' and 'teacher' in r and r['teacher'] is None
                for r in rows), 'Compact IL replay prohibits neural behavior and search targets')
    require(all(type(r.get('view')) is dict and 'history' in r['view'] for r in rows),
            'Missing compact IL view/history')
    return rows


def _json(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def pack(game):
    rows = _heuristic_game(game)
    require(all(type(r['view']['history']) is list for r in rows), 'Invalid IL history')
    history = max((r['view']['history'] for r in rows), key=len)
    # Comparing JSON also distinguishes int/bool/float and signed zero. Histories
    # which are not prefixes remain inline, without rewriting any observations.
    events = [_json(event) for event in history]
    previous = {}
    records = []
    for row in rows:
        original = row['view']
        view = dict(original)
        prefix = original['history']
        if all(_json(event) == events[i] for i, event in enumerate(prefix)):
            view['history'] = len(prefix)
        for field in FIELDS:
            if field not in original:
                previous.pop(field, None)
                continue
            edit = _delta(previous.get(field, _MISSING), original[field])
            view[field] = None if edit is _SAME else edit
            previous[field] = original[field]
        records.append(dict(row, view=view))
    return dict(history=history, game=dict(game, records=records))


def unpack(payload):
    keys(payload, ('history', 'game'))
    history, game = payload['history'], payload['game']
    require(type(history) is list, 'Invalid compact IL history table')
    rows = _heuristic_game(game)
    previous = {}
    records = []
    for row in rows:
        view = dict(row['view'])
        prefix = view['history']
        if type(prefix) is int:
            require(0 <= prefix <= len(history), 'Invalid compact IL history prefix')
            view['history'] = _copy(history[:prefix])
        else:
            require(type(prefix) is list, 'Invalid compact IL inline history')
            view['history'] = _copy(prefix)
        for field in FIELDS:
            if field not in view:
                previous.pop(field, None)
                continue
            value = _apply(previous.get(field, _MISSING), view[field])
            previous[field] = value
            view[field] = _copy(value)
        records.append(dict(row, view=view))
    return dict(game, records=records)
