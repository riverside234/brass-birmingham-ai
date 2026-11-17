"""Versioned lossless history/static-field deltas for ordinary transport."""
from bisect import bisect_right
import json

from brass_ai.learning.contracts import require
from brass_ai.recovery.admitted_inputs import FrozenDict, FrozenList, _freeze
from brass_ai.recovery.canonical_inputs import attach, dumps


class Decoder:
    def __init__(self):
        self.view = None; self.revision = None; self.history = b'';self.choices=None

    def frame(self, frame):
        if not isinstance(frame, dict) or 'history_delta' not in frame:return
        patch = frame.pop('history_delta'); reused = frame.pop('view_reuse')
        if frame.pop('choices_reuse',False):
            require(self.choices is not None and 'choices' not in frame['decision'],'Foreign candidate reuse')
            frame['decision']['choices']=self.choices
        if 'choices' in frame['decision']:self.choices=_freeze(frame['decision']['choices'])
        view = frame['decision']['view']; previous = self.view
        base = patch['base']
        require(type(base) is int and base >= 0 and patch['previous_revision'] == self.revision and
                (previous is not None or not base) and base <= (len(previous['history']) if previous else 0),
                'Stale history delta')
        require(type(reused) is list and len(reused) == len(set(reused)), 'Duplicate reused field')
        for field in reused:
            require(field in ('players','slots','links','merchants','order','own_hand','pending') and
                    field not in view and previous is not None, 'Invalid reused view field')
            view[field] = previous[field]
        view['history'] = list(previous['history'][:base] if previous else ()) + [_freeze(x) for x in patch['tail']]
        require(len(view['history']) == patch['length'], 'History delta coverage changed')
        self.view = _freeze(view); self.revision = view['context']['revision']

    def reply(self, data):
        if not isinstance(data, dict):return data
        for frame in data.get('forced', ()):self.frame(frame)
        self.frame(data.get('current', data))
        return data

    def records(self, meta, raw):
        for prefix, history, suffix, _, evidence in self.split_records(meta, raw):
            yield (prefix if history is None else b''.join((prefix,history,suffix))),evidence

    def split_records(self, meta, raw):
        """(prefix, history, suffix, base, evidence) per row; history is None on the plain wire.

        On the delta wire the row's exact bytes are prefix + history + suffix, where
        history is the raw `view.history` array and `base` its common byte prefix with
        the previous row's history.
        """
        offset = 0
        for lengths in meta['lengths']:
            if meta['wire'] == 'ordinary-records/1':
                require(len(lengths) == 2, 'Record length shape differs')
                a,b = lengths
                require(type(a) is int and type(b) is int and a > 0 and b > 0 and offset+a+b <= len(raw),
                        'Record batch lengths differ')
                yield raw[offset:offset+a], None, None, None, raw[offset+a:offset+a+b]; offset += a+b
                continue
            require(meta['wire'] == 'ordinary-records-delta/1' and len(lengths) == 5, 'Unknown record delta')
            a,b,c,d,base = lengths
            require(all(type(x) is int and x >= 0 for x in lengths) and a+c > 0 and d > 0 and
                    base <= len(self.history) and offset+a+b+c+d <= len(raw), 'Record delta bounds differ')
            prefix = raw[offset:offset+a]; offset += a
            self.history = self.history[:base] + bytes(raw[offset:offset+b]); offset += b
            suffix = raw[offset:offset+c]; offset += c
            evidence = raw[offset:offset+d]; offset += d
            yield prefix, self.history, suffix, base, evidence
        require(offset == len(raw), 'Trailing record batch bytes')


_SPLIT = '\x00history-split\x00'
_ENCODED_SPLIT = json.dumps(_SPLIT).encode()
_DECODER = json.JSONDecoder()
_SCAN = _DECODER.scan_once
_ENDS = frozenset(',}]')


class _Fallback(Exception):
    pass


class SharedRows:
    """Delta-wire rows parsed with every unchanged subtree shared with the previous row.

    row(prefix, history, suffix, base) equals json.loads(prefix + history + suffix): the same
    keys in the same order, the same values and the same types. The row dict and its fields
    other than `view` and `choices` stay plain; `view` and `choices` are immutable
    FrozenDict/FrozenList trees. A history event, a top-level view field, a player and
    `choices` reuse the previous row's object when their JSON text is byte-identical to it.
    A JSON value's text delimits itself, so identical text followed by a delimiter is an
    identical value. Only changed text is parsed, frozen and encoded. Each view carries its
    exact canonical bytes and input digest (canonical_inputs.attach), so seals, audits and
    the replay encoding need not encode it again. Anything unexpected takes the plain
    json.loads path for that row and starts sharing afresh.
    """
    def __init__(self):
        self.ends = []; self.events = []; self.previous = {}; self.memo = {}
        self.parsed = self.reused = self.fallbacks = 0

    def _history(self, history, base):
        if history[:1] != b'[' or history[-1:] != b']':raise _Fallback
        keep = bisect_right(self.ends, base)
        start = self.ends[keep-1] if keep else 1
        tail = bytes(history[start:]); text = tail.decode('utf-8'); ascii = tail.isascii()
        events = self.events[:keep]; ends = self.ends[:keep]; position = 0
        while True:
            if position >= len(text):raise _Fallback
            if text[position] == ']':
                if position != len(text)-1:raise _Fallback
                break
            if events:
                if text[position] != ',':raise _Fallback
                position += 1
            value, end = _SCAN(text, position)
            events.append(_freeze(value)); position = end
            ends.append(start + (end if ascii else len(text[:end].encode('utf-8'))))
        self.events, self.ends = events, ends
        return FrozenList(events)

    def _same(self, text, i, path):
        saved = self.previous.get(path)
        if saved is None:return None
        raw, frozen = saved; end = i+len(raw)
        if end < len(text) and text[end] in _ENDS and text.startswith(raw, i):
            self.reused += 1
            return frozen, end
        return None

    def _value(self, text, i, path):
        same = self._same(text, i, path)
        if same is not None:return same
        value, end = _SCAN(text, i)
        frozen = _freeze(value); self.previous[path] = (text[i:end], frozen); self.parsed += 1
        return frozen, end

    def _players(self, text, i):
        same = self._same(text, i, ('view', 'players'))
        if same is not None:return same
        if text[i] != '[':raise _Fallback
        start = i; i += 1; items = []
        if text[i] == ']':i += 1
        else:
            while True:
                value, i = self._value(text, i, ('player', len(items))); items.append(value)
                c = text[i]; i += 1
                if c == ']':break
                if c != ',':raise _Fallback
        frozen = FrozenList(items); self.previous[('view', 'players')] = (text[start:i], frozen)
        return frozen, i

    def _object(self, text, i, level, history):
        if text[i] != '{':raise _Fallback
        i += 1; pairs = []; seen = set()
        if text[i] == '}':raise _Fallback  # rows and views are never empty
        while True:
            if text[i] != '"':raise _Fallback
            key, i = _SCAN(text, i)
            if key in seen or text[i] != ':':raise _Fallback
            seen.add(key); i += 1
            if level == 'row':
                if key == 'view':value, i = self._object(text, i, 'view', history)
                elif key == 'choices':value, i = self._value(text, i, ('row', key))
                else:value, i = _SCAN(text, i)
            elif key == 'history':
                value, i = _SCAN(text, i)
                if value != _SPLIT:raise _Fallback
                value = history
            elif key == 'players':value, i = self._players(text, i)
            else:value, i = self._value(text, i, ('view', key))
            pairs.append((key, value))
            c = text[i]; i += 1
            if c == '}':break
            if c != ',':raise _Fallback
        if level == 'view':
            if 'history' not in seen:raise _Fallback
            return FrozenDict(pairs), i
        if 'view' not in seen or 'choices' not in seen:raise _Fallback
        return dict(pairs), i

    def _encode(self, value, depth):
        kind = type(value)
        if kind is not FrozenDict and kind is not FrozenList:return dumps(value)
        saved = self.memo.get(id(value))
        if saved is not None and saved[0] is value:return saved[1]
        if not depth:raw = dumps(value)
        elif kind is FrozenDict:
            raw = b'{' + b','.join([dumps(k) + b':' + self._encode(dict.__getitem__(value, k), depth-1)
                                    for k in sorted(dict.keys(value))]) + b'}'
        else:raw = b'[' + b','.join([self._encode(x, depth-1) for x in list.__iter__(value)]) + b']'
        self.memo[id(value)] = (value, raw)
        return raw

    def row(self, prefix, history, suffix, base):
        try:
            events = self._history(history, base)
            text = b''.join((prefix, _ENCODED_SPLIT, suffix)).decode('utf-8')
            value, end = self._object(text, 0, 'row', events)
            if end != len(text):raise _Fallback
            # FrozenList choices: a forced single choice and all-scalar lists also freeze to it.
            if type(value['choices']) is not FrozenList:raise _Fallback
        except (_Fallback, ValueError, KeyError, TypeError, IndexError, StopIteration):
            self.ends = []; self.events = []; self.previous = {}; self.fallbacks += 1
            return json.loads(b''.join((prefix, history, suffix)))
        view, choices = value['view'], value['choices']
        attach(view, choices, self._encode(view, 2), self._encode(choices, 0))
        return value
