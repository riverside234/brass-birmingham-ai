"""Versioned lossless history/static-field deltas for ordinary transport."""
from brass_ai.learning.contracts import require
from brass_ai.recovery.admitted_inputs import _freeze


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
        offset = 0
        for lengths in meta['lengths']:
            if meta['wire'] == 'ordinary-records/1':
                require(len(lengths) == 2, 'Record length shape differs')
                a,b = lengths
                require(type(a) is int and type(b) is int and a > 0 and b > 0 and offset+a+b <= len(raw),
                        'Record batch lengths differ')
                row, evidence = raw[offset:offset+a], raw[offset+a:offset+a+b]; offset += a+b
            else:
                require(meta['wire'] == 'ordinary-records-delta/1' and len(lengths) == 5, 'Unknown record delta')
                a,b,c,d,base = lengths
                require(all(type(x) is int and x >= 0 for x in lengths) and a+c > 0 and d > 0 and
                        base <= len(self.history) and offset+a+b+c+d <= len(raw), 'Record delta bounds differ')
                prefix = raw[offset:offset+a]; offset += a
                self.history = self.history[:base] + bytes(raw[offset:offset+b]); offset += b
                row = b''.join((prefix,self.history,raw[offset:offset+c])); offset += c
                evidence = raw[offset:offset+d]; offset += d
            yield row,evidence
        require(offset == len(raw), 'Trailing record batch bytes')
