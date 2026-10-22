"""Fair frozen-model queues for the single inference owner thread.

Only scheduling changes: observers, models, candidate order and reply envelopes
remain separate. New arrivals can join a model queue between GPU calls instead
of waiting for every unrelated model in the previous envelope group.
"""
from collections import defaultdict
import time


class ModelBatches:
    def __init__(self, batch_size, candidate_cells, padding_ratio=2.):
        self.batch_size, self.candidate_cells = batch_size, candidate_cells
        self.padding_ratio = padding_ratio
        self.models = defaultdict(list)
        self.count = self.peak = self.batches = self.padding_splits = self.capacity_splits = 0

    def __len__(self):
        return self.count

    def add(self, key, frame):
        self.models[frame['model']].append((key, frame))
        self.count += 1
        self.peak = max(self.peak, self.count)

    @staticmethod
    def priority(item):
        key, frame = item
        # Age bounds starvation even when a busy model never empties; urgent
        # search deadlines can still run ahead of ordinary training requests.
        return min(frame['deadline'], frame['submitted'] + .010), frame['submitted'], key

    def take(self, now=None):
        now=time.monotonic() if now is None else now
        all_rows=[(identity,item) for identity,rows in self.models.items() for item in rows]
        stopped=[pair for pair in all_rows if pair[1][1]['cancelled'] or pair[1][1]['deadline']<=now]
        if stopped:
            # A turn may expire before encoding. Such a frame intentionally
            # has encoded=None and must retire without touching its features.
            identity,anchor=min(stopped,key=lambda pair:(pair[1][1]['deadline'],self.priority(pair[1])))
            return self._pop(identity,[anchor])
        anchors=[(identity,min(rows,key=self.priority)) for identity,rows in self.models.items()]
        urgent=[pair for pair in all_rows if pair[1][1]['deadline']<=now+.005 or pair[1][1]['cancelled']]
        aged=[pair for pair in all_rows if pair[1][1]['submitted']<=now-.020]
        if urgent:
            identity,anchor=min(urgent,key=lambda pair:(pair[1][1]['deadline'],self.priority(pair[1])))
        elif aged:
            identity,anchor=min(aged,key=lambda pair:self.priority(pair[1]))
        else:
            # Prefer ready compatible work while all requests are young. A
            # thin historical queue must not repeatedly fragment a ready
            # learner batch. Age/deadline priority takes over after 20 ms.
            identity,anchor=min(anchors,key=lambda pair:(
                -min(self.batch_size,len(self.models[pair[0]])),self.priority(pair[1])))
        rows = self.models[identity]
        width = len(anchor[1]['encoded'].choice_ids)
        bucket = (width - 1).bit_length()
        others = sorted((row for row in rows if row is not anchor), key=lambda row: (
            abs((len(row[1]['encoded'].choice_ids) - 1).bit_length() - bucket), self.priority(row)))
        selected = [anchor]; maximum = total = width
        for item in others:
            size = len(item[1]['encoded'].choice_ids)
            next_max = max(maximum, size)
            if len(selected) >= self.batch_size:
                break
            if (len(selected) + 1) * next_max > self.candidate_cells:
                self.capacity_splits += 1
                continue
            # Keep a pathological long candidate list from padding many short
            # decisions. The default requires at least half the cells useful;
            # experimental padding ratios remain separately configured.
            if (len(selected) + 1) * next_max > self.padding_ratio * (total + size):
                self.padding_splits += 1
                continue
            selected.append(item); maximum = next_max; total += size
        return self._pop(identity,selected)

    def _pop(self,identity,selected):
        keys = {key for key, _ in selected}
        rest = [item for item in self.models[identity] if item[0] not in keys]
        if rest:self.models[identity] = rest
        else:del self.models[identity]
        self.count -= len(selected); self.batches += 1
        return selected

    def report(self):
        return dict(contract='streaming-model-batches/1.0.0', peak_pending_positions=self.peak,
                    batches=self.batches, padding_splits=self.padding_splits, capacity_splits=self.capacity_splits,
                    max_padding_ratio=self.padding_ratio,
                    scope='One frozen model per batch; age/deadline fairness; no extra flush wait between model batches')
