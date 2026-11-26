"""Sealed teacher-only shards; publish before commit, retire only after commit."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import digest, require
from brass_ai.step5.league import write_json
from .retention import owned, safe
from .teacher_labels import LabelRow, validate_improved_policy

CONTRACT = 'teacher-label-buffer/1.0.0'
OWNER = dict(contract=CONTRACT, purpose='policy-only search-v1 labels')


def slim(row, *, detach=False):
    """What training reads after admission: the label and its encoded inputs.

    The full view stays in the sealed shard; a cold load re-validates from it. A frozen
    view costs about 260 KB of memory per row, which an 8-iteration window cannot keep.
    ``detach`` copies the label and arrays, as publication always did.
    """
    if detach: return LabelRow(dict(teacher=copy.deepcopy(row.label)), copy.deepcopy(row.encoded))
    return LabelRow(dict(teacher=row.label), row.encoded)


def admitted(row, *, detach=False):
    """``slim(row)`` with its encoded inputs admitted once for this process (g6120).

    replay_binding.seal_encoded runs the checks collate's audit runs (layout, finite arrays, one-hot
    selectors) and keeps immutable bytes-backed copies of the arrays (detached from the caller's).
    A buffer row stays for eight iterations; each iteration's StagedRows then reuses this admission
    instead of hashing and re-checking every staged row (most of 3.7 s of teacher staging at g6118).
    """
    from .replay_binding import seal_encoded
    return LabelRow(dict(teacher=copy.deepcopy(row.label) if detach else row.label), seal_encoded(row.encoded))


class TeacherBuffer:
    def __init__(self, root, run, identity, *, receipts=(), retired=(), c_scale=.1, cotton_c_scale=None, iterations=4,
                 heldout_lag=None):
        self.root, self.run = owned(root, run)
        self.directory = safe(self.run, self.run / 'teacher-labels')
        require(iterations in (2, 4, 8), 'Invalid teacher buffer window')
        # g3270: validation labels train from `heldout_lag` iterations after their own, age counted from then.
        require(heldout_lag in (None, 1), 'Invalid teacher validation lag')
        self.identity = identity; self.c_scale = c_scale; self.iterations = iterations; self.heldout_lag = heldout_lag
        self.mixed_scales = None if cotton_c_scale is None else sorted({c_scale, cotton_c_scale})
        self.receipts = copy.deepcopy(list(receipts)); self.retired = copy.deepcopy(list(retired))
        self.pending = list(self.receipts); self._rows = {}
        require(len({r['path'] for r in self.receipts}) == len(self.receipts), 'Duplicate teacher shard receipt')
        for receipt in self.receipts: self.verify(receipt)

    def owner(self, create=False):
        marker = safe(self.run, self.directory / 'owner.json')
        if not marker.exists():
            require(create and (not self.directory.exists() or not any(self.directory.iterdir())),
                    'Teacher directory is not owned')
            self.directory.mkdir(parents=True, exist_ok=True); write_json(marker, OWNER)
        require(read(marker) == OWNER, 'Teacher directory owner differs')

    def paths(self, receipt):
        name = f"generation-{receipt['generation']:06d}-{receipt['split']}.jsonl.gz"
        require(receipt['split'] in ('train', 'heldout') and type(receipt['generation']) is int
                and receipt['generation'] >= 0 and receipt['path'] == 'teacher-labels/' + name,
                'Invalid teacher shard path')
        return safe(self.run, self.run / receipt['path']), safe(self.run, self.directory / (name + '.receipt.json'))

    def verify(self, receipt):
        """The receipt, its sidecar and the shard's presence. g6120: no content hash. A cached row was checked when
        it entered the cache, and a cold read decompresses the gzip shard (whose own CRC-32 catches a damaged
        file) and re-validates every row against the receipt. The receipt's sha256 stays the shard's identity."""
        self.owner(); path, sidecar = self.paths(receipt)
        require(receipt.get('contract') == CONTRACT and path.is_file() and sidecar.is_file() and read(sidecar) == receipt,
                'Committed teacher shard missing/changed')

    def publish(self, generation, split, rows, *, model_id, reward_id):
        self.owner(create=True)
        require(split in ('train', 'heldout'), 'Invalid teacher split')
        require(all(isinstance(r, LabelRow) and r.label['generation'] == generation
                    and r.label['model_id'] == model_id and r.label['reward_id'] == reward_id for r in rows),
                'Teacher shard lineage differs')
        require(len({r.identity for r in rows}) == len(rows), 'Duplicate teacher shard rows')
        content = ''.join(json.dumps(r.row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n' for r in rows)
        data = gzip.compress(content.encode('utf-8'), mtime=0, compresslevel=1)
        import hashlib
        receipt = dict(contract=CONTRACT, path=f'teacher-labels/generation-{generation:06d}-{split}.jsonl.gz',
            generation=generation, split=split, rows=len(rows), model_id=model_id, reward_id=reward_id,
            profile_digest=self.identity, sha256=hashlib.sha256(data).hexdigest())
        scales = {r.label['c2'] for r in rows}
        if self.mixed_scales is not None:
            require(scales.issubset(self.mixed_scales), 'Teacher shard uses an unconfigured search constant')
            receipt['c_scales'] = self.mixed_scales
        else:
            require(len(scales) <= 1, 'Teacher shard mixes search constants')
            if scales and scales != {.1}: receipt['c_scale'] = next(iter(scales))
        path, sidecar = self.paths(receipt)
        require(not path.exists() or sha(path) == receipt['sha256'], 'Uncommitted teacher shard differs; recover before recollection')
        require(not sidecar.exists() or read(sidecar) == receipt, 'Teacher receipt differs')
        if not path.exists():
            # Atomic publication; exclusive partial creation never overwrites an unknown file.
            partial = safe(self.run, path.with_name(path.name + '.partial'))
            with partial.open('xb') as f: f.write(data)
            partial.replace(path)
        if not sidecar.exists(): write_json(sidecar, receipt)
        # Detached copies of the label and arrays only; the caller's rows (with views) are released.
        self.verify(receipt); self._rows[receipt['sha256']] = tuple(admitted(r, detach=True) for r in rows)
        self.pending = [r for r in self.pending if r['path'] != receipt['path']] + [receipt]
        return receipt

    def _matches_scale(self, receipt, generation):
        if self.mixed_scales is not None:
            return receipt.get('c_scales') == self.mixed_scales
        return ('c_scales' not in receipt and
                (receipt['generation'] == generation or receipt.get('c_scale', .1) == self.c_scale))

    def lag(self, receipt):
        lag = getattr(self, 'heldout_lag', None)
        return lag if lag and receipt['split'] == 'heldout' else 0

    def age(self, receipt, generation):
        """A label's training age; validation labels count from the iteration they start training."""
        return generation - receipt['generation'] - self.lag(receipt)

    def trains(self, receipt, generation):
        return receipt['split'] == 'train' or (self.lag(receipt) > 0 and self.age(receipt, generation) >= 0)

    def window(self, generation):
        # A changed c_scale is a different search target, so older labels retire
        # with it. The current generation stays: a crash may have sealed its tables.
        keep = [r for r in self.pending if r['profile_digest'] == self.identity
                and 0 <= generation - r['generation'] < self.iterations + self.lag(r)
                and self._matches_scale(r, generation)]
        removed = [r for r in self.pending if r not in keep]
        for r in removed:
            if r not in self.retired: self.retired.append(r)
        self.pending = keep
        return keep

    def rows(self, generation, split='train'):
        """(age, row) pairs. 'train' includes validation labels from their training age 0 (g3270)."""
        from .encoding import PreparedEncoder
        result = []; encoder = PreparedEncoder()
        for receipt in sorted(self.window(generation), key=lambda r: r['generation'], reverse=True):
            if not (self.trains(receipt, generation) if split == 'train' else receipt['split'] == split): continue
            self.verify(receipt)
            if receipt['sha256'] not in self._rows:
                path, _ = self.paths(receipt)
                with gzip.open(path, 'rt', encoding='utf-8') as f: raw = [json.loads(line) for line in f]
                require(len(raw) == receipt['rows'], 'Teacher shard row count differs')
                values = []
                for row in raw:
                    label = row['teacher']
                    require(label['generation'] == receipt['generation'] and label['model_id'] == receipt['model_id']
                            and label['reward_id'] == receipt['reward_id'], 'Teacher shard row identity differs')
                    require(validate_improved_policy(label, row['choices'], view=row['view']) is not None, 'Unadmitted sealed teacher row')
                    values.append(admitted(LabelRow(row, encoder.encode(row['view'], row['choices'], [0.] * 4, [0.] * 8))))
                require(len({r.identity for r in values}) == len(values), 'Duplicate sealed teacher row')
                self._rows[receipt['sha256']] = tuple(values)
            age = self.age(receipt, generation) if split == 'train' else generation - receipt['generation']
            result.extend((age, row) for row in self._rows[receipt['sha256']])
        return result

    def payload(self):
        return dict(contract=CONTRACT, identity=self.identity, receipts=copy.deepcopy(self.pending),
                    retired=copy.deepcopy(self.retired))

    def committed(self, generation, *, state=None):
        # g3658: the state league.commit_cleanups just wrote, when it hands it over.
        state = read(safe(self.run, self.run / 'state.json')) if state is None else state
        require(state['iteration'] == generation and state.get('teacher_phase', {}).get('buffer') == self.payload(),
                'Teacher cleanup requires matching durable state commit')
        for receipt in self.retired:
            # g5466: only ages this process still holds rows for (the ones retiring now) are checked and released.
            # Every age retired before was checked when it retired, and nothing here deletes a file; re-hashing all
            # of them (975 shards since g4970, pinned by this state) cost most of the 6 s commit cleanup.
            if receipt['sha256'] not in self._rows: continue
            path, sidecar = self.paths(receipt)
            if not path.exists() and not sidecar.exists(): continue
            self.owner()
            # g6120: no content hash of a retiring shard either; nothing here deletes it.
            require(not sidecar.exists() or read(sidecar) == receipt, 'Refusing changed retired teacher receipt')
            # A prior full checkpoint may still reference this age. Retain its
            # immutable files until dependency-aware retention qualifies deletion.
            self._rows.pop(receipt['sha256'], None)
        self.receipts = copy.deepcopy(self.pending)


def sample_indices(ages, weights, count, *, seed, generation, step, stream=None):
    """Per-label age weights, independent draws per step; no global RNG use.

    ``stream`` names an independent draw (the g1491 long/short pools); None keeps
    the historical key.
    """
    if not ages or count <= 0: return []
    key = digest(dict(contract=CONTRACT, seed=seed, generation=generation, step=step,
                      **({} if stream is None else dict(stream=stream))))
    rng = np.random.default_rng(int(key[:16], 16))
    p = np.asarray([weights[a] for a in ages], dtype=np.float64); p /= p.sum()
    return rng.choice(len(ages), min(count, len(ages)), replace=False, p=p).tolist()
