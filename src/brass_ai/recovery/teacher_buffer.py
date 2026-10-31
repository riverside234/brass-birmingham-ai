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


class TeacherBuffer:
    def __init__(self, root, run, identity, *, receipts=(), retired=()):
        self.root, self.run = owned(root, run)
        self.directory = safe(self.run, self.run / 'teacher-labels')
        self.identity = identity
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
        self.owner(); path, sidecar = self.paths(receipt)
        require(receipt.get('contract') in (CONTRACT,'teacher-label-buffer/2.0.0') and path.is_file() and sidecar.is_file()
                and sha(path) == receipt['sha256'] and read(sidecar) == receipt,
                'Committed teacher shard missing/changed')

    def publish(self, generation, split, rows, *, model_id, reward_id, evaluators=None):
        self.owner(create=True)
        require(split in ('train', 'heldout'), 'Invalid teacher split')
        require(all(isinstance(r, LabelRow) and r.label['generation'] == generation
                    and r.label['model_id'] == model_id and r.label['reward_id'] == reward_id for r in rows),
                'Teacher shard lineage differs')
        require(len({r.identity for r in rows}) == len(rows), 'Duplicate teacher shard rows')
        content = ''.join(json.dumps(r.row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n' for r in rows)
        data = gzip.compress(content.encode('utf-8'), mtime=0)
        import hashlib
        from .teacher_diversity import LABEL
        version = 'teacher-label-buffer/2.0.0' if rows and rows[0].label['source'] == LABEL else CONTRACT
        require(all((r.label['source'] == LABEL) == (version != CONTRACT) for r in rows), 'Mixed teacher buffer schemas')
        receipt = dict(contract=version, path=f'teacher-labels/generation-{generation:06d}-{split}.jsonl.gz',
            generation=generation, split=split, rows=len(rows), model_id=model_id, reward_id=reward_id,
            profile_digest=self.identity, sha256=hashlib.sha256(data).hexdigest())
        if version != CONTRACT:
            from .teacher_diversity import validate_bundle
            require(evaluators is not None,'Missing sealed teacher evaluator inventory')
            for bundle in evaluators.values():validate_bundle(bundle)
            require(all(r.label['bundle_id'] in evaluators for r in rows),'Missing teacher row evaluator')
            receipt['evaluators']=copy.deepcopy(evaluators)
        path, sidecar = self.paths(receipt)
        require(not path.exists() or sha(path) == receipt['sha256'], 'Uncommitted teacher shard differs; recover before recollection')
        require(not sidecar.exists() or read(sidecar) == receipt, 'Teacher receipt differs')
        if not path.exists():
            # Atomic publication; exclusive partial creation never overwrites an unknown file.
            partial = safe(self.run, path.with_name(path.name + '.partial'))
            with partial.open('xb') as f: f.write(data)
            partial.replace(path)
        if not sidecar.exists(): write_json(sidecar, receipt)
        self.verify(receipt); self._rows[receipt['sha256']] = tuple(copy.deepcopy(rows))
        self.pending = [r for r in self.pending if r['path'] != receipt['path']] + [receipt]
        return receipt

    def window(self, generation):
        keep = [r for r in self.pending if r['profile_digest'] == self.identity and 0 <= generation - r['generation'] < 4]
        removed = [r for r in self.pending if r not in keep]
        for r in removed:
            if r not in self.retired: self.retired.append(r)
        self.pending = keep
        return keep

    def rows(self, generation, split='train'):
        from .encoding import PreparedEncoder
        result = []; encoder = PreparedEncoder()
        for receipt in sorted(self.window(generation), key=lambda r: r['generation'], reverse=True):
            if receipt['split'] != split: continue
            self.verify(receipt)
            if receipt['sha256'] not in self._rows:
                path, _ = self.paths(receipt)
                with gzip.open(path, 'rt', encoding='utf-8') as f: raw = [json.loads(line) for line in f]
                require(len(raw) == receipt['rows'], 'Teacher shard row count differs')
                values = []
                from .teacher_diversity import validate_bundle
                bundles={k:validate_bundle(v) for k,v in receipt.get('evaluators',{}).items()}
                for row in raw:
                    label = row['teacher']
                    require(label['generation'] == receipt['generation'] and label['model_id'] == receipt['model_id']
                            and label['reward_id'] == receipt['reward_id'], 'Teacher shard row identity differs')
                    require(validate_improved_policy(label, row['choices']) is not None, 'Unadmitted sealed teacher row')
                    from .teacher_diversity import LABEL, check_context
                    if label['source'] == LABEL:
                        assignment = check_context(label['seat_context'])
                        require(row['style'] == assignment['style'] and row['intent'] == assignment['intent']
                                and label['strategy_id'] == assignment['strategy_id'] and row['seat'] == assignment['seat'],
                                'Sealed teacher conditional context differs')
                        bundle=bundles[label['bundle_id']]
                        require(bundle['model_id']==label['model_id'] and bundle['export_hash']==label['export_hash']
                            and bundle['fit']['identity']==label['fit_id'] and digest(bundle['directions'])==label['direction_id']
                            and bundle['coefficients'][assignment['strategy_id']]==label['coefficient'],
                            'Sealed teacher evaluator binding differs')
                    else:
                        require(row.get('style',[0.]*4)==[0.]*4 and row.get('intent',[0.]*8)==[0.]*8, 'Changed legacy neutral row')
                    values.append(LabelRow(row, encoder.encode(row['view'], row['choices'], row.get('style',[0.]*4), row.get('intent',[0.]*8))))
                require(len({r.identity for r in values}) == len(values), 'Duplicate sealed teacher row')
                self._rows[receipt['sha256']] = tuple(values)
            result.extend((generation - receipt['generation'], row) for row in self._rows[receipt['sha256']])
        return result

    def payload(self):
        return dict(contract=CONTRACT, identity=self.identity, receipts=copy.deepcopy(self.pending),
                    retired=copy.deepcopy(self.retired))

    def committed(self, generation):
        state = read(safe(self.run, self.run / 'state.json'))
        require(state['iteration'] == generation and state.get('teacher_phase', {}).get('buffer') == self.payload(),
                'Teacher cleanup requires matching durable state commit')
        retained=state.get('resume') or state.get('history',[])[-state.get('signature',{}).get('config',{}).get('retention',{}).get('resume_checkpoints',4):]
        oldest=min((e['generation'] for e in retained),default=generation)
        protected=[]
        for receipt in self.retired:
            if receipt['generation']>=oldest-4:
                protected.append(receipt)
                continue
            path, sidecar = self.paths(receipt)
            if not path.exists() and not sidecar.exists(): continue
            self.owner()
            require(not path.exists() or sha(path) == receipt['sha256'], 'Refusing changed retired teacher shard')
            require(not sidecar.exists() or read(sidecar) == receipt, 'Refusing changed retired teacher receipt')
            if path.exists(): path.unlink()
            if sidecar.exists(): sidecar.unlink()
            self._rows.pop(receipt['sha256'], None)
        self.receipts = copy.deepcopy(self.pending)
        self.retired = protected  # Rollback checkpoints can still own earlier teacher windows.


def sample_indices(ages, weights, count, *, seed, generation, step):
    """Per-label age weights, independent draws per step; no global RNG use."""
    if not ages: return []
    key = digest(dict(contract=CONTRACT, seed=seed, generation=generation, step=step))
    rng = np.random.default_rng(int(key[:16], 16))
    p = np.asarray([weights[a] for a in ages], dtype=np.float64); p /= p.sum()
    return rng.choice(len(ages), min(count, len(ages)), replace=False, p=p).tolist()
