"""Teacher-only value rows, exact insolvency targets and sealed eight-age buffer."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
from functools import lru_cache

import numpy as np

from brass_ai.learning.contracts import digest, require
from brass_ai.learning.search_value import validate_binding
from brass_ai.learning.reward import terminal_utility
from brass_ai.step5.league import write_json

CONTRACT = 'search-value-buffer/1.0.0'


@lru_cache(maxsize=1)
def state_width():
    from brass_ai.learning.model import ModelConfig
    return ModelConfig.preset().state_dim


def validate_root(row, job):
    expected = job['expected']
    require(row['job'] == expected and all(row[k] == expected[k] for k in
            ('generation', 'model_id', 'reward_id', 'game_id')), 'Root value lineage differs')
    require(row['split'] == job['split'] and row['profile'] in ('full', 'fast') and
            row['phase'] in ('canal_early', 'canal_late', 'rail_early', 'rail_late'), 'Root profile/split differs')
    require(type(row['seat']) is int and 0 <= row['seat'] < 4 and
            row['mapping'] == [(s - row['seat']) % 4 for s in range(4)], 'Root seat mapping differs')
    value = job.get('seat_values', [job['search_value']] * 4)[row['seat']]
    require(row['search_value'] == value, 'Root value source differs')
    validate_binding(row['search_value'])
    require(type(row['index']) is int and 1 <= row['index'] <= 64 and type(row['n']) is int and row['n'] >= 0,
            'Root watermark differs')
    require(all(type(n) is int and n >= 0 for n in row['counts']) and sum(row['counts']) == row['n'], 'Root counts differ')
    for k in ('r', 'v', 'spent', 'remaining'):
        a = np.asarray(row[k], dtype=float)
        require(a.shape == (4,) and np.isfinite(a).all(), 'Invalid root ' + k)
    state = np.asarray(row['state'], dtype=np.float32)
    require(state.shape == (state_width(),) and np.isfinite(state).all(), 'Root state dimensions differ')
    returns = np.asarray(row['returns'], dtype=float)
    require(returns.shape == (len(row['counts']), 4) and np.isfinite(returns).all(), 'Root returns differ')
    oracle = (np.asarray(row['v']) + returns.sum(0)) / (1 + row['n'])
    require(np.allclose(oracle, row['r'], atol=1e-12, rtol=1e-12), 'Native/Python root value differs')
    cap = job['search']['reward']['insolvency']['max_cost']
    require(np.min(row['spent']) >= 0 and np.max(row['spent']) <= cap + 1e-9 and
            np.allclose(np.asarray(row['spent']) + row['remaining'], cap, atol=1e-9), 'Root budget differs')
    return row


def targets(rows, result, final_spent, reward, lam):
    require(lam in (.5, .8, .95, 1.) and rows, 'Invalid lambda/empty value game')
    spent = np.asarray([r['spent'] for r in rows] + [final_spent], dtype=float)
    costs = np.diff(spent, axis=0)
    require(spent.shape == (len(rows) + 1, 4) and np.isfinite(spent).all() and
            np.min(costs) >= -1e-10 and np.min(spent) >= 0 and
            np.max(spent) <= reward['insolvency']['max_cost'] + 1e-9, 'Noncausal/uncapped insolvency costs')
    final = terminal_utility(result, reward)
    outcomes = final - (spent[-1] - spent[:-1])
    g = np.empty_like(outcomes)
    following = final
    for i in reversed(range(len(rows))):
        boot = final if i == len(rows) - 1 else np.asarray(rows[i + 1]['r'])
        following = -costs[i] + (1 - lam) * boot + lam * following
        g[i] = following
    return g, outcomes


def complete_game(rows, result, final_spent, job, *, lam=1.):
    require(len(rows) == 64 and [r['index'] for r in rows] == list(range(1, 65)), 'Incomplete all-seat value game')
    require(len({tuple(r['turn_key']) for r in rows}) == 64, 'Repeated value turn')
    for row in rows: validate_root(row, job)
    require(sum(r['profile'] == 'full' for r in rows) == 16, 'Reserved full profile count differs')
    g, u = targets(rows, result, final_spent, job['search']['reward'], lam)
    return [dict(copy_row(r), target=t.tolist(), u=v.tolist(), target_lambda=lam)
            for r, t, v in zip(rows, g, u, strict=True)]


def copy_row(row):
    """Detach validated flat numeric tensors without per-number deepcopy work."""
    vectors = {'state','mapping','remaining','spent','r','v','counts','turn_key',
               'target','u','terminal_utility'}
    result = copy.deepcopy({k:v for k,v in row.items() if k not in vectors and k != 'returns'})
    result.update({k:list(v) for k,v in row.items() if k in vectors})
    if 'returns' in row: result['returns'] = [list(v) for v in row['returns']]
    return result


class ValueBuffer:
    """Separate immutable JSON shards; state commit owns the receipt list.

    Retired shards stay on disk until the existing retention workflow commits.
    No ordinary replay reader sees these paths or records.
    """
    def __init__(self, directory, reward_id, *, receipts=(), retired=()):
        self.directory = Path(directory).resolve()
        self.reward_id = reward_id
        self.receipts = copy.deepcopy(list(receipts))
        self.retired = copy.deepcopy(list(retired))
        self.cache = {}
        for r in self.receipts: self._read(r)

    def _path(self, receipt):
        path = (self.directory / receipt['file']).resolve()
        require(path.parent == self.directory and path.name.endswith('.json.gz'), 'Value shard path escaped')
        return path

    def _read(self, receipt):
        data = self._path(receipt).read_bytes()
        require(hashlib.sha256(data).hexdigest() == receipt['sha256'], 'Value buffer shard changed')
        if receipt['sha256'] not in self.cache:
            rows = json.loads(gzip.decompress(data))
            require(len(rows) == receipt['rows'] and all(r['generation'] == receipt['generation'] and
                    r['split'] == receipt['split'] and r['reward_id'] == receipt['reward_id'] for r in rows), 'Value shard identity differs')
            require('game_id' not in receipt or all(r['game_id'] == receipt['game_id'] for r in rows),
                    'Value shard game differs')
            self.cache[receipt['sha256']] = rows
        return self.cache[receipt['sha256']]

    def publish(self, generation, split, rows, *, game_id=None):
        require(split in ('train', 'heldout') and rows and all(r['generation'] == generation and r['split'] == split and
                r['reward_id'] == self.reward_id for r in rows), 'Value buffer attribution differs')
        require(len({(r['game_id'], r['index']) for r in rows}) == len(rows), 'Duplicate value row')
        require(game_id is None or all(r['game_id'] == game_id for r in rows), 'Value shard game differs')
        self.directory.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(json.dumps(rows, separators=(',', ':'), allow_nan=False).encode(), mtime=0, compresslevel=1)
        suffix = '' if game_id is None else '-' + digest(game_id)
        receipt = dict(contract=CONTRACT, file=f'{generation:06d}-{split}{suffix}.json.gz', generation=generation,
                       split=split, reward_id=self.reward_id, rows=len(rows), sha256=hashlib.sha256(data).hexdigest())
        if game_id is not None: receipt['game_id'] = game_id
        path = self._path(receipt)
        require(not path.exists() or path.read_bytes() == data, 'Uncommitted value buffer differs')
        if not path.exists():
            partial = path.with_suffix('.partial')
            with partial.open('xb') as f: f.write(data)
            partial.replace(path)
        self.receipts = [r for r in self.receipts if r['file'] != receipt['file']] + [receipt]
        # We just encoded these admitted rows. Keep an independent cache copy
        # instead of immediately decompressing and parsing the same tensors.
        # _read still rehashes the actual file; cold restore still fully decodes.
        self.cache[receipt['sha256']] = [copy_row(r) for r in rows]
        self._read(receipt)
        return receipt

    def rows(self, generation, split='train', *, stale_ages=()):
        # Caller supplies the *same* per-age label KL decision, not a second
        # expensive policy evaluation of every 8-age value tensor.
        require(split in ('train', 'heldout'), 'Invalid value split')
        keep = [r for r in self.receipts if r['reward_id'] == self.reward_id and 0 <= generation-r['generation'] < 8]
        for r in self.receipts:
            if r not in keep:
                if r not in self.retired: self.retired.append(r)
                self.cache.pop(r['sha256'], None)
        self.receipts = keep
        return [row for r in self.receipts if r['reward_id'] == self.reward_id and r['split'] == split
                and 0 <= generation-r['generation'] < 8 and (split == 'heldout' or generation-r['generation'] not in stale_ages)
                for row in self._read(r)]

    def payload(self):
        return dict(contract=CONTRACT, reward_id=self.reward_id, receipts=copy.deepcopy(self.receipts), retired=copy.deepcopy(self.retired))
