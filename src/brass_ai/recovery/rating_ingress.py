"""Streaming rating replay validation; workers return only small outcome rows.

g5466: by default a worker reads each shard's games for their outcomes only (game id, seat policies, setup and
result: all the PL fit and its identity checks use), without the full replay checks, the replay hash or the feature
cache; BRASS_RATING_REPLAY_CHECKS=1 (ratings.rating_replay_checks) restores the full read. A native-format shard
always takes the full read."""
import gzip
import time
from collections import defaultdict
from brass_ai.learning.contracts import digest,require
from .replay import read_shards


def validate_shard(task,spec_digest=None):
    from .ingress import _admission
    from .telemetry import Timings
    if _admission is None:return _validate_shard(task,spec_digest)
    with _admission.acquire(1,Timings()):return _validate_shard(task,spec_digest)


def outcomes(path):
    """The games of a rating replay shard with only the fields the PL rows read, or None for a native-format shard."""
    from brass_ai.learning.contracts import keys
    from . import il_replay_storage
    from brass_ai.learning.contracts import decode_json
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        header=decode_json(next(stream))
        if header.get('format') not in (il_replay_storage.LEGACY_FORMAT,il_replay_storage.FORMAT):return None
        keys(header,('format','compatibility','provenance'))
        require(type(header['provenance']) is dict,'Invalid rating replay provenance')
        games=[]
        for line in stream:
            game=decode_json(line)
            if header['format']==il_replay_storage.FORMAT:game=game['game']
            require(all(isinstance(game.get(k),str) and game[k] for k in ('game_id','setup_group'))
                    and type(game.get('opponents')) is list and len(game['opponents'])==4 and 'result' in game,
                    'Rating replay game fields differ')
            games.append({k:game[k] for k in ('game_id','setup_group','opponents','result')})
    return games


def _validate_shard(task,spec_digest):
    from .ratings import game_record,rating_replay_checks
    records,jobs,spec=task;began=time.perf_counter();cpu=time.process_time()
    first=records[0];path=str(first['path'])
    require(all(str(r['path'])==path and r['sha256']==first['sha256'] and r.get('features')==first.get('features')
                for r in records),'Conflicting rating shard receipts')
    games=None if rating_replay_checks() else outcomes(path)
    if games is None:
        games=read_shards([path],partition='validation',receipts={path:first.get('features')},
                          _expected_hashes={path:first['sha256']})
    loaded={g['game_id']:g for g in games}
    require(len(loaded)==len(games)==len(records) and set(loaded)=={r['game_id'] for r in records},
            'Rating shard membership differs')
    rows=[]
    for record in records:
        index=record['job'];job=jobs[index];game=loaded[record['game_id']]
        expected=[job['policies'][(s-job['rotation'])%4]['policy_id'] for s in range(4)]
        require(game['opponents']==expected and game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed'])),
                'Rating game seat/setup identity differs')
        rows.append((index,game_record(game,spec['lineups'][index],spec,spec_digest=spec_digest)))
    return rows,dict(worker_seconds=time.perf_counter()-began,worker_cpu_seconds=time.process_time()-cpu)


class RatingIngress:
    def __init__(self,pool,jobs,spec):
        self.pool,self.jobs,self.spec=pool,jobs,spec
        self.spec_digest=digest(spec)
        self.futures={};self.receipts={};self.started=time.perf_counter()

    def submit(self,records):
        for future in self.futures.values():
            if future.done():future.result()  # Fail while actors are still running.
        groups=defaultdict(list)
        for record in records:groups[record['path']].append(record)
        for path,group in groups.items():
            require(path not in self.futures,'Repeated rating shard publication')
            self.receipts[path]=group
            jobs={record['job']:self.jobs[record['job']] for record in group}
            self.futures[path]=self.pool.submit(validate_shard,(group,jobs,self.spec),self.spec_digest)

    def finish(self,records):
        from .ratings import rating_replay_checks as checks
        began=time.perf_counter();published=[r for group in self.receipts.values() for r in group]
        require(sorted(published,key=lambda r:r['job'])==sorted(records,key=lambda r:r['job']),
                'Rating ingress receipts differ from completed collection')
        ready=sum(f.done() for f in self.futures.values());rows={};worker=cpu=0.
        for future in self.futures.values():
            checked,metrics=future.result();worker+=metrics['worker_seconds'];cpu+=metrics['worker_cpu_seconds']
            for index,row in checked:
                require(index not in rows,'Duplicate rating outcome');rows[index]=row
        require(set(rows)==set(range(len(self.jobs))),'Incomplete rating outcomes')
        return [rows[i] for i in range(len(self.jobs))],dict(shards=len(self.futures),
            shards_ready_before_join=ready,worker_seconds=worker,worker_cpu_seconds=cpu,
            join_seconds=time.perf_counter()-began,stream_seconds=time.perf_counter()-self.started,
            scope='Worker time overlaps actors/readers; join is remaining wall time; '
                  +('full replay checks' if checks() else 'outcome fields only (g5466; full replay checks off)'))
