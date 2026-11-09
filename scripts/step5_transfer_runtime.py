"""Stdlib-only inventory of current native training continuation dependencies.

Replays and preparation caches deliberately do not travel. The exporter still
retains the small immutable probe, runtime evidence and live opponent state.
"""
from pathlib import Path
import json
import math
import re

from brass_ai.learning.paths import resolve_project_path
from brass_ai.learning.contracts import digest


REPLAY_POLICY = 'fresh-buffers/1.0.0'


def rating_summary(root, run, slot, *, read, require, add):
    """Retain committed PL outcome rows, never full game/action records.

    The rating loader needs these exact bytes for validation, later refits and
    history compaction. Mirror its immutable slot/spec bindings without Torch,
    NumPy, model loading or replay reconstruction in the transfer environment.
    """
    path=lambda value:resolve_project_path(root,value)
    generation=slot['generation'];record=slot['records'];source=path(record['path'])
    require(type(generation) is int and generation>0 and source.name=='games.jsonl'
        and source.parent.parent==run/'ratings/slots' and
        re.fullmatch(r'generation-'+f'{generation:06d}'+r'(-source-[0-9a-f]{16})?',source.parent.name),
        'Invalid committed rating summary path')
    spec_path=path(slot['spec']['path'])
    require(spec_path==source.parent/'spec.json','Rating summary/spec directory differs')
    add(spec_path,slot['spec']['sha256']);spec=read(spec_path)
    require(spec.get('format')=='plackett-luce-slot/1.0.0' and spec['generation']==generation,
        'Rating summary slot/spec generation differs')
    lineups={(r['kind'],r['seed'],r['seat']):r for r in spec['lineups']}
    require(lineups and len(lineups)==len(spec['lineups']),'Duplicate/empty rating summary lineups')
    expected_fields={'game_id','setup_id','setup_seed','slot','candidate_seat','kind',
        'agents','order','official_placement','vp','win_share','spec_digest'}
    found=set();games=set();identity=digest(spec)
    require(source.is_file(),'Missing committed rating outcome summary: '+str(source))
    with source.open('r',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            require(type(row) is dict and set(row)==expected_fields,'Rating outcome summary contains unsupported/replay fields')
            require(all(type(row[k]) is int for k in ('slot','setup_seed','candidate_seat')),
                'Invalid rating summary identity')
            key=(row['kind'],row['setup_seed'],row['candidate_seat']);lineup=lineups.get(key)
            require(lineup is not None and key not in found and row['slot']==generation and row['spec_digest']==identity,
                'Rating summary lineage/coverage differs')
            require(all(type(row[k]) is str and row[k] for k in ('game_id','setup_id')) and row['game_id'] not in games
                and type(row['agents']) is list and len(row['agents'])==4 and all(type(a) is str and a for a in row['agents'])
                and row['agents']==[lineup['agents'][(s-row['candidate_seat'])%4] for s in range(4)]
                and type(row['order']) is list and all(type(s) is int for s in row['order'])
                and sorted(row['order'])==list(range(4)), 'Invalid rating summary game/seats')
            for key_name in ('official_placement','vp','win_share'):
                values=row[key_name]
                require(type(values) is list and len(values)==4 and
                    all(type(v) in (int,float) and math.isfinite(v) for v in values),'Invalid rating summary outcome')
            require(all(type(v) is int and 1<=v<=4 for v in row['official_placement'])
                and all(0<=v<=1 for v in row['win_share']) and abs(sum(row['win_share'])-1)<1e-8,
                'Invalid rating summary rank/win share')
            found.add(key);games.add(row['game_id'])
    require(found==set(lineups),'Incomplete committed rating outcome summary')
    add(source,record['sha256'],rating_summary=True)
    require('bytes' not in record or source.stat().st_size==record['bytes'],'Rating summary size differs')


def native_inventory(root, run, state, *, add, read, require, runtime):
    def path(name):
        return resolve_project_path(root, name)

    def pinned(item):
        p = path(item['path']); add(p, item['sha256'])
        require('bytes' not in item or p.stat().st_size == item['bytes'], 'Native dependency size differs: '+str(p))
        return p

    def refs(value, *, required=True):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and value.get('sha256'):
                if required or path(value['path']).is_file(): pinned(value)
            for child in value.values(): refs(child, required=required)
        elif isinstance(value, list):
            for child in value: refs(child, required=required)

    probe = state.get('il_kl_limit', {}).get('probe')
    require(probe is not None, 'Native transfer requires the committed fixed IL probe')
    probe_path = pinned(probe); metadata = read(probe_path)
    require(probe_path == run/'il-kl/manifest.json' and metadata['probe_hash'] == probe['probe_hash']
            and len(metadata['rows']) == 2048, 'Transfer needs the original fixed 2048-position IL probe')
    cache = pinned(metadata['cache'])
    require(cache == run/'il-kl/probe.npz', 'Fixed IL probe array is outside its owned directory')
    # Startup-only JSON is useful report provenance after a restore. Retain it
    # when available, without requiring omitted g0 weights/replay on an already
    # restored run. Its original state receipt remains authoritative.
    seed=state['signature']['config'].get('fresh_selfplay_seed')
    if seed is not None:
        source=path(seed['manifest'])
        require(source==run/'seed/manifest.json','Fresh seed provenance is outside its owned run')
        if source.is_file():add(source,seed['sha256'])

    # These references are live metadata, weights and immutable probe arrays,
    # not the original IL replay or its .prepared feature cache.
    refs(state.get('quality_pool', {}).get('members', []))
    for key in ('policy_anchor', 'champion_mix', 'origin'):
        refs(state.get(key))
    ratings = state.get('ratings', {})
    refs({k:v for k,v in ratings.items() if k not in ('agents','slots')})
    for slot in ratings.get('slots',[]):
        require(slot['generation']<=state['iteration'],'Uncommitted future rating slot')
        rating_summary(root,run,slot,read=read,require=require,add=add)
        refs({k:v for k,v in slot.items() if k!='records'})
    refs(ratings.get('agents', []), required=False)
    for transition in state.get('source_transitions', []):
        add(run/transition['backup'])
    for key in ('auxiliary-checkpoint-inventory.json',):
        if (run/key).is_file(): add(run/key)
    if (run/'machine-transfer').is_dir():
        for p in sorted((run/'machine-transfer').glob('*.json')): add(p)

    seen = set()
    def build(item):
        p = pinned(item)
        if p in seen: return
        seen.add(p); value = read(p)
        for key in ('collector', 'binary', 'tests', 'build_log', 'performance_tests', 'reward_tests'):
            if key in value:
                file = pinned(value[key]); runtime.add(file.relative_to(root).as_posix())
        refs(value.get('bootstrap'))
        for file in (value.get('bootstrap', {}).get('binary'), *value.get('bootstrap', {}).get('dependencies', [])):
            if file: runtime.add(path(file['path']).relative_to(root).as_posix())
        # Preserve originals used to bind additive wire/verification adapters.
        for key in ('original_teacher_receipt','original_verification_receipt'):
            if key in value: build(value[key])
        for previous in value.get('previous_runtimes', []): build(previous)
        if value.get('b01'):
            b01 = pinned(value['b01'])
            fixtures = b01.parent/'inputs/fixtures.json'
            if fixtures.is_file(): add(fixtures)

    native = state['native_backend']
    q = read(pinned(native['qualification']))
    build(native['build'])
    if q.get('teacher_build'): build(q['teacher_build'])
    if native.get('shadow', {}).get('teacher_build'): build(native['shadow']['teacher_build'])
    # The rollback manifest remains pinned audit evidence. Its old diagnostic
    # replay payload is intentionally omitted; it is not a destination backup.
    pinned(native['rollback'])
    for name in ('artifacts/rust-backend/collector-performance/build.json',
                 'artifacts/rust-backend/teacher-verification/build.json',
                 'artifacts/rust-backend/teacher-zero-insolvency/build.json'):
        p = root/name
        if p.is_file():
            import hashlib
            with p.open('rb') as stream: expected = hashlib.file_digest(stream, 'sha256').hexdigest()
            build(dict(path=str(p), sha256=expected))
    # Small sealed source input for a destination-side adapter build, if it
    # wasn't prepared on the source. No IL data or compiler/target cache.
    from brass_ai.learning.post_transfer_reward import eligible
    source=root/'distribution/rust-lab-g1250/inputs.zip'
    if eligible(state['signature']['config']):
        require(source.is_file() or (root/'artifacts/rust-backend/teacher-zero-insolvency/build.json').is_file(),
            'Prepare the post-transfer reward teacher before exporting, or restore distribution/rust-lab-g1250/inputs.zip')
        if source.is_file():add(source)
    return dict(contract=REPLAY_POLICY, origin_generation=state['iteration'],
                auxiliary_buffer_iterations=8, il_probe_positions=2048)
