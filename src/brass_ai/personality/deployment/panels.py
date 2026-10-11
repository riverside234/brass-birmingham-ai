"""Prespecified matched P4 panels, collected in bounded resumable chunks."""
from pathlib import Path
import random

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.personality.search.collection import collect, read
from brass_ai.personality.search.contracts import Profile
from brass_ai.recovery.retention import owned
from brass_ai.search.common import SearchConfig
from brass_ai.step5.league import seeds_for, write_json
from . import roster
from .contracts import PANEL, identity, read_sealed, sealed

CHUNK = 'personality-roster-chunk/1.0.0'


def jobs_for(value, seeds, pool_size, *, repetitions=2, seed=92191):
    """Full seat rotations and separate real uniform-mixture/representative games."""
    require(1 <= repetitions <= 8 and 1 <= pool_size <= 4, 'Invalid bounded panel repetitions/pool')
    by_id = {m['id']: m for m in value['members']}
    recipes = {m['candidate_id']: m['profile'] for m in value['members']}
    recipes.update({c['candidate_id']: c['profile'] for c in value['controls']})
    jobs = []
    for setup in seeds:
        arms = [(dict(role='member', member=m['id'], group=m['archetype_id'], replicate=0), m)
                for m in value['members']]
        arms += [(dict(role='control', control=c['profile']['style'], replicate=0), c) for c in value['controls']]
        for group in value['archetypes']:
            for repeat in range(repetitions):
                arms.append((dict(role='representative', group=group['id'], replicate=repeat), by_id[group['representative']]))
                arms.append((dict(role='mixture', group=group['id'], replicate=repeat), None))
        for arm, fixed in arms:
            for seat in range(4):
                chosen = fixed
                if arm['role'] == 'mixture':
                    group = next(g for g in value['archetypes'] if g['id'] == arm['group'])
                    # Independent per-seat, once-per-game draw. Never choose based on score.
                    rng = random.Random(digest(dict(seed=seed, setup=setup, seat=seat, group=group['id'], repeat=arm['replicate'])))
                    chosen = by_id[rng.choice(group['members'])]
                cid = chosen['candidate_id']
                profiles = [Profile(alpha=0.).to_dict() for _ in range(4)]; profiles[seat] = recipes[cid]
                refs = [0]*4
                for offset in range(1,4): refs[(seat+offset)%4] = (setup+offset-1)%pool_size
                # Matched action randomness within arm comparisons; independent
                # representative repetitions measure normal policy randomness.
                stream = digest(dict(seed=seed, setup=setup, seat=seat, replicate=arm['replicate'],
                                     stream='variation' if arm['role'] in ('representative','mixture') else 'quality'))
                label = dict(arm, selected_member=chosen.get('id'))
                jobs.append(dict(index=len(jobs), seed=setup, sampling_seed=int(stream[:15],16),
                                 partition='validation', neural_seats=[seat], profiles=profiles,
                                 controllers=[cid]*4, parent_indices=refs, p4=label))
    return jobs


def plan(root, candidate, output, *, setups=16, repetitions=2, seed=121000000, labels=None):
    root, out = owned(root, output)
    require(not out.exists() and 2 <= setups <= 128, 'Use a new panel with 2..128 setups')
    candidate = project_path(root, candidate); value = roster.load(candidate)
    pool_size = roster.validate_sources(root, value)
    names=None
    if labels:
        names=read_json(project_path(root,labels))
        require(names['roster_id']==value['id'],'Names belong to another roster')
        ids={g['id'] for g in value['archetypes']};entries=names['labels']
        require(len(entries)==len(ids) and {e['archetype_id'] for e in entries}==ids
                and all(isinstance(e['name'],str) and e['name'].strip() and isinstance(e['description'],str)
                        and e['description'].strip() for e in entries)
                and len({e['name'].casefold().strip() for e in entries})==len(ids),'Freeze distinct names/descriptions from discovery evidence')
    require(not value['full_size'] or names is not None,'Name the full roster from discovery cards before calibration')
    seeds, cursor = [], seed
    while len(seeds) < setups:
        new, cursor = seeds_for(cursor, setups-len(seeds), 'validation')
        seeds.extend(s for s in new if digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=s)) not in value['forbidden_groups'])
    jobs = jobs_for(value, seeds, pool_size, repetitions=repetitions, seed=seed)
    result = sealed(dict(format=PANEL, contract=identity(), roster=candidate.relative_to(root).as_posix(),
                         roster_id=value['id'], roster_sha256=sha(candidate/'manifest.json'),
                         seeds=seeds, seed=seed, repetitions=repetitions, jobs=jobs, labels=names['labels'] if names else None,
                         scope='P4 calibration only; no PPO, descriptor fitting, teacher quota or final P5 test credit'))
    out.mkdir(parents=True); write_json(out/'manifest.json', result)
    write_json(out/'pins.json', dict(paths=[result['roster']+'/manifest.json']))
    return result


def load(root, directory, *, sources=False):
    directory = project_path(root, directory)
    panel = read_sealed(directory/'manifest.json', PANEL)
    value = roster.load(project_path(root, panel['roster']))
    require(value['id'] == panel['roster_id'] and sha(project_path(root,panel['roster'])/'manifest.json') == panel['roster_sha256'],
            'Panel roster changed')
    # Regeneration catches dropped controls, mislabeled mixtures or favorable seat coverage.
    pool_size = roster.validate_sources(root, value) if sources else None
    if pool_size is not None:
        require(panel['jobs'] == jobs_for(value, panel['seeds'], pool_size, repetitions=panel['repetitions'], seed=panel['seed']),
                'Panel assignments changed')
    return panel, value


def execute(root, panel_directory, output, *, start=0, games=128, tables=24, device='cpu', mono=None):
    root, out = owned(root, output)
    require(not out.exists(), 'Chunks are immutable; resume with the next uncollected job, not a favorable retry')
    panel_directory = project_path(root, panel_directory)
    panel, value = load(root, panel_directory, sources=True)
    require(0 <= start < len(panel['jobs']) and 1 <= games <= 4096, 'Invalid bounded panel slice')
    jobs = panel['jobs'][start:start+games]
    # Write the reservation before collection. A failed chunk remains incomplete;
    # analysis never discards its assigned jobs or substitutes successful trials.
    out.mkdir(parents=True)
    write_json(out/'reservation.json', dict(panel_id=panel['id'], jobs=[j['index'] for j in jobs]))
    result = collect(root, value['shared_model']['ready'], out/'games', jobs, SearchConfig(**value['search']).validate(),
                     tables=tables, device=device, mono=mono, decision_mode=value['decision_mode'])
    receipt = sealed(dict(format=CHUNK, contract=identity(), panel=panel_directory.relative_to(root).as_posix(),
                          panel_id=panel['id'], panel_sha256=sha(panel_directory/'manifest.json'),
                          collection_id=result['id'], collection_sha256=sha(out/'games/manifest.json'),
                          jobs=[j['index'] for j in jobs]))
    write_json(out/'manifest.json',receipt)
    write_json(out/'pins.json', dict(paths=[receipt['panel']+'/manifest.json', (out/'games/manifest.json').relative_to(root).as_posix()]))
    return receipt


def read_chunks(root, directory, chunks):
    panel, value = load(root, directory, sources=True)
    seen = set(); sources = []
    def pairs():
        for chunk in chunks:
            chunk = project_path(root, chunk); receipt = read_sealed(chunk/'manifest.json',CHUNK)
            require(receipt['panel_id'] == panel['id'] and receipt['panel_sha256'] == sha(project_path(root,directory)/'manifest.json')
                    and receipt['collection_sha256'] == sha(chunk/'games/manifest.json'), 'Chunk belongs to another panel')
            manifest, games = read(chunk/'games',teacher_game=False,lazy=True)
            require(receipt['collection_id'] == manifest['id'] and not manifest['spec']['calibration_probe']
                    and manifest['spec']['ready_id'] == value['shared_model']['ready_id']
                    and manifest['spec']['pool_id'] == value['pool_id']
                    and manifest['spec']['search'] == value['search']
                    and manifest['spec']['decision_mode'] == value['decision_mode'], 'Chunk controller/model/budget changed')
            require(receipt['jobs'] == [j['index'] for j in manifest['spec']['jobs']], 'Chunk coverage changed')
            sources.append(dict(path=chunk.relative_to(Path(root).resolve()).as_posix(), sha256=sha(chunk/'manifest.json')))
            for game, side in games:
                job=side['job']; index=job['index']
                require(0 <= index < len(panel['jobs']) and job == panel['jobs'][index] and index not in seen,
                        'Duplicate or unplanned calibration game')
                require(game['setup_group'] not in value['forbidden_groups'] and game['split'] == 'validation',
                        'P4 calibration overlaps discovery/selection data')
                seen.add(index); yield game,side
    return panel,value,pairs(),seen,sources
