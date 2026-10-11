"""One frozen network, actual tested controls, immutable 16-by-8 membership."""
from dataclasses import asdict
from pathlib import Path
import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.personality.contracts import PROFILES
from brass_ai.personality.repertoire import archive, descriptor, discovery
from brass_ai.personality.search.contracts import Profile, controller
from brass_ai.personality.search.readiness import load as load_ready
from brass_ai.recovery.retention import owned
from brass_ai.search.common import SearchConfig
from brass_ai.step5.league import write_json
from .contracts import (ARCHETYPE, MEMBER, ROSTER, SAMPLING, Gates, identity,
                        read_sealed, sealed, validate_roster)


def neighborhoods(entries, representatives, per_group, gates):
    """Nonoverlapping nearest-representative neighborhoods; no duplicate padding."""
    by_id = {e['candidate_id']: e for e in entries if e['qualified']}
    require(len(representatives) >= 2 and len(set(representatives)) == len(representatives)
            and set(representatives) <= by_id.keys(), 'Missing tested representatives')
    centers = np.array([by_id[i]['centroid'] for i in representatives], dtype=float)
    require(np.isfinite(centers).all(), 'Invalid representative coordinates')
    result = []
    for i, rep in enumerate(representatives):
        separation = float(min(np.linalg.norm(centers[i] - centers[j]) for j in range(len(centers)) if j != i))
        require(separation > 1e-6, 'Indistinguishable archetype representatives')
        radius = gates.radius_fraction * separation
        require(by_id[rep]['dispersion'] < radius, 'Representative variation exceeds its neighborhood')
        eligible = []
        for e in by_id.values():
            if e['candidate_id'] in representatives:
                continue
            distance = np.linalg.norm(centers - np.asarray(e['centroid']), axis=1)
            other = min(distance[j] for j in range(len(centers)) if j != i)
            noise = np.hypot(e['dispersion'], by_id[rep]['dispersion'])
            if (distance[i] + e['dispersion'] <= radius
                    and distance[i] + gates.separation_margin * separation < other
                    and distance[i] > max(1e-6, gates.variant_signal * noise)):
                eligible.append(e)
        selected = [by_id[rep]]
        while len(selected) < per_group and eligible:
            # Spread members within the bounded same-theme neighborhood.
            def spread(e):
                return min(np.linalg.norm(np.asarray(e['centroid']) - np.asarray(s['centroid']))
                           for s in selected)
            best = max(eligible, key=lambda e: (spread(e), e['quality']['control_lower'], e['candidate_id']))
            noise = max(np.hypot(best['dispersion'], s['dispersion']) for s in selected)
            if spread(best) <= max(1e-6, gates.variant_signal * noise):
                break
            selected.append(best)
            eligible.remove(best)
        require(len(selected) == per_group,
                f'Archetype {rep[:12]} has {len(selected)}/{per_group} distinguishable qualified members; expand discovery')
        result.append(dict(representative=rep, members=[s['candidate_id'] for s in selected],
                           radius=radius, nearest_other=separation))
    require(len({i for g in result for i in g['members']}) == len(result) * per_group,
            'Overlapping neighborhoods')
    return result


def load(directory):
    return validate_roster(read_sealed(Path(directory) / 'manifest.json', ROSTER))


def validate_sources(root, value):
    """Check pinned evidence and weights before collection or requalification."""
    for source in value['dependencies']:
        require(sha(project_path(root, source['path'])) == source['sha256'], 'Roster dependency changed')
    _, parents, ready, _, pool = load_ready(root, value['shared_model']['ready'])
    require(ready['id'] == value['shared_model']['ready_id']
            and ready['model']['model_id'] == value['shared_model']['model_id']
            and pool == value['pool_id'] and ready['reward'] == value['search']['reward'],
            'Shared model/pool/reward changed; rebuild and requalify every member')
    cfg = SearchConfig(**value['search']).validate()
    for m in value['members']:
        actual = controller(ready['model']['model_id'], model_id(parents[0]),
                            Profile.parse(m['profile']), cfg, pool, decision_mode=value['decision_mode'])
        require(actual['candidate_id'] == m['candidate_id'], 'Member style/model/search changed')
    for c in value['controls']:
        require(c == controller(ready['model']['model_id'], model_id(parents[0]), Profile.parse(c['profile']),
                                cfg, pool, decision_mode=value['decision_mode']), 'Control changed')
    return len(parents)


def cards(root, candidate, output):
    """Labeled public discovery examples for naming, separate from blind calibration."""
    from brass_ai.personality.search.collection import read
    from .evaluation import public_behavior
    root,out=owned(root,output);require(not out.exists(),'Use a new reference-card directory')
    candidate=project_path(root,candidate);value=load(candidate);validate_sources(root,value)
    by_id={m['id']:m for m in value['members']}
    representatives={by_id[g['representative']]['candidate_id']:g['id'] for g in value['archetypes']}
    source=next(d for d in value['dependencies'] if d['path'].endswith('manifest.json')
                and read_json(project_path(root,d['path'])).get('format')==archive.VERSION)
    a=archive.load(project_path(root,source['path']).parent);examples={g['id']:[] for g in value['archetypes']}
    for s in a['sources']:
        if s['role']!='discovery': continue
        _,pairs=read(project_path(root,s['path']),teacher_game=False,lazy=True)
        for game,side in pairs:
            job=side['job']
            for seat in job['neural_seats']:
                group=representatives.get(job['controllers'][seat])
                if group and len(examples[group])<2:
                    behavior=public_behavior(game,seat,include_boards=True)
                    examples[group].append({k:behavior[k] for k in ('public_turns','public_boards')})
    require(all(examples.values()),'Missing public representative evidence')
    out.mkdir(parents=True)
    result=sealed(dict(format='personality-style-cards/1.0.0',contract=identity(),roster_id=value['id'],examples=examples,
                       scope='Labeled discovery examples for naming; blind calibration examples are separate'))
    write_json(out/'cards.json',result)
    write_json(out/'labels-template.json',dict(roster_id=value['id'],labels=[dict(archetype_id=g['id'],name='',description='') for g in value['archetypes']]))
    write_json(out/'pins.json',dict(paths=[candidate.relative_to(root).as_posix()+'/manifest.json']))
    return result


def select(root, source_archive, output, *, groups=16, members=8, method='fps', seed=19019, gates=None):
    root, out = owned(root, output)
    require(not out.exists(), 'Roster snapshots are immutable; choose a new directory')
    require(2 <= groups <= 16 and 2 <= members <= 8, 'Use a 2..16 by 2..8 preview or the full 16 by 8 roster')
    gates = (gates or Gates()).validate()
    source_archive = project_path(root, source_archive)
    a = archive.load(source_archive)
    p = discovery.load(project_path(root, a['proposal_path']))
    require(p['id'] == a['proposal_id'], 'Archive proposal differs')
    encoder_path = project_path(root, a['encoder']['path'])
    enc, _ = descriptor.load(encoder_path)
    require(enc['id'] == a['encoder']['id'] and sha(encoder_path/'manifest.json') == a['encoder']['sha256']
            and not enc['report']['collapsed'], 'Descriptor changed/collapsed')
    _, parents, ready, _, pool = load_ready(root, p['ready'])
    require(ready['id'] == p['ready_id'] and sha(project_path(root, p['ready'])/'ready.json') == p['ready_sha256']
            and pool == a['pool_id'] == p['pool_id'], 'Archive readiness/pool changed')
    mode = p['candidates'][0]['decision_mode']
    search = p['candidates'][0]['search']
    full = groups == 16 and members == 8
    require(not full or mode == 'searched', 'A full roster requires searched-controller confirmation; direct is preview only')
    cfg = SearchConfig(**search).validate()
    known = {c['candidate_id']: c for c in p['candidates']}
    for e in a['entries']:
        c = e['controller']
        require(c == known.get(e['candidate_id']) and c == controller(ready['model']['model_id'], model_id(parents[0]),
                Profile.parse(c['profile']), cfg, pool, decision_mode=mode),
                'All members must share the exact network, pool and search; distillation requires a new P3 archive')
    entries = [e for e in a['entries'] if e['qualified'] and e['quality']['qualified']
               and e['quality']['setups'] >= gates.min_setups and e['quality']['mean_win'] >= gates.win_floor
               and e['quality']['control_lower'] >= -gates.win_tolerance]
    alternatives = {m: archive.shortlist(entries, a['elites'], count=groups, method=m, seed=seed)
                    for m in ('fps', 'kmeans')}
    require(method in alternatives and len(alternatives[method]) == groups,
            'Not enough qualified representatives; do not pad the roster')
    neighborhoods_selected = neighborhoods(entries, alternatives[method], members, gates)
    dependencies = []
    def pin(path):
        path = project_path(root, path)
        dependencies.append(dict(path=path.relative_to(root).as_posix(), sha256=sha(path)))
    pin(source_archive/'manifest.json'); pin(project_path(root, a['proposal_path'])/'manifest.json')
    pin(encoder_path/'manifest.json'); pin(encoder_path/enc['weights']['path'])
    pin(project_path(root, p['ready'])/'ready.json')
    pin(project_path(root, p['ready'])/ready['model']['path'])
    forbidden = set(enc['training_groups'])
    for source in a['sources']:
        directory = project_path(root, source['path'])
        require(sha(directory/'manifest.json') == source['manifest_sha256']
                and sha(directory/'discovery.json') == source['receipt_sha256'], 'Archive collection changed')
        manifest = read_json(directory/'manifest.json')
        require(manifest['id'] == digest({k:v for k,v in manifest.items() if k != 'id'}), 'Archive source manifest differs')
        for job in manifest['spec']['jobs']:
            forbidden.add(digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=job['seed'])))
        pin(directory/'manifest.json'); pin(directory/'discovery.json')
    by_id = {e['candidate_id']: e for e in entries}
    archetypes, variants = [], []
    for n in neighborhoods_selected:
        group_id = 'archetype-' + digest(n['representative'])[:16]
        member_ids = ['variant-' + digest(c)[:16] for c in n['members']]
        archetypes.append(dict(format=ARCHETYPE, id=group_id, name=None, description=None,
                               representative=member_ids[0], members=member_ids, supported_tiers=[],
                               neighborhood=dict(radius=n['radius'], nearest_other=n['nearest_other']),
                               sampling=dict(version=SAMPLING, weights=[1/members]*members)))
        for cid, mid in zip(n['members'], member_ids):
            e = by_id[cid]
            variants.append(dict(format=MEMBER, id=mid, archetype_id=group_id, candidate_id=cid,
                                 model_id=ready['model']['model_id'], profile=e['controller']['profile'],
                                 centroid=e['centroid'], dispersion=e['dispersion'], tier_settings={}, fallback=None,
                                 selection_quality=e['quality'], qualification='pending-fresh-P4-calibration'))
    controls = [controller(ready['model']['model_id'], model_id(parents[0]),
                           Profile(style=s, alpha=0. if s == 'neutral' or mode == 'direct' else .15), cfg, pool,
                           decision_mode=mode) for s in PROFILES]
    value = dict(format=ROSTER, contract=identity(), counts=dict(archetypes=groups, per_archetype=members),
                 shared_model=dict(ready=p['ready'], ready_id=ready['id'], model_id=ready['model']['model_id']),
                 strategy='shared-frozen-network/no-distillation', decision_mode=mode, pool_id=pool, search=search,
                 archive_id=a['id'], encoder=a['encoder'], selection=dict(method=method, seed=seed, alternatives=alternatives),
                 gates=asdict(gates), archetypes=archetypes, members=variants, controls=controls,
                 forbidden_groups=sorted(forbidden), dependencies=dependencies,
                 p4_qualified=False, release_qualified=False, full_size=full,
                 scope='Frozen P4 candidate roster; difficulty and laptop release require P5/P6')
    validate_roster(value); value=sealed(value)
    out.mkdir(parents=True); write_json(out/'manifest.json', value)
    write_json(out/'pins.json', dict(paths=[d['path'] for d in dependencies]))
    return value
