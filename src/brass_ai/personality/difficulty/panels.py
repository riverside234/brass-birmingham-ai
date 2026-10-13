"""Frozen P5 jobs: members, actual mixtures, style tables and adjacent-tier pairs."""
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import random

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.personality.deployment import roster
from brass_ai.personality.deployment.contracts import REPORT as P4_REPORT, read_sealed as read_p4
from brass_ai.personality.search.readiness import load as load_ready
from brass_ai.recovery.retention import owned
from brass_ai.step5.league import seeds_for, write_json
from .contracts import (PANEL, REPORT, STOCK, TIERS, Gates, default_profiles, identity,
                        read_sealed, resolve_search, sealed, validate_profiles)
from . import native


def setup_group(seed):
    return digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed))


def jobs_for(value, seeds, pool_size, profiles, *, repetitions=2, seed=151000000, partition='validation'):
    require(1 <= repetitions <= 8 and 1 <= pool_size <= 4, 'Invalid repetitions/reference pool')
    by_id = {m['id']: m for m in value['members']}
    groups = value['archetypes']; jobs = []
    arms = [dict(kind='member', id=m['id'], group=m['archetype_id'], repeat=0) for m in value['members']]
    arms += [dict(kind='mixture', id=g['id'], group=g['id'], repeat=r) for g in groups for r in range(repetitions)]

    def member(group, setup, seat, repeat, stream):
        rng = random.Random(digest(dict(seed=seed, setup=setup, seat=seat, group=group['id'], repeat=repeat, stream=stream)))
        return rng.choice(group['members'])

    def bot(mid, tier):
        return dict(kind='personality', member=mid, tier=tier)

    def add(setup, lineup, table, *, arm=None, seat=None, tier=None, high_seats=None, lower=None):
        stream = digest(dict(seed=seed, setup=setup, seat=seat, repeat=arm['repeat'] if arm else 0,
                             kind=arm['kind'] if arm else 'direct', high_seats=high_seats))
        jobs.append(dict(index=len(jobs), seed=setup, setup_group=setup_group(setup), partition=partition,
                         sampling_seed=int(stream[:15], 16), lineup=lineup, table=table, arm=arm,
                         seat=seat, tier=tier, higher_seats=high_seats, lower_tier=lower))

    for setup in seeds:
        for tier in TIERS:
            for arm in arms:
                group = next(g for g in groups if g['id'] == arm['group'])
                for seat in range(4):
                    # The same member draw is matched across tiers and reference lineups.
                    mid = arm['id'] if arm['kind'] == 'member' else member(group, setup, seat, arm['repeat'], 'subject')
                    table = [dict(kind='parent', index=0) for _ in range(4)]
                    for offset in range(1, 4): table[(seat+offset)%4]['index'] = (setup+offset-1)%pool_size
                    table[seat] = bot(mid, tier)
                    add(setup, 'frozen-neural', table, arm=arm, seat=seat, tier=tier)
                    if tier == 'easy':
                        stock = [dict(kind='stock', identity=STOCK) for _ in range(4)]; stock[seat] = bot(mid, tier)
                        add(setup, 'original-hard', stock, arm=arm, seat=seat, tier=tier)
            for gi, group in enumerate(groups):
                for seat in range(4):
                    arm = dict(kind='mixture', id=group['id'], group=group['id'], repeat=0)
                    for mixed in (False, True):
                        table = []
                        for s in range(4):
                            other = groups[(gi+(s-seat)%4)%len(groups)] if mixed else group
                            table.append(bot(member(other, setup, s, 0, 'style-table'), tier))
                        add(setup, 'mixed-styles' if mixed else 'homogeneous-style', table, arm=arm, seat=seat, tier=tier)
        for low, high in zip(TIERS, TIERS[1:]):
            for group in groups:
                for pair in combinations(range(4), 2):
                    table = [bot(member(group, setup, s, 0, 'direct'), high if s in pair else low) for s in range(4)]
                    add(setup, 'adjacent-direct', table, tier=high, lower=low, high_seats=list(pair),
                        arm=dict(kind='mixture', id=group['id'], group=group['id'], repeat=0))
    require(all(s.get('member') in by_id for j in jobs for s in j['table'] if s['kind']=='personality'), 'Unknown member')
    return jobs


def source_check(root, value, fixture):
    pool_size = roster.validate_sources(root, value)
    _, _, ready, handoff, _ = load_ready(root, value['shared_model']['ready'])
    if not fixture:
        require(not handoff.get('origin', {}).get('fixture') and 'fixture' not in ready.get('scope', '').lower(),
                'Fixture model cannot supply real P5 qualification')
        require(value['p4_qualified'] and value['full_size'], 'A qualified real 16/128 P4 roster is required')
        q = value['qualification']; report_path = project_path(root, q['report'])
        report = read_p4(report_path, P4_REPORT)
        require(sha(report_path) == q['sha256'], 'P4 evidence changed')
        validate_p4_evidence(value, report)
    require(fixture or all(g['name'] and g['description'] for g in value['archetypes']), 'Frozen public names are required')
    return pool_size


def validate_p4_evidence(value, report):
    """P5/P6 require the complete pinned P4 claim, not a bare roster flag."""
    expected = {'complete_panel','full_size','searched','both_eras','individual_quality','mixture_quality',
                'search_support','coherent_distinct_members','replayability','pottery_coverage','blind_recognition_and_tactics'}
    q = value['qualification']
    require(value['p4_qualified'] and report['format'] == P4_REPORT
            and report['contract'] == identity()['p4']
            and report['id'] == digest({k:v for k,v in report.items() if k!='id'})
            and report['p4_qualified'] is True and set(report['checks']) == expected
            and all(v is True for v in report['checks'].values())
            and report['roster_id'] == q['source_roster'] and report['panel_id'] == q['panel_id']
            and report['games'] == report['expected_games'] > 0 and not report['missing_jobs']
            and report['sources'] and report['human']['passed'] is True
            and set(report['members']) == {m['id'] for m in value['members']}
            and set(report['mixtures']) == {g['id'] for g in value['archetypes']}, 'Incomplete P4 evidence')


def calibration_report(root, directory):
    # Check the stage before following the evidence closure (no release cycles).
    from .evaluation import load_report
    require(read_sealed(project_path(root,directory)/'report.json', REPORT)['stage'] == 'calibration',
            'Expected calibration evidence')
    return load_report(root, directory)


def plan(root, source, output, *, profiles=None, setups=32, repetitions=2, seed=151000000,
         stage='calibration', fixture=False, calibration=None, hardware_label='lab-preview', gates=None, tables=24):
    root, out = owned(root, output)
    require(not out.exists() and 2 <= setups <= 512 and stage in ('calibration', 'release'), 'Use a new bounded P5 panel')
    require(type(tables) is int and 1 <= tables <= 24, 'Declare at most 24 evaluation tables')
    profiles = validate_profiles(profiles or default_profiles()); gates = (gates or Gates()).validate()
    source = project_path(root, source); value = roster.load(source)
    count = source_check(root, value, fixture)
    forbidden = set(value['forbidden_groups']); preceding = None
    if stage == 'release':
        require(calibration is not None, 'Freeze and review a calibration panel before the untouched release panel')
        p = project_path(root, calibration); old = calibration_report(root, p)
        require(old['roster_id'] == value['id'] and old['profiles'] == profiles and old['stage'] == 'calibration'
                and old['gates'] == asdict(gates) and old['tables'] == tables and old['hardware_label'] == hardware_label
                and old['fixture'] == fixture and old['checks']['complete_panel'] and old['checks']['no_failed_games'],
                'Calibration must be complete under these exact recipes')
        forbidden.update(old['setup_groups']); preceding = dict(path=p.relative_to(root).as_posix(), sha256=sha(p/'report.json'))
    split = 'test' if stage == 'release' else 'validation'; seeds = []; cursor = seed
    while len(seeds) < setups:
        new, cursor = seeds_for(cursor, setups-len(seeds), split)
        seeds.extend(s for s in new if setup_group(s) not in forbidden)
    host = native.runtime(root); host_manifest = read_json(host/'runtime.json')
    result = sealed(dict(format=PANEL, contract=identity(), fixture=bool(fixture), stage=stage,
        roster=source.relative_to(root).as_posix(), roster_id=value['id'], roster_sha256=sha(source/'manifest.json'),
        profiles=profiles, searches={t: resolve_search(value, profiles, t).to_dict() for t in TIERS},
        gates=asdict(gates), pool_size=count, pool_id=value['pool_id'], seed=seed, seeds=seeds,
        repetitions=repetitions, partition=split, forbidden_groups=sorted(forbidden), calibration=preceding,
        hardware_label=hardware_label, tables=tables, baseline=dict(identity=STOCK, host=host.relative_to(root).as_posix(),
            runtime_sha256=sha(host/'runtime.json'), files=host_manifest['files'],
            label='Original Hard selector on declared guarded/repaired R1 rules; not unpatched vendor rules'),
        jobs=jobs_for(value, seeds, count, profiles, repetitions=repetitions, seed=seed, partition=split),
        scope='P5 evaluation only; never PPO/IL/teacher/descriptor data. Fixtures cannot qualify tiers or release.'))
    out.mkdir(parents=True); write_json(out/'manifest.json', result)
    paths=[result['roster']+'/manifest.json',result['baseline']['host']+'/runtime.json',
           out.relative_to(root).as_posix()+'/reservations.json']
    if preceding: paths.append(preceding['path']+'/report.json')
    write_json(out/'pins.json', dict(paths=paths))
    return result


def load(root, directory, *, sources=True):
    root = Path(root).resolve(); directory = project_path(root, directory)
    panel = read_sealed(directory/'manifest.json', PANEL)
    require(panel['stage'] in ('calibration','release') and panel['partition'] ==
            ('test' if panel['stage']=='release' else 'validation')
            and (panel['calibration'] is not None) == (panel['stage']=='release'), 'Invalid panel stage/separation')
    value = roster.load(project_path(root, panel['roster']))
    require(value['id'] == panel['roster_id'] and sha(project_path(root, panel['roster'])/'manifest.json') == panel['roster_sha256'],
            'P5 roster changed')
    validate_profiles(panel['profiles']); Gates(**panel['gates']).validate()
    require(panel['searches'] == {t: resolve_search(value, panel['profiles'], t).to_dict() for t in TIERS}, 'Tier search changed')
    forbidden=set(value['forbidden_groups'])
    if sources:
        require(source_check(root, value, panel['fixture']) == panel['pool_size'], 'Reference pool changed')
        require(sha(native.runtime(root)/'runtime.json') == panel['baseline']['runtime_sha256'], 'Original Hard/runtime changed')
        if panel['calibration']:
            require(sha(project_path(root, panel['calibration']['path'])/'report.json') == panel['calibration']['sha256'], 'Calibration changed')
            old=calibration_report(root,panel['calibration']['path'])
            require(old['roster_id']==value['id'] and old['checks']['complete_panel'] and old['checks']['no_failed_games']
                    and all(old[k]==panel[k] for k in ('profiles','gates','tables','hardware_label','fixture')),
                    'Calibration recipe/coverage changed')
            forbidden.update(old['setup_groups'])
    require(forbidden <= set(panel['forbidden_groups']), 'Missing excluded training/calibration setups')
    require(len(panel['seeds']) == len(set(panel['seeds'])) and
            not {setup_group(s) for s in panel['seeds']} & set(panel['forbidden_groups']), 'Reused setup group')
    require(panel['jobs'] == jobs_for(value, panel['seeds'], panel['pool_size'], panel['profiles'],
        repetitions=panel['repetitions'], seed=panel['seed'], partition=panel['partition']), 'Frozen job assignments changed')
    return panel, value
