"""Setup-clustered difficulty gates. Never infer qualification from fixtures."""
from collections import defaultdict
from pathlib import Path
import math

import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.learning.replay import terminal_targets
from brass_ai.recovery.retention import owned
from brass_ai.step5.league import write_json
from . import collection
from .contracts import Gates, REPORT, REVIEW, TIERS, identity, sealed, read_sealed, validate_report_summary


def interval(values, gates, claims, *, width=1.):
    """Bonferroni two-sided Student-t intervals on independent SETUP means.

    Rotations/repeats are clustered first. This is an approximate parametric
    interval, not a distribution-free certificate. Zero variance uses the
    bounded Hoeffding interval instead of making a zero-uncertainty claim.
    """
    x = np.asarray(list(values), dtype=float)
    require(np.isfinite(x).all() and claims >= 1 and width > 0, 'Invalid grouped metrics')
    result = dict(mean=float(x.mean()) if len(x) else None, lower=None, upper=None, setups=len(x),
                  claims=claims, family_alpha=gates.family_alpha, method='setup-Bonferroni-Student-t')
    if len(x) < gates.min_setups: return result
    std = float(x.std(ddof=1))
    if std < 1e-12:
        half = width*math.sqrt(math.log(2*claims/gates.family_alpha)/(2*len(x)))
        result['method'] = 'setup-Bonferroni-Hoeffding-zero-variance'
    else:
        from scipy.stats import t
        half = float(t.ppf(1-gates.family_alpha/(2*claims), len(x)-1))*std/math.sqrt(len(x))
    result.update(lower=result['mean']-half, upper=result['mean']+half)
    return result


def grouped(rows, field):
    values = defaultdict(list)
    for r in rows: values[r['setup']].append(r[field])
    return {k: float(np.mean(v)) for k, v in values.items()}


def difference(a, b, gates, claims):
    common = sorted(a.keys() & b.keys())
    return interval([a[k]-b[k] for k in common], gates, claims, width=2.)


def subject_rows(records):
    rows = []
    for r in records:
        if r['failure'] is not None: continue
        targets = terminal_targets(r['result']); job = r['job']; seat = job['seat']
        if seat is None: continue
        others = [s for s in range(4) if s != seat]
        rows.append(dict(index=job['index'], setup=job['setup_group'], seat=seat, lineup=job['lineup'],
            tier=job['tier'], arm=job['arm'], win=targets['win_share'][seat], vp=targets['vp'][seat], rank=targets['placement'][seat],
            vp_margin=targets['vp'][seat]-float(np.mean([targets['vp'][s] for s in others])),
            rank_margin=targets['placement'][seat]-float(np.mean([targets['placement'][s] for s in others]))))
    return rows


def summary(rows):
    return dict(games=len(rows), setups=len({r['setup'] for r in rows}),
                **{k: float(np.mean([r[k] for r in rows])) if rows else None
                   for k in ('win', 'vp', 'rank', 'vp_margin', 'rank_margin')})


def automatic(value, panel, records):
    gates = Gates(**panel['gates']).validate(); rows = subject_rows(records)
    subjects = [('member', m['id']) for m in value['members']] + [('mixture', g['id']) for g in value['archetypes']]
    claims = len(subjects)*4+4  # Easy + three-tier equivalence + two pairs in two panels.
    members = {}; mixtures = {}; tier_targets = {}
    for tier in TIERS:
        # Equal archetype weighting, independent of member/mixture repeat counts.
        groups = [grouped([r for r in rows if r['lineup']=='frozen-neural' and r['tier']==tier
                          and r['arm']['kind']=='mixture' and r['arm']['id']==g['id']], 'win') for g in value['archetypes']]
        common = set.intersection(*(set(g) for g in groups))
        tier_targets[tier] = {s: float(np.mean([g[s] for g in groups])) for s in sorted(common)}
    for kind, ident in subjects:
        own = [r for r in rows if r['arm']['kind']==kind and r['arm']['id']==ident]
        baseline = [r for r in own if r['lineup']=='original-hard']
        win = interval(grouped(baseline, 'win').values(), gates, claims)
        corroboration = summary(baseline)
        easy = dict(win_interval=win, metrics=corroboration, threshold=.25+gates.easy_gain,
            passed=bool(win['lower'] is not None and win['lower']>.25+gates.easy_gain
                        and corroboration['vp_margin']>0 and corroboration['rank_margin']<0))
        tiers = {}
        for tier in TIERS:
            part = [r for r in own if r['lineup']=='frozen-neural' and r['tier']==tier]
            effect = difference(grouped(part, 'win'), tier_targets[tier], gates, claims)
            tiers[tier] = dict(metrics=summary(part), equivalence=effect,
                passed=bool(effect['lower'] is not None and effect['lower']>=-gates.equivalence_margin
                            and effect['upper']<=gates.equivalence_margin))
        (members if kind=='member' else mixtures)[ident] = dict(easy_original_hard=easy, tiers=tiers)
    ordering = {}
    for lower, higher in zip(TIERS, TIERS[1:]):
        common = difference(tier_targets[higher], tier_targets[lower], gates, claims)
        direct = defaultdict(list); exceptions = defaultdict(list)
        for r in records:
            j = r['job']
            if r['failure'] is not None or j['lineup']!='adjacent-direct' or j['tier']!=higher: continue
            t = terminal_targets(r['result']); win = sum(t['win_share'][s] for s in j['higher_seats'])
            direct[j['setup_group']].append(win); exceptions[j['arm']['group']].append(win)
        paired = interval([np.mean(v) for v in direct.values()], gates, claims)
        ordering[higher+'>'+lower] = dict(common_pool=common, direct_two_vs_two=paired,
            group_direct_means={g:float(np.mean(v)) for g,v in exceptions.items()},
            passed=bool(common['lower'] is not None and common['lower']>gates.ordering_gain
                        and paired['lower'] is not None and paired['lower']>.5+gates.ordering_gain))
    diagnostics = {tier: {lineup: summary([r for r in rows if r['tier']==tier and r['lineup']==lineup])
                         for lineup in ('frozen-neural', 'homogeneous-style', 'mixed-styles', 'original-hard')}
                   for tier in TIERS}
    return dict(members=members, mixtures=mixtures, ordering=ordering, diagnostics=diagnostics,
        claims=claims, same_tier_target={t:interval(v.values(),gates,claims) for t,v in tier_targets.items()},
        checks=dict(every_easy_member=all(m['easy_original_hard']['passed'] for m in members.values()),
                    every_easy_mixture=all(m['easy_original_hard']['passed'] for m in mixtures.values()),
                    same_tier_consistency=all(t['passed'] for m in list(members.values())+list(mixtures.values()) for t in m['tiers'].values()),
                    tier_ordering=all(o['passed'] for o in ordering.values())))


RATING_FIELDS = ('understandable_plans', 'varied_openings', 'resilience', 'resource_interaction',
                 'enjoyment', 'lack_of_repetition', 'appropriate_interference', 'waiting_comfort')


def review_template(panel, value):
    return dict(format=REVIEW, panel_id=panel['id'], roster_id=value['id'], fixture=panel['fixture'],
                profiles_id=digest(panel['profiles']), hardware_label=panel['hardware_label'],
                scope='Independent completed human play sessions, not automatic or fixture ratings', sessions=[])


def human_review(panel, value, review):
    if review is None: return dict(passed=False, reason='Human play/enjoyment/coherence review pending')
    template = review_template(panel,value)
    require({k:v for k,v in review.items() if k!='sessions'} == {k:v for k,v in template.items() if k!='sessions'},
            'Human review model/panel/profile/hardware differs')
    ids = {g['id'] for g in value['archetypes']}; grouped_sessions = defaultdict(list); seen = set()
    for r in review['sessions']:
        require(set(r) == {'session_id','rater','archetype_id','tier','hint_mode','completed','players','tactically_sound','ratings','evidence'},
                'Invalid human session fields')
        require(isinstance(r['session_id'],str) and r['session_id'] and r['session_id'] not in seen
                and isinstance(r['rater'],str) and r['rater'].strip() and isinstance(r['evidence'],str) and r['evidence'].strip(),
                'Duplicate or unattributed human session')
        seen.add(r['session_id'])
        require(r['archetype_id'] in ids and r['tier'] in TIERS and r['hint_mode']==panel['profiles']['hint_mode']
                and r['completed'] is True and r['players']==4 and type(r['tactically_sound']) is bool,
                'Incomplete session or different hint mode')
        require(set(r['ratings'])==set(RATING_FIELDS) and all(type(v) is int and 1<=v<=5 for v in r['ratings'].values()), 'Rate every field from 1 to 5')
        grouped_sessions[(r['archetype_id'], r['tier'])].append(r)
    gates = Gates(**panel['gates']); results = {}
    for group in sorted(ids):
        for tier in TIERS:
            sessions = grouped_sessions[(group,tier)]
            means = {k:float(np.mean([s['ratings'][k] for s in sessions])) if sessions else None for k in RATING_FIELDS}
            raters = len({s['rater'] for s in sessions})
            results[group+'/'+tier] = dict(sessions=len(sessions), raters=raters, means=means,
                passed=raters>=gates.human_raters and all(s['tactically_sound'] for s in sessions)
                       and all(v is not None and v>=gates.human_floor for v in means.values()))
    return dict(passed=all(r['passed'] for r in results.values()), groups=results,
                limitation='Human supplied records; automatic tools cannot establish enjoyment or verify reviewer independence')


def public_packet(panel, value, records):
    # Outcome-independent, fixed member/seat/setup examples. Partial chunks do
    # not substitute a more favorable available record for a missing case.
    cases = set()
    for g in value['archetypes']:
        for tier in TIERS:
            for i, member in enumerate(g['members']):
                job = next(j for j in panel['jobs'] if j['lineup']=='frozen-neural' and j['tier']==tier
                           and j['arm']['kind']=='member' and j['arm']['id']==member
                           and j['seed']==panel['seeds'][i%len(panel['seeds'])] and j['seat']==i%4)
                cases.add(job['index'])
    examples = []
    for r in records:
        if r['job']['index'] not in cases or r['failure']: continue
        j = r['job']; b = r['behavior'][str(j['seat'])]
        examples.append(dict(id=digest(dict(panel=panel['id'],job=j['index'])), archetype_id=j['arm']['group'], tier=j['tier'],
            public_turns=b['public_turns'], public_boards=b['public_boards'],
            intent_display=[{k:x[k] for k in ('era','turn','text','uncertain')} for x in r['intent_display'] if x['seat']==j['seat']]))
    return dict(format='personality-difficulty-public-cases/1.0.0', fixture=panel['fixture'],
                hint_mode=panel['profiles']['hint_mode'], examples=examples,
                scope='Public executed plans, or explicitly opted-in broad own-intent hints; no internal variants or outcomes')


def evidence_checks(value, panel, records, seen, humans, metric_checks):
    completed = [r for r in records if not r['failure']]
    support = defaultdict(int)
    for r in completed:
        for turn in r['turns']:
            spec = r['job']['table'][turn['seat']]
            support[(spec['member'],spec['tier'])] += turn['completed']
    return dict(metric_checks, complete_panel=seen==set(range(len(panel['jobs']))),
        full_roster=value['full_size'], qualified_p4=value['p4_qualified'], real_artifacts=not panel['fixture'],
        fresh_release_panel=panel['stage']=='release',
        no_failed_games=bool(records) and all(r['failure'] is None and not r['baseline_failures'] for r in records),
        searched=not any(r['fixture_direct'] for r in records), both_eras=all(r['both_eras'] for r in completed),
        every_controller_searched=all(support[(m['id'],t)]>0 for m in value['members'] for t in TIERS),
        one_measured_environment=len({digest(r['environment']) for r in records})==1,
        human_play_review=humans['passed'])


def check(root, directory, chunks, output, *, review=None):
    root, out = owned(root, output); require(not out.exists(), 'Use a new immutable analysis directory')
    panel, value, stream, seen, sources = collection.read_chunks(root, directory, chunks)
    records = list(stream); require(records, 'No reserved P5 games supplied')
    metrics = automatic(value,panel,records)
    failures = [dict(index=r['job']['index'], **r['failure']) for r in records if r['failure']]
    humans = human_review(panel,value,read_json(project_path(root,review)) if review else None)
    checks = evidence_checks(value,panel,records,seen,humans,metrics.pop('checks'))
    qualified = bool(all(checks.values()))
    timing = dict(game_seconds=sum(r['seconds'] for r in records),
                  **{k:sum(t.get(k,0) for r in records for t in r['telemetry'].values())
                     for k in ('searches','completed','interrupted','fallbacks','seconds','prior_seconds','overruns')})
    result = sealed(dict(format=REPORT, contract=identity(), panel_id=panel['id'], roster_id=value['id'],
        fixture=panel['fixture'], stage=panel['stage'], profiles=panel['profiles'], gates=panel['gates'],
        tables=panel['tables'], hardware_label=panel['hardware_label'],
        panel=project_path(root,directory).relative_to(root).as_posix(),
        panel_sha256=sha(project_path(root,directory)/'manifest.json'),
        review_source=dict(path=project_path(root,review).relative_to(root).as_posix(),sha256=sha(project_path(root,review))) if review else None,
        checks=checks, p5_qualified=qualified, qualified_tiers=list(TIERS) if qualified else [], release_qualified=False,
        status='P5 qualified for the measured configuration; P6 pending' if qualified else
               ('FIXTURE — unqualified' if panel['fixture'] else 'Difficulty qualification pending'),
        games=len(records), expected_games=len(panel['jobs']), missing_jobs=sorted(set(range(len(panel['jobs'])))-seen),
        setup_groups=sorted({r['job']['setup_group'] for r in records}), sources=sources, failures=failures,
        stock_failures=[dict(index=r['job']['index'], failures=r['baseline_failures']) for r in records if r['baseline_failures']],
        baseline=panel['baseline'], timing=timing, environment=records[0]['environment'], human=humans, **metrics,
        limits='Multiplicity-adjusted setup-level Student-t intervals assume sufficiently regular independent setup means; '
               '32 groups is a minimum, not guaranteed power. Wide intervals remain unresolved. VP is corroborative, '
               'not capped/normalized. This is not laptop timing, human tournament strength or P6 deployment authorization.'))
    out.mkdir(parents=True); write_json(out/'report.json',result)
    write_json(out/'review-template.json',review_template(panel,value))
    write_json(out/'public-cases.json',public_packet(panel,value,records))
    paths = [project_path(root,directory)/'manifest.json']+[project_path(root,s['path'])/'manifest.json' for s in sources]
    if review: paths.append(project_path(root,review))
    write_json(out/'pins.json',dict(paths=[p.relative_to(root).as_posix() for p in paths]))
    return result


def load_report(root, directory):
    """Validate pinned evidence before exposing a supported-tier claim to a UI."""
    report=read_sealed(project_path(root,directory)/'report.json',REPORT)
    require(sha(project_path(root,report['panel'])/'manifest.json')==report['panel_sha256'], 'Report panel changed')
    if report['review_source']:
        r=report['review_source'];require(sha(project_path(root,r['path']))==r['sha256'],'Human review evidence changed')
    for source in report['sources']:
        require(sha(project_path(root,source['path'])/'manifest.json')==source['sha256'],'Qualification chunk changed')
    panel,value,records,seen,_=collection.read_chunks(root,report['panel'],[s['path'] for s in report['sources']])
    records=list(records)
    validate_report_summary(report,value)
    require(panel['id']==report['panel_id'] and len(records)==report['games']
            and len(panel['jobs'])==report['expected_games']
            and sorted(set(range(len(panel['jobs'])))-seen)==report['missing_jobs']
            and sorted({r['job']['setup_group'] for r in records})==report['setup_groups'], 'Report coverage differs')
    require(all(report[k]==panel[k] for k in ('fixture','stage','profiles','gates','tables','hardware_label','baseline')),
            'Report recipe differs')
    metrics=automatic(value,panel,records)
    humans=human_review(panel,value,read_json(project_path(root,report['review_source']['path'])) if report['review_source'] else None)
    require(report['checks']==evidence_checks(value,panel,records,seen,humans,metrics.pop('checks'))
            and report['human']==humans and all(report[k]==v for k,v in metrics.items()), 'Report results differ from pinned evidence')
    return report
