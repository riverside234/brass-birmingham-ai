"""Re-encode deployed controls, compare quality/variety, and record blind review."""
from collections import Counter, defaultdict
from dataclasses import asdict
import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.learning.replay import terminal_targets
from brass_ai.personality.repertoire import descriptor
from brass_ai.personality.repertoire.traces import FIELDS, trajectory
from brass_ai.recovery.retention import owned
from brass_ai.search.common import turn_key
from brass_ai.step5.league import write_json
from . import panels
from .contracts import REPORT, Gates, identity, sealed

ERAS = ('CanalEra', 'RailEra')
PUBLIC_ACTIONS = ('build', 'develop', 'bonus_develop', 'network', 'sell', 'loan', 'pass', 'scout')


def public_board(view, seat):
    """Allowlist the visible board for tactical review; never copy the raw view."""
    slots=[]
    for s in view['slots']:
        tile=s.get('tile');t=None
        if tile:
            t={k:tile[k] for k in ('type','level','resource','flipped')}
            t['relative_owner']=(tile['seat']-seat)%4
        slots.append(dict(location=s['id'],types=s['types'],tile=t))
    links=[dict(location=l['id'],regions=l['regions'],relative_owner=(l['owner']-seat)%4 if 'owner' in l else None)
           for l in view['links']]
    players=[dict(relative_seat=(p['seat']-seat)%4,cash=p['cash'],income=p['income'],
                  remaining_industries=[dict(industry=m['type'],levels=[t['level'] for t in m['tiles']]) for m in p['mat']])
             for p in view['players']]
    return dict(coal_market=view['coal_market'],iron_market=view['iron_market'],slots=slots,links=links,
                players=sorted(players,key=lambda p:p['relative_seat']),
                merchants=[{k:m[k] for k in ('id','types','beer','bonus') if k in m} for m in view['merchants']])


def public_behavior(game, seat, *, include_boards=False):
    """Canonical public plans, ignoring card/payment IDs and interchangeable subchoices."""
    turns = {era: {} for era in ERAS}; boards={era:{} for era in ERAS}; opportunities = 0; pottery_built = 0
    for r in game['records']:
        if r['seat'] != seat or r['view']['context']['phase'] == 'debt':
            continue
        c = r['choices'][r['chosen']]; kind = c['kind']; view = r['view']; key = turn_key(view['context'])
        opportunities += int(any(a['kind'] == 'build' and a.get('industry') == 'Pottery' for a in r['choices']))
        pottery_built += int(kind == 'build' and c.get('industry') == 'Pottery')
        if kind not in PUBLIC_ACTIONS:
            continue
        # Link/slot locations are meaningful public route differences. Merchant
        # payment IDs, discard choices, worker IDs and hidden cards are excluded.
        target = (c.get('target') or '') if kind in ('build','network','sell') else ''
        industry = c.get('industry') or ''
        level = 0
        if kind == 'build':
            mat = next(m['tiles'] for m in view['players'][seat]['mat'] if m['type'] == industry)
            level = mat[0]['level'] if mat else 0
        turns[key[0]].setdefault(key,[]).append((kind,industry,target,level))
        if include_boards and key not in boards[key[0]]:boards[key[0]][key]=public_board(view,seat)
    openings = {}; later = {}; summaries = {}
    for era in ERAS:
        # Preserve ordering of unlike actions (e.g. loan before build). Only
        # canonicalize adjacent choices of the same kind, such as the two links
        # of a rail action or the order of industries in a group sale.
        sequence=[]
        for actions in turns[era].values():
            runs=[]
            for action in actions:
                if not runs or runs[-1][0][0]!=action[0]:runs.append([])
                runs[-1].append(action)
            sequence.append(tuple(tuple(sorted(run)) for run in runs))
        openings[era] = digest(sequence[:2])
        later[era] = digest(sequence[2:])
        summaries[era] = [[list(a) for a in actions] for actions in turns[era].values()]
    return dict(openings=openings, later=later, public_turns=summaries,
                public_boards={e:list(boards[e].values()) for e in ERAS},
                pottery_opportunities=opportunities, pottery_built=pottery_built,
                both_eras=all(turns.values()))


def interval(values):
    """Setup-clustered preliminary interval; P5 owns multiplicity-adjusted release."""
    x = np.asarray(values,dtype=float)
    mean = float(x.mean()) if len(x) else 0.
    half = float(1.96*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None
    return dict(mean=mean, lower=mean-half if half is not None else None,
                upper=mean+half if half is not None else None, setups=len(x))


def setup_means(rows, field):
    grouped = defaultdict(list)
    for r in rows: grouped[r['setup']].append(r['quality'][field])
    return {k: float(np.mean(v)) for k,v in grouped.items()}


def quality(rows, control, gates):
    report = {}
    for field in ('win_share','vp','placement'):
        a,b = setup_means(rows,field),setup_means(control,field); common=sorted(a.keys() & b.keys())
        report[field] = dict(absolute=interval([a[k] for k in common]), delta=interval([a[k]-b[k] for k in common]))
    w,v,r = (report[k] for k in ('win_share','vp','placement'))
    n=w['delta']['setups']
    report['passed'] = bool(n>=gates.min_setups and w['absolute']['mean']>=gates.win_floor
        and w['delta']['lower']>=-gates.win_tolerance and v['delta']['lower']>=-gates.vp_tolerance
        and r['delta']['upper']<=gates.rank_tolerance)
    return report


def diversity(signatures):
    n=len(signatures)
    return 1-sum(v*(v-1) for v in Counter(signatures).values())/(n*(n-1)) if n>1 else 0.


def variety(mixture, representative, gates):
    result={}
    for period in ('openings','later'):
        for era in ERAS:
            def by_setup(rows):
                grouped=defaultdict(list)
                for r in rows: grouped[r['setup']].append(r['behavior'][period][era])
                return {k:diversity(v) for k,v in grouped.items()}
            a,b=by_setup(mixture),by_setup(representative);common=sorted(a.keys() & b.keys())
            effect=interval([a[k]-b[k] for k in common])
            # Later behavior may preserve competence with different openings;
            # report both, require at least one clear gain in EACH era.
            result[period+'/'+era]=dict(effect,passed=bool(len(common)>=gates.min_setups
                and effect['lower'] is not None and effect['lower']>0 and effect['mean']>=gates.variety_gain))
    result['passed']=all(any(result[p+'/'+era]['passed'] for p in ('openings','later')) for era in ERAS)
    return result


def repeatability(points, labels):
    groups={k:points[np.array(labels)==k] for k in sorted(set(labels))}
    centers=np.asarray([v.mean(0) for v in groups.values()])
    within=float(np.mean([np.square(v-v.mean(0)).sum(-1).mean() for v in groups.values()]))
    between=float(np.var(centers,axis=0).sum())
    return between/max(within,1e-8)


def search_metrics(rows):
    totals={key:sum(r['telemetry'].get(key,0) for r in rows)
            for key in ('searches','completed','interrupted','fallbacks','seconds','prior_seconds','overruns')}
    totals['simulations_per_search_second']=totals['completed']/max(totals['seconds'],1e-8)
    totals['mean_game_seconds']=float(np.mean([r['seconds'] for r in rows])) if rows else 0.
    return totals


def geometry(value, rows, points, gates):
    members={m['id']:m for m in value['members']}; groups=value['archetypes']
    ids=[g['id'] for g in groups]
    prototypes=np.array([members[g['representative']]['centroid'] for g in groups])
    qualified={};recognition={};member_points={}
    for mid in members:
        ix=[i for i,r in enumerate(rows) if r['arm']['role']=='member' and r['arm']['member']==mid]
        z=points[ix]; member_points[mid]=z
    for g in groups:
        z=np.concatenate([member_points[mid] for mid in g['members']])
        index=ids.index(g['id']);pred=np.argmin(np.square(z[:,None]-prototypes).sum(-1),axis=1)
        recognition[g['id']]=float(np.mean(pred==index))
        representative=member_points[g['representative']]; center=representative.mean(0)
        for mid in g['members']:
            z=member_points[mid]; centroid=z.mean(0);dispersion=float(np.sqrt(np.square(z-centroid).sum(-1).mean()))
            own=float(np.linalg.norm(centroid-center))
            others=[member_points[h['representative']].mean(0) for h in groups if h['id']!=g['id']]
            other=float(min(np.linalg.norm(centroid-h) for h in others))
            same_theme=own+dispersion<=g['neighborhood']['radius'] and own+gates.separation_margin*g['neighborhood']['nearest_other']<other
            distinct=True
            for neighbor in g['members']:
                if neighbor==mid: continue
                nz=member_points[neighbor];nc=nz.mean(0);nd=float(np.sqrt(np.square(nz-nc).sum(-1).mean()))
                distinct &= np.linalg.norm(centroid-nc)>max(1e-6,gates.variant_signal*np.hypot(dispersion,nd))
            qualified[mid]=dict(centroid=centroid.tolist(),dispersion=dispersion,same_theme=bool(same_theme),
                                 distinct=bool(distinct),passed=bool(same_theme and distinct))
    ix=[i for i,r in enumerate(rows) if r['arm']['role']=='member']
    labels=[rows[i]['arm']['group'] for i in ix]
    ratio=repeatability(points[ix],labels)
    pilot_ix=[i for i,r in enumerate(rows) if r['arm']['role']=='control' and r['arm']['control']!='neutral']
    pilot=repeatability(points[pilot_ix],[rows[i]['arm']['control'] for i in pilot_ix])
    return dict(members=qualified,recognition=recognition,between_to_within=ratio,pilot_between_to_within=pilot,
                passed=bool(all(v['passed'] for v in qualified.values()) and min(recognition.values())>=gates.recognition_floor and ratio>pilot),
                recognition_scope='Held-out nearest discovery representative; human recognition is checked separately')


def blind_jobs(value, jobs, gates):
    chosen=[]
    for g in value['archetypes']:
        # Cover every member, then balance setups and seats. Taking each member's
        # first job previously chose only the first deal and seat zero.
        candidates=[j for j in jobs if j['p4']['role']=='member' and j['p4']['group']==g['id']]
        candidates.sort(key=lambda j:j['index'])
        selected=[];setups=Counter();seats=Counter();members=Counter()
        def choose(part):
            best=min(part,key=lambda j:(setups[j['seed']],seats[j['neural_seats'][0]],members[j['p4']['member']],j['index']))
            selected.append(best);setups[best['seed']]+=1;seats[best['neural_seats'][0]]+=1;members[best['p4']['member']]+=1
        for mid in g['members']:
            part=[j for j in candidates if j['p4']['member']==mid]
            if part:choose(part)
        while len(selected)<max(gates.human_examples,len(g['members'])):
            remaining=[j for j in candidates if j not in selected]
            if not remaining:break
            choose(remaining)
        chosen.extend(j['index'] for j in selected)
    return set(chosen)


def blind_packet(value, panel, rows, gates):
    # Partial chunks never replace missing prespecified review cases with other
    # available games. Selection is independent of chunk order and outcomes.
    selected=blind_jobs(value,panel['jobs'],gates)
    examples=[dict(id=digest(dict(panel=panel['id'],job=r['index'])), public_turns=r['behavior']['public_turns'],
                   public_boards=r['behavior'].get('public_boards',{})) for r in rows if r['index'] in selected]
    examples.sort(key=lambda e:e['id'])
    return sealed(dict(format='personality-blind-packet/1.0.0',contract=identity(),panel_id=panel['id'],
                       examples=examples,scope='Public plans and visible opportunities only; no model/style identity, outcome, hidden cards or descriptor coordinates'))


def human_review(value,panel,packet,rows,review,gates):
    if review is None: return dict(passed=False,reason='Blind recognition/tactical review not supplied')
    require(review['format']=='personality-blind-review/1.0.0' and review['panel_id']==panel['id']
            and review['packet_id']==packet['id'], 'Blind review belongs to another panel/packet')
    ids={g['id'] for g in value['archetypes']};labels=review['labels']
    require(panel['labels'] is not None and labels==panel['labels'],'Recognition names must be frozen before calibration')
    require(len(labels)==len(ids) and {l['archetype_id'] for l in labels}==ids
            and all(isinstance(l['name'],str) and l['name'].strip() and isinstance(l['description'],str)
                    and l['description'].strip() for l in labels)
            and len({l['name'].casefold().strip() for l in labels})==len(ids), 'Supply distinct evidence-based names/descriptions')
    expected={e['id'] for e in packet['examples']}
    truth={digest(dict(panel=panel['id'],job=r['index'])):r['arm']['group'] for r in rows if r['arm']['role']=='member'}
    ratings=defaultdict(list);seen=set()
    for r in review['ratings']:
        require(r['example'] in expected and isinstance(r['rater'],str) and r['rater'].strip()
                and r['guess'] in ids and type(r['tactically_sound']) is bool, 'Invalid blind response')
        key=(r['example'],r['rater']);require(key not in seen,'Duplicate blind rating');seen.add(key);ratings[r['example']].append(r)
    complete=all(len(ratings[e])>=gates.human_raters for e in expected)
    results={}
    for group in sorted(ids):
        examples=[e for e in expected if truth[e]==group]
        scored=[float(np.mean([r['guess']==group for r in ratings[e]])) for e in examples if ratings[e]]
        accuracy=float(np.mean(scored)) if scored else 0.
        sound=all(r['tactically_sound'] for e in examples for r in ratings[e])
        results[group]=dict(examples=len(scored),accuracy=accuracy,tactically_sound=sound,
                            passed=complete and len(scored)>=gates.human_examples and accuracy>=gates.recognition_floor and sound)
    return dict(passed=all(r['passed'] for r in results.values()),groups=results,labels=labels,
                scope='Small prespecified blind P4 screen; player enjoyment and tier certification remain P5')


def check(root, panel_directory, chunks, output, *, review=None, device='cpu'):
    root,out=owned(root,output);require(not out.exists(),'Use a new immutable check directory')
    panel,value,pairs,seen,sources=panels.read_chunks(root,panel_directory,chunks)
    enc,payload=descriptor.load(project_path(root,value['encoder']['path']))
    require(enc['id']==value['encoder']['id'],'Requalification descriptor changed')
    gates=Gates(**value['gates']).validate()
    selected_for_review=blind_jobs(value,panel['jobs'],gates)
    rows=[];features=[]
    for game,side in pairs:
        job=side['job'];seat=job['neural_seats'][0];x,mask=trajectory(game,seat)
        targets=terminal_targets(game['result'])
        rows.append(dict(index=job['index'],arm=job['p4'],setup=game['setup_group'],seat=seat,
                         quality={k:targets[k][seat] for k in ('win_share','placement','vp')},
                         behavior=public_behavior(game,seat,include_boards=job['index'] in selected_for_review),seconds=side['seconds'],telemetry=side['telemetry']))
        features.append(dict(values=x,mask=mask))
    require(rows,'No completed P4 calibration games')
    complete=seen==set(range(len(panel['jobs'])))
    neutral=[r for r in rows if r['arm']['role']=='control' and r['arm']['control']=='neutral']
    quality_by_member={m['id']:quality([r for r in rows if r['arm']['role']=='member' and r['arm']['member']==m['id']],neutral,gates)
                       for m in value['members']}
    mixtures={g['id']:quality([r for r in rows if r['arm']['role']=='mixture' and r['arm']['group']==g['id']],neutral,gates)
              for g in value['archetypes']}
    variations={g['id']:variety([r for r in rows if r['arm']['role']=='mixture' and r['arm']['group']==g['id']],
                               [r for r in rows if r['arm']['role']=='representative' and r['arm']['group']==g['id']],gates)
                for g in value['archetypes']}
    geom=geometry(value,rows,descriptor.transform(payload,features,device=device),gates) if complete else dict(passed=False,reason='Incomplete panel')
    packet=blind_packet(value,panel,rows,gates)
    review_data=read_json(project_path(root,review)) if review else None
    human=human_review(value,panel,packet,rows,review_data,gates)
    member_rows=[r for r in rows if r['arm']['role']=='member']
    throughput={m['id']:search_metrics([r for r in member_rows if r['arm']['member']==m['id']]) for m in value['members']}
    pottery=dict(opportunity_decisions=sum(r['behavior']['pottery_opportunities'] for r in member_rows),
                 builds=sum(r['behavior']['pottery_built'] for r in member_rows))
    pottery['flips']=int(sum(f['values'][f['mask'],FIELDS.index('flips_Pottery')].sum()
                             for f,r in zip(features,rows) if r['arm']['role']=='member'))
    # An opportunity-conditioned coverage gate, NOT a per-style action quota or reward.
    pottery['passed']=pottery['opportunity_decisions']>0 and pottery['builds']>0 and pottery['flips']>0
    tests=dict(complete_panel=complete,full_size=value['full_size'],searched=value['decision_mode']=='searched',
               both_eras=all(r['behavior']['both_eras'] for r in rows),
               individual_quality=all(r['passed'] for r in quality_by_member.values()),
               mixture_quality=all(r['passed'] for r in mixtures.values()),
               search_support=all(r['completed']>0 for r in throughput.values()),
               coherent_distinct_members=geom['passed'],replayability=all(v['passed'] for v in variations.values()),
               pottery_coverage=pottery['passed'],blind_recognition_and_tactics=human['passed'])
    result=sealed(dict(format=REPORT,contract=identity(),panel_id=panel['id'],roster_id=value['id'],gates=asdict(gates),
                       games=len(rows),expected_games=len(panel['jobs']),missing_jobs=sorted(set(range(len(panel['jobs']))) - seen),
                       sources=sources,checks=tests,p4_qualified=all(tests.values()),release_qualified=False,
                       members=quality_by_member,mixtures=mixtures,variety=variations,geometry=geom,pottery=pottery,human=human,
                       timing=dict(game_seconds=sum(r['seconds'] for r in rows),members=throughput,
                                   neutral=search_metrics(neutral)),
                       scope='P4 calibration at this exact shared model/search recipe. Preliminary grouped intervals; P5 must calibrate tiers and multiplicity-adjusted release gates.'))
    out.mkdir(parents=True);write_json(out/'report.json',result);write_json(out/'blind-packet.json',packet)
    template=dict(format='personality-blind-review/1.0.0',panel_id=panel['id'],packet_id=packet['id'],
                  labels=panel['labels'] or [dict(archetype_id=g['id'],name='',description='') for g in value['archetypes']],
                  ratings=[dict(example=e['id'],rater='',guess='',tactically_sound=None) for e in packet['examples']])
    write_json(out/'review-template.json',template)
    updated={k:v for k,v in value.items() if k!='id'}
    updated.update(p4_qualified=result['p4_qualified'],qualification=dict(report=(out/'report.json').relative_to(root).as_posix(),sha256=sha(out/'report.json'),
                   source_roster=value['id'],panel_id=panel['id']))
    if review_data:
        names={l['archetype_id']:l for l in review_data['labels']}
        updated['archetypes']=[dict(g,name=names[g['id']]['name'],description=names[g['id']]['description']) for g in value['archetypes']]
    updated['members']=[dict(m,qualification='P4-qualified' if result['p4_qualified'] else 'P4-unresolved') for m in value['members']]
    updated['forbidden_groups']=sorted(set(value['forbidden_groups'])|{r['setup'] for r in rows})
    pins=[project_path(root,panel_directory)/'manifest.json',out/'report.json']+[project_path(root,s['path'])/'manifest.json' for s in sources]
    if review: pins.append(project_path(root,review))
    updated['dependencies']=value['dependencies']+[dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in pins]
    write_json(out/'manifest.json',sealed(updated))
    write_json(out/'pins.json',dict(paths=[d['path'] for d in updated['dependencies']]))
    return result
