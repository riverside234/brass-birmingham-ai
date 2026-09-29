"""Predeclared four-player qualification, clustered uncertainty and fail-closed pools.

Reports do not turn failed games into wins. A small reliable tournament can admit
a labelled niche training opponent; it cannot establish human enjoyment or a
general improved-Hard release. Timing here is the headless whole turn, not UI IPC.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import statistics
import time
from pathlib import Path
from .bank import FrozenOpponent, digest, load_bank, read, require, sha

STYLES = ('balanced', 'rail-tempo', 'cotton', 'goods-beer', 'supplier', 'pottery-flex')


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def make_spec(root, bank):
    manifest = load_bank(root, bank)
    generation = 'v3' if all(s+'-v3' in manifest['payload']['profiles'] for s in STYLES) else 'v2'
    main_profiles = [s+'-'+generation for s in STYLES]
    pair = [] if generation == 'v3' else ['balanced-s2-v2', 'balanced-s1-10s-v2']
    jobs = []
    def add(stage, seed, table, target=None):
        data = dict(stage=stage, seed=seed, table=table, target=target)
        jobs.append(dict(id=digest(data)[:20], **data))
    for profile in main_profiles+pair:
        for seed in (range(91001,91005) if generation == 'v3' else range(31001,31005)):
            for seat in range(4):
                table = [None]*4; table[seat] = profile
                add('guarded', seed, table, seat)
    tables = [['balanced-v2','cotton-v2','supplier-v2','pottery-flex-v2'],
              ['rail-tempo-v2','goods-beer-v2','balanced-v2','pottery-flex-v2']]
    if generation == 'v3': tables = [[n.replace('-v2','-v3') for n in table] for table in tables]
    for i, table in enumerate(tables):
        for arrangement in (table, list(reversed(table))):
            for rotation in range(4):
                add('mixed', (92001 if generation == 'v3' else 32001)+i, arrangement[rotation:]+arrangement[:rotation])
    for style in STYLES:
        add('identical', 93001 if generation == 'v3' else 33001, [style+'-'+generation]*4)
        if generation == 'v2': add('identical-s2', 33002, [style+'-s2-v2']*4)
    require(all(p in manifest['payload']['profiles'] for j in jobs for p in j['table'] if p), 'Spec profile missing')
    payload = dict(format='legacy-qualification-spec/1.0.0', bank=bank, player_count=4,
        bank_manifest_sha256=sha(Path(root)/'training/opponents/bank'/bank/'manifest.json'), jobs=jobs,
        stopping='Run every prespecified job once. No early strength stopping or retuning. Resume only missing jobs.',
        concurrency=6 if generation=='v3' else 3, bootstrap_seed=731, bootstrap_samples=10000,
        admission=dict(min_guarded_games=16, min_mixed_games=4, min_identical_games=1,
            maximum_failures=0, maximum_economic_stalls=0, minimum_guarded_mean_rank=2.5,
            held_out_family='pottery-flex', general_hard_min_setup_groups=30),
        release_scope='Headless niche benchmark/training admission. Human enjoyment, UI latency, difficulty calibration and Linux qualification are separate.',
        generation=generation, equal_budget_pair=pair)
    return dict(id='qualification-'+digest(payload)[:20], payload=payload)


def run_spec(root, spec, output, *, mono=None):
    root=Path(root).resolve(); output=Path(output); output.mkdir(parents=True, exist_ok=True)
    require(spec['id']=='qualification-'+digest(spec['payload'])[:20], 'Evaluation spec changed')
    p=spec['payload']; bank=p['bank']; manifest=load_bank(root,bank)
    require(sha(root/'training/opponents/bank'/bank/'manifest.json')==p['bank_manifest_sha256'], 'Bank changed after preregistration')
    path=output/'spec.json'
    if path.exists(): require(read(path)==spec,'Cannot resume a different evaluation')
    else: write_new(path,spec)
    def execute(job):
        path=output/(job['id']+'.json')
        if path.exists():
            record=read(path);require(record['job']==job and record['spec_id']==spec['id'],'Mismatched completed job')
            return record
        table=[manifest['payload']['profiles'][n] if n else None for n in job['table']]
        profile=next(x for x in job['table'] if x)
        began=time.perf_counter(); opponent=None
        try:
            opponent=FrozenOpponent(root,bank+'/'+profile,purpose='diagnostic',mono=mono)
            match=opponent.worker.call('run',seed=job['seed'],player_count=4,profile=opponent.profile,
                table_profiles=table,fixed_work=opponent.fixed_work,record_selectors=True)
        except Exception as error:
            match=dict(completed=False,failure=str(error),trace=[],turn_reports=[],result=None)
        finally:
            if opponent is not None: opponent.close()
        record=dict(spec_id=spec['id'],job=job,elapsed_seconds=time.perf_counter()-began,match=match)
        write_new(path,record)
        return record
    pending=list(p['jobs']);random.Random(9141).shuffle(pending)
    with ThreadPoolExecutor(max_workers=p['concurrency']) as executor:
        for future in as_completed([executor.submit(execute,j) for j in pending]):
            r=future.result();m=r['match']
            print(json.dumps(dict(job=r['job']['id'],stage=r['job']['stage'],table=r['job']['table'],
                completed=m['completed'],failure=m.get('failure'),seconds=round(r['elapsed_seconds'],2))),flush=True)


def grouped_interval(rows, field, *, seed=731, samples=10000):
    """Equal-weight setup clusters; all rotations stay in their sampled cluster."""
    groups=defaultdict(list)
    for r in rows: groups[r['seed']].append(r[field])
    means=[statistics.mean(v) for _,v in sorted(groups.items())]
    if len(means)<2: return None
    rng=random.Random(seed)
    draws=sorted(statistics.mean(rng.choices(means,k=len(means))) for _ in range(samples))
    return [draws[int(.025*samples)],draws[min(samples-1,int(.975*samples))]]


def percentile(values,q):
    return sorted(values)[min(len(values)-1,int(q*(len(values)-1)))] if values else None


def summarize(root,spec,output,checks):
    root=Path(root).resolve(); output=Path(output); p=spec['payload']; bank=p['bank']
    require(spec['id']=='qualification-'+digest(p)[:20], 'Evaluation spec changed')
    require(checks.get('passed') is True and checks.get('bank')==bank, 'Correctness checks must match this frozen bank')
    records=[]; evidence={}
    profiles=load_bank(root,bank)['payload']['profiles']
    for job in p['jobs']:
        path=output/(job['id']+'.json');r=read(path)
        require(r['job']==job and r['spec_id']==spec['id'],'Evaluation record mismatch')
        match=r['match']
        if match['completed'] and not match.get('failure'):
            expected=[profiles[n]['id'] if n else 'stock-hard-guarded-1.1.0' for n in job['table']]
            require(match['table_profiles']==expected and match['seed']==job['seed'] and match['player_count']==4,'Match did not execute the prescribed table/setup')
            require(match['turns']==64 and match['actions']==124,'Incomplete four-player lifecycle')
            require(match['result']['objective']=='win-share/1.0.0','Unexpected match objective')
        records.append(r);evidence[path.name]=sha(path)
    stats={}
    for name in profiles:
        outcomes=[]; failures=[]; timings=[]; mode_counts=Counter(); reply_failures=0
        horizons=Counter(); cash=[]; builds=Counter(); opening=Counter(); recovery=Counter()
        stages=Counter(); games=0; economic_stalls=[]
        for r in records:
            j=r['job'];m=r['match'];seats=[s for s,n in enumerate(j['table']) if n==name]
            if not seats:continue
            games+=1;stages[j['stage']]+=1
            if not m['completed'] or m.get('failure'):
                failures.append(dict(job=j['id'],failure=m.get('failure') or 'incomplete'));continue
            results=m['result']['seats']
            for s in seats:
                result=results[s];cash.append(result['cash'])
                if result['vp']==0 and result['cash']==0 and result['income'] < -7:
                    economic_stalls.append(dict(job=j['id'],seat=s,seed=j['seed']))
                outcomes.append(dict(stage=j['stage'],seed=j['seed'],seat=s,win=result['utility'],rank=result['rank'],vp=result['vp'],
                    margin=result['vp']-statistics.mean(x['vp'] for x in results if x['seat']!=s)))
            timings.extend(t['whole_turn_ms'] for t in m['turn_reports'] if t['seat'] in seats)
            for t in m['trace']:
                if t.get('seat') not in seats:continue
                d=t.get('decision') or {};replies=d.get('replies') or {}
                if replies:
                    mode_counts[replies['status']]+=1;reply_failures+=replies.get('failures',0)
                    horizons.update(map(str,replies['intervening_turns']))
                intent=(d.get('selected') or {}).get('intent') or {}
                if intent: recovery[intent.get('milestone','unknown')]+=1
                for choice in t.get('choices',[]):
                    parts=choice.split('|')
                    if parts[0]=='build':builds[parts[2]]+=1
                if t.get('context',{}).get('era')=='CanalEra' and t.get('context',{}).get('round')==1:
                    opening[';'.join(t.get('choices',[]))]+=1
        guarded=[r for r in outcomes if r['stage']=='guarded']
        win=statistics.mean(r['win'] for r in guarded) if guarded else None
        rank=statistics.mean(r['rank'] for r in guarded) if guarded else None
        interval=grouped_interval(guarded,'win',seed=p['bootstrap_seed'],samples=p['bootstrap_samples'])
        gates=p['admission']; reliable=not failures and not reply_failures
        niche=(reliable and len(guarded)>=gates['min_guarded_games'] and stages['mixed']>=gates['min_mixed_games']
               and stages['identical']>=gates['min_identical_games'] and rank<gates['minimum_guarded_mean_rank'])
        if 'maximum_economic_stalls' in gates:
            niche = niche and len(economic_stalls)<=gates['maximum_economic_stalls']
        heldout=profiles[name]['tactic']==gates['held_out_family']
        stats[name]=dict(policy_id=bank+'/'+name,games=games,stages=dict(stages),completed_games=games-len(failures),failures=failures,
            guarded=dict(games=len(guarded),setup_groups=len({r['seed'] for r in guarded}),win_share=win,win_share_setup_bootstrap_95=interval,
                mean_rank=rank,mean_vp=statistics.mean(r['vp'] for r in guarded) if guarded else None,
                mean_score_margin=statistics.mean(r['margin'] for r in guarded) if guarded else None),
            headless_whole_turn_ms=dict(p50=percentile(timings,.5),p95=percentile(timings,.95),maximum=max(timings,default=None)),
            reply_status=dict(mode_counts),reply_failures=reply_failures,reply_horizons=dict(horizons),
            built_industries=dict(builds),opening_signatures=len(opening),intent_milestones=dict(recovery),mean_terminal_cash=statistics.mean(cash) if cash else None,
            economic_stalls=economic_stalls,reliable_in_observed_games=reliable,niche_qualified=niche,training_eligible=niche and not heldout,
            evaluation_eligible=niche and heldout,general_hard_eligible=bool(niche and interval and interval[0]>.25 and len({r['seed'] for r in guarded})>=gates['general_hard_min_setup_groups']),
            live_deployment_eligible=False,linux_qualified=False,held_out_family=heldout)
    comparison=dict(profiles=[],benefit_demonstrated=False,complete=False,scope='S1 v3 qualification; prior S2 bank unchanged')
    if p['equal_budget_pair']:
        first,second=p['equal_budget_pair'];paired=[]
        require(profiles[first]['milliseconds']==profiles[second]['milliseconds'],'Unequal whole-turn caps in S2 comparison')
        by_key={}
        for r in records:
            j=r['job'];m=r['match']
            if j['stage']!='guarded' or not m['completed'] or m.get('failure'):continue
            s=j['target'];name=j['table'][s]
            if name in (first,second):by_key[(name,j['seed'],s)]=m['result']['seats'][s]['utility']
        for name,seed,s in by_key:
            if name==first and (second,seed,s) in by_key:paired.append(dict(seed=seed,delta=by_key[(first,seed,s)]-by_key[(second,seed,s)]))
        interval=grouped_interval(paired,'delta',seed=p['bootstrap_seed'],samples=p['bootstrap_samples'])
        pair_complete=len(paired)==16 and stats[first]['reliable_in_observed_games'] and stats[second]['reliable_in_observed_games']
        comparison=dict(profiles=[first,second],milliseconds=profiles[first]['milliseconds'],paired_games=len(paired),
            win_share_delta=statistics.mean(r['delta'] for r in paired) if paired else None,setup_bootstrap_95=interval,
            benefit_demonstrated=bool(pair_complete and interval and interval[0]>0),complete=pair_complete)
    return dict(format='legacy-qualification/1.0.0',spec_id=spec['id'],bank=bank,player_count=4,checks=checks,
        scope_amendment=p.get('amendment'),complete_original_study='amendment' not in p,
        spec_sha256=digest(spec),raw_evidence=evidence,profiles=stats,total_games=len(records),
        failed_games=sum(not r['match']['completed'] or bool(r['match'].get('failure')) for r in records),
        reply_comparison=comparison,
        limits=([('User-requested limited validation; '+str(len(records))+' completed games retained. Coverage differs by profile; no full-study or calibrated-strength claim.')]
            if 'amendment' in p else ['Four held-out setup groups per main profile: wide uncertainty and no general improved-Hard label.'])+[
            'Concurrent headless execution: timing is not a 135U foreground Unity latency certificate.',
            'Rollout heuristics may mispredict people; human enjoyment and equal-difficulty personalities remain untested.',
            'Specialist S2 profiles have identical-table coverage only; no bank admission without the full qualification matrix.'])


def publish_pool(root,report,report_path):
    """The report is immutable; the index may append new identities, never retarget one."""
    root=Path(root).resolve();report_path=Path(report_path).resolve()
    require(report_path.is_relative_to(root),'Qualification report must be in project')
    require(read(report_path)==report,'Saved qualification differs')
    pool_path=root/'training/opponents/pool.json'
    pool=read(pool_path) if pool_path.exists() else dict(format='legacy-pools/1.0.0',training={},evaluation={})
    for name,result in report['profiles'].items():
        for purpose in ('training','evaluation'):
            if not result[purpose+'_eligible']:continue
            entry=dict(qualification=report_path.relative_to(root).as_posix(),sha256=sha(report_path),profile=name)
            identity=result['policy_id']
            require(identity not in pool[purpose] or pool[purpose][identity]==entry,'Cannot change an admitted identity')
            require(identity not in pool['evaluation' if purpose=='training' else 'training'],'Training and evaluation pools must be disjoint')
            pool[purpose][identity]=entry
    pool_path.parent.mkdir(parents=True,exist_ok=True)
    pool_path.write_text(json.dumps(pool,indent=2)+'\n',encoding='utf-8')
    return pool
