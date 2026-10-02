"""Small reproducible decision panel and 20-game, seat-rotated strength screen.

The control is the immutable pre-repair bank, never the freshly compiled worker.
Only detached observations enter a planner. Match hosts and result files are private.
"""
import argparse
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import random
from itertools import product
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from brass_ai.bank import FrozenOpponent, read
from brass_ai.host import RulesHost
from brass_ai.legacy import LegacyWorker
from brass_ai.step5.league import write_json

CONTROL = 'bank-927c8ddec148897fa542'
STYLES = ('balanced', 'rail-tempo', 'cotton', 'goods-beer', 'supplier', 'pottery-flex')


def positions(count):
    paths = sorted((ROOT / 'artifacts/masked-exit').glob('integration-*/fresh-run/iteration-0001/*.jsonl.gz'))
    rows = []
    sources = []
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            next(stream)
            for line in stream:
                game = json.loads(line)
                rows.extend(r for r in game['records'] if r['view']['context']['phase'] == 'action' and
                            r['view']['context']['actions_remaining'] == 2)
        sources.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(rows) >= 60: break
    if not rows: raise ValueError('No saved complete-game observations available for the decision panel')
    # Cover each era/round first, then fill across cash and sale opportunities.
    chosen = []
    groups = {}
    for row in rows:
        c = row['view']['context']
        groups.setdefault((c['era'], c['round']), []).append(row)
    for group in groups.values():
        chosen.append(min(group, key=lambda r: next(p['cash'] for p in r['view']['players'] if p['seat'] == r['seat'])))
    for row in rows:
        if len(chosen) >= count: break
        if all(row['step'] != r['step'] or row['view'] != r['view'] for r in chosen): chosen.append(row)
    return chosen[:count], sources


def diagnose(out, count):
    rows, sources = positions(count)
    results = []
    config = read(ROOT / 'configs/legacy-ai/balanced-v3-masked-strength.json')
    with LegacyWorker(ROOT) as candidate, closing(FrozenOpponent(ROOT, CONTROL + '/balanced-v3-masked', purpose='diagnostic')) as control:
        for i, row in enumerate(rows):
            variants = [('control', control, control.profile), ('combined', candidate, config)]
            if i in (0, 5, 9, len(rows)-1):
                variants += [('unseen-first', candidate, dict(config, selection_policy='unseen_first')),
                             ('broad-k', candidate, dict(config, root_top_k=16, interior_top_k=8)),
                             ('one-reply', candidate, dict(config, opponent_turn_limit=1)),
                             ('raw-q', candidate, dict(config, q_transform='none')),
                             ('incumbent-only', candidate, dict(config, incumbent_only=True))]
            for name, worker, profile in variants:
                d = getattr(worker, 'worker', worker).call('plan', observation=row['view'], choices=row['choices'], profile=profile,
                                fixed_work=False, sampling_seed=411)
                s = d['search']; edges = s['root'].values()
                results.append(dict(position=i, era=row['view']['context']['era'], round=row['view']['context']['round'],
                    seat=row['seat'], variant=name, elapsed_ms=d['elapsed_ms'], failures=d['failures'],
                    completed=s['completed'], revisited_edges=sum(e['visits'] >= 2 for e in edges),
                    selected=d['selected']['signature'], score=d['selected']['score'], search=s))
            write_json(out / 'decisions.json', dict(format='masked-strength-decisions/1.1.0', sources=sources,runtime=read(ROOT/'artifacts/legacy-ai/runtime/runtime.json'),positions=len(rows), records=results))
            print(json.dumps(dict(position=i, combined=next(r['completed'] for r in results if r['position']==i and r['variant']=='combined'))), flush=True)
    return results


def metrics(records, result):
    seats = {str(s): dict(cash_distress=0, unused_canal_l1=0, unflipped_endgame=0,
                        merchant_beer=0, completed=0, revisited_plans=0, fallbacks=0, turns=0,
                        industry_opportunities={}, industries={}) for s in range(4)}
    for row in records:
        m = seats[str(row['seat'])]; view = row['view']; c = view['context']
        m['cash_distress'] += view['players'][row['seat']]['debt'] > 0
        for kind in {x['industry'] for x in row['choices'] if x['kind'] == 'build'}:
            m['industry_opportunities'][kind] = m['industry_opportunities'].get(kind, 0) + 1
        for action in row['plan']['actions']:
            for identity in action:
                parts = identity.split('|')
                if parts[0] == 'beer' and parts[1].startswith('merchant:'): m['merchant_beer'] += 1
            if action[0].startswith('build|'):
                kind = action[0].split('|')[2]; m['industries'][kind] = m['industries'].get(kind, 0) + 1
        stats = row['search']; m['completed'] += stats['completed']; m['revisited_plans'] += stats.get('revisited_plans', 0)
        m['fallbacks'] += stats.get('fallback', stats['completed'] == 0); m['turns'] += 1
        if c['round'] == 8:
            # Count immediately AFTER this seat's final turn, before removal/scoring.
            for slot in row['after']['slots']:
                t = slot.get('tile')
                if t and t['seat'] == row['seat'] and not t['flipped']:
                    if c['era'] == 'CanalEra' and t['level'] == 1: m['unused_canal_l1'] += 1
                    if c['era'] == 'RailEra': m['unflipped_endgame'] += 1
    for s in result['seats']: seats[str(s['seat'])].update(vp=s['vp'], rank=s['rank'], utility=s['utility'])
    return seats


def clock_panel(out):
    rows,sources=positions(16);records=[]
    with LegacyWorker(ROOT) as worker:
        base=read(ROOT/'configs/legacy-ai/balanced-v3-masked-strength.json')
        for index in (0,6,13):
            row=rows[index]
            for ms,k,interior,replies in ((2000,4,2,1),(5000,6,3,2),(12000,8,4,2)):
                p=dict(base,milliseconds=ms,root_top_k=k,interior_top_k=interior,opponent_turn_limit=replies)
                d=worker.call('plan',observation=row['view'],choices=row['choices'],profile=p,fixed_work=False,sampling_seed=411)
                records.append(dict(position=index,ms=ms,profile=p,decision=d))
                write_json(out/'laptop-clocks.json',dict(format='masked-strength-clock-panel/1.1.0',sources=sources,records=records))
                print(json.dumps(dict(position=index,ms=ms,completed=d['search']['completed'],revisited=d['search']['revisited_plans'])),flush=True)
    return records


def match(index, out, block, rotation):
    seed = 740100 + block
    candidate_seat, control_seat = rotation, (rotation+1) % 4
    lineup = [None]*4; lineup[candidate_seat] = 'candidate'; lineup[control_seat] = 'control'
    others = [s for s in range(4) if lineup[s] is None]
    for seat, style in zip(others, (STYLES[(block+rotation+1)%6], STYLES[(block+rotation+3)%6])): lineup[seat] = style
    workers = []; records = []; began = time.monotonic()
    try:
        for seat, label in enumerate(lineup):
            if label == 'candidate':
                w = LegacyWorker(ROOT); p = read(ROOT / 'configs/legacy-ai/balanced-v3-masked-strength.json')
            else:
                style = 'balanced' if label == 'control' else label
                w = FrozenOpponent(ROOT, CONTROL + '/' + style + '-v3-masked', purpose='diagnostic'); p = w.profile
            workers.append((w, p))
        with RulesHost(ROOT / 'artifacts/r1-host/guarded') as host:
            context = host.reset(seed); intents = [None]*4
            while not context['terminal']:
                if len(records) > 160: raise ValueError('Game failed to progress')
                seat = context['actor']; view = host.observe(seat); choices = host.choices(); w, p = workers[seat]
                d = getattr(w, 'worker', w).call('plan', observation=view, choices=choices, profile=p, fixed_work=False,
                           sampling_seed=seed, intent=intents[seat])
                if d['failures']: raise ValueError('Search legality failure')
                intents[seat] = d['selected'].get('intent')
                # Observe the final settled own afterstate before committing the turn.
                after = None
                for action in d['selected']['actions']:
                    for identity in action:
                        legal = host.choices()
                        if identity not in {c['id'] for c in legal}: raise ValueError('Illegal continuation')
                        # Last finish advances the live host; public before-finish tiles
                        # already reflect all resource and sale choices.
                        if identity.startswith('finish|'): after = host.observe(seat)
                        context = host.apply(context, next(c for c in legal if c['id']==identity))
                records.append(dict(seat=seat, view=view, choices=choices, plan=d['selected'], after=after or view, search=d['search']))
            result = host.call('result')
        report = dict(format='masked-strength-match/1.1.0', game=index, block=block, rotation=rotation, seed=seed,
                      completed=True, candidate=candidate_seat, control=control_seat, lineup=lineup,
                      seconds=1., workers_per_search=1, result=result, metrics=metrics(records, result), elapsed=time.monotonic()-began)
        write_json(out / f'game-{index:02}.json', report)
        return report
    finally:
        for w, _ in workers: w.close()


def screen(out, workers):
    reports = []; failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(match, i, out, i//4, i%4): i for i in range(20)}
        for f in as_completed(futures):
            try: reports.append(f.result())
            except Exception as error: failures.append(dict(game=futures[f], error=str(error)))
            print(json.dumps(dict(finished=len(reports), failed=len(failures), latest=futures[f])), flush=True)
            write_json(out / 'screen-progress.json', dict(completed=len(reports), failures=failures))
    differences = []
    for block in range(5):
        panel = [r for r in reports if r['block'] == block]
        if len(panel) == 4:
            differences.append(sum(r['metrics'][str(r['candidate'])]['vp'] - r['metrics'][str(r['control'])]['vp'] for r in panel)/4)
    def average(role, key):
        return sum(r['metrics'][str(r[role])][key] for r in reports)/max(1,len(reports))
    boot=sorted(sum(sample)/5 for sample in product(differences,repeat=5)) if len(differences)==5 else []
    interval=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]] if boot else None
    report = dict(format='masked-strength-screen/1.1.0', games=len(reports), failures=failures,
        vp_difference_block_bootstrap_95=interval,
        control=CONTROL, seconds=1., workers_per_search=1, independent_setups=5,
        block_vp_differences=differences, candidate={k:average('candidate', k) for k in ('vp','rank','utility','cash_distress','unused_canal_l1','unflipped_endgame','merchant_beer','completed','revisited_plans','fallbacks')},
        control_metrics={k:average('control', k) for k in ('vp','rank','utility','cash_distress','unused_canal_l1','unflipped_endgame','merchant_beer','completed','fallbacks')},
        qualified=len(reports)==20 and not failures and len(differences)==5 and sum(differences)>0 and average('candidate','rank')<average('control','rank'),
        limitation='Five setup blocks; this is a screening comparison, not proof of human-level strength.')
    write_json(out / 'screen.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('diagnose','screen','clocks'))
    parser.add_argument('--out', default='artifacts/masked-strength/fix'); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--positions', type=int, default=16)
    a = parser.parse_args(); out = (ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Reports must remain inside this project')
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'diagnose': diagnose(out, a.positions)
    elif a.mode=='clocks':clock_panel(out)
    else: print(json.dumps(screen(out, a.workers)), flush=True)


if __name__ == '__main__': main()
