"""Summarize prespecified four-seat pilot groups without treating failures as wins."""
import argparse
import json
from pathlib import Path
import statistics

def percentile(values,p):
    values=sorted(values)
    return values[min(len(values)-1,int((len(values)-1)*p))] if values else None

def summarize(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    games=[json.loads(p.read_text()) for p in sorted(directory.glob('[0-9]*.json'))]
    qualified=[r for r in games if r['completed'] and not r.get('failure') and r['candidate_seat'] in range(4)]
    rows=[r['result']['seats'][r['candidate_seat']] for r in qualified]
    turns=[];nodes=[];failures=[];terminal_checks=0
    for r in games:
        turn={}
        for t in r['trace']:
            if t.get('source','').startswith('legacy-'):
                c=t['context'];key=(c['era'],c['round'],c['turn'],t['seat'])
                turn[key]=turn.get(key,0)+t.get('milliseconds',0)
                if 'decision' in t:
                    nodes.append(t['decision']['nodes']);failures.append(t['decision'].get('failures',0))
                    ledger=t['decision']['selected'].get('ledger',{})
                    if r['completed'] and ledger.get('win_share') is not None:
                        actual=r['result']['seats'][t['seat']]
                        if (ledger['win_share'],ledger['rank'],ledger['banked'])!=(actual['utility'],actual['rank'],actual['vp']):
                            raise ValueError(f"Terminal projection differs from actual outcome: {directory}, seed {r['seed']}, seat {t['seat']}")
                        terminal_checks+=1
        turns.extend(turn.values())
    groups={seed:[r['result']['seats'][r['candidate_seat']]['utility'] for r in qualified if r['seed']==seed] for seed in sorted({r['seed'] for r in qualified})}
    return dict(profile=manifest['profile'],games=len(games),complete=len(qualified),failures=len(games)-len(qualified),
                seeds=manifest['seeds'],seats=manifest['seats'],win_share=sum(r['utility'] for r in rows)/len(games) if games else 0,
                mean_rank=statistics.mean(r['rank'] for r in rows) if rows else None,
                mean_vp=statistics.mean(r['vp'] for r in rows) if rows else None,
                vp_margin_vs_best_rival=statistics.mean(r['result']['seats'][r['candidate_seat']]['vp']-max(s['vp'] for s in r['result']['seats'] if s['seat']!=r['candidate_seat']) for r in qualified) if qualified else None,
                decision_execution_ms_per_turn=dict(median=percentile(turns,.5),p95=percentile(turns,.95),maximum=max(turns) if turns else None),
                timing_scope='Sum of recorded decision/execution timers per turn; excludes outer observation capture, IPC and UI presentation',
                mean_search_nodes=statistics.mean(nodes) if nodes else None,rejected_candidates=sum(failures),terminal_projection_checks=terminal_checks,
                per_setup_win_share={str(k):statistics.mean(v) for k,v in groups.items()},
                qualification='Small grouped pilot only; no H5 promotion or population-wide confidence claim',
                runtime=manifest['runtime'])

def main():
    p=argparse.ArgumentParser();p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    reports=[summarize(d) for d in args.directories];args.output.write_text(json.dumps(reports,indent=2))
    for r in reports:print(json.dumps({k:v for k,v in r.items() if k!='runtime'}))

if __name__=='__main__':main()
