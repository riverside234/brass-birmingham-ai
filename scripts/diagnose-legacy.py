"""H0 repeated/control diagnosis; never calls a stock selector to explain it again."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.legacy import LAB_ROOT,LegacyWorker

def semantic(report):
    return [(t.get('seat'),t.get('stock'),t.get('after'),t.get('context')) for t in report['trace']]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--seed',type=int,default=9301)
    args=p.parse_args();out=args.root/'artifacts/legacy-ai/h0';out.mkdir(parents=True,exist_ok=True)
    with LegacyWorker(args.root,timeout=600) as w:
        profile=w.profile('balanced-l1-v1');profile.update(work_limit=32,milliseconds=0)
        reports=[]
        for name,diagnosis,record in [('plain',False,False),('recorded',False,True),('diagnosed',True,True)]:
            r=w.call('run',seed=args.seed,candidate_seat=-1,player_count=4,profile=profile,
                     diagnosis=diagnosis,record_selectors=record,fixed_work=True,check_live_observer=True)
            (out/(name+'.json')).write_text(json.dumps(r,separators=(',',':')))
            if not r['completed'] or r.get('failure'):raise RuntimeError(r.get('failure'))
            reports.append(r);print(name+': completed',flush=True)
        assert semantic(reports[0])==semantic(reports[1])==semantic(reports[2])
        assert reports[0]['result']==reports[1]['result']==reports[2]['result']
        assert [t.get('selected_rule') for t in reports[1]['trace']]==[t.get('selected_rule') for t in reports[2]['trace']]
        decisions=[t['decision'] for t in reports[2]['trace'] if 'decision' in t]
        assert len(decisions)==124 and all(d['alternatives'] for d in decisions)
        assert sum(len(d['alternatives'])>1 for d in decisions)>100
        original=[]
        for name in ('stock-hard-original','stock-hard-guarded-1.1.0'):
            directory=args.root/'artifacts/h0'/name
            instrument=json.loads((directory/'capture.json').read_text())
            plain=json.loads((directory/'capture-plain.json').read_text())
            first=instrument['games'][0];control=plain['games'][0]
            def events(r):return [(t['seat'],t['era'],t['round'],t['action'],t['events']) for t in r['trace']]
            assert first['completed'] and control['completed'] and events(first)==events(control)
            assert first['scores']==control['scores']
            original.append(dict(identity=name,seed=first['seed'],completed_games=len(instrument['games']),trace_control_equal=True,assembly_sha256=instrument['assembly_sha256']))
        result=dict(passed=True,seed=args.seed,actions=124,independent_candidate_decisions=len(decisions),multiple_alternatives=sum(len(d['alternatives'])>1 for d in decisions),
                    purity='identical public action/state/outcome trace; selected rule unchanged; live RNG/hash checked before every decision',
                    live_observer_parity='complete both-era game including stock Scout/wildcard plays',stock_controls=original,runtime=w.manifest)
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runtime'}))

if __name__=='__main__':main()
