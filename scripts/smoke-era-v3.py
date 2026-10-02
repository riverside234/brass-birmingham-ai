"""Eight matched four-seat games, new vs frozen previous Masked v3, 1s/4 owners."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import ExitStack,closing
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT,load_bank,read,sha
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import require,project_path
from brass_ai.learning.replay import terminal_targets
from brass_ai.recovery.era_bank import PREVIOUS_BANK
from brass_ai.recovery.legacy_search import LabLegacyPolicy
from brass_ai.step5.league import write_json


def old_factory(root):
    import importlib.util
    path=root/'training/opponents/bank'/PREVIOUS_BANK/'source/src/brass_ai/recovery/legacy_search.py'
    spec=importlib.util.spec_from_file_location('brass_ai.recovery._previous_masked',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.LabLegacyPolicy


def play(root,out,new_bank,index,seed,rotation,mono):
    styles=('balanced','rail-tempo') if index//4==0 else ('cotton','goods-beer')
    entries=[('new',styles[0]),('old',styles[0]),('new',styles[1]),('old',styles[1])]
    assigned=[entries[(s-rotation)%4] for s in range(4)]
    budget=dict(role='reference',seconds=1.,search_workers=4,global_workers=60,work_multiplier=1)
    began=time.monotonic();turns=[]
    with ExitStack() as stack:
        policies=[]
        for role,style in assigned:
            factory=LabLegacyPolicy if role=='new' else old_factory(root)
            parent=(new_bank+'/'+style+'-v3-masked-era') if role=='new' else (PREVIOUS_BANK+'/'+style+'-v3-masked-strength')
            policies.append(stack.enter_context(closing(factory(root,parent,budget,mono=mono,purpose='diagnostic'))))
        host=stack.enter_context(RulesHost(root/'artifacts/r1-host/guarded',mono=mono))
        context=host.reset(seed);states=[dict(intent=None,work=0,turn=None) for _ in range(4)]
        while not context['terminal']:
            require(len(turns)<160,'Game did not progress')
            seat=context['actor'];view=host.observe(seat);choices=host.choices();state=states[seat]
            key=tuple(context[k] for k in ('era','round','turn','turn_owner'))
            if key!=state['turn']:state.update(turn=key,work=0,seed=seed+len(turns)*101,deadline=time.monotonic()+1.)
            started=time.monotonic();before=policies[seat].telemetry['nodes']
            plan=policies[seat].decision(view,choices,state);elapsed=time.monotonic()-started
            require(plan['actions'] and all(plan['actions']),'Missing complete continuation')
            state['intent']=plan.get('intent')
            for action in plan['actions']:
                for identity in action:
                    legal={c['id']:c for c in host.choices()}
                    require(identity in legal,'Illegal complete continuation')
                    context=host.apply(context,legal[identity])
            detail=policies[seat].last_decision
            player=view['players'][seat]
            turns.append(dict(seat=seat,era=view['context']['era'],seconds=elapsed,
                nodes=policies[seat].telemetry['nodes']-before,
                actions=[a[0].split('|')[0] for a in plan['actions']],
                complete_actions=plan['actions'],cash=player['cash'],income=player['income'],ledger=plan.get('ledger'),
                completed=detail.get('completed',0),cpu_batching=detail.get('search',{}).get('cpu_batching',{})))
            if view['context']['era']=='RailEra' and view['context']['round'] in (1,4,7):
                write_json(out/f'position-{index:02d}-{len(turns):02d}.json',dict(view=view,choices=choices,selected=plan,role=assigned[seat][0]))
        result=host.call('result');targets=terminal_targets(result)
    rows=[]
    for seat,(role,style) in enumerate(assigned):
        mine=[t for t in turns if t['seat']==seat]
        rows.append(dict(role=role,style=style,seat=seat,vp=targets['vp'][seat],win_share=targets['win_share'][seat],
            first_place=int(targets['placement'][seat]==1),rank=targets['placement'][seat],
            mean_turn_seconds=statistics.mean(t['seconds'] for t in mine),max_turn_seconds=max(t['seconds'] for t in mine),
            nodes=sum(t['nodes'] for t in mine),rail_actions=sum(a=='network' for t in mine if t['era']=='RailEra' for a in t['actions']),
            rail_industry_actions=sum(a=='build' for t in mine if t['era']=='RailEra' for a in t['actions'])))
    report=dict(index=index,seed=seed,rotation=rotation,completed=True,lineup=assigned,result=result,metrics=rows,turns=turns,seconds=time.monotonic()-began)
    write_json(out/f'game-{index:02d}.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=LAB_ROOT)
    parser.add_argument('--output',default='artifacts/rail-era/smoke-1');parser.add_argument('--tables',type=int,default=2)
    parser.add_argument('--mono');parser.add_argument('--seed',type=int,default=842100)
    parser.add_argument('--bank',help='Frozen candidate bank; default local candidate or transferred current bank')
    args=parser.parse_args();root=args.root.resolve()
    candidate=root/'artifacts/rail-era/candidate-bank.json'
    new_bank=args.bank or read(candidate if candidate.exists() else root/'training/opponents/current.json')['bank']
    require(1<=args.tables<=24,'Invalid table count')
    old=load_bank(root,PREVIOUS_BANK);new=load_bank(root,new_bank);out=project_path(root,args.output)
    require(not out.exists(),'Use a new smoke output');out.mkdir(parents=True)
    rows=[];failures=[]
    with ThreadPoolExecutor(max_workers=args.tables) as pool:
        jobs={pool.submit(play,root,out,new_bank,block*4+r,args.seed+block,r,args.mono):block*4+r for block in range(2) for r in range(4)}
        for future in as_completed(jobs):
            try:rows.append(future.result())
            except Exception as e:failures.append(dict(index=jobs[future],error=str(e)))
            print(dict(completed=len(rows),failures=failures,total=8),flush=True)
    summary={}
    for role in ('new','old'):
        seats=[m for r in rows for m in r['metrics'] if m['role']==role]
        if seats:
            summary[role]=dict(seat_games=len(seats),mean_vp=statistics.mean(m['vp'] for m in seats),
                win_share=sum(m['win_share'] for m in seats),win_rate=statistics.mean(m['win_share'] for m in seats),
                first_places=sum(m['first_place'] for m in seats),mean_rank=statistics.mean(m['rank'] for m in seats),
                mean_turn_seconds=statistics.mean(m['mean_turn_seconds'] for m in seats),max_turn_seconds=max(m['max_turn_seconds'] for m in seats),
                mean_rail_actions=statistics.mean(m['rail_actions'] for m in seats),mean_rail_industry_actions=statistics.mean(m['rail_industry_actions'] for m in seats))
    passed=len(rows)==8 and not failures
    winner=('new' if summary['new']['win_share']>summary['old']['win_share'] else 'old' if summary['old']['win_share']>summary['new']['win_share'] else 'tie') if passed else None
    report=dict(format='masked-era-smoke/1.0.0',passed=passed,games=len(rows),failures=failures,new_bank=new_bank,previous_bank=PREVIOUS_BANK,
        new_runtime_files=new['payload']['runtime_files'],previous_runtime_files=old['payload']['runtime_files'],seconds=1.,search_workers=4,
        local_tables=args.tables,lab_tables=24,logical_cpus=os.cpu_count(),seed=args.seed,policies=summary,winner=winner,
        selection_rule='Strength screen reports official win share; explicit user instruction on 2026-10-01 authorizes installation after correctness/package checks',
        evidence={f'game-{r["index"]:02d}.json':sha(out/f'game-{r["index"]:02d}.json') for r in rows},
        limitation='Eight games, two setups and four seat rotations. Directional smoke evidence; no statistically established strength claim.')
    write_json(out/'report.json',report);print({k:report[k] for k in ('passed','games','new_bank','previous_bank','policies','winner','failures')},flush=True)
    require(passed,'Smoke failed')


if __name__=='__main__':main()
