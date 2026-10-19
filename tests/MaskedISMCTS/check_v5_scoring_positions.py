"""Bounded old/new v5 probes on replayed supplier positions; never a tournament."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from brass_ai.bank import FrozenOpponent,read,sha
from brass_ai.host import RulesHost
from brass_ai.legacy import LegacyWorker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT/'artifacts') and not out.exists();out.mkdir(parents=True)
    game=ROOT/'artifacts/v5-vs-v4-all6-8s10s1w-12games-20261018'
    source=read(game/'game-09.json')
    turns=[json.loads(line) for line in (game/'game-09-turns.jsonl').read_text().splitlines()]
    indices=(7,21,43,45,55,63);positions=[]
    with RulesHost(ROOT/'artifacts/r1-host/guarded') as host:
        context=host.reset(source['seed'])
        for turn in turns:
            assert context['actor']==turn['seat']
            if turn['index'] in indices:
                positions.append(dict(index=turn['index'],checkpoint=host.call('checkpoint'),context=context,
                    view=host.observe(context['actor']),choices=host.choices()))
            for action in turn['actions']:
                for identity in action:
                    choice=next(c for c in host.choices() if c['id']==identity)
                    context=host.apply(context,choice)
        assert context['terminal'] and host.call('result')==source['result']
    records=[];began=time.monotonic()
    with ExitStack() as stack:
        old=FrozenOpponent(ROOT,'bank-357f50e6bcbdd119ea8c/supplier-v5-masked-industry',purpose='diagnostic')
        stack.callback(old.close)
        new=stack.enter_context(LegacyWorker(ROOT,runtime_directory=args.runtime,timeout=30))
        host=stack.enter_context(RulesHost(ROOT/'artifacts/r1-host/guarded'))
        workers=dict(previous=old.worker,revised=new)
        for index,position in enumerate(positions):
            profile=dict(old.profile,milliseconds=2000.)
            for role in (('previous','revised') if index%2==0 else ('revised','previous')):
                assert time.monotonic()-began<120,'Bounded smoke deadline'
                start=time.monotonic()
                result=workers[role].call('plan_forest',observation=position['view'],choices=position['choices'],
                    profile=profile,search_workers=1,sampling_seed=918500+position['index'],intent=None)
                elapsed=time.monotonic()-start
                assert result['failures']==0 and result['search']['completed']>0
                context=host.call('restore',checkpoint=position['checkpoint']);seat=position['context']['actor']
                assert host.observe(seat)==position['view']
                sold=[];built=[]
                for action in result['selected']['actions']:
                    for identity in action:
                        assert context['actor']==seat
                        choice=next(c for c in host.choices() if c['id']==identity)
                        if choice['kind'] in ('sell','build'):
                            view=host.observe(seat)
                            tile=(next(s for s in view['slots'] if s['id']==choice['target'])['tile'] if choice['kind']=='sell' else
                                  next(m for m in view['players'][seat]['mat'] if m['type']==choice['industry'])['tiles'][0])
                            (sold if choice['kind']=='sell' else built).append(dict(type=tile['type'],level=tile['level']))
                        context=host.apply(context,choice)
                keys=('era','round','turn','turn_owner')
                assert context['terminal'] or tuple(context[k] for k in keys)!=tuple(position['context'][k] for k in keys)
                row=dict(index=position['index'],era=position['context']['era'],round=position['context']['round'],
                    role=role,seconds=elapsed,legal=True,sold=sold,built=built,result=result)
                records.append(row)
                print(json.dumps(dict(index=row['index'],role=role,completed=result['search']['completed'],
                    interrupted=result['search']['interrupted'],seconds=elapsed,actions=[a[0] for a in result['selected']['actions']])),flush=True)
    summary={role:dict(completed=sum(r['result']['search']['completed'] for r in records if r['role']==role),
        interrupted=sum(r['result']['search']['interrupted'] for r in records if r['role']==role),
        seconds=sum(r['seconds'] for r in records if r['role']==role),
        mean_turn_seconds=statistics.mean(r['seconds'] for r in records if r['role']==role)) for role in workers}
    runtime=args.runtime.resolve()
    report=dict(passed=True,previous_bank='bank-357f50e6bcbdd119ea8c',candidate_runtime=runtime.relative_to(ROOT).as_posix(),
        candidate_executable_sha256=sha(runtime/'BrassLegacyAI.exe'),candidate_source_pins=read(runtime/'runtime.json')['sources'],
        source_game=str(game.relative_to(ROOT)/'game-09.json'),source_sha256=sha(game/'game-09-turns.jsonl'),
        replay_verified=True,budget_ms=2000,workers=1,positions=len(positions),summary=summary,records=records,
        limitation='Six selected positions, one sample per version, no win-rate or playing-strength evidence.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,summary=summary)),flush=True)


if __name__=='__main__':main()
