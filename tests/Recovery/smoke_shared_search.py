"""Bounded official-v4 parity and IL/self-play adapter smoke, no full games."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import time

from brass_ai.bank import read,sha
from brass_ai.host import RulesHost
from brass_ai.legacy import LegacyWorker
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.config import LEGACY_BUDGET,LEGACY_TIME_BUDGET,LEGACY_OPPONENT_WORK_BUDGET
from brass_ai.recovery.legacy_search import LabLegacyPolicy,legacy_spec
from brass_ai.search.legacy_runtime import build,runtime_directory

ROOT=SOURCE_ROOT;BANK='bank-3dc65253a610191d66a5';PARENT=BANK+'/balanced-v3-masked-economy'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT/'artifacts') and not out.exists();out.mkdir(parents=True)
    began=time.monotonic();protected=[ROOT/'training/opponents/pool.json',ROOT/'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll',
        ROOT/f'training/opponents/bank/{BANK}/manifest.json',ROOT/'configs/step5/joint7-latest.json']
    before={str(p):sha(p) for p in protected};ref=build(ROOT,BANK)
    positions=read(ROOT/'artifacts/v4-search-vs-fast-2s2w-20261017/final/positions.json')
    rows=[]
    with ExitStack() as stack:
        host=stack.enter_context(RulesHost(ROOT/'artifacts/r1-host/guarded'))
        original=stack.enter_context(LegacyWorker(ROOT,runtime_directory=f'artifacts/legacy-ai/bank/{BANK}/runtime',timeout=30))
        current=stack.enter_context(LegacyWorker(ROOT,runtime_directory=runtime_directory(ROOT,ref),timeout=30))
        for i in (0,3,5,7):
            position=positions[i];profile=dict(read(ROOT/'configs/legacy-ai/balanced-v3-masked-economy.json'),simulations=8,work_limit=1000000)
            request=dict(observation=position['view'],choices=position['choices'],profile=profile,fixed_work=True,sampling_seed=881000+i)
            a=original.call('plan',**request);b=current.call('plan',**request)
            assert a['selected']==b['selected'] and a['search']['root']==b['search']['root'] and a['search']['plans']==b['search']['plans']
            assert a['search']['completed']==b['search']['completed']==8 and b['search']['deadline_stops']==0
            assert b['search']['network_candidates']==0
            timed={}
            profile.update(simulations=100000,milliseconds=1000.)
            # Fixed checks above warm both native processes; alternate order.
            order=(('old',original),('shared',current)) if len(rows)%2==0 else (('shared',current),('old',original))
            for name,worker in order:
                start=time.monotonic()
                d=worker.call('plan_forest',observation=position['view'],choices=position['choices'],profile=profile,
                              search_workers=2,sampling_seed=881000+i)
                assert d['failures']==0 and d['search']['opponent_nodes']==0
                timed[name]=dict(seconds=time.monotonic()-start,completed=d['search']['completed'],
                    interrupted=d['search']['interrupted'],deadline_stops=d['search'].get('deadline_stops',0))
            context=host.call('restore',checkpoint=position['checkpoint'])
            for action in b['selected']['actions']:
                for identity in action:
                    choice=next(c for c in host.choices() if c['id']==identity);context=host.apply(context,choice)
            # Exercise the exact lab adapter used by IL generation and self-play.
            budget=dict(LEGACY_BUDGET,search_workers=2)
            reports={}
            for fast in (False,True):
                policy=LabLegacyPolicy(ROOT,PARENT,budget,search_free=fast,work_budget=LEGACY_TIME_BUDGET)
                try:
                    choice=policy.select_batch([(position['view'],position['choices'],random.Random(881000+i))])[0]
                    assert 0<=choice[0]<len(position['choices']) and policy.telemetry['failures']==0
                    if fast:assert policy.telemetry['completed']==0
                    reports['fast' if fast else 'searched']=dict(policy.telemetry)
                finally:policy.close()
            rows.append(dict(position=i,phase=position['view']['context']['era'],fixed_completed=8,
                exact_bank_plan_and_value_parity=True,lab=reports,warm_native=timed))
            print(json.dumps(dict(position=i,passed=True)),flush=True)
            assert time.monotonic()-began<150,'Bounded smoke exceeded 150s'
    assert before=={str(p):sha(p) for p in protected}
    result=dict(passed=True,reference=ref,positions=rows,seconds=time.monotonic()-began,protected_files_unchanged=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,seconds=result['seconds'],report=str(out/'report.json'))))

if __name__=='__main__':main()
