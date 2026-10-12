"""Two matched native games, two PPO games, and optional four-worker NN turns.

Local CPU correctness/performance evidence, never a strength qualification.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import random
import time
import torch
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.bank import read
from brass_ai.recovery.host_runtime import training_host
from brass_ai.recovery.telemetry import Timings,instrument_host
from brass_ai.recovery.league import train
from brass_ai.step5.league import write_json,implementation_id


def native_game(fast,seed):
    timings=Timings();trace=[];rng=random.Random(seed+4327);forced_count=0
    path=training_host(SOURCE_ROOT) if fast else SOURCE_ROOT/'artifacts/r1-host/guarded'
    with RulesHost(path) as host:
        instrument_host(host,timings);context=host.reset(seed)
        if fast:require(host.enable_fastpath(),'Missing optimized runtime')
        began=time.perf_counter();buffer=deque();decision=None
        while not context['terminal']:
            require(len(trace)<2000,'Native smoke did not terminate')
            if buffer:
                entry=buffer.popleft();current=entry['before'];after=entry['after'];forced_count+=1
            else:
                current=(decision or host.decision()) if fast else dict(view=host.observe(context['actor']),choices=host.choices())
                after=None
            view,choices=current['view'],current['choices']
            choice=choices[rng.choices(range(len(choices)),k=1)[0]]
            trace.append(dict(view=view,choices=choices,chosen=choice['id']))
            if after is not None:context=after
            elif fast:
                reply=host.step_decision(context,choice,16);buffer.extend(reply['forced']);decision=reply['decision']
                context=buffer[0]['before']['context'] if buffer else decision['context']
            else:context=host.apply(context,choice)
        seconds=time.perf_counter()-began
        result=host.call('result');state_hash=host.call('canonical_hash')
        return dict(seconds=seconds,decisions=len(trace),forced_batched=forced_count,trace=digest(trace),
                    final_hash=state_hash,result=result,timings=timings.report())

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--host-only',action='store_true');p.add_argument('--ppo-only',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();out=a.output.resolve();require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a fresh artifact directory')
    torch.set_num_threads(2)
    pairs=[]
    for i,seed in enumerate(() if a.ppo_only else (851001,851002)):
        modes=[False,True] if i==0 else [True,False]
        results={str(fast):native_game(fast,seed) for fast in modes}
        old,new=results['False'],results['True']
        require(all(old[k]==new[k] for k in ('decisions','trace','final_hash','result')),'Fused native game differs')
        row=dict(seed=seed,old=old,new=new,speedup=old['seconds']/new['seconds']);pairs.append(row)
        write_json(out/'native.json',pairs);print(dict(seed=seed,speedup=row['speedup'],old_seconds=old['seconds'],new_seconds=new['seconds']),flush=True)
    report=dict(native=pairs,implementation=implementation_id(),scope='Local CPU, matched random legal play; not lab GPU or playing strength')
    if not a.host_only:
        cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
        cfg.update(games_per_iteration=2,threads=2,audit_workers=2,search_every_games=0)
        cfg['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1)
        cfg['update']['epochs']=1;cfg['evaluation_schedule']['enabled']=False
        cfg['legacy_boost']=dict(stages=[dict(games=20,selfplay=95,historical=0,legacy=5)])
        began=time.perf_counter();result=train(SOURCE_ROOT,out/'ppo',cfg,iterations=1,device=a.device)
        summary=result['summaries'][0]
        require(summary['actor']['optimizer_steps']>0,'No PPO update')
        record=read(out/'ppo/iteration-0000/report.json')
        require(record['utilization']['replay_ingress']['streaming'],'Replay did not prepare during collection')
        require(record['inference']['shared_requests']>0,'Shared inference transport not used')
        report['ppo']=dict(seconds=time.perf_counter()-began,checkpoint=result['checkpoint'],games=2,
                           optimizer_steps=summary['actor']['optimizer_steps'],utilization=record['utilization'])
    write_json(out/'report.json',report);print(out/'report.json',flush=True)

if __name__=='__main__':main()
