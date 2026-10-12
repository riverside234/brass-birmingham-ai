"""Small matched CPU pipeline benchmark; optional CUDA learner, no campaign.

Uses existing complete replays as IO/IL fixtures, two ordinary PPO games, and
the three active native search turns. Run in a fresh owned artifact directory.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT, require, digest
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.recovery.replay import read_shards, SerializedGame, write_shard, cache_receipt
from brass_ai.recovery import execution
from brass_ai.recovery.updates import prepared, Dataset, supervised_epoch
from brass_ai.step5.league import implementation_id, write_json


@contextmanager
def measured(report):
    stop=threading.Event();owner=psutil.Process();seen={};base={};peaks=[]
    def sample():
        total=0
        for p in [owner]+owner.children(recursive=True):
            try:
                key=(p.pid,p.create_time());cpu=p.cpu_times();seconds=cpu.user+cpu.system
                if key not in base:base[key]=seconds if p.pid==owner.pid else 0.
                seen[key]=seconds;total+=p.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        peaks.append(total)
    def watch():
        while not stop.wait(.25):sample()
    sample();thread=threading.Thread(target=watch,daemon=True);start=time.perf_counter();thread.start()
    try:yield
    finally:
        stop.set();thread.join();sample();elapsed=time.perf_counter()-start
        cpu=sum(seen[k]-base[k] for k in seen)
        report.update(elapsed_seconds=elapsed,cpu_seconds_sampled=cpu,average_cpu_cores=cpu/elapsed,
                      peak_tree_rss_bytes=max(peaks),gpu_utilization_percent=None,
                      resource_note='250 ms process-tree sampling; brief child processes may be missed; RSS includes shared pages')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--skip-search',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a fresh owned artifact directory')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(71717)
    spec=read(a.fixture);report=dict(implementation=implementation_id(),device=a.device)
    # Full validation for the same authoritative bytes in both versions. This
    # deliberately does not credit a stale derived-cache receipt after an update.
    phase={}
    with measured(phase):games=read_shards(spec['paths'])
    phase.update(games=len(games),rows=sum(len(g['records']) for g in games));report['replay_ingress']=phase
    phase={}
    with measured(phase):
        # The actual writer receives fully validated games from play_game.
        # These fixtures have passed the same schema checks in read_shards.
        for game in games:game.actor_validated=True
        writer=getattr(execution,'_ActorSerializedGame',SerializedGame)
        values=[writer(g) for g in games]
        write_shard(out/'fixture.jsonl.gz',values,dict(purpose='training',benchmark=True))
    phase['games_per_second']=len(games)/phase['elapsed_seconds'];report['replay_publication']=phase
    phase={}
    with measured(phase):
        cached=read_shards([out/'fixture.jsonl.gz'],receipts={str(out/'fixture.jsonl.gz'):cache_receipt(out/'fixture.jsonl.gz')})
        data=prepared(cached,neutral=True)
    report['cached_il_ingress']=phase
    cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    model=FourSeatNet(ModelConfig.preset('medium')).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    phase={}
    with measured(phase):
        loss=supervised_epoch(model,optimizer,Dataset(data,a.device,policy=False),cfg['update'])
    phase.update(result=loss,model_hash=digest({k:v.tolist() for k,v in model.state_dict().items()}))
    report['il_epoch']=phase
    if a.device=='cuda':report['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    else:report['gpu_peak_allocated_bytes']=None
    write_json(out/'report.json',report)
    # The existing smoke holds its own configuration and seeds constant.
    for name,script,args in [('ppo','smoke_throughput.py',['--ppo-only']),
                             ('search','smoke_time_budgets.py',['--time-only'])]:
        if name=='search' and a.skip_search:continue
        phase={}
        with measured(phase), (out/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name(script)),
                            '--output',str(out/name),'--device',a.device,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        phase['result']=read(out/name/'report.json');report[name]=phase
        if name=='ppo':phase['games_per_second']=2/phase['elapsed_seconds']
        else:
            for row in phase['result']['results']:
                row['qualified_samples_per_second']=row['qualified_samples']/row['elapsed_seconds']
        write_json(out/'report.json',report);print(name,phase['elapsed_seconds'],flush=True)
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
