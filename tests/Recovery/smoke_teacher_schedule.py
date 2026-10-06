"""One native teacher game, cached replay re-read and a tiny distillation update.

Run explicitly; never part of unittest discovery or a training campaign.
"""
import argparse
from pathlib import Path
import time

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery.execution import ActorPool
from brass_ai.recovery.config import PPO_LEARNING_RATE
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.teacher_schedule import CONTRACT,collection_report
from brass_ai.recovery.updates import distill_update
from brass_ai.step5.league import seeds_for,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--mono')
    args=parser.parse_args();output=args.output.resolve()
    require(output.is_relative_to(SOURCE_ROOT/'artifacts') and not output.exists(),'Use a new workspace artifacts directory')
    output.mkdir(parents=True)
    cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    execution=dict(cfg['execution'],processes=1,heavy_processes=1)
    # Main lab search mechanics/clock; cap completed simulations for this small
    # plumbing check. The actual training simulation cap remains unchanged.
    search=dict(read(SOURCE_ROOT/'configs/step5/lab.json')['search'],simulations=16)
    torch.set_num_threads(2);torch.manual_seed(5012026)
    model=FourSeatNet(ModelConfig.preset('medium')).to(args.device)
    seed=seeds_for(22515000,1)[0][0];sampling=99015000;began=time.monotonic()
    with InferenceService(execution,args.device) as service:
        identity=service.register(model)
        spec=dict(kind='masked_ismcts',model_id=identity,opponents=[identity],search=search,
            focus='neutral',cpu_role='reference',teacher_schedule=CONTRACT)
        provenance=dict(purpose='training',kind='masked-ismcts-teacher',seed=seed,sampling_seed=sampling,spec=digest(spec))
        job=dict(index=0,seed=seed,rotation=0,sampling_seed=sampling,policies=[spec]*4,
            path=str(output/'teacher.jsonl.gz'),provenance=provenance)
        with ActorPool(SOURCE_ROOT,cfg['host'],service,execution,mono=args.mono,heavy=True) as pool:
            records=pool.collect([job]);inference=service.report()
            cached=pool.collect([job])
            require(cached[0]['cached'] and cached[0]['teacher_phases']==records[0]['teacher_phases'],'Cached phase receipts changed')
    games=read_shards([r['path'] for r in records],partition='train',receipts={r['path']:r['features'] for r in records})
    teaching=collection_report(records,games)
    for phase in teaching['phases'].values():
        require(phase['scheduled_turns']==4 and phase['reached_turns']==4,'Missing reserved phase turns')
        require(phase['searched_turns']<=4,'Phase turn allocation exceeded')
    update=distill_update(model,torch.optim.AdamW(model.parameters(),lr=PPO_LEARNING_RATE),games,cfg['update'],
        steps=cfg['distill_steps'],weight=cfg['distill_weight'])
    report=dict(games=len(games),seed=seed,sampling_seed=sampling,seconds_per_turn=search['seconds'],workers=4,
        simulations_cap_per_worker=16,elapsed_seconds=time.monotonic()-began,device=args.device,
        replay_cache_verified=True,teaching=teaching,distillation=update,inference=inference,
        collection=records,note='Plumbing/coverage smoke with a fresh random network; not strength or lab GPU throughput evidence.')
    write_json(output/'report.json',report)
    print('Teacher smoke:',output/'report.json',flush=True)
    for name,counts in teaching['phases'].items():
        print(name,counts,flush=True)
    print('Distillation:',update,flush=True)


if __name__=='__main__':main()
