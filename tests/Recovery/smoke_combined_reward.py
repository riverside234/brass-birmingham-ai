"""Two ordinary PPO games and one native game with one four-worker NN teacher.

Explicit smoke only: no automatic discovery, installed-game writes or campaign.
"""
import argparse
from pathlib import Path
import time
import torch

from brass_ai.bank import read
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.learning.reward import CostLedger,replay_returns
from brass_ai.recovery.execution import ActorPool,neural_spec
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.league import train
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.teacher_schedule import CONTRACT
from brass_ai.recovery.updates import distill_update
from brass_ai.recovery.utility import calibration_ready
from brass_ai.step5.league import seeds_for,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Choose a new workspace artifact directory')
    cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    cfg.update(games_per_iteration=2,threads=2,search_every_games=0)
    cfg['update']['epochs']=1;cfg['evaluation_schedule']['enabled']=False;cfg['audit_workers']=2
    began=time.monotonic();trained=train(SOURCE_ROOT,out/'league',cfg,iterations=1)
    checkpoint=SOURCE_ROOT/trained['checkpoint']['path'];model,raw=load_checkpoint(checkpoint)
    require(calibration_ready(model),'New utility head did not calibrate from fresh rollouts')
    actor=trained['summaries'][0]['actor']
    require(actor['reward']['spec']==model.config.utility and actor['optimizer_steps']>0,'Missing combined PPO update')
    # Preserve actual lab turn clocks/workers; only the simulation cap is small.
    search=dict(read(SOURCE_ROOT/'configs/step5/lab.json')['search'],simulations=8)
    ex=dict(cfg['execution'],processes=1,heavy_processes=1)
    seed=seeds_for(22516000,1)[0][0]
    with InferenceService(ex,'cpu') as service:
        identity=service.register(model)
        teacher=dict(kind='masked_ismcts',model_id=identity,opponents=[identity],search=search,
                     focus='neutral',cpu_role='reference',teacher_schedule=CONTRACT)
        neural=neural_spec(identity,.05,'neutral')
        provenance=dict(purpose='training',kind='masked-ismcts-teacher',seed=seed,sampling_seed=99016000,spec=digest(teacher))
        job=dict(index=0,seed=seed,rotation=0,sampling_seed=99016000,policies=[teacher,neural,neural,neural],
                 path=str(out/'teacher.jsonl.gz'),provenance=provenance)
        with ActorPool(SOURCE_ROOT,cfg['host'],service,ex,heavy=True) as pool:records=pool.collect([job])
        inference=service.report()
    games=read_shards([r['path'] for r in records],partition='train',receipts={r['path']:r.get('features') for r in records})
    # Verify streaming path accounting against offline labels on an actual game.
    episode=games[0];labels,components=replay_returns(episode['records'],episode['result'],model.config.utility)
    ledger=CostLedger(model.config.utility)
    for row,label in zip(episode['records'],labels):
        ledger.observe(row['view'])
        require(all(abs(a-b)<1e-8 for a,b in zip(ledger.remaining,label['remaining'])),'Live/offline budgets differ')
        ledger.chosen(row['view'],row['choices'][row['chosen']])
    labels_count=sum(r['teacher'] is not None for r in episode['records'])
    update=distill_update(model,torch.optim.AdamW(model.parameters(),lr=1e-5),games,cfg['update'])
    require(not update['updated'],'One smoke setup must not bypass the two-setup teacher gate')
    write_json(out/'report.json',dict(games=3,ordinary_games=2,teacher_games=1,teacher_seats=1,
        seconds=time.monotonic()-began,checkpoint=str(checkpoint),parameters=model.parameter_count,
        ppo_steps=actor['optimizer_steps'],reward=actor['reward'],utility_error=actor['epochs'][-1]['utility_prediction'],
        calibration=raw['training']['utility_calibration'],qualified_teacher_labels=labels_count,
        teacher_phases=records[0]['teacher_phases'],teacher_update=update,inference=inference,
        live_offline_budget_parity=True,teacher_reward=components['reward'],
        note='Small CPU execution check, eight simulations per worker; not strength or lab GPU throughput evidence.'))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
