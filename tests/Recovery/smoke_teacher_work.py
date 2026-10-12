"""A native teacher game exercising the active fixed simulation allowance.

An authored ordinary replay calibrates a fresh small utility head solely to
exercise the runtime contract. This is not a trained model or strength test.
"""
import argparse
import copy
from pathlib import Path
import time

import torch

from fixtures import game
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import TEACHER_WORK_BUDGET
from brass_ai.recovery.execution import ActorPool,neural_spec
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.replay import cache_receipt,read_shards
from brass_ai.recovery.teacher_schedule import CONTRACT,PHASES,reserved_rounds
from brass_ai.recovery.updates import Dataset,prepared
from brass_ai.recovery.utility import calibrate,calibration_ready,reward_spec
from brass_ai.step5.league import implementation_id,seeds_for,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mono')
    parser.add_argument('--modes',nargs='+',choices=('time','work'),default=['work'])
    out=parser.parse_args();directory=out.output.resolve()
    require('time' not in out.modes,'Time-mode comparison is archived; use preserved teacher-work-smoke-20261011 evidence. Active lab teachers use fixed work.')
    require(directory.is_relative_to(SOURCE_ROOT/'artifacts') and not directory.exists(),
            'Choose a new workspace artifact directory')
    torch.set_num_threads(2);torch.manual_seed(91417)
    lab=read(SOURCE_ROOT/'configs/step5/lab.json')
    cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(lab)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lab['learning_rate'])
    ordinary=game();ordinary['game_id']='authored-calibration-fixture'
    data=prepared([ordinary],insolvency=model.config.utility['insolvency'],reward=model.config.utility)
    fit=calibrate(model,optimizer,Dataset(data,'cpu',policy=False),cfg['update'])
    require(calibration_ready(model),'Authored fixture utility calibration failed')
    # Explicitly mark synthetic fit scope; its value fit is not game evidence.
    model.utility_calibration=dict(model.utility_calibration,scope='Authored ordinary replay fixture; smoke only')
    execution=dict(cfg['execution'],processes=1,heavy_processes=1,search_workers_per_turn=4)
    search=dict(lab['search'],workers=4);require(search['seconds']==12.,'Expected lab safety clock')
    seed=seeds_for(25516000,1)[0][0];sampling_seed=99116000
    require(len(set(out.modes))==len(out.modes),'Duplicate smoke mode')
    outputs={};began=time.monotonic();source_started=implementation_id()
    with InferenceService(execution,'cpu') as service:
        identity=service.register(model);neural=neural_spec(identity,.05,'neutral')
        for name in out.modes:
            budget=None if name=='time' else TEACHER_WORK_BUDGET
            print(dict(starting_mode=name,safety_seconds=search['seconds'],workers=4,work_budget=budget),flush=True)
            teacher=dict(kind='masked_ismcts',model_id=identity,opponents=[identity],search=search,
                focus='neutral',cpu_role='reference',teacher_schedule=CONTRACT)
            if budget is not None:teacher['teacher_work_budget']=copy.deepcopy(budget)
            provenance=dict(purpose='training',kind='masked-ismcts-teacher',seed=seed,
                sampling_seed=sampling_seed,spec=digest(teacher))
            job=dict(index=0,seed=seed,rotation=0,sampling_seed=sampling_seed,
                policies=[teacher,neural,neural,neural],path=str(directory/name/'teacher.jsonl.gz'),provenance=provenance)
            started=time.monotonic()
            with ActorPool(SOURCE_ROOT,cfg['host'],service,execution,heavy=True,mono=out.mono) as pool:
                records=pool.collect([job])
            elapsed=time.monotonic()-started
            receipts={r['path']:r.get('features') for r in records}
            for record in records:
                require(record.get('features')==cache_receipt(record['path']),'Feature receipt differs')
            cached=read_shards([r['path'] for r in records],partition='train',receipts=receipts)
            plain=read_shards([r['path'] for r in records],partition='train')
            require(digest(cached)==digest(plain),'Derived feature cache changed authoritative replay')
            require(len(cached)==1,'Expected one native game per mode')
            episode=cached[0];phase=records[0]['teacher_phases']
            require(len(phase)==1 and phase[0]['seat']==0,'Teacher seat misattributed')
            require(episode['opponents'][0].startswith('step5-phase-teacher/') and
                    episode['opponents'][1:]==[neural['policy_id']]*3,'Frozen opponent identities differ')
            receipt=phase[0]
            require(receipt['reserved_rounds']==reserved_rounds(seed,sampling_seed,0),'Teacher reservations changed')
            require(receipt['workers']==4 and receipt['seconds_per_turn']==12.,'Whole-turn worker/clock changed')
            counts=receipt['phases'];labels=sum(row['teacher'] is not None for row in episode['records'])
            require(labels==sum(p['qualified_samples'] for p in counts.values()),'Teaching sample receipt differs')
            if budget is not None:
                require(receipt['work_budget']==budget,'Work quota absent from runtime receipt')
                for name_,_,_,_ in PHASES:
                    entry=counts[name_];limit=budget['simulations_per_phase'][name_]
                    require(entry['allocated_simulations']<=limit and
                            entry['attempted_simulations']<=entry['allocated_simulations'],
                            'Search exceeded one whole-turn work allowance')
                    require(entry['actual_searches']<=1,'Work allowance restarted during a turn')
            outputs[name]=dict(seconds=elapsed,teacher_seconds=sum(p['elapsed_seconds'] for p in counts.values()),
                labels=labels,phases=counts,decisions=len(episode['records']),teacher_policy_id=episode['opponents'][0],
                result=episode['result'],cache_roundtrip=True,implementation_started=source_started,
                implementation_finished=implementation_id())
            # Persist each completed mode before starting another host. A
            # cancelled comparison must not discard its completed evidence.
            write_json(directory/name/'report.json',outputs[name])
            print(dict(mode=name,seconds=elapsed,teacher_seconds=outputs[name]['teacher_seconds'],labels=labels),flush=True)
        inference=service.report()
    if {'time','work'}<=set(outputs):
        require(outputs['time']['teacher_policy_id']!=outputs['work']['teacher_policy_id'],
                'Work and time policies must have different provenance identities')
    write_json(directory/'report.json',dict(implementation=implementation_id(),native_games=len(outputs),
        seconds=time.monotonic()-began,seed=seed,sampling_seed=sampling_seed,parameters=model.parameter_count,
        calibration=fit,work_budget=TEACHER_WORK_BUDGET,comparison=outputs,inference=inference,
        note='CPU/laptop fresh-model execution smoke, one teacher and three neural opponents per game. '
             'Same initial weights/setup/seeds, different game trajectories after teacher decisions. '
             'Authored calibration fixture; no trained strength or lab GPU speed claim.'))
    print(directory/'report.json',flush=True)


if __name__=='__main__':main()
