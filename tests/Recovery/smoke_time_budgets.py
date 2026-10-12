"""Matched native turns for time-primary NN/v4 versus archived simulation work.

One small random calibrated model, no training campaign or strength comparison.
"""
import argparse
import copy
import random
import time
from pathlib import Path

import torch
from fixtures import game
from brass_ai.bank import read
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.reward import CostLedger
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import (NN_WORK_BUDGET,TEACHER_WORK_BUDGET,NN_TIME_BUDGET,
    TEACHER_TIME_BUDGET,LEGACY_TIME_BUDGET,LEGACY_OPPONENT_WORK_BUDGET,LEGACY_BUDGET)
from brass_ai.recovery.execution import make_policy
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.legacy_search import LabLegacyPolicy
from brass_ai.recovery.resources import TurnLease
from brass_ai.recovery.teacher_schedule import CONTRACT,reserved_rounds
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset,prepared
from brass_ai.recovery.utility import calibrate,calibration_ready,reward_spec
from brass_ai.search.common import turn_key
from brass_ai.step5.league import implementation_id,write_json


def root_position(host):
    context=host.reset(25516000)
    # Reach a real two-action turn without synthetic edits to the engine.
    for _ in range(48):
        if context['round']==2 and context['phase']=='action':return context
        choices=host.choices()
        selected=max(choices,key=lambda c:(c['kind']=='pass',c['kind']=='finish'))
        context=host.apply(context,selected)
    raise AssertionError('Could not reach second Canal round')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--time-only',action='store_true',help='Only the three active time-budget turns')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Choose a new artifact directory')
    torch.set_num_threads(2);torch.manual_seed(91417)
    lab=read(SOURCE_ROOT/'configs/step5/lab.json');cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(lab))).to(args.device)
    data=prepared([game()],insolvency=model.config.utility['insolvency'],reward=model.config.utility)
    calibrate(model,torch.optim.AdamW(model.parameters(),lr=lab['learning_rate']),Dataset(data,args.device,policy=False),cfg['update'])
    require(calibration_ready(model),'Fixture calibration not ready')
    model.utility_calibration=dict(model.utility_calibration,scope='Authored replay fixture; runtime smoke only')
    execution=dict(cfg['execution'],processes=1,heavy_processes=1,search_workers_per_turn=4)
    source=implementation_id();reports=[]
    with InferenceService(execution,args.device) as service:
        identity=service.register(model)
        parent=next(p for p in read(SOURCE_ROOT/'training/opponents/pool.json')['training'] if 'balanced' in p)
        # Reverse pair order for teacher vs evaluation; no concurrent benchmarks.
        for kind,modes in [('evaluation',('fixed','time')),('teacher',('time','fixed')),('v4',('fixed','time'))]:
            if args.time_only:modes=('time',)
            for mode in modes:
                if kind=='v4':
                    policy=LabLegacyPolicy(SOURCE_ROOT,parent,LEGACY_BUDGET,
                        work_budget=LEGACY_TIME_BUDGET if mode=='time' else LEGACY_OPPONENT_WORK_BUDGET)
                else:
                    spec=dict(kind='masked_ismcts',model_id=identity,opponents=[identity],
                        search=dict(lab['search'],workers=4),focus='neutral',cpu_role='reference')
                    if kind=='teacher':spec.update(teacher_schedule=CONTRACT,
                        teacher_work_budget=copy.deepcopy(TEACHER_TIME_BUDGET if mode=='time' else TEACHER_WORK_BUDGET))
                    else:spec['nn_work_budget']=copy.deepcopy(NN_TIME_BUDGET if mode=='time' else NN_WORK_BUDGET)
                    channels=service.channels(5)
                    policy=make_policy(spec,SOURCE_ROOT,[channels.acquire() for _ in range(5)],execution,None)
                lease=TurnLease(service.cpu,Timings());rng=random.Random(921)
                print(dict(starting=kind,mode=mode),flush=True)
                try:
                    with RulesHost(SOURCE_ROOT/cfg['host']) as host:
                        context=root_position(host);seat=context['actor'];key=turn_key(context)
                        sampling=next(i for i in range(1000) if reserved_rounds(25516000,i,seat)['canal_early']==2)
                        if kind=='teacher':policy.begin_teacher_game(rng,seat,25516000,sampling)
                        elif kind=='evaluation':policy.begin_work_game(rng,seat)
                        ledger=CostLedger(model.config.utility);decisions=[]
                        lease.update(key+(seat,),4);began=time.monotonic()
                        for step in range(96):
                            if context['terminal'] or context['actor']!=seat or turn_key(context)!=key:break
                            if kind=='teacher':policy.admit_teacher_turn(context,rng)
                            elif kind=='evaluation':policy.admit_work_turn(context,rng)
                            else:policy.admit_legacy_turn(context,rng)
                            view=host.observe(seat);choices=host.choices();ledger.observe(view)
                            if kind!='v4':policy.set_reward_context(ledger)
                            chosen,_,teacher=policy.select_batch([(view,choices,rng)])[0]
                            require(0<=chosen<len(choices),'Illegal selection')
                            if kind!='teacher':require(teacher is None,'Nonteacher exported a label')
                            decisions.append(dict(phase=context['phase'],remaining=context['actions_remaining'],
                                chosen=choices[chosen]['id'],label=teacher is not None))
                            ledger.chosen(view,choices[chosen]);context=host.apply(context,choices[chosen])
                        else:raise AssertionError('Current-turn completion cap exceeded')
                        elapsed=time.monotonic()-began
                        if kind=='v4':counts=copy.deepcopy(policy.telemetry)
                        else:
                            receipt=policy.teacher_report(rng) if kind=='teacher' else policy.work_report(rng)
                            counts=receipt['phases']['canal_early']
                        row=dict(kind=kind,mode=mode,elapsed_seconds=elapsed,counts=counts,decisions=decisions,
                            qualified_samples=sum(d['label'] for d in decisions),workers=4)
                        reports.append(row);write_json(out/(kind+'-'+mode+'.json'),row);print(row,flush=True)
                finally:lease.close();policy.close()
        require(service.cpu.report()['active']==0,'CPU lease leaked')
        inference=service.report()
    write_json(out/'report.json',dict(implementation_started=source,implementation_finished=implementation_id(),
        results=reports,inference=inference,note=f'{len(reports)} native two-action turns on {args.device}; random small fixture model, not strength evidence. Includes cold native planning; compare completed work and labels, not VRAM.'))


if __name__=='__main__':main()
