"""Two native turns, not games: generic evaluation and scheduled NN teacher."""
import argparse
import copy
from pathlib import Path
import random
import time

import torch

from fixtures import game
from brass_ai.bank import read
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.reward import CostLedger
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import NN_WORK_BUDGET,TEACHER_WORK_BUDGET
from brass_ai.recovery.execution import make_policy
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.resources import TurnLease
from brass_ai.recovery.teacher_schedule import CONTRACT,reserved_rounds
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset,prepared
from brass_ai.recovery.utility import calibrate,calibration_ready,reward_spec
from brass_ai.search.common import turn_key
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Choose a new artifact directory')
    torch.set_num_threads(2);torch.manual_seed(91417)
    lab=read(SOURCE_ROOT/'configs/step5/lab.json');cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(lab)))
    data=prepared([game()],insolvency=model.config.utility['insolvency'],reward=model.config.utility)
    calibrate(model,torch.optim.AdamW(model.parameters(),lr=lab['learning_rate']),Dataset(data,'cpu',policy=False),cfg['update'])
    require(calibration_ready(model),'Fixture utility fit not ready')
    model.utility_calibration=dict(model.utility_calibration,scope='Authored ordinary replay fixture, runtime smoke only')
    execution=dict(cfg['execution'],processes=1,heavy_processes=1,search_workers_per_turn=4)
    seed=25516000;reports={};started=implementation_id()
    with InferenceService(execution,'cpu') as service:
        identity=service.register(model)
        for mode in ('evaluation','teacher'):
            spec=dict(kind='masked_ismcts',model_id=identity,opponents=[identity],
                search=dict(lab['search'],workers=4),focus='neutral',cpu_role='reference')
            if mode=='teacher':spec.update(teacher_schedule=CONTRACT,teacher_work_budget=copy.deepcopy(TEACHER_WORK_BUDGET))
            else:spec['nn_work_budget']=copy.deepcopy(NN_WORK_BUDGET)
            channels=service.channels(5);endpoints=[channels.acquire() for _ in range(5)]
            policy=make_policy(spec,SOURCE_ROOT,endpoints,execution,None)
            lease=TurnLease(service.cpu,Timings());rng=random.Random(921)
            try:
                with RulesHost(SOURCE_ROOT/cfg['host']) as host:
                    context=host.reset(seed);seat=context['actor'];key=turn_key(context)
                    sampling=next(i for i in range(1000) if reserved_rounds(seed,i,seat)['canal_early']==context['round'])
                    if mode=='teacher':policy.begin_teacher_game(rng,seat,seed,sampling)
                    else:policy.begin_work_game(rng,seat)
                    ledger=CostLedger(model.config.utility);decisions=[];began=time.monotonic()
                    for step in range(96):
                        if context['terminal'] or context['actor']!=seat or turn_key(context)!=key:break
                        lease.update(key+(seat,),4)
                        if mode=='teacher':policy.admit_teacher_turn(context,rng)
                        else:policy.admit_work_turn(context,rng)
                        view=host.observe(seat);choices=host.choices();ledger.observe(view);policy.set_reward_context(ledger)
                        chosen,_,teacher=policy.select_batch([(view,choices,rng)])[0]
                        require(0<=chosen<len(choices),'Illegal native selection')
                        if mode=='evaluation':require(teacher is None,'Evaluation exported a teacher label')
                        decisions.append(dict(phase=context['phase'],choices=len(choices),chosen=choices[chosen]['id'],label=teacher is not None))
                        ledger.chosen(view,choices[chosen]);context=host.apply(context,choices[chosen])
                    else:raise AssertionError('Native current-turn completion cap exceeded')
                    report=policy.teacher_report(rng) if mode=='teacher' else policy.work_report(rng)
                    counts=report['phases']['canal_early'];expected=32 if mode=='teacher' else 16
                    require(counts['allocated_simulations']==counts['attempted_simulations']==expected,'Native quota not completed')
                    require(counts['actual_searches']==1,'Subchoice restarted native work')
                    require(counts['timeouts']==0 and counts['deadline_limited_turns']==0,'Watchdog limited normal native work')
                    require(counts['completed_rollouts']>0,'Native search produced no completed evidence')
                    reports[mode]=dict(seconds=time.monotonic()-began,counts=counts,decisions=decisions,workers=4,quota=expected)
                    write_json(out/(mode+'.json'),reports[mode]);print(dict(mode=mode,**reports[mode]),flush=True)
            finally:lease.close();policy.close()
        inference=service.report()
        require(service.cpu.report()['active']==0,'CPU lease leaked')
    write_json(out/'report.json',dict(native_turns=2,implementation_started=started,implementation_finished=implementation_id(),
        results=reports,inference=inference,note='Fresh small model and authored utility calibration; native correctness, not strength or lab GPU speed evidence.'))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
