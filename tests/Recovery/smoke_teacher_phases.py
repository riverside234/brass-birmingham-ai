"""Bounded native phase searches; paired 12/24-second late-Rail positions.

Uses observed positions from complete ordinary training replays, never edits the
engine state. The separate authored five-game update checks sample accounting;
neither part is a playing-strength test or modifies a training run.
"""
import argparse
import copy
from pathlib import Path
import random
import time

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.reward import CostLedger
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import TEACHER_TIME_BUDGET,teacher_time_seconds
from brass_ai.recovery.execution import make_policy
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.resources import TurnLease
from brass_ai.recovery.teacher_schedule import CONTRACT,PHASES,phase_for,reserved_rounds,counters
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset,prepared,distill_update
from brass_ai.recovery.utility import calibrate,calibration_ready,reward_spec
from brass_ai.search.common import SearchConfig,turn_key
from brass_ai.search.ismcts import visit_teacher
from brass_ai.searchlab.spiel import BrassSpielGame,action_id
from brass_ai.step5.league import implementation_id,write_json


def positions(games,limit,reward):
    found={p:[] for p,_,_,_ in PHASES}
    for episode in games:
        ledger=CostLedger(reward);seen=set()
        for row in episode['records']:
            view=row['view'];ledger.observe(view);phase=phase_for(view['context'])
            if (phase not in seen and phase in found and len(found[phase])<limit and
                    view['context']['phase']=='action' and view['context']['actions_remaining']==2 and len(row['choices'])>1):
                found[phase].append((episode,row,copy.deepcopy(ledger)));seen.add(phase)
            ledger.chosen(view,row['choices'][row['chosen']])
    require(all(len(rows)==limit for rows in found.values()),'Need enough complete replays with two-action roots in every phase')
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--positions-per-phase',type=int,default=2,choices=(1,2,3,4))
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Choose a new artifact directory')
    source=implementation_id();began=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(736150)
    cfg=read(SOURCE_ROOT/'configs/step5/lab.json');reward=reward_spec(cfg)
    games=read_shards(args.replay,partition='train');roots=positions(games,args.positions_per_phase,reward)
    model=UtilityNet(utility_config(ModelConfig.preset('small'),reward))
    ordinary=Dataset(prepared(games,insolvency=reward['insolvency'],reward=reward),'cpu',policy=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    calibration=calibrate(model,optimizer,ordinary,cfg['update']);require(calibration_ready(model),'Smoke calibration failed')
    execution=dict(cfg['execution'],processes=1,heavy_processes=1)
    budgets={'current':copy.deepcopy(TEACHER_TIME_BUDGET),'baseline':copy.deepcopy(TEACHER_TIME_BUDGET)}
    budgets['baseline']['seconds_per_phase']['rail_late']=12.
    results=[]
    with InferenceService(execution,'cpu') as service:
        identity=service.register(model);policies={};driver=BrassSpielGame(root=SOURCE_ROOT,fast=True)
        lease=TurnLease(service.cpu,Timings())
        try:
            for mode,work in budgets.items():
                channels=service.channels(5)
                spec=dict(kind='masked_ismcts',model_id=identity,opponents=[identity],search=cfg['search'],
                    focus='neutral',cpu_role='reference',teacher_schedule=CONTRACT,teacher_work_budget=work)
                policies[mode]=make_policy(spec,SOURCE_ROOT,[channels.acquire() for _ in range(5)],execution,None)
            for phase,rows in roots.items():
                for index,(episode,row,ledger_original) in enumerate(rows):
                    modes=(('baseline','current') if index%2==0 else ('current','baseline')) if phase=='rail_late' else ('current',)
                    for mode in modes:
                        policy=policies[mode];view=copy.deepcopy(row['view']);seat=view['seat']
                        state=driver.from_observation(view,736150+index)
                        require([c['id'] for c in state.legal]==[c['id'] for c in row['choices']],'Reconstructed root legality differs')
                        rng=random.Random(9413+index);seed=736150+index
                        sampling=next(i for i in range(10000) if reserved_rounds(seed,i,seat)[phase]==view['context']['round'])
                        policy.begin_teacher_game(rng,seat,seed,sampling)
                        ledger=copy.deepcopy(ledger_original);key=turn_key(state.context);decisions=[]
                        lease.update(key+(seat,),4);started=time.monotonic()
                        print(dict(phase=phase,index=index,mode=mode,seconds=teacher_time_seconds(budgets[mode],phase)),flush=True)
                        for step in range(96):
                            if state.is_terminal() or state.is_chance_node() or state.current_player()!=seat or turn_key(state.context)!=key:break
                            live=state.views[seat];choices=state.legal
                            policy.admit_teacher_turn(live['context'],rng);ledger.observe(live);policy.set_reward_context(ledger)
                            chosen,_,teacher=policy.select_batch([(live,choices,rng)])[0]
                            require(0<=chosen<len(choices),'Illegal teacher selection')
                            if teacher is not None:
                                require(teacher['search']['seconds']==teacher_time_seconds(budgets[mode],phase),'Label clock differs')
                            decisions.append(dict(phase=live['context']['phase'],actions_remaining=live['context']['actions_remaining'],
                                legal=len(choices),chosen=choices[chosen]['id'],qualified_label=teacher is not None))
                            ledger.chosen(live,choices[chosen]);state.apply_action(action_id(choices[chosen]['id']))
                        else:raise AssertionError('Continuation cap exceeded')
                        receipt=policy.teacher_report(rng);lease.close()
                        result=dict(phase=phase,index=index,mode=mode,original_game=episode['game_id'],
                            position=digest(dict(view=row['view'],choices=row['choices'])),model_id=identity,
                            elapsed_seconds=time.monotonic()-started,counts=receipt['phases'][phase],
                            decisions=decisions,receipt=receipt)
                        results.append(result);write_json(out/f'{phase}-{index}-{mode}.json',result)
                        print(result['counts'],flush=True)
            require(service.cpu.report()['active']==0,'CPU lease leaked')
            inference=service.report()
        finally:
            lease.close();driver.close()
            for policy in policies.values():policy.close()
    # A separate, explicitly authored five-game collection exercises 80 distinct
    # eligible rows and the real optimizer. Native partial-turn probes are never
    # represented as complete teacher games or used for distillation.
    from test_selfplay_review import teacher_games
    authored=teacher_games(5);identity=model_id(model)
    for episode in authored:
        for row in episode['records']:
            phase=phase_for(row['view']['context']);search=SearchConfig(**dict(cfg['search'],seconds=teacher_time_seconds(TEACHER_TIME_BUDGET,phase)))
            choices=row['choices'];stats=dict(expansion=[True]*len(choices),root={c['id']:dict(visits=n) for c,n in zip(choices,[9,1,1])})
            row['teacher']=visit_teacher(choices,stats,[.8,.1,.1],model=identity,observer=row['seat'],config=search)
    update=distill_update(model,optimizer,authored,cfg['update'],steps=cfg['distill_steps'],quality=cfg['distillation'])
    aggregates={}
    for mode in budgets:
        aggregates[mode]={}
        for phase,_,_,_ in PHASES:
            selected=[r for r in results if r['phase']==phase and r['mode']==mode]
            if selected:aggregates[mode][phase]={k:sum(r['counts'][k] for r in selected) for k in counters()}
    require(source==implementation_id(),'Source changed during smoke')
    report=dict(implementation=source,elapsed_seconds=time.monotonic()-began,
        inputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in args.replay],budgets=budgets,
        calibration=calibration,native=results,phases=aggregates,inference=inference,
        native_usage=dict(qualified_labels=sum(r['counts']['qualified_samples'] for r in results),distinct_labels_used=0,
            reason='Matched partial-turn probes, not complete teacher games; never enter training'),
        authored_five_game_distillation=update,
        scope='Local CPU, small random model calibrated on ordinary smoke replay. Matched observed late-Rail roots, alternating budget order. Coverage/work validation only; authored update is separate; no playing-strength conclusion.')
    write_json(out/'report.json',report);print('Smoke report:',out/'report.json',flush=True)


if __name__=='__main__':main()
