"""Bounded width-304 fresh IL -> six ordinary games -> resume -> four eval games.

Optional four real teacher turns use the exact current phase clocks and quality
gates. Partial teacher probes never masquerade as complete training games.
"""
import argparse
import copy
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,require
from brass_ai.learning.reward import CostLedger
from brass_ai.recovery.config import latest_model_config,teacher_time_seconds
from brass_ai.recovery.demonstrations import train_bc,completed_imitation
from brass_ai.recovery.league import train
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import calibration_ready,reward_spec
from brass_ai.step5.league import implementation_id,write_json


def teacher_probe(model,games,cfg,out):
    from brass_ai.recovery.config import time_budget_config
    cfg=time_budget_config(cfg)
    from smoke_teacher_phases import positions
    from brass_ai.recovery.execution import make_policy
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.resources import TurnLease
    from brass_ai.recovery.teacher_schedule import CONTRACT,reserved_rounds
    from brass_ai.recovery.telemetry import Timings
    from brass_ai.search.common import turn_key
    from brass_ai.searchlab.spiel import BrassSpielGame,action_id
    require(calibration_ready(model),'Keep teacher calibration safeguards; smoke model is not ready')
    roots=positions(games,1,reward_spec(cfg));results=[]
    execution=cfg['execution']
    with InferenceService(execution,'cpu') as service:
        identity=service.register(model);channels=service.channels(5)
        policy=make_policy(dict(kind='masked_ismcts',model_id=identity,opponents=[identity],
            search=cfg['search'],focus='neutral',cpu_role='reference',teacher_schedule=CONTRACT,
            teacher_work_budget=cfg['teacher_work_budget']),SOURCE_ROOT,[channels.acquire() for _ in range(5)],execution,None)
        driver=BrassSpielGame(root=SOURCE_ROOT,fast=True);lease=TurnLease(service.cpu,Timings())
        try:
            for phase,rows in roots.items():
                episode,row,original=rows[0];view=row['view'];seat=view['seat'];seed=65413
                state=driver.from_observation(view,seed);require(state.legal==row['choices'],'Teacher root legality differs')
                rng=random.Random(seed);sampling=next(i for i in range(10000) if reserved_rounds(seed,i,seat)[phase]==view['context']['round'])
                policy.begin_teacher_game(rng,seat,seed,sampling);ledger=copy.deepcopy(original)
                key=turn_key(state.context);lease.update(key+(seat,),4);began=time.monotonic();labels=[]
                print(dict(teacher_phase=phase,seconds=teacher_time_seconds(cfg['teacher_work_budget'],phase)),flush=True)
                for _ in range(96):
                    if state.is_terminal() or state.is_chance_node() or state.current_player()!=seat or turn_key(state.context)!=key:break
                    live=state.views[seat];choices=state.legal
                    policy.admit_teacher_turn(live['context'],rng);ledger.observe(live);policy.set_reward_context(ledger)
                    chosen,_,label=policy.select_batch([(live,choices,rng)])[0]
                    require(0<=chosen<len(choices),'Illegal teacher result')
                    if label:labels.append(label)
                    ledger.chosen(live,choices[chosen]);state.apply_action(action_id(choices[chosen]['id']))
                else:raise ValueError('Teacher continuation cap exceeded')
                receipt=policy.teacher_report(rng);lease.close()
                result=dict(phase=phase,elapsed_seconds=time.monotonic()-began,counts=receipt['phases'][phase],
                    qualified_labels=len(labels),distinct_labels_used=0,receipt=receipt)
                results.append(result);write_json(out/f'{phase}.json',result)
            require(service.cpu.report()['active']==0,'Teacher CPU lease leaked')
            inference=service.report()
        finally:lease.close();driver.close();policy.close()
    return dict(phases=results,inference=inference,policy_drift=0.,distinct_labels_used=0,
        scope='Four observed whole-turn probes, no parameter updates or complete teacher games; all native labels remain diagnostic only.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--teacher',action='store_true')
    parser.add_argument('--ordinary-games-per-iteration',type=int,choices=(1,3),default=3)
    parser.add_argument('--teacher-from',type=Path,help='Read a completed isolated smoke league; probe current clocks without further training')
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new smoke output directory')
    torch.set_num_threads(2);began=time.monotonic();source=implementation_id()
    if args.teacher_from:
        from brass_ai.recovery.replay import read_shards
        run=args.teacher_from.resolve();require(run.is_relative_to(SOURCE_ROOT/'artifacts/feature-engineering'),'Use an isolated feature smoke league')
        state=read(run/'state.json');entry=state['history'][-1]
        model,_=load_checkpoint(SOURCE_ROOT/entry['path'])
        games=read_shards(sorted(run.glob('iteration-*/*.jsonl.gz')),partition='train')
        report=teacher_probe(model,games,state['signature']['config'],out)
        write_json(out/'report.json',report);return
    require(args.manifest is not None,'Pass the separately converted --manifest')
    cfg=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');update=copy.deepcopy(cfg['update'])
    update.update(effective_batch_size=64,microbatch_size=32,max_candidate_cells=8192,critic_batch_size=64,kl_probe_rows=128,epochs=1)
    il=out/'imitation'
    bc=train_bc(SOURCE_ROOT,il,args.manifest,None,update,epochs=1,min_epochs=1,patience=1,workers=1)
    original,origin=completed_imitation(SOURCE_ROOT,il)
    require(original.config.hidden==304 and original.config.residual_blocks==5 and original.parameter_count==2766477,
        'Smoke must use production width 304 and five residual blocks')
    # Same-version BC resume of the completed one-epoch run preserves progress.
    before_bc=read(il/'state.json');before_il=sha(SOURCE_ROOT/origin['path'])
    train_bc(SOURCE_ROOT,il,args.manifest,None,update,epochs=1,min_epochs=1,patience=1,workers=1)
    require(before_bc==read(il/'state.json') and before_il==sha(SOURCE_ROOT/origin['path']),'BC resume changed completed progress')
    cfg=latest_model_config(cfg,origin);cfg.update(games_per_iteration=args.ordinary_games_per_iteration,threads=2,audit_workers=1,
        rehearsal_manifest=None,rehearsal_weight=0.,update=update)
    cfg['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1)
    cfg['evaluation_schedule']['enabled']=False;cfg['retention']['enabled']=False
    write_json(out/'smoke-config.json',cfg);run=out/'league';receipts=[]
    def restored(raw,optimizer,**kwargs):
        value=resume_optimizer(raw,optimizer,**kwargs)
        require(_same_payload(raw['optimizer'],optimizer.state_dict()),'Adam resume differs')
        require(torch.equal(raw['rng'],torch.get_rng_state()),'RNG resume differs')
        receipts.append(dict(step=raw['step'],adam_states=len(optimizer.state),rng_exact=True));return value
    with patch('brass_ai.recovery.league.resume_optimizer',side_effect=restored):
        first=train(SOURCE_ROOT,run,cfg,iterations=1,imitation_run=il,fresh=True)
        initial,raw=load_checkpoint(run/'initial.pt')
        require(raw['optimizer']['state']=={} and raw['step']==0,'PPO inherited IL optimizer state')
        require(_same_payload(initial.state_dict(),original.state_dict()),'PPO did not inherit the fresh IL weights')
        prior=read(run/'state.json');path=SOURCE_ROOT/first['checkpoint']['path'];before=sha(path)
        second=train(SOURCE_ROOT,run,cfg,iterations=1,imitation_run=il,expected_generation=1)
    state=read(run/'state.json');reports=[first['summaries'][0],second['summaries'][0]]
    require(state['iteration']==2 and state['games_done']==2*args.ordinary_games_per_iteration and sha(path)==before,'PPO resume reset progress/checkpoint')
    require(state['legacy_boost']==prior['legacy_boost'] and state['signature']['imitation_source']==prior['signature']['imitation_source'],
            'PPO resume changed curriculum/IL lineage')
    require(reports[1]['incoming_model_id']==reports[0]['post_update_model_id'],'Updated model was not synchronized')
    from brass_ai.recovery.evaluation import evaluate
    evaluation=evaluate(SOURCE_ROOT,out/'evaluation',second['checkpoint']['path'],[str(run/'initial.pt')],
        read(SOURCE_ROOT/'configs/step5/evaluation.json'),setups=1,seed=54119,partition='validation',
        include_legacy=False,search=False,lineups=['recent-neural'])
    teacher=None
    if args.teacher:
        from brass_ai.recovery.replay import read_shards
        games=read_shards(sorted(run.glob('iteration-*/*.jsonl.gz')),partition='train')
        model,_=load_checkpoint(SOURCE_ROOT/second['checkpoint']['path'])
        teacher=teacher_probe(model,games,cfg,out/'teacher')
    require(source==implementation_id(),'Source changed during smoke')
    write_json(out/'report.json',dict(implementation=source,compatibility=compatibility(),elapsed_seconds=time.monotonic()-began,
        IL=dict(epochs=bc['epochs'],width=304,parameters=original.parameter_count,resume_exact=True,origin=origin),
        PPO=dict(iterations=2,games=state['games_done'],initial_weights_exact=True,initial_optimizer_empty=True,resume=receipts,
            previous_checkpoint_unchanged=True,sampling=[r['sampling'] for r in reports],
            updates=[r['actor'] for r in reports],inference=[r['inference'] for r in reports]),
        evaluation=evaluation,teacher=teacher,
        scope='Local CPU functional smoke: two real audited heuristic games, one fresh IL epoch, two PPO iterations and one four-seat-rotated evaluation setup. Reduced actors/minibatches/epochs for bounded validation; production recipe unchanged. No strength claim.'))
    print('FE5 smoke report:',out/'report.json',flush=True)


if __name__=='__main__':main()
