"""Eleven bounded native games: resumed PPO, each rollout stage and gate format."""
import argparse
import copy
from pathlib import Path
import random
import time

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery.champion import jobs_for,gate_rows,decision
from brass_ai.recovery import champion_mix
from brass_ai.recovery.config import champion_mix_config
from brass_ai.recovery.execution import neural_spec
from brass_ai.recovery.execution import ActorPool
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.league import train
from brass_ai.recovery.replay import read_shards
from brass_ai.step5.league import seeds_for,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new workspace artifact directory')
    cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    cfg.update(games_per_iteration=2,threads=2,audit_workers=2,search_every_games=0)
    cfg['update']['epochs']=1;cfg['evaluation_schedule']['enabled']=False
    cfg['legacy_boost']=dict(stages=[dict(games=20,selfplay=95,historical=0,legacy=5)])
    began=time.monotonic();first=train(SOURCE_ROOT,out/'league',cfg,iterations=1)
    best_hash=sha(out/'league/best.pt')
    second=train(SOURCE_ROOT,out/'league',cfg,iterations=1)
    state=read(out/'league/state.json')
    require(second['completed_iteration']==2 and state['games_done']==4,'Resume counters changed')
    require(second['best']['generation']==0 and sha(out/'league/best.pt')==best_hash,'Untested learner replaced champion')
    model,raw=load_checkpoint(SOURCE_ROOT/second['checkpoint']['path'])
    require(model.config.utility['vp']==.001 and model.config.utility['insolvency']['max_cost']==.075,'Wrong resumed reward')
    reports=[first['summaries'][0],second['summaries'][0]]
    for report in reports:
        require(report['actor']['reward']['trajectories']==2,'Opponent actions incorrectly trained as learner')
        require(report['selfplay_opponents']['tables_by_champion_count']=={'1':2,'2':0},'Bootstrap champion seat cap differs')
    require(state['champion_mix']['origin_games']==0 and state['champion_mix']['stage_games']==10,'Resume reset/stretched rollout origin')
    # Controlled sampler smoke: these learned checkpoints have NOT passed a
    # real champion gate. Exercise actual frozen policies without promoting them.
    cfg=champion_mix_config(cfg);simulated=copy.deepcopy(state);simulated['champion']=second['checkpoint']
    champion_mix.promoted(simulated,first['checkpoint'])
    origin=simulated['games_done'];span=simulated['champion_mix']['stage_games']
    stage_jobs=[];stage_receipts=[];selected=seeds_for(1960000,6)[0]
    learner=neural_spec(second['checkpoint']['model_id'],.05,'mixed')
    for phase in range(3):
        simulated['games_done']=origin+phase*span
        for i in range(2):
            index=2*phase+i;rotation=index%4
            entries,receipt=champion_mix.opponents(simulated,cfg,random.Random(index),transition_index=i)
            policies=champion_mix.policies(learner,entries)
            stage_jobs.append(dict(index=index,seed=selected[index],rotation=rotation,sampling_seed=selected[index]*7,
                policies=policies,focuses={str(rotation):'neutral'},path=str(out/'stages'/f'game-{index}.jsonl.gz'),
                provenance=dict(purpose='training',kind='selfplay',champion_rollout=receipt)))
            stage_receipts.append(receipt)
    with InferenceService(cfg['execution'],'cpu') as service:
        for e in (first['checkpoint'],second['checkpoint'],state['evaluation_anchor']):service.register_checkpoint(SOURCE_ROOT/e['path'])
        with ActorPool(SOURCE_ROOT,cfg['host'],service,cfg['execution']) as pool:stage_records=pool.collect(stage_jobs)
    stage_games=read_shards([r['path'] for r in stage_records],partition='train',receipts={r['path']:r.get('features') for r in stage_records})
    for game in stage_games:
        require(game['opponents'].count(learner['policy_id'])==1,'Stage smoke attributed opponents to PPO learner')
        champion_policy=neural_spec(second['checkpoint']['model_id'],.05,'neutral')['policy_id']
        require(game['opponents'].count(champion_policy)<=2,'Native game exceeded champion seat cap')
        require(neural_spec(state['evaluation_anchor']['model_id'],.05,'neutral')['policy_id'] in game['opponents'],'Native table lost stable anchor')
    require([sorted(r['champion_opponents'] for r in stage_receipts[i:i+2]) for i in (0,2,4)]==[[1,1],[1,2],[2,2]],'Stage allocation differs')
    seeds=seeds_for(1944000,1,'validation')[0]
    jobs=jobs_for(seeds,second['checkpoint'],second['best'],out/'gate','native-smoke')[:1]
    with InferenceService(cfg['execution'],'cpu') as service:
        for entry in (second['checkpoint'],second['best']):service.register_checkpoint(SOURCE_ROOT/entry['path'])
        with ActorPool(SOURCE_ROOT,cfg['host'],service,cfg['execution']) as pool:records=pool.collect(jobs)
    games=read_shards([r['path'] for r in records],partition='validation',receipts={r['path']:r.get('features') for r in records})
    rows=gate_rows(games,records,jobs)
    try:decision(rows,cfg['champion'])
    except ValueError:pass
    else:raise AssertionError('Incomplete smoke gate could promote')
    write_json(out/'report.json',dict(native_games=11,ordinary_games=4,rollout_stage_games=6,gate_games=1,seconds=time.monotonic()-began,
        iterations=state['iteration'],best_generation=state['champion']['generation'],learner_generation=second['checkpoint']['generation'],
        reward=model.config.utility,learner_trajectories=[r['actor']['reward']['trajectories'] for r in reports],
        ppo_steps=[r['actor']['optimizer_steps'] for r in reports],gate_outcomes=rows,rollout_receipts=stage_receipts,
        resume_passed=True,untested_candidate_not_promoted=True,incomplete_gate_rejected=True,
        scope='CPU execution smoke only; no strength claim or full 48-game promotion gate'))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
