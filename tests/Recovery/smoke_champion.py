"""Four native learner/incumbent games across resume, plus one gate-format game."""
import argparse
from pathlib import Path
import time

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery.champion import jobs_for,gate_rows,decision
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
    require(model.config.utility['vp']==.001 and model.config.utility['insolvency']['max_cost']==.025,'Wrong resumed reward')
    reports=[first['summaries'][0],second['summaries'][0]]
    for report in reports:
        require(report['actor']['reward']['trajectories']==2,'Opponent actions incorrectly trained as learner')
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
    write_json(out/'report.json',dict(native_games=5,ordinary_games=4,gate_games=1,seconds=time.monotonic()-began,
        iterations=state['iteration'],best_generation=state['champion']['generation'],learner_generation=second['checkpoint']['generation'],
        reward=model.config.utility,learner_trajectories=[r['actor']['reward']['trajectories'] for r in reports],
        ppo_steps=[r['actor']['optimizer_steps'] for r in reports],gate_outcomes=rows,
        resume_passed=True,untested_candidate_not_promoted=True,incomplete_gate_rejected=True,
        scope='CPU execution smoke only; no strength claim or full 48-game promotion gate'))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
