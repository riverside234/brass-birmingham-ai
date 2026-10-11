"""Four native ordinary games across a reviewed old-source checkpoint boundary.

Creates a small format-compatible fixture using the preceding source receipt;
does not import or claim to evaluate the trained model still on the lab.
"""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery.league import train
from brass_ai.recovery.utility import calibration_ready
from brass_ai.step5.league import implementation_id,write_json

OLD='826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new workspace smoke directory')
    cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
    cfg.update(games_per_iteration=2,threads=2,audit_workers=2,search_every_games=0)
    cfg['update']['epochs']=1;cfg['evaluation_schedule']['enabled']=False
    cfg['legacy_boost']=dict(stages=[dict(games=20,selfplay=95,historical=0,legacy=5)])
    began=time.monotonic();run=out/'league'
    # This small ordinary-only fixture uses unchanged PPO/model/data contracts.
    # Exercise the real source gate, not a mock of checkpoint/optimizer loading.
    with patch('brass_ai.recovery.league.implementation_id',return_value=OLD):
        first=train(SOURCE_ROOT,run,cfg,iterations=1)
    previous=read(run/'state.json');saved=SOURCE_ROOT/first['checkpoint']['path'];before=sha(saved)
    best_before=sha(run/'best.pt')
    second=train(SOURCE_ROOT,run,cfg,iterations=1,resume_reviewed_source=True)
    state=read(run/'state.json');model,raw=load_checkpoint(SOURCE_ROOT/second['checkpoint']['path'])
    require(state['iteration']==2 and state['games_done']==4,'Resume reset learner progress')
    require(sha(saved)==before and sha(run/'best.pt')==best_before,'Existing learner/incumbent bytes changed')
    require(len(state['source_transitions'])==1 and state['signature']['implementation']==implementation_id(),
            'Missing explicit reviewed transition')
    require(state['legacy_boost']==previous['legacy_boost'],'Recovery opponent window restarted')
    require(calibration_ready(model) and second['summaries'][0]['actor']['utility_calibration']['required'],
            'Fresh ordinary returns did not recalibrate the resumed utility head')
    require(state['champion']['generation']==0,'Untested learner silently replaced accepted best')
    write_json(out/'report.json',dict(scope='CPU/native old-source format fixture; no lab model or strength claim',
        seconds=time.monotonic()-began,native_games=4,iterations=2,from_implementation=OLD,
        to_implementation=implementation_id(),resume_passed=True,previous_checkpoint_unchanged=True,
        incumbent_unchanged=True,opponent_window_preserved=True,fresh_utility_recalibration=True,
        ppo_steps=[r['actor']['optimizer_steps'] for r in (first['summaries'][0],second['summaries'][0])],
        checkpoint=second['checkpoint'],reward=model.config.utility))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
