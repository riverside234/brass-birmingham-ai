"""Six native games in a new fixture run: pinned IL handoff, PPO and resume.

Uses authored small IL-format test weights, not the lab's trained joint-7 model.
The production directory is never touched. Ordinary v4 clocks are unchanged.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
from unittest.mock import patch
import torch

from latest_model_fixture import imitation_fixture
from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery import champion,champion_mix
from brass_ai.recovery.league import train
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new workspace smoke directory')
    torch.set_num_threads(2);began=time.monotonic();source=implementation_id()
    il,original,cfg=imitation_fixture(out);run=out/'league'
    receipts=[]
    def restored(raw,optimizer,**kwargs):
        result=resume_optimizer(raw,optimizer,**kwargs)
        require(_same_payload(raw['optimizer'],optimizer.state_dict()),'Adam was not restored exactly')
        require(torch.equal(torch.get_rng_state(),raw['rng']),'CPU RNG was not restored exactly')
        receipts.append(dict(step=raw['step'],optimizer_states=len(optimizer.state),rng_restored=True))
        return result
    with ExitStack() as stack:
        for owner,names in ((champion,('initialize','schedule','publish_best','selfplay_policies')),
                            (champion_mix,('initialize','opponents','summarize'))):
            for name in names:stack.enter_context(patch.object(owner,name,side_effect=AssertionError('Unexpected champion call')))
        stack.enter_context(patch('brass_ai.recovery.league.resume_optimizer',side_effect=restored))
        stack.enter_context(patch('brass_ai.recovery.demonstrations.fresh_imitation',side_effect=AssertionError('Repeated IL')))
        first=train(SOURCE_ROOT,run,cfg,iterations=1,imitation_run=il,fresh=True)
        initial,raw=load_checkpoint(run/'initial.pt')
        require(raw['optimizer']['state']=={} and raw['step']==0,'PPO inherited IL optimizer/progress')
        require(_same_payload(initial.state_dict(),original.state_dict()),'PPO did not start from pinned IL policy')
        checkpoint=SOURCE_ROOT/first['checkpoint']['path'];before_hash=sha(checkpoint)
        before=read(run/'state.json')
        second=train(SOURCE_ROOT,run,cfg,iterations=1,imitation_run=il,expected_generation=1)
    state=read(run/'state.json');reports=[first['summaries'][0],second['summaries'][0]]
    require(sha(checkpoint)==before_hash,'Resume changed the preceding checkpoint')
    require(state['iteration']==2 and state['games_done']==6,'Resume changed ordinary progress')
    require(state['legacy_boost']==before['legacy_boost'] and state['legacy_boost']['start_games']==0,'Curriculum reset')
    require(state['signature']['imitation_source']==before['signature']['imitation_source'],'IL lineage changed')
    require(reports[1]['incoming_model_id']==reports[0]['post_update_model_id'],'Collection did not synchronize to updated learner')
    require(not any(k in state for k in ('champion','champion_gate','champion_mix')) and not (run/'best.pt').exists(),'Gate artifacts appeared')
    for report in reports:
        counts=report['composition'];expected=4*counts.get('selfplay',0)+counts.get('historical',0)+counts.get('legacy',0)
        require(report['actor']['learner_trajectories']==report['sampling']['learner_trajectories']==expected,'Learner seat attribution differs')
        require(report['search'].get('collection_count',0)==0,'Smoke consumed teacher collection games')
        require(report['inference']['error'] is None,'Native inference failed')
    require(implementation_id()==source,'Source changed during smoke')
    write_json(out/'report.json',dict(implementation=source,elapsed_seconds=time.monotonic()-began,
        native_games=6,completed_iteration=state['iteration'],ordinary_games=state['games_done'],
        original_il=cfg['imitation_checkpoint'],initial_optimizer_empty=True,resume_receipts=receipts,
        no_gate_calls=True,old_checkpoint_unchanged=True,curriculum_window=state['legacy_boost'],
        sampling=[r['sampling'] for r in reports],
        updates=[{k:r['actor'][k] for k in ('optimizer_steps','learner_decisions','meaningful_decisions','learner_trajectories','sample_usage','kl_early_stop','epochs')} for r in reports],
        teacher_clocks=cfg['teacher_work_budget'],teacher_updates=0,
        scope='CPU/native functional smoke with small authored IL-format fixture weights and reduced actor concurrency. No production model, run iterations or strength evaluation.'))
    print('Smoke report:',out/'report.json',flush=True)


if __name__=='__main__':main()
