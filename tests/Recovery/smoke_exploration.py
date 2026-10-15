"""Bounded native 0.05 collection followed by a reviewed 0.08 continuation.

Run baseline before changing source, then continuation after registering the
review. All artifacts and authored small IL-format weights are separate from
production. Each stage plays three ordinary games with unchanged v4 clocks.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import time
from unittest.mock import patch

import torch

from latest_model_fixture import imitation_fixture
from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery import champion,champion_mix,ordinary
from brass_ai.recovery.league import train
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.updates import Dataset,probabilities,ppo_update
from brass_ai.step5.league import implementation_id,write_json


def behavior_probe(model,data,behavior,config):
    """All nonforced learner decisions, before any PPO/calibration updates."""
    ds=Dataset(data,next(model.parameters()).device)
    ids=[i for i,r in enumerate(ds.rows) if len(r['choices'])>1]
    families=Counter();industries=Counter();entropy=counterfactual=error=0.
    previous=model.training;model.eval()
    with torch.no_grad():
        parts=ds.microbatches(ids,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])
        for part,ix,inputs,q in ds.policy_batches(parts,behavior.exploration):
            out=model(*inputs,remaining=ds.remaining[ix]) if ds.utility is not None else model(*inputs)
            p=probabilities(out[0],q,behavior.exploration,behavior.temperature)
            old=probabilities(out[0],q,.05,1.)
            require(torch.allclose(p.sum(-1),torch.ones(len(part)),rtol=1e-5,atol=1e-6),'Unnormalized mixture')
            chosen=p.gather(1,ds.chosen[ix,None]).squeeze(1)
            require(torch.allclose(chosen,ds.old[ix],rtol=3e-3,atol=3e-6),'Recorded behavior differs')
            error=max(error,float((chosen-ds.old[ix]).abs().max()))
            entropy+=float(-(p*p.clamp_min(1e-30).log()).sum())
            counterfactual+=float(-(old*old.clamp_min(1e-30).log()).sum())
            for i in part:
                row=ds.rows[i];choice=row['choices'][row['chosen']]
                require(bool(torch.all(p[part.index(i),len(row['choices']):]==0)),'Illegal padded probability')
                families[choice['kind']]+=1
                if choice['kind'] in ('build','develop'):industries[choice.get('industry','unspecified')]+=1
    model.train(previous)
    return dict(exploration=behavior.exploration,temperature=behavior.temperature,decisions=len(ids),
        mean_behavior_entropy_nats=entropy/len(ids),same_positions_entropy_at_005_nats=counterfactual/len(ids),
        maximum_selected_probability_error=error,
        action_families={k:dict(count=v,fraction=v/len(ids)) for k,v in sorted(families.items())},
        build_develop_industries=dict(industries),
        scope='All nonforced ordinary learner decisions in this three-game collection, before PPO/utility calibration. Frequencies are sampled legal-choice kinds; the 0.05 comparison uses these same states/model/contexts, not another strength trial.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('baseline','continuation'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts'),'Use a workspace artifact directory')
    began=time.monotonic();torch.set_num_threads(2);source=implementation_id()
    if args.stage=='baseline':
        require(not out.exists(),'Baseline needs a new directory')
        il,_,cfg=imitation_fixture(out);cfg['exploration']=.05
        write_json(out/'config005.json',cfg)
    else:
        require((out/'baseline.json').is_file() and not (out/'report.json').exists(),'Complete baseline first; continuation may run once')
        cfg=read(out/'config005.json');cfg['exploration']=.08;il=out/'fixture-il'
    run=out/'league';before=read(run/'state.json') if args.stage=='continuation' else None
    checkpoint=SOURCE_ROOT/before['history'][-1]['path'] if before else None
    checkpoint_hash=sha(checkpoint) if checkpoint else None
    probes=[];restored_receipts=[];collection_receipts=[]
    def update(model,optimizer,games,behavior,config,**kwargs):
        probes.append(behavior_probe(model,kwargs['learner_data'],behavior,config))
        return ppo_update(model,optimizer,games,behavior,config,**kwargs)
    def restore(raw,optimizer,**kwargs):
        result=resume_optimizer(raw,optimizer,**kwargs)
        require(_same_payload(raw['optimizer'],optimizer.state_dict()),'Adam restore differs')
        require(torch.equal(torch.get_rng_state(),raw['rng']),'CPU RNG restore differs')
        restored_receipts.append(dict(step=raw['step'],optimizer_states=len(optimizer.state),rng_exact=True))
        return result
    planner=ordinary.plan_collection
    def plan(root,directory,state,config,signature,service,current,*args):
        if before:
            require(current==before['history'][-1]['model_id'],'Resume changed incoming model weights')
            require({k:v for k,v in state.items() if k not in ('signature','source_transitions')}==
                    {k:v for k,v in before.items() if k not in ('signature','source_transitions')},'Continuation reset progress/population')
            collection_receipts.append(dict(incoming_model_exact=True,counters_population_lineage_exact=True))
        return planner(root,directory,state,config,signature,service,current,*args)
    with ExitStack() as stack:
        for owner,names in ((champion,('initialize','schedule','publish_best')),
                            (champion_mix,('initialize','opponents','summarize'))):
            for name in names:stack.enter_context(patch.object(owner,name,side_effect=AssertionError('Unexpected gate call')))
        stack.enter_context(patch('brass_ai.recovery.league.ppo_update',side_effect=update))
        stack.enter_context(patch('brass_ai.recovery.league.resume_optimizer',side_effect=restore))
        stack.enter_context(patch.object(ordinary,'plan_collection',side_effect=plan))
        result=train(SOURCE_ROOT,run,cfg,iterations=1,imitation_run=il,
            fresh=args.stage=='baseline',expected_generation=1 if before else None,resume_reviewed_source=bool(before))
    report=result['summaries'][0];state=read(run/'state.json')
    require(implementation_id()==source,'Source changed while collecting')
    receipt=dict(implementation=source,elapsed_seconds=time.monotonic()-began,probe=probes[0],
        sampling=report['sampling'],actor=report['actor'],resume=restored_receipts,
        collection=collection_receipts,inference_error=report['inference']['error'],teacher_collections=report['search'].get('collection_count',0))
    if args.stage=='baseline':write_json(out/'baseline.json',receipt)
    else:
        require(sha(checkpoint)==checkpoint_hash,'Continuation changed old checkpoint bytes')
        require(state['iteration']==2 and state['games_done']==6,'Ordinary progress differs')
        require(state['signature']['config']['exploration']==.08,'Exploration not committed')
        require(receipt['sampling']['learner_trajectories']==6,'Learner attribution differs')
        write_json(out/'report.json',dict(baseline=read(out/'baseline.json'),continuation=receipt,
            previous_checkpoint_unchanged=True,ordinary_games=6,iteration=2,
            source_transitions=state['source_transitions'],
            scope='Six native games with authored small IL-format fixture weights, CPU inference, reduced actor concurrency and no rehearsal/diagnostic panels/retention. Production checkpoints are untouched; no teacher update or playing-strength evaluation.'))
    print('Smoke stage complete:',args.stage,out,flush=True)


if __name__=='__main__':main()
