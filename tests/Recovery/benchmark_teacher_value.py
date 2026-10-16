"""Isolated saved-data component comparison; no searches or strength claim.

The authored game/label fixtures exercise real encoding, heads, Adam and
calibration. They are not native played-game outcomes or a full iteration.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import platform
import statistics
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer,save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.recovery.replay import read_shards,write_shard
from brass_ai.recovery.teacher_value import TeacherValue,provenance
from brass_ai.recovery.updates import Dataset,distill_update,prepared
from brass_ai.recovery.utility import calibrate
from brass_ai.step5.league import implementation_id,write_json,seeds_for
from test_teacher_value import TeacherValueTests


def optimizer_identity(optimizer):
    state=optimizer.state_dict();h=hashlib.sha256(digest(state['param_groups']).encode())
    for i,values in sorted(state['state'].items()):
        for k,v in sorted(values.items()):
            h.update(str((i,k)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=6)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifact directory')
    require(2<=args.repeats<=10,'Use 2-10 bounded repeated comparisons')
    out.mkdir(parents=True);TeacherValueTests.setUpClass()
    initial,opt,games,ordinary,options,update,quality,bindings=TeacherValueTests().fixture(games=5,model_size='medium',dense_labels=True)
    # Match the production default sample cap; all 80 authored rows fit.
    options=dict(options,samples_per_game=24)
    write_shard(out/'teacher.jsonl.gz',games,dict(purpose='training',kind='authored-current-source-test-data',
        target=provenance(model_id(initial),initial.config.utility,options)))
    games=read_shards([out/'teacher.jsonl.gz'],partition='train')
    ordinary_game=copy.deepcopy(dict(games[0]));keys=set(ordinary_game['records'][0])
    ordinary_game.update(game_id=ordinary[0][2][0],
        setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seeds_for(85500,1)[0][0])),
        records=[{k:v for k,v in d[0].items() if k in keys} for d in ordinary])
    write_shard(out/'ordinary.jsonl.gz',[ordinary_game],dict(purpose='training',kind='authored-ordinary-test-data'))
    ordinary=prepared(read_shards([out/'ordinary.jsonl.gz'],partition='train'),
                      insolvency=initial.config.utility['insolvency'],reward=initial.config.utility)
    save_checkpoint(out/'initial.pt',initial,opt,step=0,training=dict(mode='teacher-value-isolated-authored-benchmark'))
    saved={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    # Ordinary data and both label families are fixed for every repetition.
    ordinary_dataset=Dataset(ordinary,'cpu',policy=False)
    observations=[];examples={};modes=('policy_only','auxiliary','auxiliary_diagnostics')
    for repeat in range(-1,args.repeats):
        order=modes if repeat%2==0 else tuple(reversed(modes))
        for mode in order:
            model,raw=load_checkpoint(out/'initial.pt');optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.0001)
            resume_optimizer(raw,optimizer)
            opts=dict(options,diagnostics=mode=='auxiliary_diagnostics')
            expected=provenance(model_id(model),model.config.utility,opts)
            bindings={g['game_id']:dict(expected,game_digest=digest(g)) for g in games}
            began=time.perf_counter()
            aux=(TeacherValue(model,games,ordinary,opts,update,bindings,ordinary_policy_id='fixture')
                 if mode!='policy_only' else None)
            report=distill_update(model,optimizer,games,update,steps=8,quality=quality,auxiliary=aux)
            calibrated=calibrate(model,optimizer,ordinary_dataset,update,auxiliary=aux)
            elapsed=time.perf_counter()-began
            summary=dict(mode=mode,repeat=repeat,seconds=elapsed,optimizer_steps=report['optimizer_steps'],
                calibration_steps=calibrated['receipt']['steps'],model_id=model_id(model),
                diagnostic_seconds=aux.probe.seconds if aux and aux.probe else 0.,
                auxiliary_training_seconds=aux.training_seconds if aux else 0.,
                teacher_calibration_seconds=calibrated['receipt'].get('teacher_seconds',0.),
                optimizer_id=optimizer_identity(optimizer),rng_id=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            if repeat>=0:observations.append(summary)
            examples[mode]=dict(distillation=report,calibration=calibrated,teacher_value=aux.report() if aux else None)
            print(f'{mode} repeat={repeat}: {elapsed:.4f}s, {summary["optimizer_steps"]} distillation steps + '
                  f'{summary["calibration_steps"]} existing calibration steps',flush=True)
    for repeat in range(args.repeats):
        off=next(r for r in observations if r['repeat']==repeat and r['mode']=='auxiliary')
        on=next(r for r in observations if r['repeat']==repeat and r['mode']=='auxiliary_diagnostics')
        require(all(off[k]==on[k] for k in ('model_id','optimizer_id','rng_id','optimizer_steps','calibration_steps')),
                'Telemetry changed training state')
    require(saved=={name:sha(out/name) for name in saved},'Saved benchmark input/checkpoint changed')
    medians={mode:statistics.median(r['seconds'] for r in observations if r['mode']==mode) for mode in modes}
    incremental=medians['auxiliary_diagnostics']-medians['auxiliary']
    paired=[next(r['seconds'] for r in observations if r['repeat']==i and r['mode']=='auxiliary_diagnostics')-
            next(r['seconds'] for r in observations if r['repeat']==i and r['mode']=='auxiliary') for i in range(args.repeats)]
    report=dict(format='teacher-value-component-comparison/1.0.0',implementation=implementation_id(),
        hardware=dict(platform=platform.platform(),processor=platform.processor(),device='cpu',threads=torch.get_num_threads(),torch=torch.__version__),
        data=dict(games=5,positions=sum(len(g['records']) for g in games),source='Authored four-phase native-board fixtures with synthetic current-source search labels and terminal outcomes',
            correlated=True,ordinary_positions=len(ordinary),files=saved),
        model=dict(size='medium',parameters=initial.parameter_count),warmups=1,repeats=args.repeats,
        ordering='alternating forward/reverse modes; identical saved checkpoint/data and restored RNG',
        median_seconds=medians,diagnostic_incremental_seconds=incremental,
        paired_diagnostic_deltas=dict(median_seconds=statistics.median(paired),minimum_seconds=min(paired),maximum_seconds=max(paired)),
        median_instrumented_diagnostic_seconds=statistics.median(r['diagnostic_seconds'] for r in observations if r['mode']=='auxiliary_diagnostics'),
        diagnostic_component_percent=100*incremental/medians['auxiliary'],
        full_iteration_overhead_target=dict(percent=1.,status='unverified',
            reason='Component benchmark omits actual collection, search, full PPO and representative calibration workload'),
        telemetry_training_state_exact=True,inputs_unchanged=True,measurements=observations,examples=examples,
        scope='Bounded implementation and component-cost evidence only; no playing-strength, stability or full-iteration overhead conclusion')
    write_json(out/'report.json',report)
    print(dict(median_seconds=medians,diagnostic_incremental_seconds=incremental,full_iteration_target='unverified'),flush=True)


if __name__=='__main__':main()
