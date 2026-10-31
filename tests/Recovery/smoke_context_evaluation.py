"""28 real games in a new directory, including interrupted/missing-job reuse."""
import argparse
import copy
from pathlib import Path
import random
import time
import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require,digest
from brass_ai.learning.batching import model_id
from brass_ai.recovery import ratings
from brass_ai.recovery.context_evaluation import isolated_rng
from brass_ai.recovery import context_evaluation_config as C,context_evaluation_spec as S
from brass_ai.recovery import context_evaluation_collect as E,context_evaluation_analysis as A
from brass_ai.recovery.context_evaluation_plot import render
from brass_ai.step5.league import implementation_id,write_json,seeds_for


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();out=project_path(SOURCE_ROOT,a.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated output')
    out.mkdir(parents=True);began=time.perf_counter();torch.set_num_threads(2)
    from test_domino import fixture
    model,optimizer,_=fixture();model.eval();path=out/'authored-smoke.pt'
    save_checkpoint(path,model,optimizer,step=0,training={'purpose':'untrained context-evaluation smoke only'})
    c=C.configure(read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-domino-entropy010-g1326.json'))
    c['execution'].update(processes=4,hosts_per_process=1,worker_threads=1,feature_cache=False,shard_games=1)
    e=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=model_id(model),generation=50)
    seed=seeds_for(c['rating']['seed'],1,'validation')[0][0]
    # One repeated frozen model is an explicit startup fixture. One GPU/CPU
    # model identity services all seven contexts; no trained-run files touched.
    agents={'candidate':dict(kind='model',checkpoint=e),'generation-0':dict(kind='model',checkpoint=dict(e,generation=0))}
    parent=dict(format='plackett-luce-slot/1.0.0',generation=50,candidate='candidate',agents=agents,
        lineups=[dict(seed=seed,seat=s,kind='rating',agents=['candidate','generation-0','generation-0','generation-0']) for s in range(4)],
        config=c['rating'],execution=c['execution'],host=c['host'],implementation=implementation_id(),partition='validation',
        startup_reference_fallback=True,scope='Reduced test-only untrained model; not a live rating slot')
    spec=S.specification_for(SOURCE_ROOT,parent,c,test_setups=1);ratings.write_once(out/'spec.json',spec)
    before=sha(path);py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone()
    neutral_started=time.perf_counter();selected=[]
    def sink(game,job,record):
        value=E.summary(game,job,spec,0,job['index'],ratings.receipt(SOURCE_ROOT,Path(record['path'])))
        E.save_summary(out,value);selected.append(value)
    sink.finish=lambda:None
    with isolated_rng(model):
        neutral=ratings.collect_slot(SOURCE_ROOT,out/'neutral',parent,c,'cpu',None,None,context_sink=sink)
    neutral_seconds=time.perf_counter()-neutral_started;require(len(neutral)==4,'Missing neutral rotations')
    # Interrupt immediately after some completed publications. Retry submits
    # the exact recipe; ActorPool validates cached parts and runs only missing.
    acknowledged=[];interrupted=False;first=time.perf_counter()
    def stop(records):
        acknowledged.extend(r['job'] for r in records)
        if len(acknowledged)>=4:raise KeyboardInterrupt
    try:
        with isolated_rng(model):E.collect(SOURCE_ROOT,out,spec,device='cpu',on_records=stop)
    except KeyboardInterrupt:interrupted=True
    interrupted_seconds=time.perf_counter()-first
    require(interrupted,'Interruption exercise did not run')
    with isolated_rng(model):metrics=E.collect(SOURCE_ROOT,out,spec,device='cpu')
    require(metrics['reused_completed_jobs']>=4,'Completed context jobs were recollected')
    rows=E.load_summaries(out,spec);require(len(rows)==28,'Expected 4 neutral plus 24 new unique real games')
    report=A.analyze(spec,rows);report['cost']=dict(metrics,additional_seconds=interrupted_seconds+metrics['collection_seconds']+metrics['extraction_seconds'],
        incremental_wall_fraction=None,neutral_smoke_seconds=neutral_seconds,scope='Test-only CPU; no normal training denominator')
    report['history']=[dict(generation=50,protocol_id=report['protocol_id'],mixture_vp=report['contexts'][-1]['paired']['vp'],
        incremental_wall_fraction=None,additional_seconds=report['cost']['additional_seconds'])]
    ratings.append_games(out/'summaries.jsonl',rows);ratings.append_games(out/'games.jsonl',neutral)
    ratings.write_once(out/'report.json',report);(out/'report.md').write_text(A.markdown(report),encoding='utf-8')
    visual=render(read(out/'report.json'),out/'progress.svg',preview=out/'preview.png')
    require(py==random.getstate() and np.array_equal(nr[1],np.random.get_state()[1]) and torch.equal(tr,torch.get_rng_state()),'Evaluation changed training RNG')
    require(before==sha(path) and not model.training,'Model/checkpoint changed')
    require(not list(out.rglob('*.features*')),'Unexpected feature cache')
    write_json(out/'smoke.json',dict(passed=True,seconds=time.perf_counter()-began,unique_real_games=28,
        acknowledged_before_interrupt=len(acknowledged),retry=metrics,visualization=visual,rng_preserved=True,checkpoint_preserved=True,
        parameter_count=sum(p.numel() for p in model.parameters()),device='cpu',scope=parent['scope']))
    print(out/'smoke.json',flush=True)


if __name__=='__main__':main()
