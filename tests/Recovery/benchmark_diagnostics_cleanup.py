"""Isolated CPU before/after diagnostics and journal benchmark; no training run.

Run with tests/Recovery and tests/FeatureEngineering on PYTHONPATH. Baseline
modules come from the stated Git revision, never from a production checkpoint.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import il_kl_probe,local_scoring_auxiliary,retention,updates
from brass_ai.recovery.config import RETENTION
from brass_ai.step5.league import write_json
from test_diagnostics_cleanup import probe_fixture,retention_fixture
from test_gae_value_phase import fixture
from test_local_scoring import annotate


def baseline(directory,name,revision):
    file=directory/(name+'.py')
    data=subprocess.check_output(['git','show',revision+':src/brass_ai/recovery/'+name+'.py'],cwd=SOURCE_ROOT)
    file.write_bytes(data)
    spec=importlib.util.spec_from_file_location('brass_ai.recovery._before_'+name,file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def close(left,right,path=''):
    if isinstance(left,dict):
        assert set(left)==set(right),(path,set(left)^set(right))
        for k in left:close(left[k],right[k],path+'/'+k)
    elif isinstance(left,list):
        assert len(left)==len(right),path
        for i,(a,b) in enumerate(zip(left,right)):close(a,b,path+'/'+str(i))
    elif isinstance(left,(float,np.floating)):
        assert np.isclose(left,right,rtol=1e-6,atol=1e-7),(path,left,right)
    else:assert left==right,(path,left,right)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref',default='c8fa5a5')
    parser.add_argument('--output',default='artifacts/diagnostics-cleanup-benchmark.json')
    parser.add_argument('--files',type=int,default=1000)
    a=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(533)
    result=dict(baseline=a.baseline_ref,device='cpu',threads=2,training_updates=0,
        scope='Authored fixed inputs, identical weights and temporary owned files; not production GPU timing')
    with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
        directory=Path(tmp);old_probe=baseline(directory,'il_kl_probe',a.baseline_ref)
        old_aux=baseline(directory,'local_scoring_auxiliary',a.baseline_ref)
        old_retention=baseline(directory,'retention',a.baseline_ref)
        current=probe_fixture(2048);previous=old_probe.Probe(current.metadata,current.arrays,current.config)
        model=FourSeatNet(ModelConfig.preset('small')).eval()
        update=dict(microbatch_size=512,max_candidate_cells=131072)
        active=current.arrays['reference'].copy();samples={k:[] for k in ('before','after')}
        # Alternate order to limit first-call/warm filesystem effects.
        reports={}
        for cycle in range(3):
            for name,probe in ([('before',previous),('after',current)] if cycle%2==0 else [('after',current),('before',previous)]):
                began=time.perf_counter();reports[name]=probe.measure(model,update,reference=active)
                samples[name].append(time.perf_counter()-began)
        for field in ('positions','kl_mean','groups'):close(reports['before'][field],reports['after'][field],field)
        result['probe']=dict(rows=2048,candidates='2 or 3 per row',seconds=samples,
            median_seconds={k:statistics.median(v) for k,v in samples.items()},
            input_cache_bytes=current._input_bytes,parity=True,original_il_comparison_after=False)
        print('probe',result['probe'],flush=True)
        model,opt,cfg,(_,data)=fixture(games=32,decisions=16);annotate(data)
        dataset=updates.Dataset(data,'cpu');ids=list(range(len(data)))
        companion=local_scoring_auxiliary.Companion(model,copy.deepcopy(local_scoring_auxiliary.DEFAULT))
        # Bind the old method to the SAME heads, model and immutable dataset.
        samples={k:[] for k in ('before','after')};reports={};rng=torch.get_rng_state().clone()
        for cycle in range(2):
            for name,method in ([('before',old_aux.Companion.diagnostics),('after',local_scoring_auxiliary.Companion.diagnostics)] if cycle==0 else
                    [('after',local_scoring_auxiliary.Companion.diagnostics),('before',old_aux.Companion.diagnostics)]):
                began=time.perf_counter();reports[name]=method(companion,model,dataset,ids,maximum=2048)
                samples[name].append(time.perf_counter()-began)
        retained={k:{field:v[field] for field in reports['after']['groups'][k]} for k,v in reports['before']['groups'].items()}
        close(retained,reports['after']['groups'])
        assert torch.equal(rng,torch.get_rng_state())
        result['auxiliary']=dict(rows=reports['after']['rows'],games=reports['after']['games'],
            seconds=samples,median_seconds={k:statistics.median(v) for k,v in samples.items()},parity=True)
        reduced_times=[]
        for _ in range(2):
            began=time.perf_counter();reduced=companion.diagnostics(model,dataset,ids)
            reduced_times.append(time.perf_counter()-began)
        old_reduced=old_aux.Companion.diagnostics(companion,model,dataset,ids,maximum=512)
        retained={k:{field:v[field] for field in reduced['groups'][k]} for k,v in old_reduced['groups'].items()}
        close(retained,reduced['groups'])
        result['auxiliary']['reduced_default']=dict(rows=reduced['rows'],games=reduced['games'],seconds=reduced_times,
            every_iterations=5,median_seconds=statistics.median(reduced_times),
            amortized_seconds_per_iteration=statistics.median(reduced_times)/5,parity_on_same_selected_rows=True)
        print('auxiliary',result['auxiliary'],flush=True)
        result['cleanup']={}
        for name,module in (('before',old_retention),('after',retention)):
            root=directory/name;run=retention_fixture(root,a.files)
            with patch.object(module,'write_json',wraps=write_json) as writes:
                began=time.perf_counter();r=module.prune(root,run,dict(RETENTION,quarantine_hours=0),apply=True)
                seconds=time.perf_counter()-began
            assert r['permanently_deleted_files']==a.files
            assert (run/'keep.pt').read_bytes()==b'live'
            result['cleanup'][name]=dict(seconds=seconds,journal_writes=writes.call_count,
                deleted_files=r['permanently_deleted_files'],deleted_bytes=r['permanently_deleted_bytes'],
                timings=r.get('timings'))
            print('cleanup',name,result['cleanup'][name],flush=True)
    write_json(SOURCE_ROOT/a.output,result)


if __name__=='__main__':main()
