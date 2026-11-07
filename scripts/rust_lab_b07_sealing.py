"""Two matched frozen-table publication pairs; no search, weights or training."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def measure(output, evidence, archive):
    from brass_ai.learning.contracts import digest, require
    from brass_ai.recovery.admitted_inputs import _freeze
    from brass_ai.recovery.encoding import PreparedEncoder
    from brass_ai.recovery.search_value_rows import ValueBuffer,state_width
    from brass_ai.recovery.teacher_buffer import TeacherBuffer
    from brass_ai.recovery.teacher_labels import LabelRow
    from brass_ai.recovery import search_value_phase as current_phase
    from brass_ai.rust_backend import lab_teacher_finalization as current_verifier
    from brass_ai.runtime_contracts import validator
    from brass_ai.step5.league import write_json
    with zipfile.ZipFile(archive) as z:
        old_phase=ModuleType('brass_ai.recovery.b07_old_phase')
        old_verifier=ModuleType('brass_ai.rust_backend.b07_old_verifier')
        for module,name in [(old_phase,'src/brass_ai/recovery/search_value_phase.py'),
                            (old_verifier,'src/brass_ai/rust_backend/lab_teacher_finalization.py')]:
            module.__package__=module.__name__.rsplit('.',1)[0]
            exec(compile(z.read(name),name,'exec'),module.__dict__)
    jobs=json.loads((evidence/'jobs.json').read_text())
    tables=json.loads(gzip.decompress((evidence/'tables.json.gz').read_bytes()))
    encoder=PreparedEncoder();state_width();validator('result')
    encoded=[[LabelRow(r,encoder.encode(r['view'],r['choices'],[0.]*4,[0.]*8)) for r in t['rows']] for t in tables]
    output.mkdir(parents=True,exist_ok=False);samples=[];expected=None
    for pair in range(2):
        for arm in (('before','after') if pair==0 else ('after','before')):
            vm,pm=(old_verifier,old_phase) if arm=='before' else (current_verifier,current_phase)
            work=copy.deepcopy(tables);accepted=copy.deepcopy(encoded);stage=[]
            # Reproduce only snapshot preparation. NN/legality verification was
            # already measured in the lab; it is not simulated by this timing.
            for table,rows in zip(work,accepted,strict=True):
                start=time.perf_counter()
                if arm=='after':
                    table['rows']=[_freeze(r) for r in table['rows']]
                    rows[:]=[LabelRow(_freeze(r.row),r.encoded) for r in rows]
                    raw=list(table['rows'])
                else:raw=copy.deepcopy(table['rows'])
                stage.append((raw,time.perf_counter()-start))
            directory=output/f'{pair}-{arm}';directory.mkdir()
            policy=TeacherBuffer(ROOT,directory/'policy','frozen-component-only')
            phase=pm.SearchValuePhase.__new__(pm.SearchValuePhase)
            phase.machine=SimpleNamespace(state={'lam':None});phase.label_ages=[];phase.report={}
            phase.reward_id=jobs[0]['expected']['reward_id'];phase.buffer=ValueBuffer(directory/'values',phase.reward_id)
            policy_times={};value_times={};overruns=[]
            for table,job,rows,(raw,_) in zip(work,jobs,accepted,stage,strict=True):
                key=job['expected']['job_id'];v=vm.TurnVerifier.__new__(vm.TurnVerifier)
                v.model=None;v.model_id=job['expected']['model_id'];v.diagnostic=False;v.cancel=threading.Event()
                v.jobs={key:job};v.cache={key:dict(raw=raw,accepted=rows,rejected={},last=64,timings=[])}
                start=time.perf_counter()
                with patch.object(vm,'model_id',return_value=v.model_id):selected,_=v.take(table,job,None)
                receipt=policy.publish(1250,job['split'],selected,model_id=v.model_id,reward_id=phase.reward_id)
                policy_times[key]=time.perf_counter()-start
                try:phase.admit([table],[job],1250,post_spent=policy_times[key])
                except ValueError as error:
                    if 'Value sealing exceeded' not in str(error):raise
                    overruns.append(str(error))
                value_times[key]=phase.report['value_admission_per_table_seconds'][key]
            restored=TeacherBuffer(ROOT,directory/'policy',policy.identity,receipts=policy.pending)
            cold=ValueBuffer(directory/'values',phase.reward_id,receipts=phase.buffer.receipts)
            checks={}
            for split in ('train','heldout'):
                policy_rows=policy.rows(1250,split);reloaded=restored.rows(1250,split)
                require(digest([r.row for _,r in policy_rows])==digest([r.row for _,r in reloaded]),'Cold policy rows differ')
                import numpy as np
                for (_,a),(_,b) in zip(policy_rows,reloaded,strict=True):
                    for field in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(a.encoded,field),getattr(b.encoded,field))
                require(cold.rows(1250,split)==phase.buffer.rows(1250,split),'Cold value rows differ')
                checks[split]=dict(policy=digest([r.row for _,r in policy_rows]),values=digest(cold.rows(1250,split)))
            if expected is None:expected=checks
            require(checks==expected,'Matched teacher training rows/targets differ')
            sample=dict(pair=pair,arm=arm,policy_seconds=policy_times,value_seconds=value_times,
                preparation_seconds=sum(s for _,s in stage),finalization_seconds=sum(policy_times.values())+sum(value_times.values()),
                overruns=overruns,outputs=checks)
            samples.append(sample);print(json.dumps({k:v for k,v in sample.items() if k!='outputs'}),flush=True)
    medians={arm:{key:statistics.median(s[key] for s in samples if s['arm']==arm)
                  for key in ('preparation_seconds','finalization_seconds')} for arm in ('before','after')}
    report=dict(passed=True,outputs_equal=True,samples=samples,medians=medians,
        scope='Two AB/BA frozen pairs. Real encoders, publication and cold readers; excludes unchanged model fingerprint and NN verification. Snapshot work is reported, not free. Local timing is not a lab deadline pass.')
    write_json(output/'measurement.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'distribution/rust-lab-b07-g1250/rust-lab-b07-g1250-r5.zip')
    a=p.parse_args();measure(a.out,a.evidence,a.baseline)
