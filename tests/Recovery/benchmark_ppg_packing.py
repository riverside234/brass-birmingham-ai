"""Bounded paired mmap benchmark on existing permitted-view position fixtures."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.learning.seat_descriptors import unknown_numpy
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery import ppg_packing as K
from brass_ai.recovery.value_buffer import row_dtype
from brass_ai.step5.league import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();source=project_path(SOURCE_ROOT,a.positions);out=project_path(SOURCE_ROOT,a.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a fresh artifacts output')
    if source.is_dir():
        paths=sorted(source.glob('position-*.json'));paths=[paths[i] for i in np.linspace(0,len(paths)-1,min(64,len(paths)),dtype=int)]
        positions=[json.loads(p.read_text()) for p in paths]
    else:positions=json.loads(source.read_text())[:64]
    require(positions,'No benchmark positions');out.mkdir(parents=True)
    encoder=PreparedEncoder();encoded=[encoder.encode(x['view'],x['choices']) for x in positions]
    raw=np.zeros(len(encoded),dtype=row_dtype())
    for i,e in enumerate(encoded):
        raw[i]['state']=e.state;raw[i]['action']=e.actions[0];raw[i]['mapping']=e.stable_to_relative
        raw[i]['descriptors']=unknown_numpy();raw[i]['game']=i;raw[i]['seat']=positions[i]['view']['seat']
    raw=np.tile(raw,max(1,8192//len(raw)))
    actions=np.concatenate([e.actions for e in encoded])
    np.save(out/'dense.npy',raw,allow_pickle=False);np.save(out/'actions.npy',actions,allow_pickle=False)
    del raw,actions
    raw=np.load(out/'dense.npy',mmap_mode='r');actions=np.load(out/'actions.npy',mmap_mode='r')
    start=time.perf_counter();layout=K.pack(lambda ext:out/ext,raw,actions,len(encoded));write_seconds=time.perf_counter()-start
    packed=K.CriticArray(lambda ext:out/ext,dict(rows=len(raw),packing=layout));policy=K.ActionArray(out/'policy-actions.npy',layout)
    fields=('state','action','mapping','targets','utility','remaining','descriptors')
    rng=np.random.default_rng(614);batches=[rng.integers(len(raw),size=512) for _ in range(16)]
    try:
        for field in raw.dtype.names:require(raw[field].tobytes()==packed.column(field,slice(None)).tobytes(),'Changed FP32 bits')
        require(actions.tobytes()==policy[:].tobytes(),'Changed legal action inputs')
        times={}
        for compact in (False,True):
            samples=[]
            for repeat in range(6):
                began=time.perf_counter()
                for ids in batches:
                    for name in fields:x=packed.column(name,ids) if compact else raw[name][ids]
                if repeat:samples.append(time.perf_counter()-began)
            times['packed' if compact else 'dense']=dict(median_seconds=statistics.median(samples),samples=samples)
        report=dict(passed=True,positions=len(encoded),rows=len(raw),policy_candidates=len(actions),
            dense_bytes=raw.nbytes+actions.nbytes,packed_bytes=layout['packed_array_bytes'],
            state_columns={k:len(layout['state'][k]) for k in ('constant','binary','floating')},
            pack_seconds=write_seconds,reads=times,bit_exact=True,source=a.positions,
            scope='Existing permitted-view fixtures re-encoded with current Features-4 and repeated to ~8192 rows. Warm CPU mmap, same 16 random batches of 512 rows. Targets/descriptors are authored placeholders. Not a production compression ratio or GPU/phase-throughput measurement.')
        write_json(out/'report.json',report);print(json.dumps(report))
    finally:packed.close();policy.close();raw._mmap.close();actions._mmap.close()


if __name__=='__main__':main()
