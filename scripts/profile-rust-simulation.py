"""Diagnostic adapter wall time, separate from native Criterion/allocator probes."""
import argparse
import json
from pathlib import Path
import statistics
import time

from brass_ai.native.simulation import NativeSpielGame


def profile(binary, fixtures, out):
    records={}
    with NativeSpielGame(executable=binary) as game:
        for name in ('initial','complex'):
            dto=json.loads((fixtures/(name+'.json')).read_text())
            with game.import_state(dto) as state:
                view=state.views[state.current_player()];recall=state.recall[state.current_player()]
                actions=state.legal_actions(); saved=state.clone()
                def sample():
                    with game.from_observation(view,890,recall):pass
                def fork():
                    with state.clone() as clone:clone.restore(state)
                def step():
                    state.apply_action(actions[0]);state.restore(saved)
                def snapshot():game._call('snapshot',handle=state.handle)
                try:
                    for label, operation in [('sample_cached_root',sample),('clone_restore_release',fork),('apply_restore',step),('snapshot',snapshot)]:
                        operation(); times=[]
                        for _ in range(30):
                            start=time.perf_counter_ns();operation();times.append(time.perf_counter_ns()-start)
                        records[name+':'+label]=dict(samples_ns=times,median_ns=statistics.median(times),operations=30)
                finally:saved.close()
        result=dict(build=game.worker.hello['build'],measurements=records,
            note='Diagnostic IPC/JSON/Python copies included; not a search-speed benchmark. No NN executed.')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();profile(args.binary,args.fixtures,args.out)
