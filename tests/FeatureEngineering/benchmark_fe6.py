"""Paired CPU encoder/native-game benchmark, using an immutable pre-FE4 snapshot.

The after arm zeroes added input weights only as a timing control. This is not
an artifact importer and never produces a production checkpoint or trained run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('before','after'),required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();snapshot=args.snapshot.resolve();out=args.output.resolve()
    sys.path.insert(0,str((snapshot if args.arm=='before' else ROOT)/'src'))
    import numpy as np
    import torch
    from brass_ai.learning.features import Encoder
    from brass_ai.learning.model import FourSeatNet
    from brass_ai.learning.batching import collate
    from brass_ai.recovery.encoding import PreparedEncoder
    from brass_ai.recovery.execution import ActorPool,neural_spec
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.replay import read_shards
    from brass_ai.recovery.telemetry import iteration_utilization,environment
    from brass_ai.learning.contracts import require,digest
    from brass_ai.step5.league import write_json
    # Import the measurement helper without changing its pinned imports.
    sys.path.insert(0,str(ROOT/'scripts'))
    from importlib.machinery import SourceFileLoader
    measure=SourceFileLoader('fe_bench_helper',str(ROOT/'scripts/benchmark-feature-engineering.py')).load_module().measure
    # The helper adds the live source path. Spawned actors must inherit the
    # selected immutable runtime, just like this parent process.
    selected=str((snapshot if args.arm=='before' else ROOT)/'src')
    sys.path=[selected]+[p for p in sys.path if p not in (selected,str(ROOT/'src'))]
    require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'Use a new benchmark arm directory')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(52791)
    positions=json.loads(args.inputs.read_text(encoding='utf-8'))
    model=FourSeatNet().eval()
    if args.arm=='before':torch.save(model.state_dict(),out.parent/'control-weights.pt')
    else:
        old=torch.load(out.parent/'control-weights.pt',weights_only=True);weights=model.state_dict()
        for name,value in old.items():
            if value.shape==weights[name].shape:weights[name].copy_(value)
            elif name=='state_encoder.0.weight':
                weights[name].zero_();weights[name][:,:3195]=value[:,:3195];weights[name][:,-12:]=value[:,-12:]
            elif name=='action_encoder.0.weight':weights[name].zero_();weights[name][:,:184]=value
            else:raise ValueError('Unexpected benchmark control layout')
        model.load_state_dict(weights)
    encoder=PreparedEncoder();outputs={}
    with torch.inference_mode():
        for i,row in enumerate(positions):
            result=model(*collate([encoder.encode(row['view'],row['choices'])]))
            for k,array in enumerate(result):outputs[f'{i}_{k}']=array.numpy()
    np.savez_compressed(out/'control-outputs.npz',**outputs)
    if args.arm=='after':
        with np.load(out.parent/'before/control-outputs.npz') as previous:
            for key,value in outputs.items():np.testing.assert_allclose(value,previous[key],rtol=2e-4,atol=2e-4)
    class Cold:
        def encode(self,view,choices):return PreparedEncoder().encode(view,choices)
    timing=dict(reference=measure(Encoder,positions),prepared_cold=measure(Cold,positions),prepared_warm=measure(PreparedEncoder,positions))
    cfg=json.loads((ROOT/'configs/step5/joint7-latest.json').read_text())['execution']
    cfg=dict(cfg,processes=1,hosts_per_process=1,heavy_processes=1)
    service_start=time.monotonic();collections=[]
    with InferenceService(cfg,'cpu') as service:
        identity=service.register(model);policy=neural_spec(identity,.08,'neutral')
        with ActorPool(ROOT,ROOT/'artifacts/r1-host/guarded',service,cfg) as actors:
            for index,seed in enumerate((52793,52794,52795)):
                directory=out/f'game-{index}';started=time.monotonic();before=service.report()
                records=actors.collect([dict(index=0,seed=seed,rotation=0,sampling_seed=987611+index,
                    policies=[policy]*4,path=str(directory/'game.jsonl.gz'),
                    provenance=dict(producer='fe6-paired-benchmark',purpose='training',seed=seed))])
                seconds=time.monotonic()-started;after=service.report()
                games=read_shards([r['path'] for r in records]);game=games[0]
                collections.append(dict(warmup=index==0,seed=seed,seconds=seconds,decisions=len(game['records']),
                    choice_sequence=digest([(r['seat'],r['choices'][r['chosen']]['id']) for r in game['records']]),
                    utilization=iteration_utilization(before,after,records,seconds,1,cfg),
                    memory=after['memory']))
        inference=service.report()
    require(inference['error'] is None,'Benchmark inference failed')
    result=dict(arm=args.arm,inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),positions=len(positions),
        candidates=sum(len(r['choices']) for r in positions),timing=timing,collections=collections,
        environment=environment('cpu'),total_pipeline_seconds=time.monotonic()-service_start,inference=inference,
        controlled_outputs_match=True,parameters=model.parameter_count,
        scope='Same saved corpus, game seeds, sampling seeds, actor/thread counts and preserved old-weight predictions with added input weights zeroed in the after arm only. New dimensions/model work are included. One warmup plus two timed native games per arm. CPU only; no search/teacher work, no learning/strength comparison. Memory is sampled resident/Python allocations, not total peak GPU/RSS.')
    write_json(out/'report.json',result);print('Paired benchmark:',out/'report.json',flush=True)


if __name__=='__main__':main()
