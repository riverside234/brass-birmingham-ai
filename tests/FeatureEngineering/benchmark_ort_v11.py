"""Bounded v11 CPU ONNX microbenchmark using the saved v9 input fixtures."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.learning.model import ModelConfig, dense_multiply_adds
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.personality.contracts import context, context_vector
from brass_ai.personality.model import import_parent
from brass_ai.personality.laptop.export import BASE_INPUTS, Graph, Named, compare, export, samples
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.step5.league import implementation_id


HISTORICAL=Path('artifacts/feature-engineering/deploy-v9-w352-12s-20261016')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def summary(values):
    ordered=sorted(values)
    return dict(n=len(values), median_ms=statistics.median(values),
                p95_ms=ordered[int(.95*(len(values)-1))], samples_ms=values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delete-models', action='store_true')
    parser.add_argument('--reverse-head-order', action='store_true',
                        help='Repeat with actor before utility to expose order/time sensitivity')
    args=parser.parse_args()
    output=args.output.resolve()
    require(output.is_relative_to(SOURCE_ROOT/'artifacts/feature-engineering') and not output.exists(),
            'Use a new isolated feature-engineering artifact directory')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(610352)
    source=implementation_id();started=time.perf_counter()
    historical=SOURCE_ROOT/HISTORICAL
    inputs_path=historical/'inputs.json';historical_path=historical/'prepared.json'
    input_hash=sha(inputs_path);historical_hash=sha(historical_path)
    rows=json.loads(inputs_path.read_text());prior=json.loads(historical_path.read_text())
    require(input_hash==prior['inputs_sha256'], 'Saved v9 input fixtures changed')
    chosen={name:next(row for row in rows if row['name']==name)
            for name in ('rail-action', 'settled-next-round')}
    encoded={name:PreparedEncoder().encode(row['view'], row['choices']) for name,row in chosen.items()}
    require([len(chosen[k]['choices']) for k in chosen]==[119,387], 'V9 fixture shapes changed')
    utility=UtilityNet(utility_config(ModelConfig.preset(), specification())).eval()
    actor,_=import_parent(utility);actor.eval()
    require(utility.parameter_count==5928616 and actor.parameter_count==6030433, 'V11 architecture changed')
    output.mkdir(parents=True);models=output/'models';models.mkdir()
    report=dict(format='v11-onnx-runtime-benchmark/1.0.0', source=source, seed=610352,
        environment=dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
            torch=str(torch.__version__), device='CPUExecutionProvider', precision='float32',
            intra_op_threads=2, inter_op_threads=1, execution='ORT_SEQUENTIAL', allow_spinning=False),
        inputs=dict(path=inputs_path.relative_to(SOURCE_ROOT).as_posix(), sha256=input_hash,
            cases={name:len(row['choices']) for name,row in chosen.items()}),
        method=dict(warmups=3, blocks=3, samples_per_block=7, samples_per_backend_shape=21,
            alternating_backend_order=True, timing_unit='whole-batch milliseconds'),
        models={}, micro=[], historical=dict(path=historical_path.relative_to(SOURCE_ROOT).as_posix(),
            sha256=historical_hash, comparison='Previously measured v9; matching fixtures and settings, different run/time. '
                'Not a controlled simultaneous architecture comparison.'),
        scope='Initialized v11 utility/personality graphs; named inference wrapper. Excludes encoding, game rules, '
            'queues, sampling and search. No training, checkpoint migration, deployment or strength claim.')
    graph_paths=[]
    heads=[('utility',utility),('actor',actor)]
    if args.reverse_head_order:heads.reverse()
    report['method']['head_order']=[name for name,_ in heads]
    with torch.inference_mode():
        for head,model in heads:
            path=models/f'v11-{head}.onnx';meta=export(model,path);graph_paths.append(path)
            graph=Graph(path,threads=2);wrapper=Named(model).eval()
            invariance={k:0. for k in graph.outputs}
            for batch in samples(model.core.config.to_dict() if head=='actor' else model.config.to_dict(),head=='actor'):
                actual=graph.run(batch)
                for i in range(len(batch['states'])):
                    one=graph.run({k:v[i:i+1] for k,v in batch.items()})
                    for name,error in compare({k:v[i:i+1] for k,v in actual.items()},one,batch['legal_mask'][i:i+1]).items():
                        invariance[name]=max(invariance[name],error)
            report['environment']['onnxruntime']=graph.version
            report['models'][head]=dict(parameters=model.parameter_count, onnx_sha256=meta['sha256'],
                onnx_bytes=path.stat().st_size, model_id=meta['model_id'], config=meta['config'],
                parity=meta['parity'], batch_invariance_max_abs=invariance, macs=dense_multiply_adds(model))
            for count in (1,2,3,4,6,8,16):
                for case,position in encoded.items():
                    batch=dict(zip(BASE_INPUTS[:4],[tensor.numpy() for tensor in collate([position]*count)]))
                    batch['remaining']=np.full((count,4),.075,np.float32)
                    if head=='actor':batch['personality']=np.stack([context_vector(context())]*count).astype('float32')
                    tensors=tuple(torch.from_numpy(batch[k]) for k in graph.inputs)
                    calls=dict(ort=lambda:graph.run(batch), torch=lambda:wrapper(*tensors))
                    actual=graph.run(batch);expected=dict(zip(graph.outputs,[v.numpy() for v in wrapper(*tensors)]))
                    compare(expected,actual,batch['legal_mask'])
                    measured={k:[] for k in calls}
                    for call in calls.values():
                        for _ in range(3):call()
                    for block in range(3):
                        for backend in (list(calls) if block%2==0 else list(reversed(calls))):
                            for _ in range(7):
                                began=time.perf_counter();calls[backend]()
                                measured[backend].append(1000*(time.perf_counter()-began))
                    prior_row=next(row for row in prior['micro'] if row['model']=='v9' and row['head']==head and
                                   row['batch']==count and row['case']==case)
                    record=dict(head=head,batch=count,candidates=len(chosen[case]['choices']),case=case,
                        **{k:summary(v) for k,v in measured.items()},
                        historical_v9={k:prior_row[k] for k in ('ort','torch')})
                    report['micro'].append(record)
                    print(json.dumps(dict(head=head,batch=count,candidates=record['candidates'],
                        ort_ms=record['ort']['median_ms'],torch_ms=record['torch']['median_ms'])),flush=True)
                    write(output/'report.json',report)
            del graph
    require(implementation_id()==source and sha(inputs_path)==input_hash and sha(historical_path)==historical_hash,
            'Source or historical inputs changed during benchmark')
    report['source_unchanged']=True
    report['elapsed_seconds']=time.perf_counter()-started
    report['model_cleanup']=[]
    if args.delete_models:
        for path in graph_paths:
            require(path.resolve().is_relative_to(models.resolve()) and path.name in ('v11-utility.onnx','v11-actor.onnx'),
                    'Unexpected cleanup target')
            receipt=dict(path=path.relative_to(output).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
            path.unlink();receipt['deleted']=not path.exists();report['model_cleanup'].append(receipt)
    write(output/'report.json',report)
    print('V11 ORT benchmark passed: '+str(output/'report.json'),flush=True)


if __name__=='__main__':main()
