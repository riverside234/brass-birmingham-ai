"""Paired dense-selector versus indexed-selector v11, same nonzero weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
import onnx

from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.learning.model import ModelConfig, dense_multiply_adds
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.learning.reward import specification
from brass_ai.personality.model import import_parent
from brass_ai.personality.contracts import context, context_vector
from brass_ai.personality.laptop.export import BASE_INPUTS, Named, Graph, compare, export
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.step5.league import implementation_id
from test_v11_selectors import dense_copy, trained_branches


def stats(values):
    ordered=sorted(values)
    return dict(median_ms=statistics.median(values),p95_ms=ordered[int(.95*(len(values)-1))],samples_ms=values)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts/feature-engineering') and not out.exists(),'Use a fresh isolated output directory')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(716)
    source=implementation_id();began=time.perf_counter();out.mkdir(parents=True)
    rows=json.loads((SOURCE_ROOT/'artifacts/feature-engineering/deploy-v9-w352-12s-20261016/inputs.json').read_text())
    positions={r['name']:PreparedEncoder().encode(r['view'],r['choices']) for r in rows
               if r['name'] in ('rail-action','settled-next-round')}
    utility=UtilityNet(utility_config(ModelConfig.preset(),specification())).eval();trained_branches(utility)
    actor,_=import_parent(utility);torch.nn.init.normal_(actor.context_projection.weight,std=.02);actor.eval()
    report=dict(format='v11-indexed-selectors-benchmark/1.0.0',source=source,seed=716,
        environment=dict(platform=platform.platform(),processor=platform.processor(),torch=str(torch.__version__),
            device='CPUExecutionProvider',precision='float32',intra_op_threads=2,inter_op_threads=1,
            execution='ORT_SEQUENTIAL',spinning=False),
        method=dict(samples=21,warmups=3,blocks=3,block_samples=7,alternating_order=True,
            weights='Identical nonzero role, residual-final, utility and personality conditioning weights; diagnostic initialization, not trained.'),
        models={},micro=[],scope='Pure tensor-ready forward, identical weights and fixtures. Excludes encoding, transport, sampling and search.')
    paths=[]
    with torch.inference_mode():
        for head,current in (('utility',utility),('actor',actor)):
            variants=dict(dense=dense_copy(current),indexed=current);graphs={};wrappers={};metadata={}
            for name,model in variants.items():
                path=out/f'{head}-{name}.onnx';meta=export(model,path);paths.append(path)
                graphs[name]=Graph(path,threads=2);wrappers[name]=Named(model)
                counts=Counter(node.op_type for node in onnx.load(str(path)).graph.node)
                metadata[name]=dict(parameters=model.parameter_count,model_id=meta['model_id'],sha256=meta['sha256'],
                    bytes=path.stat().st_size,parity=meta['parity'],operators=dict(counts))
            require(metadata['dense']['model_id']==metadata['indexed']['model_id'],'Benchmark weights/identity differ')
            require(metadata['dense']['operators']['MatMul']-metadata['indexed']['operators']['MatMul']==3,
                    'Expected three removed selector MatMuls')
            require(metadata['indexed']['operators'].get('ArgMax',0)-metadata['dense']['operators'].get('ArgMax',0)==3,
                    'Expected three categorical ArgMax nodes')
            report['models'][head]=metadata;report['environment']['onnxruntime']=graphs['indexed'].version
            for batch in (1,2,4,8):
                for case,encoded in positions.items():
                    data=dict(zip(BASE_INPUTS[:4],[t.numpy() for t in collate([encoded]*batch)]))
                    data['remaining']=np.full((batch,4),.075,np.float32)
                    if head=='actor':data['personality']=np.stack([context_vector(context())]*batch).astype('float32')
                    tensors=tuple(torch.from_numpy(data[k]) for k in graphs['indexed'].inputs)
                    errors=compare(graphs['dense'].run(data),graphs['indexed'].run(data),data['legal_mask'])
                    calls={}
                    for name in variants:
                        calls[name+'/ort']=lambda n=name:graphs[n].run(data)
                        calls[name+'/torch']=lambda n=name:wrappers[n](*tensors)
                    measured={k:[] for k in calls}
                    for call in calls.values():
                        for _ in range(3):call()
                    for block in range(3):
                        for key in (list(calls) if block%2==0 else list(reversed(calls))):
                            for _ in range(7):
                                start=time.perf_counter();calls[key]();measured[key].append(1000*(time.perf_counter()-start))
                    row=dict(head=head,batch=batch,candidates=len(encoded.choice_ids),case=case,parity=errors,
                             times={k:stats(v) for k,v in measured.items()})
                    report['micro'].append(row)
                    print(json.dumps(dict(head=head,batch=batch,candidates=row['candidates'],
                        ort_dense=row['times']['dense/ort']['median_ms'],ort_indexed=row['times']['indexed/ort']['median_ms'])),flush=True)
            del graphs,wrappers,variants
    require(implementation_id()==source,'Source changed during benchmark')
    report['source_unchanged']=True;report['elapsed_seconds']=time.perf_counter()-began
    report['dense_macs']=dict(indexed=dense_multiply_adds(utility),removed_per_candidate=22880)
    report['cleanup']=[]
    for path in paths:
        require(path.resolve().parent==out,'Unexpected cleanup target')
        size=path.stat().st_size;path.unlink();report['cleanup'].append(dict(path=path.name,bytes=size,deleted=not path.exists()))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Paired v11 selector benchmark passed: '+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
