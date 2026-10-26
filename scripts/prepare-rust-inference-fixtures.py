"""Convert the frozen exporter's real categorical test inputs for Rust tests."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def prepare(manifest,out):
    import numpy as np
    from brass_ai.native.inference import ReferenceEvaluator
    m=json.loads(manifest.read_text());g=ReferenceEvaluator(manifest.parent/'model.onnx')
    batches=[]
    with np.load(manifest.parent/'model.parity.npz',allow_pickle=False) as data:
        for i in range(6):
            x={k:data[f'{i}:input:{k}'] for k in g.graph.inputs};y=g.evaluate(x)
            rows=[]
            for j in range(len(x['states'])):
                n=int(x['legal_mask'][j].sum());assert x['legal_mask'][j,:n].all()
                mapping=x['stable_to_relative'][j]
                rows.append(dict(observer=j%4,state=x['states'][j].tolist(),actions=x['actions'][j,:n].reshape(-1).tolist(),
                    mapping=mapping.tolist(),remaining=x['remaining'][j].tolist(),descriptors=x['descriptors'][j].reshape(-1).tolist(),
                    stable_descriptors=x['descriptors'][j,mapping].tolist(),
                    torch=[data[f'{i}:output:{k}'][j,:n if k=='policy_logits' else 4].tolist() for k in g.graph.outputs],
                    ort=[y[k][j,:n if k=='policy_logits' else 4].tolist() for k in g.graph.outputs]))
            batches.append(rows)
    assert not out.exists();out.write_text(json.dumps(batches,separators=(',',':')),encoding='utf-8')
    print(json.dumps(dict(batches=len(batches),positions=sum(map(len,batches)),manifest=m['export_sha256'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(a.reference.resolve()/'src')]+[str(ROOT/d) for d in ('artifacts/r2-pythondeps','artifacts/r4-pythondeps','artifacts/r0-pythonlibs','artifacts/r0-validation-deps')]
    prepare(a.manifest,a.out)
