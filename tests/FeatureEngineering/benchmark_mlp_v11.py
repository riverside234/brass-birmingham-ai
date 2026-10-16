"""Small CPU, tensor-ready v11 inference diagnostic; no strength comparison."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from brass_ai.learning.local_entities import entity_layout
from brass_ai.learning.model import FourSeatNet, dense_multiply_adds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new benchmark output file')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(710)
    model=FourSeatNet().eval();rows=[]
    with torch.inference_mode():
        for batch,candidates in ((1,32),(1,128),(1,512),(8,128)):
            states=torch.randn(batch,3128)*.1
            actions=torch.randn(batch,candidates,327)*.1
            for field in entity_layout()['action_fields'].values():
                actions[:,:,field['start']:field['stop']]=0
                indices=torch.arange(candidates)%(field['stop']-field['start'])+field['start']
                actions[:,torch.arange(candidates),indices]=1
            mask=torch.ones(batch,candidates,dtype=torch.bool)
            mapping=torch.arange(4).expand(batch,-1)
            for _ in range(10):model(states,actions,mask,mapping)
            elapsed=[]
            for _ in range(50):
                began=time.perf_counter();output=model(states,actions,mask,mapping)
                elapsed.append(1000*(time.perf_counter()-began))
            if not all(torch.isfinite(x).all() for x in output):raise ValueError('Non-finite inference')
            ordered=sorted(elapsed)
            rows.append(dict(batch=batch,candidates=candidates,median_ms=statistics.median(elapsed),
                p95_ms=ordered[47],minimum_ms=min(elapsed),samples=len(elapsed),warmups=10))
    report=dict(format='mlp-v11-inference-benchmark/1.0.0',architecture=model.config.architecture,
        parameters=model.parameter_count,dense_macs=dense_multiply_adds(model),rows=rows,
        environment=dict(platform=platform.platform(),processor=os.environ.get('PROCESSOR_IDENTIFIER',platform.processor()),
            torch=torch.__version__,device='cpu',precision='float32',intraop_threads=2,interop_threads=1),
        scope='Synthetic tensor-ready inputs with one-hot references; full base-model forward including policy and four-seat heads. '
              'Excludes game rules, feature extraction, IPC and action sampling. No baseline, end-to-end speedup or strength claim.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
