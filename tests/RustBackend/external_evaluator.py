"""One synchronous CPU evaluator for the research-only G_ref controls.

Consumes native-encoded little-endian tensors over bounded stdio frames. This
process is never used by the deployed backend. stdout is exclusively protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
LIMIT=64*1024*1024
HEADS=('win_share_4','placement_4','vp_4','remaining_utility_4')


def read_exact(stream,n):
    chunks=[]
    while n:
        chunk=stream.read(n)
        if not chunk:raise EOFError('truncated evaluator frame')
        chunks.append(chunk);n-=len(chunk)
    return b''.join(chunks)


def send(value):
    data=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    assert len(data)<=LIMIT
    sys.stdout.buffer.write(struct.pack('<I',len(data))+data);sys.stdout.buffer.flush()


class Model:
    def __init__(self,mode,folder,checkpoint):
        import numpy as np
        self.np=np;self.mode=mode
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert hashlib.sha256((folder/'model.onnx').read_bytes()).hexdigest()==manifest['export_sha256']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
        self.manifest=manifest
        if mode=='torch':
            import torch
            from brass_ai.learning.checkpoint import load_checkpoint
            if torch.get_num_threads()!=1:torch.set_num_threads(1)
            if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
            self.torch=torch;self.model,raw=load_checkpoint(checkpoint);self.model.eval()
            assert raw['model_id']==manifest['model_id']
        elif mode=='ort':
            from brass_ai.native.inference import ReferenceEvaluator
            self.model=ReferenceEvaluator(folder/'model.onnx')
        else:assert mode=='scripted'

    def evaluate(self,inputs,choice_ids=None):
        np=self.np
        if self.mode=='torch':
            with self.torch.inference_mode():
                values=self.model(**{k:self.torch.from_numpy(v.copy()) for k,v in inputs.items()})
                assert len(values)==5
                return {k:v.detach().cpu().numpy() for k,v in zip(('policy_logits',*HEADS),values)}
        if self.mode=='ort':return self.model.evaluate(inputs)
        n=inputs['actions'].shape[1]
        logits=np.asarray([[float(c.startswith('pass|')) for c in choice_ids]],np.float32)
        return dict(policy_logits=logits,win_share_4=np.full((1,4),.25,np.float32),placement_4=np.zeros((1,4),np.float32),
                    vp_4=np.zeros((1,4),np.float32),remaining_utility_4=np.asarray([[.4,.3,.2,.1]],np.float32))


def main(args):
    for p in reversed([args.reference_source/'src',ROOT/'artifacts/r2-pythondeps',ROOT/'artifacts/r4-pythondeps',ROOT/'artifacts/r0-pythonlibs',ROOT/'artifacts/r0-validation-deps']):
        sys.path.insert(0,str(p))
    model=Model(args.mode,args.model,args.checkpoint);np=model.np
    # Kernels and model load are outside every measured turn.
    inputs=dict(states=np.zeros((1,3128),np.float32),actions=np.zeros((1,1,327),np.float32),legal_mask=np.ones((1,1),bool),
                stable_to_relative=np.arange(4,dtype=np.int64)[None],remaining=np.full((1,4),.075,np.float32),descriptors=np.zeros((1,4,34),np.float32))
    inputs['descriptors'][:,:,4]=1
    model.evaluate(inputs,['pass|||||'])
    send(dict(ready=True,threads=1,model_id=model.manifest['model_id'],export_sha256=model.manifest['export_sha256'],mode=args.mode))
    while True:
        header=sys.stdin.buffer.read(4)
        if not header:return
        assert len(header)==4
        size=struct.unpack('<I',header)[0];assert 4<=size<=LIMIT
        data=read_exact(sys.stdin.buffer,size);m=struct.unpack_from('<I',data)[0];assert m<=size-4
        metadata=json.loads(data[4:4+m]);offset=4+m;rows=[];compute=0
        assert 1<=len(metadata['rows'])<=4
        for row in metadata['rows']:
            n=len(row['choice_ids']);assert 1<=n<=8192
            inputs={}
            for name,dtype,shape in [('states','<f4',(1,3128)),('actions','<f4',(1,n,327)),('legal_mask','?',(1,n)),
                ('stable_to_relative','<i8',(1,4)),('remaining','<f4',(1,4)),('descriptors','<f4',(1,4,34))]:
                count=int(np.prod(shape));length=count*np.dtype(dtype).itemsize
                assert offset+length<=size
                inputs[name]=np.frombuffer(data,dtype=dtype,count=count,offset=offset).reshape(shape);offset+=length
            begin=time.perf_counter_ns();output=model.evaluate(inputs,row['choice_ids']);compute+=time.perf_counter_ns()-begin
            rows.append(dict(id=row['id'],choice_ids=row['choice_ids'],logits=output['policy_logits'][0].tolist(),
                values=[output[k][0].tolist() for k in HEADS]))
        assert offset==size
        send(dict(rows=rows,compute_ns=compute))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['torch','ort','scripted'],required=True)
    for name in ('model','checkpoint','reference-source'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
