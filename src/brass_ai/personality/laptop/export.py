"""Float32 named ONNX graphs. Shared actor and pinned rollout references."""
from pathlib import Path
import copy
import time
import numpy as np
import torch
from torch import nn

from brass_ai.bank import sha
from brass_ai.learning.contracts import require,digest,compatibility
from brass_ai.learning.batching import model_id as base_id
from brass_ai.personality.checkpoint import model_id
from brass_ai.personality.contracts import CONTEXT_DIM,FACTORS
from brass_ai.personality.model import PersonalityNet
from .contracts import EXPORT,BACKENDS,write,read

BASE_INPUTS=['states','actions','legal_mask','stable_to_relative','remaining']
BASE_OUTPUTS=['policy_logits','win_share_4','placement_4','vp_4','remaining_utility_4']
GOAL_OUTPUTS=['goal_'+k for k in FACTORS]+['confidence_logits']


def validate_metadata(meta):
    from brass_ai.learning.utility_model import UtilityConfig
    UtilityConfig.from_dict(meta['config'])
    require(meta['format']==EXPORT and type(meta['actor']) is bool
            and meta['compatibility']==compatibility() and meta['precision']=='float32' and meta['opset']==17
            and meta['inputs']==BASE_INPUTS+(['personality'] if meta['actor'] else [])
            and meta['outputs']==BASE_OUTPUTS+(GOAL_OUTPUTS if meta['actor'] else [])
            and meta['reward_id']==digest(meta['config']['utility']), 'Incompatible named export/seat/reward contract')
    return meta


class Named(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,states,actions,legal_mask,stable_to_relative,remaining,personality=None):
        if isinstance(self.model,PersonalityNet):
            out=self.model(states,actions,legal_mask,stable_to_relative,remaining=remaining,personality=personality)
            return tuple(out[k] for k in BASE_OUTPUTS)+tuple(out['goals'][k] for k in FACTORS)+(out['confidence_logits'],)
        return self.model(states,actions,legal_mask,stable_to_relative,remaining=remaining)


def samples(config,personality,seed=661):
    rng=np.random.default_rng(seed);cap=config['utility']['insolvency']['max_cost']
    result=[]
    for batch,candidates in ((1,1),(2,7),(4,31),(1,257)):
        masks=np.ones((batch,candidates),bool)
        if candidates>1:masks[-1,candidates//2:]=False
        data=dict(states=rng.normal(0,.2,(batch,config['state_dim'])).astype('float32'),
            actions=rng.normal(0,.2,(batch,candidates,config['action_dim'])).astype('float32'),
            legal_mask=masks,stable_to_relative=np.array([[(s-i)%4 for s in range(4)] for i in range(batch)],dtype='int64'),
            remaining=rng.uniform(0,cap,(batch,4)).astype('float32'))
        if personality:data['personality']=rng.uniform(0,1,(batch,CONTEXT_DIM)).astype('float32')
        result.append(data)
    return result


class Graph:
    """One synchronous native request; return only after device work completes."""
    def __init__(self,path,backend='ort-cpu',threads=2):
        require(backend in BACKENDS and threads in (1,2),'Unknown backend/thread profile')
        self.path=Path(path);self.meta=validate_metadata(read(self.path.with_suffix('.json')));self.backend=backend
        require(self.meta['format']==EXPORT and sha(self.path)==self.meta['sha256'],'Export bytes changed')
        self.inputs=self.meta['inputs'];self.outputs=self.meta['outputs'];self.calls=0;self.seconds=0.
        if backend=='ort-cpu':
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry('session.intra_op.allow_spinning','0')
            self.engine=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
            require([x.name for x in self.engine.get_inputs()]==self.inputs and
                    [x.name for x in self.engine.get_outputs()]==self.outputs,'Graph tensor names differ')
            self.version=ort.__version__;self.hardware='CPUExecutionProvider'
        else:
            import openvino as ov
            core=ov.Core();device='GPU' if backend.endswith('gpu') else 'CPU'
            require(any(x.split('.')[0]==device for x in core.available_devices),'Requested OpenVINO device unavailable')
            cfg={'INFERENCE_PRECISION_HINT':'f32','NUM_STREAMS':'1'}
            if device=='CPU':cfg['INFERENCE_NUM_THREADS']=str(threads)
            self.engine=core.compile_model(str(path),device,cfg);self.request=self.engine.create_infer_request()
            self.version=ov.__version__;self.hardware=core.get_property(device,'FULL_DEVICE_NAME')

    def run(self,inputs):
        require(set(inputs)==set(self.inputs),'Wrong inference inputs')
        start=time.perf_counter()
        if self.backend=='ort-cpu':out=self.engine.run(self.outputs,inputs)
        else:
            self.request.infer(inputs,share_inputs=False,share_outputs=False)
            out=[self.request.get_tensor(name).data.copy() for name in self.outputs]
        self.seconds+=time.perf_counter()-start;self.calls+=1
        return dict(zip(self.outputs,out))


def compare(expected,actual,mask):
    require(set(actual)==set(expected),'Missing exported head')
    errors={}
    for name,reference in expected.items():
        observed=actual[name]
        require(reference.shape==observed.shape,'Exported tensor shape differs')
        # NaN == NaN in assert_allclose is not evidence of a usable model.
        # Only masked policy logits may be nonfinite, and they must be -inf.
        if name=='policy_logits':
            require(reference.shape==mask.shape and np.isfinite(reference[mask]).all()
                    and np.isfinite(observed[mask]).all() and np.isneginf(reference[~mask]).all()
                    and np.isneginf(observed[~mask]).all(), 'Nonfinite legal policy outputs')
        else:
            require(np.isfinite(reference).all() and np.isfinite(observed).all(), 'Nonfinite exported value/goal outputs')
        finite=np.isfinite(reference)
        require(np.array_equal(finite,np.isfinite(observed)),'Mask/nonfinite parity differs')
        np.testing.assert_allclose(observed,reference,rtol=2e-4,atol=2e-4,err_msg=name)
        errors[name]=float(np.max(np.abs(reference[finite]-observed[finite]))) if finite.any() else 0.
    require(np.isneginf(actual['policy_logits'][~mask]).all(),'Export admitted a masked choice')
    return errors


def export(model,path):
    import onnx
    model.validate_parameter_count()
    if isinstance(model,PersonalityNet):model.core.validate_parameter_count()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'Do not overwrite an immutable export')
    actor=isinstance(model,PersonalityNet);config=(model.core.config if actor else model.config).to_dict()
    inputs=BASE_INPUTS+(['personality'] if actor else []);outputs=BASE_OUTPUTS+(GOAL_OUTPUTS if actor else [])
    batches=samples(config,actor);frozen=Named(copy.deepcopy(model).cpu().eval())
    axes={name:{0:'positions'} for name in inputs+outputs}
    for name in ('actions','legal_mask','policy_logits'):axes[name][1]='candidates'
    with torch.inference_mode():
        torch.onnx.export(frozen,tuple(torch.from_numpy(batches[0][k]) for k in inputs),str(path),
            input_names=inputs,output_names=outputs,dynamic_axes=axes,opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    meta=dict(format=EXPORT,sha256=sha(path),model_id=model_id(model) if actor else base_id(model),
        actor=actor,config=config,inputs=inputs,outputs=outputs,compatibility=compatibility(),
        reward_id=digest(config['utility']),precision='float32',opset=17,parity=None)
    write(path.with_suffix('.json'),meta);runtime=Graph(path)
    errors={name:0. for name in outputs};stored={}
    for i,batch in enumerate(batches):
        with torch.inference_mode():out=frozen(*(torch.from_numpy(batch[k]) for k in inputs))
        expected={k:v.numpy() for k,v in zip(outputs,out)}
        for key,value in compare(expected,runtime.run(batch),batch['legal_mask']).items():errors[key]=max(errors[key],value)
        for key,value in batch.items():stored[f'{i}:input:{key}']=value
        for key,value in expected.items():stored[f'{i}:output:{key}']=value
    np.savez_compressed(path.with_suffix('.parity.npz'),**stored)
    meta['parity']=dict(shapes=[[len(b['states']),b['actions'].shape[1]] for b in batches],errors=errors,
                        backend='ort-cpu',runtime=runtime.version,passed=True)
    write(path.with_suffix('.json'),meta);return meta


def check_backend(graph):
    errors={k:0. for k in graph.outputs}
    with np.load(graph.path.with_suffix('.parity.npz'),allow_pickle=False) as data:
        for i in range(4):
            inputs={k:data[f'{i}:input:{k}'] for k in graph.inputs}
            expected={k:data[f'{i}:output:{k}'] for k in graph.outputs}
            for name,value in compare(expected,graph.run(inputs),inputs['legal_mask']).items():errors[name]=max(errors[name],value)
    return dict(passed=True,backend=graph.backend,version=graph.version,device=graph.hardware,errors=errors)
