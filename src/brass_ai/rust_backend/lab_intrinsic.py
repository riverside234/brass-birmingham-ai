"""Read-only intrinsic inference, sharing the base trunk with task inference."""
import copy
import numpy as np
import torch
from brass_ai.learning.batching import model_id,collate
from brass_ai.learning.contracts import require
from brass_ai.learning.seat_descriptors import descriptor,unknown_tensor
from brass_ai.recovery.local_scoring_auxiliary import bundle_id
from brass_ai.recovery.teacher_diversity import validate_bundle


def register(service, model, head, bundle):
    validate_bundle(bundle)
    require(model_id(model)==bundle['model_id'] and bundle_id(head.state_dict())==bundle['head_id'],
            'Intrinsic inference weights differ from frozen bundle')
    with service.lock:
        values=getattr(service,'teacher_evaluators',None)
        if values is None:values=service.teacher_evaluators={}
        require(bundle['identity'] not in values,'Evaluator already registered')
        values[bundle['identity']]=(copy.deepcopy(bundle),copy.deepcopy(head).to(service.device).eval().requires_grad_(False))


def predict(model,encoded,device,limit,remaining,specs,registry):
    inputs=collate(encoded,device)
    entities=model.encode_state(inputs[0]);prepared=model.prepare_candidates(entities)
    logits=torch.cat([model.score_prepared(inputs[1][:,start:start+limit],prepared)
                      for start in range(0,inputs[1].shape[1],limit)],1).masked_fill(~inputs[2],-torch.inf)
    # Task descriptors stay unknown. Only the separate scalar head sees self.
    outputs=(logits,*model.heads_from_encoded(entities,inputs[3],remaining))
    require(len(outputs)==5,'Intrinsic search requires four-seat task utility')
    descriptors=unknown_tensor(entities.context);heads=[]
    for i,spec in enumerate(specs):
        if spec is None:heads.append(None);continue
        require(set(spec)=={'bundle_id','strategy_id'} and spec['bundle_id'] in registry,'Unregistered intrinsic evaluator')
        bundle,head=registry[spec['bundle_id']];z=spec['strategy_id']
        require(type(z) is int and 1<=z<7,'Invalid intrinsic context')
        descriptors[i,0]=descriptors.new_tensor(descriptor('learner',0,bundle['fingerprints'][z]['values']))
        heads.append(head)
    embeddings=model.descriptor_embeddings(entities,descriptors)
    joined=torch.cat((entities.context,embeddings[:,0]),-1)
    intrinsic=joined.new_zeros((len(encoded),1))
    # Batch heterogeneous strategies together under the one frozen companion.
    for head in set(h for h in heads if h is not None):
        indices=[i for i,h in enumerate(heads) if h is head]
        intrinsic[indices]=head(joined[indices]).float()
    raw=logits.float().clone().masked_fill(~inputs[2],0.)
    parts=[logits.float().softmax(-1),*[v.float() for v in outputs[1:]],raw,intrinsic]
    offsets=np.cumsum([0,*[v.shape[1] for v in parts]]);packed=torch.cat(parts,1).cpu().numpy()
    return [packed[:,offsets[i]:offsets[i+1]] for i in range(len(parts))]
