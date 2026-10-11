"""Versioned named-output checkpoints; Step 5 readers intentionally reject them."""
import hashlib
from pathlib import Path
import torch

from brass_ai.learning.contracts import digest,keys,require
from .contracts import CHECKPOINT,contract
from .model import PersonalityConfig,PersonalityNet


def model_id(model):
    result=hashlib.sha256(digest(dict(contract=contract(),config=model.config.to_dict())).encode())
    for name,tensor in sorted(model.state_dict().items()):
        result.update(name.encode());result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def save(path,model,optimizer=None,*,training=None):
    model.validate_parameter_count();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(format=CHECKPOINT,contract=contract(),config=model.config.to_dict(),model_id=model_id(model),
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict() if optimizer else None,training=training or {},
        utility_calibration=model.core.utility_calibration,rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary=path.with_name(path.name+'.tmp')
    try:torch.save(payload,temporary);temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
    return payload['model_id']


def load(path,device='cpu'):
    raw=torch.load(path,map_location='cpu',weights_only=True)
    keys(raw,('format','contract','config','model_id','state_dict','optimizer','training','utility_calibration','rng','cuda_rng'))
    require(raw['format']==CHECKPOINT and raw['contract']==contract(),'Personality checkpoint contract/source differs')
    with torch.random.fork_rng(devices=[]): model=PersonalityNet(PersonalityConfig.from_dict(raw['config']))
    model.load_state_dict(raw['state_dict'],strict=True);model.core.utility_calibration=raw['utility_calibration']
    require(all(torch.isfinite(t).all() for t in model.state_dict().values()),'Nonfinite personality weights')
    require(raw['model_id']==model_id(model),'Personality weights changed')
    return model.to(device),raw
