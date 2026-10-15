"""Versioned named-output checkpoints; Step 5 readers intentionally reject them."""
import hashlib
from pathlib import Path
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,keys,require
from .contracts import CHECKPOINT,contract,local_file,accepted_contract
from .model import PersonalityConfig,PersonalityNet

PILOT='personality-pilot/1.1.0'


def model_id(model):
    return _model_id(model,contract())


def _model_id(model,recorded_contract):
    result=hashlib.sha256(digest(dict(contract=recorded_contract,config=model.config.to_dict())).encode())
    for name,tensor in sorted(model.state_dict().items()):
        result.update(name.encode());result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def committed_path(run,state):
    """Resolve the authoritative pilot checkpoint, including the original layout."""
    require(state.get('format') in ('personality-pilot/1.0.0',PILOT),'Unsupported P1 pilot layout')
    require(state['format']!=PILOT or 'model_path' in state,'Missing committed P1 checkpoint pointer')
    path=local_file(run,state.get('model_path','model.pt'))
    require(sha(path)==state['model_sha256'],'P1 saved model changed')
    return path


def restore_rng(raw):
    """Restore the pilot stream independently of its optimizer parameter set."""
    torch.set_rng_state(raw['rng'])
    if raw['cuda_rng'] and torch.cuda.is_available():
        require(len(raw['cuda_rng'])==torch.cuda.device_count(),'CUDA topology changed on P1 resume')
        torch.cuda.set_rng_state_all(raw['cuda_rng'])


def save(path,model,optimizer=None,*,training=None):
    model.validate_parameter_count();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    training=dict(training or {})
    if hasattr(model,'personality_migration'):training.setdefault('source_migration',model.personality_migration)
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
    require(raw['format']==CHECKPOINT,'Personality checkpoint format differs')
    config=PersonalityConfig.from_dict(raw['config'])
    migration=accepted_contract(raw['contract'])
    with torch.random.fork_rng(devices=[]): model=PersonalityNet(config)
    model.load_state_dict(raw['state_dict'],strict=True);model.core.utility_calibration=raw['utility_calibration']
    model.validate_parameter_count()
    require(all(torch.isfinite(t).all() for t in model.state_dict().values()),'Nonfinite personality weights')
    require(raw['model_id']==_model_id(model,raw['contract']),'Personality weights changed')
    if migration:
        model.personality_migration=dict(migration,from_model_id=raw['model_id'],to_model_id=model_id(model))
        raw['source_migration']=model.personality_migration
    return model.to(device),raw
