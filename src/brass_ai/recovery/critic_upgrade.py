"""Explicit, new-directory v11.1 completed IL -> v11.2 critic-only extension."""
from dataclasses import replace
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
from brass_ai.learning.contracts import compatibility,digest,require,project_path
from brass_ai.learning.model import FourSeatNet,ARCHITECTURE,LEGACY_ARCHITECTURE
from brass_ai.step5.league import implementation_id,write_json

CONTRACT='completed-il-critic-upgrade/1.0.0'


def upgrade_model(old):
    require(type(old) is FourSeatNet and old.config.architecture==LEGACY_ARCHITECTURE,
            'Upgrade only completed v11.1 base IL, never a PPO/utility checkpoint')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(112032);new=FourSeatNet(replace(old.config,architecture=ARCHITECTURE))
    weights=new.state_dict()
    for name,value in old.state_dict().items():
        if name=='outcomes.0.weight':
            weights[name].zero_();weights[name][:,:value.shape[1]].copy_(value)
        else:
            require(weights[name].shape==value.shape,'Unexpected IL upgrade tensor layout')
            weights[name].copy_(value)
    new.load_state_dict(weights);new.validate_parameter_count()
    return new


def upgrade(root,imitation_run,output):
    from .demonstrations import completed_imitation
    from .retention import owned
    root,out=owned(root,output)
    require(not out.exists(),'Use a new IL upgrade directory; completed IL is immutable')
    old,origin=completed_imitation(root,imitation_run,_legacy_upgrade=True)
    new=upgrade_model(old)
    receipt=dict(contract=CONTRACT,parent=origin,compatibility=compatibility(),implementation=implementation_id(),
        architecture=new.config.to_dict(),from_architecture=old.config.to_dict(),parameters=new.parameter_count,
        extra_parameters=new.parameter_count-old.parameter_count,optimizer='Fresh self-play Adam; no IL optimizer conversion',
        initialization='Copied policy/trunk/old critic tensors; descriptor head columns exactly zero; encoder seeded 112032')
    out.mkdir(parents=True)
    path=out/'candidate.pt'
    identity=save_checkpoint(path,new,training=dict(mode=CONTRACT,upgrade=digest(receipt)))
    receipt['candidate']=dict(path=path.relative_to(root).as_posix(),sha256=sha(path),model_id=identity)
    write_json(out/'upgrade.json',receipt)
    return receipt


def completed(root,run,seen):
    from .demonstrations import completed_imitation,bound_reference
    r=read(run/'upgrade.json')
    require(r['contract']==CONTRACT and r['compatibility']==compatibility(),
            'IL critic upgrade source/layout differs; regenerate into a new directory')
    old,parent=completed_imitation(root,r['parent']['bc_run'],_seen=seen,_legacy_upgrade=True)
    parent=bound_reference(root,parent,r['parent'],('bc_run','path','manifest'))
    path=project_path(root,r['candidate']['path'])
    require(path.parent==run and sha(path)==r['candidate']['sha256'],'Upgraded IL checkpoint changed')
    with torch.random.fork_rng(devices=[]):model,raw=load_checkpoint(path)
    require(raw['model_id']==r['candidate']['model_id'] and raw['optimizer'] is None and raw['step']==0
        and raw['training']==dict(mode=CONTRACT,upgrade=digest({k:v for k,v in r.items() if k!='candidate'}))
        and model.config.to_dict()==r['architecture'] and old.config.to_dict()==r['from_architecture'],
        'Upgraded IL identity/lineage differs')
    expected=upgrade_model(old).state_dict()
    require(all(torch.equal(v,expected[k]) for k,v in model.state_dict().items()),
            'Upgraded IL tensors differ from the explicit zero-column extension')
    return model,dict(bc_run=run.relative_to(root).as_posix(),path=path.relative_to(root).as_posix(),
        sha256=r['candidate']['sha256'],model_id=raw['model_id'],manifest=parent['manifest'],
        manifest_sha256=parent['manifest_sha256'],bc_signature=digest(r))
