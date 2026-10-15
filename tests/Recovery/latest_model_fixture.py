"""Small authored completed-IL-format fixture, never the production joint-7 model."""
from pathlib import Path
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery.config import latest_model_config
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.step5.league import implementation_id,write_json


def imitation_fixture(directory):
    directory=Path(directory);run=directory/'fixture-il';run.mkdir(parents=True)
    manifest=directory/'fixture-manifest.json'
    write_json(manifest,dict(scope='Authored handoff test fixture; no production IL data'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4781);model=FourSeatNet(ModelConfig.preset('small'))
    adam=torch.optim.AdamW(model.parameters(),lr=1e-4)
    # Nonempty old optimizer makes accidental IL-optimizer import detectable.
    sum(p.square().mean() for p in model.parameters()).backward();adam.step();adam.zero_grad()
    signature=dict(format='bc-training/2.2.0',initialization='random',epochs=1,
        architecture=model.config.to_dict(),implementation=implementation_id(),
        manifest_path=manifest.relative_to(SOURCE_ROOT).as_posix(),manifest_sha256=sha(manifest))
    path=run/'epoch-0001.pt'
    save_checkpoint(path,model,adam,step=1,training=dict(mode='step5-v3-bc',initialization='random',parent=None,
        fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(signature),manifest_sha256=sha(manifest)))
    write_json(run/'state.json',dict(signature=signature,epoch=1,stopped=False,best=dict(path=path.name,sha256=sha(path))))
    write_json(run/'report.json',dict(format='bc-training-report/1.0.0',signature=signature,epochs=1,candidate_sha256=sha(path)))
    _,origin=completed_imitation(SOURCE_ROOT,run)
    base=read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json')
    base.update(model_size='small',threads=2,audit_workers=1,rehearsal_manifest=None,rehearsal_weight=0.,
                games_per_iteration=3)
    base['evaluation_schedule']['enabled']=False;base['retention']['enabled']=False
    base['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1)
    return run,model,latest_model_config(base,origin)
