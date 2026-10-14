"""Verify an exported joint-7 generation and optionally preserve its actual run.

No collection, optimizer steps, source transition, cleanup, or game installation.
The lab checkpoints and complete IL lineage must exist; exported JSON alone is
insufficient. A backup is an archival copy, not a relabelled training run.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.learning.contracts import LAB_ROOT, compatibility, digest, project_path, require
from brass_ai.recovery.config import time_budget_config, champion_mix_config, validate_config
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.recovery.resume import apply_optimizer_config, load_resume
from brass_ai.recovery.retention import owned, run_lock, link
from brass_ai.step5.league import implementation_id, write_json


def inspect(root, run, config, evidence_path=None):
    import torch
    evidence_path=project_path(root,evidence_path or 'results/selfplay-joint/state.json')
    if evidence_path.is_dir():evidence_path=evidence_path/'state.json'
    evidence=read(evidence_path)
    state=read(run/'state.json')
    generation=evidence['iteration']
    require(generation>0 and state['iteration']==generation and state['games_done']==evidence['games_done'],
            f'Expected the exported generation-{generation} run')
    require(state['history'][-1]==evidence['history'][-1],f'Generation-{generation} checkpoint receipt differs')
    require(state['origin']==evidence['origin'],'Joint IL lineage differs')
    for key in ('history','population','champion','next_seed','context_counter','next_search_games',
                'ordinary_seconds','search_seconds','legacy_boost'):
        require(state.get(key)==evidence.get(key),f'Exported generation-{generation} {key} differs')
    config=champion_mix_config(time_budget_config(config));validate_config(config)
    signature=dict(state['signature'],config=config,implementation=implementation_id())
    require(signature['compatibility']==compatibility(),'Learning contracts changed')
    require(read(root/'training/opponents/pool.json')['training']==signature['legacy_pool'],'Opponent bank changed')
    _, origin=completed_imitation(root,state['origin']['bc_run'])
    require(origin==signature['imitation_source'],'Completed IL receipt differs')
    model,raw,changed=load_resume(root,run,state,signature,reviewed=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    resume_optimizer(raw,optimizer,restore_rng=False)
    rates=apply_optimizer_config(optimizer,config)
    require(raw['step']==generation and raw['optimizer']['state'],'Missing committed Adam state')
    return dict(iteration=state['iteration'],games_done=state['games_done'],checkpoint=state['history'][-1],
        state_sha256=sha(run/'state.json'),source_transition_required=changed,
        source_before=state['signature']['implementation'],source_after=signature['implementation'],
        config_id=digest(config),optimizer=rates,adam_parameters=len(raw['optimizer']['state']),
        population=len(state['population']),champion=state['champion'],origin=origin,
        counters={k:state[k] for k in ('next_seed','context_counter','next_search_games','ordinary_seconds','search_seconds')},
        utility_calibration=model.utility_calibration,
        note='Verified actual lab files; no training or source transition performed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=LAB_ROOT)
    p.add_argument('--run',default='artifacts/step5/v4-selfplay-joint')
    p.add_argument('--config',default='configs/step5/joint7-review.json')
    p.add_argument('--evidence',default='results/selfplay-joint/state.json',
                   help='Exported state.json or report-export directory; defaults to the earlier generation-60 review')
    p.add_argument('--backup',help='New directory under root/artifacts; copy the entire inactive run without overwriting')
    a=p.parse_args();root=a.root.resolve();_,run=owned(root,a.run)
    require((run/'state.json').is_file(),'Actual lab run is unavailable; retained results JSON cannot resume Adam')
    with run_lock(run):
        receipt=inspect(root,run,read(project_path(root,a.config)),a.evidence)
        if a.backup:
            backup=project_path(root,a.backup)
            require(backup.is_relative_to(root/'artifacts') and not backup.is_relative_to(run)
                    and not run.is_relative_to(backup) and not backup.exists(),'Use a new separate artifact backup directory')
            require(not any(link(path) for path in run.rglob('*')),'Archival copy requires a run without links/reparse points')
            shutil.copytree(run,backup,ignore=shutil.ignore_patterns('.active-run.json'))
            relative=project_path(root,receipt['checkpoint']['path']).relative_to(run)
            require(sha(backup/relative)==receipt['checkpoint']['sha256'] and
                    sha(backup/'state.json')==receipt['state_sha256'] and
                    sha(run/'state.json')==receipt['state_sha256'],'Backup/source changed during archival copy')
            write_json(backup/'joint7-review-backup.json',receipt)
            receipt['backup']=str(backup)
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__=='__main__':main()
