"""Generate the reviewed sixteen-age packed PPG continuation; never edit a run."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery import dcpg_config
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id,write_json
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json'
        and not output.is_relative_to(run),'Write the config under configs/step5, outside the run')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; never rewind a committed run')
        old=state['signature']['config'];require(old['value_phase']['enabled'],'Expected active DCPG/PPG')
        entry=state['history'][-1]
        require(sha(safe(run,project_path(root,entry['path'])))==entry['sha256'],'Committed checkpoint missing/changed')
        config=validate_config(dcpg_config.sixteen_options(old))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),generation=state['iteration'],run_modified=False,
            first_phase=expected_generation+16 if not dcpg_config.sixteen(old['value_phase']) else 'restore checkpoint schedule',
            buffer_iterations=16,heldout_iterations=16,phase_interval=16,weighting='uniform',
            cache_format=dcpg_config.PACKED_CACHE,
            kl_control='penalty-only',kl_cap=None,beta_mode='fixed-checkpoint',
            selection=config['value_phase']['selection'],
            scope='Retain existing sealed caches; pack only new generations. Preserve model, both Adams, beta, RNG, progress, population and gate evidence. No fresh-buffer reset.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
