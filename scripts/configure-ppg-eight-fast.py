"""Generate the reviewed dense eight-age PPG continuation; never rewind a run."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery import dcpg_config
    from brass_ai.recovery.exploration import low_branch_config
    from brass_ai.recovery.lr_controller import cut_config
    from brass_ai.recovery.ppo_kl_control import configure as exact_kl_config
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id,write_json
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require((run/'state.json').is_file(),'No committed run state; this command cannot restore a missing checkpoint')
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json',
        'Write the config under configs/step5')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; do not rewind state.json')
        old=state['signature']['config'];require(old['value_phase']['enabled'],'Expected active penalty-only PPG')
        entry=state['history'][-1];checkpoint=safe(run,project_path(root,entry['path']))
        require(entry['generation']==expected_generation and checkpoint.is_file() and sha(checkpoint)==entry['sha256'],
            'Committed full checkpoint missing/changed; a PPG cache or population weight file is not a substitute')
        config=validate_config(exact_kl_config(cut_config(low_branch_config(dcpg_config.fast_options(old)))))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        changed=not dcpg_config.fast(old['value_phase'])
        return dict(config=output.relative_to(root).as_posix(),generation=expected_generation,run_modified=False,
            first_phase=expected_generation+8 if changed else 'restore checkpoint schedule',
            buffer_iterations=8,validation_iterations=8,phase_interval=8,cache_format=dcpg_config.DENSE_CACHE,
            fresh_cache_reset_on_startup=changed,selection=config['value_phase']['selection'],
            kl_cap=None,beta='preserve fixed checkpoint value',preparation_workers=config['update'].get('preparation_workers',0),
            branch_rate_per_decision=config['branching']['rate_per_decision'],
            ppo_lr_cut_divisor=config['ppo_lr_controller']['cut_divisor'],
            ppo_kl_control=config['update']['kl_control'],
            scope='Only value_phase, branch probability, PPO LR cut divisor and exact-only PPO KL control change. Startup clears owned old PPG replay after reviewed adoption, then collects eight fresh dense ages. Preserve model, Adams, RNG, counters, reward, population, entropy baseline and controller evidence.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
