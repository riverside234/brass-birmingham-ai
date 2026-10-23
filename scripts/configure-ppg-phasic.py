"""Create a reviewed four-iteration PPG/PPO-detach config without editing the run."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery import ppg_config,ppg_phasic
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import write_json,implementation_id
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json' and not output.is_relative_to(run),
        'Write the config under configs/step5, outside the run')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; never rewind a committed run')
        old=state['signature']['config']
        require(old['value_phase']['enabled'] and old['value_phase']['contract'] in
            (ppg_config.COVERAGE_CONTRACT,ppg_config.FIXED_CONTRACT,ppg_config.LINKED_CONTRACT),
            'Expected an active eight-age one-SE PPG run')
        entry=state['history'][-1]
        require(sha(safe(run,project_path(root,entry['path'])))==entry['sha256'],'Committed checkpoint missing/changed')
        config=validate_config(ppg_phasic.configure(old))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),generation=state['iteration'],run_modified=False,
            phase_interval=4,buffer_iterations=8,heldout_iterations=8,shuffle='fresh uniform permutation each epoch',
            peak_lr='effective LR of this iteration final PPO step; existing phase warmup/decay retained',
            ppo_critic_gradient=config['update']['value_gradient'],epoch_cap=5,acceptance_steps=32,
            kl_check_steps=32,patience=1,heldout_standard_errors=1.5,
            changes={k:dict(before=old.get(k),after=v) for k,v in config.items() if old.get(k)!=v},
            scope='Preserve checkpoints, both Adams, RNG, replay, progress, population and gate evidence. No fresh-buffer reset.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
