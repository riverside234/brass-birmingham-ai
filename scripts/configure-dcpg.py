"""Generate a reviewed delayed-critic continuation config; never edit the run."""
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
    from brass_ai.recovery import ppg_config,dcpg_config
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
        require(old['value_phase']['enabled'] and ppg_config.linked(old['value_phase']),
            'Expected an active PPO-linked eight-age PPG run')
        entry=state['history'][-1]
        require(sha(safe(run,project_path(root,entry['path'])))==entry['sha256'],'Committed checkpoint missing/changed')
        config=validate_config(dcpg_config.configure(old))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),generation=state['iteration'],run_modified=False,
            phase_interval=8,buffer_iterations=8,heldout_iterations=8,shuffle='fresh uniform permutation each epoch',
            learning_rate='0.35 times this iteration actual final PPO LR, fixed throughout phase; no warmup/decay',
            ppo_critic=config['update']['value_gradient'],epoch_cap=5,acceptance_steps=16,
            kl_check_steps=32,patience=1,heldout_standard_errors=1.5,
            changes={k:dict(before=old.get(k),after=v) for k,v in config.items() if old.get(k)!=v},
            scope='Preserve weights, all Adam/RNG state, replay, progress, population and gate state. No fresh-buffer reset; no fresh-target fitting during PPO/calibration.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
