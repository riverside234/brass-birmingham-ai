"""Generate the explicit g210-style PPG continuation without changing run files."""
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
    from brass_ai.recovery import ppg_g210
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id,write_json
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Write config under configs/step5')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; do not rewind state.json')
        entry=state['history'][-1];checkpoint=safe(run,project_path(root,entry['path']))
        require(entry['generation']==expected_generation and checkpoint.is_file() and sha(checkpoint)==entry['sha256'],
            'Committed full checkpoint missing/changed; reports or cached rows cannot replace it')
        old=state['signature']['config'];config=validate_config(ppg_g210.options(old))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        changing=not ppg_g210.active(old['value_phase'])
        return dict(config=output.relative_to(root).as_posix(),generation=expected_generation,run_modified=False,
            first_phase=expected_generation+1,buffer_iterations=2,validation_iterations=1,phase_interval=1,
            beta_on_adoption=4. if changing else 'preserve checkpoint beta',beta_range=[1.,32.],
            learning_rate='0.65 * actual PPO last-applied LR',max_epochs=3,kl_cap=.002,
            acceptance_steps=16,kl_check_steps=config['value_phase']['kl_check_steps'],fresh_cache_reset_on_startup=changing,
            scope='Only value_phase changes. Reviewed startup resets beta once and retires incompatible auxiliary caches. Preserve model, all Adam moments, RNG, PPO objective, ordinary progress and controller evidence.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
