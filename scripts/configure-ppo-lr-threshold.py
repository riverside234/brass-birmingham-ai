"""Prepare only the reviewed 50%-of-target LR threshold from committed g560 onward."""
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
    from brass_ai.recovery.lr_controller import threshold_config
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import write_json,implementation_id
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(expected_generation>=560,'This hotfix starts from committed generation 560 or later')
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json' and not output.is_relative_to(run),
            'Write the config under configs/step5, outside the run')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; never rewind a committed run')
        entry=state['history'][-1]
        require(entry['generation']==expected_generation and sha(safe(run,project_path(root,entry['path'])))==entry['sha256'],
                'Committed checkpoint missing/changed')
        old=state['signature']['config'];config=threshold_config(old);validate_config(config)
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=str(output),committed_generation=expected_generation,first_collection_generation=expected_generation+1,
            low_ratio_before=old['ppo_lr_controller']['low_ratio'],low_ratio_after=.50,
            raise_below_kl=.50*config['ppo_lr_controller']['target_kl'],run_modified=False,
            scope='Config only. Reviewed startup preserves LR multiplier/window/cooldown, warmup, both core and head Adams, RNG, PPG replay/clock, and all training progress.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run',required=True)
    p.add_argument('--expected-generation',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
