"""Prepare g1247 PPG continuation: floor .125, PPO peak factor .65, every iteration."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation=1247):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery.ppg_g210 import every_iteration_options
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import write_json,implementation_id
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json' and
        not output.is_relative_to(run),'Write config under configs/step5, outside the run')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation==1247,'Committed generation differs; never rewind')
        entry=state['history'][-1];checkpoint=safe(run,project_path(root,entry['path']))
        require(entry['generation']==expected_generation and checkpoint.is_file() and sha(checkpoint)==entry['sha256'],
            'Committed full checkpoint missing/changed; reports cannot replace it')
        pending=safe(run,run/f'iteration-{expected_generation:04d}')
        require(not pending.exists() or pending.is_dir() and not any(pending.iterdir()),
            'Recover the interrupted collection before changing training settings')
        require(not list(run.glob(f'generation-{expected_generation+1:04d}.pt*')),'Recover the uncommitted checkpoint first')
        config=validate_config(every_iteration_options(state['signature']['config']))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),committed_generation=expected_generation,run_modified=False,
            value_phase=config['value_phase'],first_phase=1248,
            scope='First collection and PPG g1248. Reset beta once to .125; preserve trip history, model/Adam/RNG, counters and compatible replay.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,default=1247)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
