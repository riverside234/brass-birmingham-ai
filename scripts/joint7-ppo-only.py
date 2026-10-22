"""Disable PPG using the committed run recipe; do not edit checkpoints or progress."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.bank import read
from brass_ai.learning.contracts import project_path,require
from brass_ai.recovery.config import validate_config,ppo_microbatch_config
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.value_phase import disabled_options
from brass_ai.step5.league import implementation_id,write_json


def generate(root,run_path,output,expected_generation=None,microbatch_size=None,max_candidate_cells=None):
    root,run=owned(root,run_path);output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json',
        'Output must be under configs/step5/')
    with run_lock(run):
        state=read(run/'state.json');before=state['signature']['config']
        require(expected_generation is None or expected_generation==state['iteration'],
            'Unexpected committed generation')
        require(microbatch_size is None or type(microbatch_size) is int and microbatch_size==2048,
            'Only the reviewed 2048-row PPO microbatch change is supported')
        require(max_candidate_cells is None or type(max_candidate_cells) is int and max_candidate_cells==524288,
            'Only the reviewed 524288-cell PPO candidate limit is supported')
        after=disabled_options(before)
        if microbatch_size is not None or max_candidate_cells is not None:after=ppo_microbatch_config(after)
        validate_config(after)
        signature=dict(copy.deepcopy(state['signature']),config=after,implementation=implementation_id())
        check_signature(state,signature,reviewed=True)
        require(not output.exists() or read(output)==after,'Existing output differs; choose a new --output')
        write_json(output,after)
        return dict(config=output.relative_to(root).as_posix(),committed_generation=state['iteration'],
            changed_blocks=[k for k in after if before.get(k)!=after[k]],value_phase_enabled=False,
            ppo_microbatch_size=after['update']['microbatch_size'],
            ppo_max_candidate_cells=after['update']['max_candidate_cells'],
            critic_gate_evidence='PPO critic updates and unchanged whole-game holdout',
            cache_cleanup='Both owned PPG caches retired on verified training startup, before collection',
            run_modified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only.json')
    p.add_argument('--expected-generation',type=int)
    p.add_argument('--microbatch-size',type=int,choices=(2048,))
    p.add_argument('--max-candidate-cells',type=int,choices=(524288,),
        help='Opt into the reviewed 2048-row / 524288-cell PPO batching pair')
    a=p.parse_args()
    print(json.dumps(generate(ROOT,a.run,a.output,a.expected_generation,a.microbatch_size,a.max_candidate_cells),indent=2))


if __name__=='__main__':main()
