"""Generate reviewed throughput, snapshot-floor and anchor-driven entropy settings."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.bank import read
from brass_ai.learning.contracts import project_path,require
from brass_ai.recovery.config import throughput_config,validate_config
from brass_ai.recovery.critic_data import floor_config
from brass_ai.recovery.exploration import anchor_floor_config
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json


def generate(root,run_path,output,expected_generation=None):
    root,run=owned(root,run_path);output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Output must be under configs/step5/')
    with run_lock(run):
        state=read(run/'state.json')
        require(expected_generation is None or expected_generation==state['iteration'],'Unexpected committed generation')
        config=validate_config(floor_config(anchor_floor_config(throughput_config(state['signature']['config']),state)))
        signature=dict(copy.deepcopy(state['signature']),config=config,implementation=implementation_id())
        check_signature(state,signature,reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),committed_generation=state['iteration'],
            microbatch_size=config['update']['microbatch_size'],max_candidate_cells=config['update']['max_candidate_cells'],
            preparation_workers=config['update']['preparation_workers'],
            snapshot_weight_floor=config['critic_data']['rollout']['reduced_snapshot_weight'],
            entropy_floor_drop_per_promotion=config['entropy_control']['floor_drop_per_anchor_promotion'],run_modified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-throughput-g470.json')
    p.add_argument('--expected-generation',type=int)
    a=p.parse_args();print(json.dumps(generate(ROOT,a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
