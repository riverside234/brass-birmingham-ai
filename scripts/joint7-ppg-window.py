"""Write a reviewed continuation config; never edit the run or checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.bank import read
from brass_ai.learning.contracts import project_path,require
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.gae import health_options
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.value_phase import interval_options
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    parser.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-ppg-window.json')
    args=parser.parse_args()
    run=project_path(ROOT,args.run);output=project_path(ROOT,args.output)
    require(output.resolve().is_relative_to((ROOT/'configs').resolve()),'Output must be under configs/')
    state=read(run/'state.json');before=state['signature']['config']
    after=validate_config(interval_options(health_options(before)))
    require(after['value_phase'].get('acceptance_steps')==16 and
        after['advantage']['gate'].get('window_iterations')==20,
        'This helper requires the existing stratified PPG and 20-iteration GAE recipe')
    signature=dict(copy.deepcopy(state['signature']),config=after,implementation=implementation_id())
    check_signature(state,signature,reviewed=True)
    require(not output.exists() or read(output)==after,'Existing output has different settings; choose a new --output')
    write_json(output,after)
    print(json.dumps(dict(config=output.relative_to(ROOT).as_posix(),generation=state['iteration'],
        changed_blocks=[k for k in before if before[k]!=after[k]],acceptance_steps=16,
        observation_iterations=20,minimum_accepted_steps=16,minimum_accepted_iterations=3,
        run_modified=False),indent=2))


if __name__=='__main__':main()
