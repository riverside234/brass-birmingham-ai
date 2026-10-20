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
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.value_phase import disabled_options
from brass_ai.step5.league import implementation_id,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only.json')
    a=p.parse_args();run=project_path(ROOT,a.run);output=project_path(ROOT,a.output)
    require(output.resolve().is_relative_to((ROOT/'configs').resolve()),'Output must be under configs/')
    state=read(run/'state.json');before=state['signature']['config']
    after=validate_config(disabled_options(before))
    signature=dict(copy.deepcopy(state['signature']),config=after,implementation=implementation_id())
    check_signature(state,signature,reviewed=True)
    require(not output.exists() or read(output)==after,'Existing output differs; choose a new --output')
    write_json(output,after)
    print(json.dumps(dict(config=output.relative_to(ROOT).as_posix(),committed_generation=state['iteration'],
        changed_blocks=[k for k in after if before.get(k)!=after[k]],value_phase_enabled=False,
        critic_gate_evidence='PPO critic updates and unchanged whole-game holdout',
        run_modified=False),indent=2))


if __name__=='__main__':main()
