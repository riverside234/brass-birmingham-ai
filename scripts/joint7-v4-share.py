"""Generate the reviewed v4-to-history/8-8-2 holdout continuation config."""
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
from brass_ai.recovery.v4_share import configure,migrate_state
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    parser.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-v4-history.json')
    args=parser.parse_args();run=project_path(ROOT,args.run);output=project_path(ROOT,args.output)
    require(output.resolve().is_relative_to((ROOT/'configs').resolve()),'Output must be under configs/')
    state=read(run/'state.json');before=state['signature']['config']
    after=validate_config(configure(before))
    signature=dict(copy.deepcopy(state['signature']),config=after,implementation=implementation_id())
    check_signature(state,signature,reviewed=True)
    _,migration=migrate_state(state,after)
    require(not output.exists() or read(output)==after,'Existing output differs; choose a new --output')
    write_json(output,after)
    print(json.dumps(dict(config=output.relative_to(ROOT).as_posix(),committed_generation=state['iteration'],
        changed_blocks=[k for k in after if before.get(k)!=after[k]],run_modified=False,
        base_tables=[160,144,16],gated_tables=[160,154,6],critic_holdout=[8,8,2],
        pfsp_gated_opponent_seats=308,pfsp_gated_comparisons=616,gate_migration=migration,
        signal='Latest completed held-out v4 panel; strict win share > 0.95; one-way latch'),indent=2))


if __name__=='__main__':main()
