"""Prepare a reviewed continuation whose additive VP reward stops at 180 VP."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.bank import read
from brass_ai.learning.contracts import project_path,require
from brass_ai.recovery.config import validate_config,vp_cap_config
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.ppg_config import active
from brass_ai.step5.league import implementation_id,write_json


def generate(root,run_path,output,*,config_path=None,expected_generation=None):
    root,run=owned(root,run_path);output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Output must be under configs/step5/')
    with run_lock(run):
        state=read(run/'state.json');old=state['signature']['config']
        require(expected_generation is None or expected_generation==state['iteration'],'Unexpected committed generation')
        base=read(project_path(root,config_path)) if config_path is not None else old
        config=validate_config(vp_cap_config(base))
        signature=dict(copy.deepcopy(state['signature']),config=config,implementation=implementation_id())
        check_signature(state,signature,reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        fresh=('vp_cap' not in old['reward'] and config['value_phase']['enabled'] and active(config['value_phase']))
        return dict(config=output.relative_to(root).as_posix(),committed_generation=state['iteration'],
            changed_blocks=[k for k in config if config[k]!=old[k]],run_modified=False,
            reward=config['reward'],insolvency=config['insolvency'],
            first_fresh_ppg_generation=state['iteration']+8 if fresh else None,
            scope='Cap only the additive VP reward; official win/rank/VP stay raw. Restore weights, both Adams, RNG and progress. Exclude old-target PPG caches after adoption and collect eight fresh iterations before the next auxiliary phase.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--config',help='Optional already-prepared reviewed recipe; otherwise use committed config')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-vp180-g540.json')
    p.add_argument('--expected-generation',type=int)
    a=p.parse_args();print(json.dumps(generate(ROOT,a.run,a.output,
        config_path=a.config,expected_generation=a.expected_generation),indent=2))


if __name__=='__main__':main()
