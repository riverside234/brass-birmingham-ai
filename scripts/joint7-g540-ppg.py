"""Generate a reviewed continuation with one-SE PPG retention and .020 insolvency cost."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.bank import read
from brass_ai.learning.contracts import project_path,require
from brass_ai.recovery.config import insolvency_020_config,validate_config
from brass_ai.recovery.ppg_config import coverage_options
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json


def configure(config):
    require(config['value_phase']['enabled'],'This continuation requires active PPG')
    require(config['insolvency'] in (dict(event_cost=.010,max_cost=.075),dict(event_cost=.020,max_cost=.080)),
        'Expected the current .010/.075 insolvency recipe')
    return validate_config(coverage_options(insolvency_020_config(config)))


def generate(root,run_path,output,expected_generation=None):
    root,run=owned(root,run_path);output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Output must be under configs/step5/')
    with run_lock(run):
        state=read(run/'state.json');old=state['signature']['config']
        require(expected_generation is None or expected_generation==state['iteration'],'Unexpected committed generation')
        config=configure(old);signature=dict(copy.deepcopy(state['signature']),config=config,implementation=implementation_id())
        check_signature(state,signature,reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),committed_generation=state['iteration'],
            changed_blocks=[k for k in config if config[k]!=old[k]],run_modified=False,
            insolvency=config['insolvency'],selection=config['value_phase']['selection'],
            first_fresh_ppg_generation=state['iteration']+8,
            scope='Preserve weights, both Adams, RNG and ordinary progress. Explicitly reset PPG cache/schedule at this generation, with epoch cap/floor two. Startup deletes owned old caches after reviewed adoption; new targets use .020.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-one-se-g540.json')
    p.add_argument('--expected-generation',type=int)
    a=p.parse_args();print(json.dumps(generate(ROOT,a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
