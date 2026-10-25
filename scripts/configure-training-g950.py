"""Generate the g950 PPG-only continuation config; trainer applies beta once on adoption."""
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
    from brass_ai.recovery.ppg_g210 import linear_options
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
            'Committed full checkpoint missing/changed; reports cannot replace it')
        config=validate_config(linear_options(state['signature']['config']))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),generation=expected_generation,run_modified=False,
            value_phase=config['value_phase'],beta_on_adoption=1.,beta_reset='Once on v15-to-v16 adoption; subsequent restarts restore saved beta',
            scope='PPG-only: keep weights, Adam/RNG, cached targets, ordinary progress, population and independent gate evidence.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
