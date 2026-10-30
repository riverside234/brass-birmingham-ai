"""Bind DOMiNO adoption to a real committed full checkpoint; do not train here."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation,mode='apply',*,entropy_010=False):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery.domino_config import configure
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id,write_json
    from brass_ai.recovery.domino_fingerprints import inventory
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Write config under configs/step5')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; do not rewind state.json')
        entry=state['history'][-1];checkpoint=safe(run,project_path(root,entry['path']))
        require(entry['generation']==expected_generation and checkpoint.is_file() and sha(checkpoint)==entry['sha256'],
                'Committed full checkpoint missing/changed; reports cannot replace it')
        inventory(root,state)
        config=configure(state['signature']['config'],state,mode)
        if entropy_010:
            from brass_ai.recovery.exploration import fixed_010_config
            config=fixed_010_config(config)
        config=validate_config(config)
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),generation=expected_generation,run_modified=False,
            before=state['signature']['config'].get('diversity',{'mode':'off'}),after=config['diversity'],
            fixed_entropy=config['update'].get('fixed_entropy'),
            cache_refresh='Empty new PPG window; expire old owned caches only after new checkpoint commits',
            pfsp='Archive neutral-context scores; same members/prior/half-life, uniform strategy per opponent seat',
            preserved='Base weights, PPO/value/local Adam, RNG, progress, population, IL, KL anchor, economic baseline and GAE/snapshot controls')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('shadow','apply'),default='apply')
    p.add_argument('--entropy-010',action='store_true',help='Explicitly set all four fixed PPO entropy coefficients to 0.10')
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation,a.mode,entropy_010=a.entropy_010),indent=2))


if __name__=='__main__':main()
