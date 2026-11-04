"""Configure committed-health entropy annealing and explicit reward-step restarts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation,command,*,start=None,reset=False,reverse=False):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery import entropy_schedule as S
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id,write_json
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json' and
        not output.is_relative_to(run),'Write config under configs/step5, outside the run')
    with run_lock(run):
        state=read(safe(run,run/'state.json'));g=state['iteration']
        require(g==expected_generation,'Stale expected generation; do not rewind')
        entry=state['history'][-1];path=safe(run,project_path(root,entry['path']))
        require(entry['generation']==g and path.is_file() and sha(path)==entry['sha256'],
            'Committed full checkpoint missing/changed')
        pending=safe(run,run/f'iteration-{g:04d}')
        require(not pending.exists() or pending.is_dir() and not any(pending.iterdir()),
            'Recover the interrupted collection before configuring a schedule restart')
        require(not list(run.glob(f'generation-{g+1:04d}.pt*')),'Recover the uncommitted checkpoint first')
        config=validate_config(S.configure(state['signature']['config'],state,command,start=start,reset=reset,reverse=reverse))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        schedule=S.restore_run(config,state)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),generation=g,run_modified=False,
            entropy_schedule=schedule,insolvency=config['insolvency'],
            reward_changed=config['insolvency']!=state['signature']['config']['insolvency'],
            scope='Config only. Reviewed resume preserves weights/Adam/RNG/progress/baselines. Reward steps invalidate calibration and refresh PPG caches; on/off alone preserves reward and replay.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('on','off','insolvency-step'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--output',required=True)
    p.add_argument('--start',help='next or the first generation to collect (on only)')
    p.add_argument('--reset',action='store_true',help='Explicitly restart entropy progress (on only, before reward step two)')
    p.add_argument('--reverse',action='store_true',help='Explicitly reverse the latest applied reward step; entropy progress stays unchanged')
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation,a.command,
        start=a.start,reset=a.reset,reverse=a.reverse),indent=2))


if __name__=='__main__':main()
