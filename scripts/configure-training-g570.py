"""Prepare g570 continuation with entropy annealing and its first reward step."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery.training_revision_g570 import configure
    from brass_ai.recovery import entropy_schedule
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.recovery.rating_coverage import seed_reference
    from brass_ai.recovery.v4_history import migrate,policies,report
    from brass_ai.step5.league import write_json,implementation_id
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json' and
        not output.is_relative_to(run),'Write config under configs/step5, outside the run')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation==570,'This hotfix requires committed generation 570; never rewind')
        entry=state['history'][-1];checkpoint=safe(run,project_path(root,entry['path']))
        require(entry['generation']==expected_generation and checkpoint.is_file() and sha(checkpoint)==entry['sha256'],
            'Committed full checkpoint missing/changed; reports cannot replace it')
        pending=safe(run,run/f'iteration-{expected_generation:04d}')
        require(not pending.exists() or pending.is_dir() and not any(pending.iterdir()),
            'Recover the interrupted collection before configuring g570')
        require(not list(run.glob(f'generation-{expected_generation+1:04d}.pt*')),'Recover the uncommitted checkpoint first')
        config=validate_config(entropy_schedule.configure(configure(state['signature']['config']),state,'on',start='next'))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        baseline=seed_reference(root,state,config)
        pool,receipt=migrate(state['quality_pool'],config,policies(root,config))
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=output.relative_to(root).as_posix(),committed_generation=570,first_collection_generation=571,
            entropy=config['update']['fixed_entropy']['weights'],exploration=config['exploration'],
            entropy_schedule=config['update']['entropy_schedule'],insolvency=config['insolvency'],
            normal_opening=config['opening'],branching=config['branching'],v4_history=report(pool),
            baseline=baseline,run_modified=False,
            scope='Manual v4 move and entropy schedule start apply during reviewed resume. Keep model/Adam/RNG, progress, GAE and PPG cadence. Reward step one invalidates utility calibration and refreshes incompatible PPG targets; first phase uses one fresh age.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
