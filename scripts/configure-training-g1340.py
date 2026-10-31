"""Adopt g1340 reward/exploration and DOMiNO context evaluation, without editing a run."""
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
    from brass_ai.recovery.training_revision_g1340 import configure,CURRENT
    from brass_ai.recovery.policy_observations import configure as observations_config
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id,write_json
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Write config under configs/step5')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; do not rewind training')
        entry=state['history'][-1];checkpoint=safe(run,project_path(root,entry['path']))
        require(entry['generation']==expected_generation and checkpoint.is_file() and sha(checkpoint)==entry['sha256'],
            'Committed full checkpoint missing/changed; reports cannot replace it')
        config=validate_config(observations_config(configure(state['signature']['config'])))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        interval=config['rating']['every_iterations'];period=config['context_evaluation']['every_iterations']
        due=state.get('context_evaluation',{}).get('next_due_generation',
            ((expected_generation+period+interval-1)//interval)*interval)
        return dict(config=output.relative_to(root).as_posix(),generation=expected_generation,run_modified=False,
            insolvency=CURRENT,exploration=config['exploration'],next_context_generation=due,
            policy_observations=config['policy_observations'],
            cache_refresh='Automatic on reviewed reward adoption; owned PPG targets are retired after durable config commit. '
                'Retries are safe; new-reward committed data are not repeatedly cleared.',
            scope='Preserve model/optimizer/RNG, progress, population and lineage. Rebuild reward-dependent DOMiNO '
                'quality evidence and PPG data under the existing freshness rules; preserve duals and their Adam state.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation),indent=2))


if __name__=='__main__':main()
