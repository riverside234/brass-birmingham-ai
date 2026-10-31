"""Generate an evaluation-only continuation config; leave all run files intact."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation,mode='report',every_iterations=50):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery.context_evaluation_config import configure
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id,write_json
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Write config under configs/step5')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; do not rewind training')
        e=state['history'][-1];path=safe(run,project_path(root,e['path']))
        require(e['generation']==expected_generation and path.is_file() and sha(path)==e['sha256'],'Missing/changed committed checkpoint')
        config=validate_config(configure(state['signature']['config'],mode,every_iterations))
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        interval=config['rating']['every_iterations']
        due=state.get('context_evaluation',{}).get('next_due_generation',
            ((expected_generation+every_iterations+interval-1)//interval)*interval)
        return dict(config=output.relative_to(root).as_posix(),generation=expected_generation,next_due_generation=due,
            run_modified=False,scope='Evaluation only. No training, optimizer, RNG, PPG cache, population or baseline changes.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-generation',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('off','report'),default='report');p.add_argument('--every-iterations',type=int,choices=(50,100),default=50)
    a=p.parse_args();print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation,a.mode,a.every_iterations),indent=2))


if __name__=='__main__':main()
