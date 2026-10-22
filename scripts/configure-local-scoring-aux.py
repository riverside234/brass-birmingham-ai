"""Prepare an exact recorded-config continuation; never edit checkpoints/caches."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def generate(root,run_path,output,expected_generation,*,mode='apply',fresh_ppg_buffer=False):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.recovery.retention import owned,safe,run_lock
    from brass_ai.recovery.local_scoring_adoption import configure
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import write_json,implementation_id
    root,run=owned(root,project_path(root,run_path));output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json' and not output.is_relative_to(run),
        'Write the config under configs/step5, outside the run')
    with run_lock(run):
        state=read(safe(run,run/'state.json'))
        require(state['iteration']==expected_generation,'Stale expected generation; never rewind a committed run')
        old=state['signature']['config'];first='local_scoring_auxiliary' not in old
        require(not first or (fresh_ppg_buffer and mode!='off'),'Initial adoption requires --fresh-ppg-buffer and apply/shadow')
        require(first or not fresh_ppg_buffer,'The fresh PPG reset is one-time; omit --fresh-ppg-buffer after adoption')
        entry=state['history'][-1]
        require(sha(safe(run,project_path(root,entry['path'])))==entry['sha256'],'Committed checkpoint missing/changed')
        config=configure(state,mode);validate_config(config)
        check_signature(state,dict(state['signature'],config=config,implementation=implementation_id()),reviewed=True)
        require(not output.exists() or read(output)==config,'Existing output differs; choose a new --output')
        write_json(output,config)
        return dict(config=str(output),generation=state['iteration'],mode=mode,branching=.01,run_modified=False,
            lr_raise_below_ratio=config['ppo_lr_controller']['low_ratio'],
            lr_raise_below_kl=config['ppo_lr_controller']['low_ratio']*config['ppo_lr_controller']['target_kl'],
            next_fresh_ppg=state['iteration']+8 if first else None,
            changes={k:dict(before=old.get(k),after=v) for k,v in config.items() if old.get(k)!=v},
            scope='Config only. Locked training startup validates, commits adoption, then retires old owned caches.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run',required=True)
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--mode',choices=('off','shadow','apply'),default='apply')
    p.add_argument('--fresh-ppg-buffer',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(generate(a.root.resolve(),a.run,a.output,a.expected_generation,
        mode=a.mode,fresh_ppg_buffer=a.fresh_ppg_buffer),indent=2))


if __name__=='__main__':main()
