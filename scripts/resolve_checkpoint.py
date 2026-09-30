"""Print an immutable, hash-checked BC/league checkpoint path for shell scripts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import LAB_ROOT,project_path,require

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--run',required=True)
    p.add_argument('--select',choices=['best','latest','oldest-opponent','latest-opponent'],default='latest')
    a=p.parse_args();root=a.root.resolve();run=project_path(root,a.run);state=read(run/'state.json')
    if a.select=='best': entry=state['best'];path=run/entry['path']
    elif 'history' not in state: entry=state['last'];path=run/entry['path']
    elif a.select=='latest': entry=state['history'][-1];path=project_path(root,entry['path'])
    else:
        entries=[e for e in state['population'] if e['model_id']!=state['history'][-1]['model_id']]
        require(entries,'No distinct historical opponent yet')
        entries.sort(key=lambda e:e['generation']);entry=entries[0 if a.select=='oldest-opponent' else -1]
        path=project_path(root,entry['path'])
    require(path.is_file() and sha(path)==entry['sha256'],'Committed checkpoint missing or changed')
    print(path)
