"""Retain every completed game after an explicitly requested study-size reduction.

No games run here and no result is discarded. The original spec/results remain
unchanged. This records a post-start scope amendment, not a preregistered study.
"""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT,below,digest,read,require,sha
from brass_ai.qualification import write_new


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT)
    p.add_argument('--spec',required=True);p.add_argument('--matches',required=True)
    p.add_argument('--requested-games',type=int,required=True);p.add_argument('--reason',required=True)
    a=p.parse_args();root=a.root.resolve();source=read(below(root,a.spec));records=[]
    require(source['id']=='qualification-'+digest(source['payload'])[:20],'Source spec changed')
    require(source['payload']['generation']=='v3' and a.requested_games>=25 and a.reason.strip(),
            'Expected an explicit v3 study reduction with at least 25 games')
    for job in source['payload']['jobs']:
        path=below(root,a.matches)/(job['id']+'.json')
        if not path.exists():continue
        r=read(path)
        require(r['spec_id']==source['id'] and r['job']==job,'Source record changed')
        records.append((path,r))  # Include failures too; qualification will reject them.
    require(len(records)>=a.requested_games,'Not enough completed records')
    payload=copy.deepcopy(source['payload'])
    payload['jobs']=[r['job'] for _,r in records]
    payload['amendment']=dict(kind='user-requested-limited-validation',reason=a.reason,
        requested_games=a.requested_games,retained_games=len(records),parent_spec=source['id'],
        parent_spec_sha256=sha(below(root,a.spec)),original_admission=payload['admission'].copy(),
        selection='Every result file completed before the runner stopped; no outcome filtering.')
    payload['admission'].update(min_guarded_games=1,min_mixed_games=0,min_identical_games=0)
    payload['stopping']='User stopped the original study; retain all completed results. No new games are required.'
    payload['release_scope']='User-accepted limited Windows validation for niche opponents; no complete strength or all-table coverage claim.'
    spec=dict(id='qualification-'+digest(payload)[:20],payload=payload)
    directory=root/'training/opponents/evaluations'/spec['id'];output=root/'artifacts/legacy-ai'/spec['id']
    def save(path,value):
        if path.exists():require(read(path)==value,'Existing amendment differs')
        else:write_new(path,value)
    save(directory/'spec.json',spec);save(output/'spec.json',spec)
    for path,r in records:
        imported=copy.deepcopy(r);imported['spec_id']=spec['id']
        imported['imported_from']=dict(path=path.relative_to(root).as_posix(),sha256=sha(path),spec_id=source['id'])
        save(output/path.name,imported)
    for name in ('platform-check.json','r2-integration.json'):
        evidence=root/'training/opponents/evaluations'/source['id']/name
        save(directory/name,read(evidence))
    print(spec['id'])


if __name__=='__main__':main()
