"""Immutable original-source snapshots and evidence-gated legacy opponent pools."""
from __future__ import annotations
import hashlib
import json
import re
import shutil
from pathlib import Path
from .legacy import LAB_ROOT, LegacyWorker

ADAPTER_FILES=('src/brass_ai/legacy.py','src/brass_ai/bank.py',
               'src/brass_ai/learning/legacy_policy.py','src/brass_ai/learning/runner.py')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)
def below(root,path):
    root=Path(root).resolve();out=(root/path).resolve()
    require(out.is_relative_to(root),'Bank path escapes project root');return out


def freeze(root,profile_names,*,fixed_work=False):
    require(profile_names and len(set(profile_names))==len(profile_names) and all(re.fullmatch(r'[a-z0-9-]+',n) for n in profile_names),'Invalid/duplicate profile name')
    root=Path(root).resolve();runtime=read(root/'artifacts/legacy-ai/runtime/runtime.json')
    files={r['path']:r['sha256'] for r in runtime['sources']}
    for pattern in ('src/BrassLegacyAI/Unity/*.cs',*ADAPTER_FILES,'configs/contracts/v0.2.0/*.json',
                    'configs/rules/runtime-dependencies-1.1.0.json'):
        for path in root.glob(pattern):files[path.relative_to(root).as_posix()]=sha(path)
    profiles={name:read(below(root,f'configs/legacy-ai/{name}.json')) for name in profile_names}
    require(profiles and all(p['player_count']==4 for p in profiles.values()),'Four-player profiles required')
    for name in profiles:files[f'configs/legacy-ai/{name}.json']=sha(root/f'configs/legacy-ai/{name}.json')
    for path,h in files.items():require(sha(below(root,path))==h,'Rebuild before freeze: '+path)
    for f in runtime['files']:require(sha(root/'artifacts/legacy-ai/runtime'/f['name'])==f['sha256'],'Runtime changed before freeze')
    payload=dict(format='legacy-bank/1.0.0',player_count=4,rules=runtime['rules'],wire=runtime['protocol'],
                 source_files=files,runtime_files=runtime['files'],runtime_source_paths=[f['path'] for f in runtime['sources']],
                 profiles=profiles,fixed_work=fixed_work,platform='windows-dotnet-unity-mono',sampling_seed=1,
                 adapter_files=list(ADAPTER_FILES),compiler_sha256=sha(root/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'))
    identity='bank-'+digest(payload)[:20];directory=root/'training/opponents/bank'/identity
    require(not directory.exists(),'Refusing to rewrite frozen bank '+identity)
    directory.mkdir(parents=True)
    for path in files:
        destination=directory/'source'/path;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/path,destination)
    (directory/'manifest.json').write_text(json.dumps(dict(id=identity,payload=payload),indent=2)+'\n',encoding='utf-8')
    private=root/'artifacts/legacy-ai/bank'/identity/'runtime';private.mkdir(parents=True,exist_ok=False)
    for f in runtime['files']:shutil.copyfile(root/'artifacts/legacy-ai/runtime'/f['name'],private/f['name'])
    frozen=dict(runtime,sources=[dict(path=(directory/'source'/p).relative_to(root).as_posix(),sha256=files[p]) for p in payload['runtime_source_paths']])
    (private/'runtime.json').write_text(json.dumps(frozen,indent=2)+'\n',encoding='utf-8')
    return identity


def load_bank(root,identity):
    root=Path(root).resolve();path=below(root,'training/opponents/bank/'+identity)
    manifest=read(path/'manifest.json');payload=manifest['payload']
    require(identity==manifest['id']=='bank-'+digest(payload)[:20],'Frozen bank manifest changed')
    require(payload['player_count']==4 and payload['wire']=='0.2.0' and payload['rules']=='steam-brass-patch-1.1.0','Incompatible bank contract')
    for name,h in payload['source_files'].items():require(sha(below(path/'source',name))==h,'Frozen source changed: '+name)
    for name,h in payload['source_files'].items():
        if name.startswith('configs/contracts/') or name.startswith('configs/rules/') or name in payload['adapter_files']:
            require(sha(below(root,name))==h,'Active contract differs from frozen bank')
    return manifest


def admission(root,policy_id,purpose='training'):
    require(purpose in ('training','evaluation','diagnostic'),'Unknown bank purpose')
    if purpose=='diagnostic':return
    pool=read(Path(root)/'training/opponents/pool.json')
    selected=pool[purpose];require(policy_id in selected,'Opponent is not admitted to '+purpose+' pool')
    record=selected[policy_id];path=below(root,record['qualification'])
    require(sha(path)==record['sha256'],'Qualification report changed')
    report=read(path);require(report['bank']==policy_id.split('/')[0],'Qualification belongs to another bank')
    result=report['profiles'][record['profile']]
    require(result['policy_id']==policy_id and result[purpose+'_eligible'] and result['niche_qualified'],
            'Qualification does not admit this opponent')


class FrozenOpponent:
    def __init__(self,root,policy_id,*,purpose='training',mono=None):
        self.root=Path(root).resolve()
        parts=policy_id.split('/')
        require(len(parts)==2,'Use a frozen bank/profile identity')
        identity,name=parts;manifest=load_bank(self.root,identity);payload=manifest['payload']
        require(name in payload['profiles'],'Profile missing from frozen bank')
        admission(self.root,policy_id,purpose)
        self.policy_id=policy_id;self.profile=payload['profiles'][name];self.fixed_work=payload['fixed_work']
        runtime='artifacts/legacy-ai/bank/'+identity+'/runtime'
        actual=read(self.root/runtime/'runtime.json')
        require(actual['files']==payload['runtime_files'],'Frozen runtime manifest changed')
        expected=[dict(path=f'training/opponents/bank/{identity}/source/{p}',sha256=payload['source_files'][p]) for p in payload['runtime_source_paths']]
        require(actual['sources']==expected,'Frozen source mapping changed')
        self.worker=LegacyWorker(self.root,mono=mono,timeout=600,runtime_directory=runtime)
    def close(self):self.worker.close()
