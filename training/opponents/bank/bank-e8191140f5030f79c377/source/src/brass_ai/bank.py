"""Immutable original-source snapshots and evidence-gated legacy opponent pools."""
from __future__ import annotations
import hashlib
import json
import re
import shutil
from pathlib import Path
from .legacy import LAB_ROOT, LegacyWorker

ADAPTER_FILES=('src/brass_ai/legacy.py','src/brass_ai/bank.py',
               'src/brass_ai/learning/legacy_policy.py','src/brass_ai/learning/runner.py',
               'src/brass_ai/recovery/legacy_search.py','src/brass_ai/search/common.py',
               'src/brass_ai/recovery/economy_bank.py')


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
    if directory.exists():
        load_bank(root,identity)
        return identity
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
        # Archived adapters are immutable evidence, not a restriction on all future
        # clients. Wire/rules contracts still must agree with the active host.
        if name.startswith('configs/contracts/') or name.startswith('configs/rules/'):
            require(sha(below(root,name))==h,'Active contract differs from frozen bank')
    return manifest


def admission(root,policy_id,purpose='training'):
    require(purpose in ('training','evaluation','diagnostic'),'Unknown bank purpose')
    if purpose=='diagnostic':return
    pool=read(Path(root)/'training/opponents/pool.json')
    selected=pool[purpose];require(policy_id in selected,'Opponent is not admitted to '+purpose+' pool')
    record=selected[policy_id]
    validate_admission(root,policy_id,record,purpose)


def validate_admission(root,policy_id,record,purpose='training'):
    """Verify current or archived evidence without admitting an archived policy."""
    require(purpose in ('training','evaluation'),'Unknown admission purpose')
    if record.get('admission') in ('masked-economy-smoke/1.0.0','masked-economy-smoke/1.1.0','masked-economy-smoke/1.2.0'):
        from .recovery.economy_bank import CONTROLS, HISTORICAL_CONTROLS
        advanced=record['admission']=='masked-economy-smoke/1.2.0'
        controls=CONTROLS if advanced else HISTORICAL_CONTROLS
        identity,name=policy_id.split('/'); manifest=load_bank(root,identity)
        require(record['manifest_sha256']==sha(below(root,'training/opponents/bank/'+identity+'/manifest.json')),'Economy manifest changed')
        path=below(root,record['qualification']); require(sha(path)==record['sha256'],'Economy receipt changed')
        report=read(path); p=manifest['payload']['profiles'][name]
        require(report['new_bank']==identity and report['passed'] and report['promote'] and report['games']==12*len(controls) and not report['failures'],'Missing completed winning screen')
        require(set(report['comparisons'])==set(controls),'Required immutable controls differ')
        for bank,c in report['comparisons'].items():
            require(c['games']==12 and c['runtime_files']==load_bank(root,bank)['payload']['runtime_files'],'Frozen control differs')
            require(c['policies']['new']['win_share']>c['policies']['old']['win_share'],'Fewer/equal wins cannot promote')
            if record['admission'] in ('masked-economy-smoke/1.1.0','masked-economy-smoke/1.2.0') and bank==CONTROLS[-1]:
                n,o=c['policies']['new'],c['policies']['old']
                require(report['format']==record['admission'] and c['all_metrics_improved'] and
                        n['first_places']>o['first_places'] and n['mean_vp']>o['mean_vp'] and n['mean_rank']<o['mean_rank'],
                        'Economic control must improve on first places, VP AND rank')
        require(report['new_runtime_files']==manifest['payload']['runtime_files'] and report['seconds']==1. and report['search_workers']==4,'Economy runtime/budget differs')
        require(p['mode']=='masked_ismcts' and p['milliseconds']==1000 and p['cash_policy']==p['conversion_policy']==1,'Economy teacher contract differs')
        require('src/BrassMaskedSearch/CompetitiveEconomy.cs' in manifest['payload']['runtime_source_paths'],'Economic/tempo repair missing')
        if record['admission'] in ('masked-economy-smoke/1.1.0','masked-economy-smoke/1.2.0'):
            require('src/BrassMaskedSearch/TournamentOpenings.cs' in manifest['payload']['runtime_source_paths'],'Attributed opening books missing')
        require((purpose=='evaluation')==name.startswith('pottery-flex'),'Pottery holdout changed')
        return
    if record.get('admission')=='masked-industry-smoke/1.0.0':
        identity,name=policy_id.split('/');manifest=load_bank(root,identity)
        require(record['manifest_sha256']==sha(below(root,'training/opponents/bank/'+identity+'/manifest.json')),'Industry bank manifest changed')
        report_path=below(root,record['qualification']);require(sha(report_path)==record['sha256'],'Industry smoke receipt changed')
        report=read(report_path);p=manifest['payload']['profiles'][name]
        require(report['new_bank']==identity and report['passed'] and report['promote'] and report['games']==24 and not report['failures'],'Missing completed/improved industry smoke')
        require(set(report['comparisons'])=={'bank-3f7221cadcd472aafdb9','bank-d82438ccec287dbc8a64'},'Both requested controls are required')
        for bank,comparison in report['comparisons'].items():
            require(comparison['games']==12 and comparison['runtime_files']==load_bank(root,bank)['payload']['runtime_files'],'Control evidence differs')
            candidate,control=comparison['policies']['new'],comparison['policies']['old']
            require(candidate['win_share']>control['win_share'] or candidate['mean_vp']>control['mean_vp'] or candidate['mean_rank']<control['mean_rank'],'No requested metric improved against a control')
        require(report['new_runtime_files']==manifest['payload']['runtime_files'] and report['seconds']==1. and report['search_workers']==4,'Candidate runtime/budget differs')
        require(p['mode']=='masked_ismcts' and p['milliseconds']==1000 and p['cash_policy']==p['conversion_policy']==1,'Industry teacher contract changed')
        require('src/BrassMaskedSearch/SaleTiming.cs' in manifest['payload']['runtime_source_paths'],'Group-sale repair missing')
        require((purpose=='evaluation')==name.startswith('pottery-flex'),'Pottery opponent holdout changed')
        return
    if record.get('admission')=='masked-era-smoke/1.0.0':
        identity,name=policy_id.split('/');manifest=load_bank(root,identity)
        require(record['manifest_sha256']==sha(below(root,'training/opponents/bank/'+identity+'/manifest.json')),'Era bank manifest changed')
        report_path=below(root,record['qualification']);require(sha(report_path)==record['sha256'],'Era smoke receipt changed')
        report=read(report_path);p=manifest['payload']['profiles'][name]
        require(report['new_bank']==identity and report['passed'] and report['games']==8 and not report['failures'],'Missing completed era smoke')
        require(report['new_runtime_files']==manifest['payload']['runtime_files'],'Era smoke/runtime differs')
        require(p['mode']=='masked_ismcts' and p['milliseconds']==1000 and p['cash_policy']==p['conversion_policy']==1,'Era teacher contract changed')
        require('src/BrassMaskedSearch/RootForest.cs' in manifest['payload']['runtime_source_paths'] and
                'src/BrassMaskedSearch/EraEconomy.cs' in manifest['payload']['runtime_source_paths'],'Era repair/forest source missing')
        require((purpose=='evaluation')==name.startswith('pottery-flex'),'Pottery opponent holdout changed')
        return
    if record.get('admission')=='user-directed-v3-s1/1.0.0':
        bank,name=policy_id.split('/');manifest=load_bank(root,bank)
        require(record['manifest_sha256']==sha(below(root,'training/opponents/bank/'+bank+'/manifest.json')),'S1 bank manifest changed')
        profile=manifest['payload']['profiles'][name]
        origin=manifest['payload']['training_origin']
        require(origin['source_bank']=='bank-b880ac1fd5f1dfe8bcc2' and origin['backend']=='S1','Original S1 source identity changed')
        source=load_bank(root,origin['source_bank'])['payload']
        require(source['runtime_files']==manifest['payload']['runtime_files'],'S1 worker differs from frozen original')
        require(profile['mode']=='S1' and profile['milliseconds']==1000 and profile['cash_policy']==profile['conversion_policy']==1,
                'One-second S1 contract changed')
        require((purpose=='evaluation')==name.startswith('pottery-flex'),'Pottery identity holdout changed')
        return
    if record.get('admission')=='masked-strength-screen/1.1.0':
        identity,name=policy_id.split('/');manifest=load_bank(root,identity)
        require(record['manifest_sha256']==sha(below(root,'training/opponents/bank/'+identity+'/manifest.json')),'Screened bank manifest changed')
        report_path=below(root,record['qualification']);require(sha(report_path)==record['sha256'],'Strength receipt changed')
        report=read(report_path);p=manifest['payload']['profiles'][name]
        require(report['bank']==identity and report['qualified'] and report['screen']['games']==20 and not report['screen']['failures'],'Missing matched strength screen')
        require(report['runtime_files']==manifest['payload']['runtime_files'],'Strength screen/runtime identity differs')
        require(p['mode']=='masked_ismcts' and p['milliseconds']==1000 and p['search_contract']=='masked-ismcts/1.1.0' and p['evaluator_kind']=='heuristic','Screened search contract changed')
        require((purpose=='evaluation')==name.startswith('pottery-flex'),'Pottery identity holdout changed')
        return
    if record.get('admission')=='user-directed-masked-ismcts/1.0.0':
        bank,name=policy_id.split('/');manifest=load_bank(root,bank)
        require(record['manifest_sha256']==sha(below(root,'training/opponents/bank/'+bank+'/manifest.json')),'Direct bank manifest changed')
        profile=manifest['payload']['profiles'][name]
        require(profile['mode']=='masked_ismcts' and profile['milliseconds']==1000 and profile['cash_policy']==profile['conversion_policy']==1,
                'Direct one-second v3 contract changed')
        require((purpose=='evaluation')==('pottery-flex' in name),'Pottery opponent holdout changed')
        return
    path=below(root,record['qualification'])
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
