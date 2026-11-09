"""Retain/repair exact frozen managed opponents without rebuilding old identities.

Transport only: no Torch, learner imports, game execution or state rewrites.
"""
import os
import re
import shutil
import uuid

from brass_ai.learning.contracts import digest
from brass_ai.learning.paths import resolve_project_path
from brass_ai.search.legacy_runtime import directory,recorded,source_hash,validate_reference

SUPPLEMENT='step5-frozen-runtime-supplement/1.0.0'


def live_rating_inputs(state):
    rating=state.get('ratings',{})
    return rating.get('agents',{}) if rating.get('config',{}).get('enabled') else {}


def references(root,value,*,require):
    found={}
    def visit(item):
        if isinstance(item,dict):
            if isinstance(item.get('recipe'),dict) and item['recipe'].get('contract')=='shared-heuristic-search/1.0.0':
                validate_reference(item)
                require(item['id'] not in found or found[item['id']]==item,'Conflicting frozen runtime references')
                found[item['id']]=item
            policy=item.get('search_runtime')
            if isinstance(policy,dict) and item.get('parent'):
                validate_reference(policy);bank,profile=item['parent'].split('/',1)
                require(bank==policy['recipe']['bank'],'Frozen runtime bank differs from its parent')
                require(re.fullmatch(r'[A-Za-z0-9_.-]+',profile) is not None,'Invalid frozen profile name')
                path=resolve_project_path(root,'configs/legacy-ai/'+profile+'.json')
                require(path.is_file() and source_hash(path)==policy['recipe']['profiles'].get(profile),
                        'Frozen rating profile changed: '+profile)
            for child in item.values():visit(child)
        elif isinstance(item,list):
            for child in item:visit(child)
    visit(value)
    return list(found.values())


def pins(metadata,*,require):
    result=metadata['files']
    require(isinstance(result,list) and result and all(isinstance(p,dict) and
        isinstance(p.get('name'),str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',p['name']) and
        isinstance(p.get('sha256'),str) and re.fullmatch(r'[0-9a-f]{64}',p['sha256']) for p in result),
        'Invalid frozen runtime file pins')
    require(len({p['name'].casefold() for p in result})==len(result) and
        sum(p['name']=='BrassLegacyAI.exe' for p in result)==1,'Duplicate/missing frozen executable pin')
    return result


def inventory(root,values,*,read,require,add,runtime):
    """Typed references contain id/recipe, not path/sha256; visit explicitly.

    Frozen workers verify saved binary receipts, not today's policy sources.
    Retain those exact receipts and ALL dependencies, even in repo mode.
    """
    for ref in references(root,values,require=require):
        base=directory(root,ref['id']);receipt=base/'search.json'
        require(receipt.is_file(),'Missing frozen runtime dependency: '+str(receipt)+
            '; repair the exact saved runtime before export')
        require(recorded(root,ref['id'])==ref,'Frozen runtime receipt/reference differs')
        manifest=base/'runtime/runtime.json';metadata=read(manifest)
        add(receipt);add(manifest,read(receipt)['runtime_sha256'])
        paths=[receipt,manifest]
        for pin in pins(metadata,require=require):
            path=resolve_project_path(root,(base/'runtime'/pin['name']).relative_to(root).as_posix())
            add(path,pin['sha256']);paths.append(path)
        runtime.update(p.relative_to(root).as_posix() for p in paths)


def repair(root,values,*,read,sha,require,publish,dry_run=False):
    """Rehydrate a sealed authored runtime with already restored private DLLs.

    All references, source bytes and destination conflicts are checked before
    any write. Publish search.json last; repeat/partial repairs are idempotent.
    """
    refs=references(root,values,require=require);planned={};blocked=[];reused=0
    distribution=read(root/'training/runtime/distribution.json')
    available={}
    for name,h in distribution['private_runtime_files'].items():
        available.setdefault(h,[]).append(resolve_project_path(root,name))
    hashes={}
    def matches(path,h):
        if not path.is_file():return False
        if path not in hashes:hashes[path]=sha(path)
        return hashes[path]==h
    for ref in refs:
        base=directory(root,ref['id'])
        if (base/'search.json').is_file() and (base/'runtime/runtime.json').is_file():
            require(recorded(root,ref['id'])==ref,'Existing frozen runtime reference differs')
            existing=pins(read(base/'runtime/runtime.json'),require=require)
            if all(matches(resolve_project_path(root,(base/'runtime'/p['name']).relative_to(root).as_posix()),p['sha256'])
                   for p in existing):
                reused+=len(existing)+2
                continue
        bank=ref['recipe']['bank']
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',bank) is not None,'Invalid frozen runtime bank')
        sealed=resolve_project_path(root,'training/runtime/shared-search/'+bank+'/frozen/'+ref['id'])
        catalog=sealed/'supplement.json'
        require(catalog.is_file(),'Exact frozen runtime supplement missing: '+str(catalog)+
            '; retain the original runtime, never substitute a newly built policy')
        binding=read(catalog)
        require(binding.get('contract')==SUPPLEMENT and binding.get('reference')==ref and
            binding.get('id')==digest({k:v for k,v in binding.items() if k!='id'}) and
            set(binding.get('files',{}))=={'search.json','runtime.json','BrassLegacyAI.exe'},
            'Frozen runtime supplement binding differs')
        for name,pin in binding['files'].items():
            source=resolve_project_path(root,(sealed/name).relative_to(root).as_posix())
            require(matches(source,pin['sha256']) and source.stat().st_size==pin['bytes'],
                    'Frozen runtime supplement file changed: '+str(source))
        receipt=read(sealed/'search.json');metadata=read(sealed/'runtime.json')
        require(receipt['reference']==ref and receipt['runtime_sha256']==sha(sealed/'runtime.json'),
                'Frozen runtime supplement receipt differs')
        files=pins(metadata,require=require)
        specs=[]
        for pin in files:
            sources=([sealed/'BrassLegacyAI.exe'] if pin['name']=='BrassLegacyAI.exe' else available.get(pin['sha256'],[]))
            specs.append((base/'runtime'/pin['name'],pin['sha256'],sources))
        # The public receipt is the final commit point for a runtime.
        specs.extend([(base/'runtime/runtime.json',receipt['runtime_sha256'],[sealed/'runtime.json']),
                      (base/'search.json',binding['files']['search.json']['sha256'],[sealed/'search.json'])])
        for target,h,sources in specs:
            target=resolve_project_path(root,target.relative_to(root).as_posix())
            if target.exists():
                if matches(target,h):reused+=1
                else:blocked.append('Conflicting frozen runtime file; nothing overwritten: '+str(target))
                continue
            source=next((p for p in sources if matches(p,h)),None)
            if source is None:
                blocked.append('Missing exact frozen runtime input: '+str(target));continue
            if target in planned:require(planned[target][1]==h,'Conflicting frozen runtime target bindings')
            planned[target]=(source,h)
    require(not blocked,'Frozen runtime repair preflight failed; nothing overwritten:\n'+'\n'.join(blocked))
    if not dry_run:
        for target,(source,h) in planned.items():
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name('.repair-'+uuid.uuid4().hex+'.partial')
            try:
                with source.open('rb') as incoming,temporary.open('xb') as output:
                    shutil.copyfileobj(incoming,output,1024*1024);output.flush();os.fsync(output.fileno())
                require(sha(temporary)==h,'Frozen runtime input changed during repair: '+str(source))
                if target.exists():require(sha(target)==h,'Frozen runtime target appeared with different bytes')
                else:publish(temporary,target)
            finally:temporary.unlink(missing_ok=True)
        def check(path,expected=None):
            require(path.is_file() and (expected is None or sha(path)==expected),'Repaired frozen runtime changed: '+str(path))
        inventory(root,values,read=read,require=require,add=check,runtime=set())
    return dict(contract='step5-frozen-runtime-repair/1.0.0',dry_run=dry_run,
        runtimes=[r['id'] for r in refs],files_to_install=len(planned),files_reused=reused,
        repaired=not dry_run,rebuilt=False,training_started=False)
