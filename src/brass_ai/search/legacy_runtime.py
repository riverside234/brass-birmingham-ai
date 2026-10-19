"""Shared v5 funded-industry strategy/search for new collections, with provenance.

The strategy baseline is bank-3dc65253a610191d66a5, not Network/Scout.
The bank names the profile ancestry; source pins identify the active v5 logic.
Frozen receipts do not establish strength of a new timed search engine.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from brass_ai.bank import below, digest, load_bank, read, require, sha

CONTRACT = 'shared-heuristic-search/1.0.0'
PREBUILT = 'shared-search-prebuilt/1.0.0'
# These policy files are authoritative in the game. The lab links the SAME
# files, not the older copies under BrassLegacyAI. Platform hosts/coordinators
# remain separate (the lab is four-player; the game supports two to four).
GAME_POLICY = ('Behavior.cs','CashManagement.cs','Conversion.cs','Evaluator.cs','Tactics.cs')


def source_paths(root):
    root=Path(root)
    sources=[p for p in (root/'src/BrassHost').glob('*.cs') if p.name!='Program.cs']
    sources += [p for p in (root/'src/BrassLegacyAI').glob('*.cs') if p.name not in GAME_POLICY]
    sources += list((root/'src/BrassMaskedSearch').glob('*.cs'))
    sources += [root/'src/BrassLocalAI/Policy'/name for name in GAME_POLICY]
    return sorted(sources)


def source_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode()).hexdigest()


def validate_reference(ref):
    require(isinstance(ref,dict) and set(ref)=={'id','recipe'},'Invalid shared search reference')
    recipe=ref['recipe']
    require(recipe.get('contract')==CONTRACT and ref['id']=='search-'+digest(recipe)[:24],
            'Shared search reference changed')
    require(set(recipe)=={'contract','bank','sources','profiles','adapter'} and recipe['sources'] and recipe['profiles'],
            'Incomplete shared strategy provenance')
    for h in [recipe['adapter'],*recipe['sources'].values(),*recipe['profiles'].values()]:
        require(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Invalid source fingerprint')
    return ref


def reference(root, bank):
    root=Path(root).resolve();bank=bank.split('/')[0]
    payload=load_bank(root,bank)['payload']
    profiles={}
    for name in payload['profiles']:
        path=root/'configs/legacy-ai'/(name+'.json');profile=read(path)
        require(profile['mode']=='masked_ismcts' and profile['player_count']==4,'Current lab profile must use four-player Masked search')
        profiles[name]=source_hash(path)
    recipe=dict(contract=CONTRACT,bank=bank,sources={p.relative_to(root).as_posix():source_hash(p) for p in source_paths(root)},
                profiles=profiles,adapter=source_hash(Path(__file__)))
    return dict(id='search-'+digest(recipe)[:24],recipe=recipe)


def directory(root, identity):
    require(isinstance(identity,str) and len(identity)==31 and identity.startswith('search-') and
            all(c in '0123456789abcdef' for c in identity[7:]), 'Invalid shared search runtime identity')
    return below(Path(root).resolve(), 'artifacts/legacy-ai/search/'+identity)


def recorded(root, identity):
    path = directory(root, identity)/'search.json'
    require(path.exists(), 'Shared search runtime missing; run python scripts/build-shared-search.py --root .')
    result=read(path);validate_reference(result['reference'])
    require(result['reference']['id']==identity,'Shared search runtime receipt changed')
    require(sha(path.parent/'runtime/runtime.json')==result['runtime_sha256'], 'Shared runtime manifest changed')
    return result['reference']


def runtime_directory(root, ref):
    require(recorded(root,ref['id'])==ref and reference(root,ref['recipe']['bank'])==ref,
            'Shared search source changed; rebuild and recollect with a fresh policy identity')
    return directory(root,ref['id'])/'runtime'


def bundle_directory(root, bank):
    return below(root,'training/runtime/shared-search/'+bank)


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify_file(path, expected, message):
    require(path.is_file(),message+' (missing: '+str(path)+')')
    require(sha(path)==expected,message+' (fingerprint mismatch: '+str(path)+')')


def publish(root, ref):
    """Ship only the authored managed executable, never licensed game DLLs."""
    root=Path(root).resolve();runtime=runtime_directory(root,ref)
    payload=load_bank(root,ref['recipe']['bank'])['payload']
    manifest=read(runtime/'runtime.json')
    dependencies=[f for f in payload['runtime_files'] if f['name']!='BrassLegacyAI.exe']
    require([f for f in manifest['files'] if f['name']!='BrassLegacyAI.exe']==dependencies,
            'Shared runtime dependencies differ from the bank contract')
    pins=[f for f in manifest['files'] if f['name']=='BrassLegacyAI.exe']
    require(len(pins)==1,'Shared executable pin missing or duplicated')
    verify_file(runtime/'BrassLegacyAI.exe',pins[0]['sha256'],'Shared executable changed')
    out=bundle_directory(root,ref['recipe']['bank']);out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(runtime/'BrassLegacyAI.exe',out/'BrassLegacyAI.exe')
    write_json(out/'prebuilt.json',dict(format=PREBUILT,reference=ref,compiler_sha256=payload['compiler_sha256'],
        executable=pins[0],dependencies=dependencies))
    return out


def build(root, bank, *, managed=None, compiler=None, mono=None, use_prebuilt=True):
    root = Path(root).resolve(); ref = reference(root,bank); bank = ref['recipe']['bank']
    payload = load_bank(root,bank)['payload']; out = directory(root,ref['id'])
    if (out/'search.json').exists():
        runtime=runtime_directory(root,ref)
        for pin in read(runtime/'runtime.json')['files']:
            verify_file(below(runtime,pin['name']),pin['sha256'],'Existing shared runtime changed')
        return ref
    dependencies = [f for f in payload['runtime_files'] if f['name']!='BrassLegacyAI.exe']
    bundle=bundle_directory(root,bank);prebuilt=None
    if use_prebuilt and compiler is None and (bundle/'prebuilt.json').exists():
        prebuilt=read(bundle/'prebuilt.json')
        require(prebuilt.get('format')==PREBUILT and prebuilt.get('reference')==ref,
                'Prebuilt shared search is stale: pull matching source and training/runtime/shared-search files, '
                'or rebuild locally with --compile --publish')
        require(prebuilt.get('compiler_sha256')==payload['compiler_sha256'] and prebuilt.get('dependencies')==dependencies,
                'Prebuilt compiler/dependency contract changed')
        pin=prebuilt.get('executable',{})
        require(pin.get('name')=='BrassLegacyAI.exe' and isinstance(pin.get('sha256'),str),'Invalid prebuilt executable pin')
        verify_file(bundle/'BrassLegacyAI.exe',pin['sha256'],'Prebuilt shared executable changed')
    else:
        compiler=below(root,compiler or 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe')
        require(compiler.is_file(), 'Pinned C# compiler missing and no matching prebuilt selected. '
                'Pull training/runtime/shared-search (BrassLegacyAI.exe and prebuilt.json), then rerun this command. '
                'To compile here, provide the complete Roslyn 4.12.0 toolset via --compiler; do not copy csc.exe alone.')
        verify_file(compiler,payload['compiler_sha256'],'Use the pinned compiler')
        if os.name!='nt':
            require(shutil.which(mono or 'mono') is not None,'Mono is required to compile on Linux; use --mono /path/to/mono')
    deps = below(root,managed or 'artifacts/legacy-ai/bank/'+bank+'/runtime')
    # Validate all inputs before writing a runtime; no partial manifest is usable.
    for pin in dependencies:
        verify_file(deps/pin['name'],pin['sha256'],'Private runtime dependency changed; use --managed with the pinned DLL directory')
    runtime = out/'runtime'; runtime.mkdir(parents=True,exist_ok=True)
    sources=source_paths(root)
    for pin in dependencies:
        shutil.copyfile(deps/pin['name'],runtime/pin['name'])
    exe = runtime/'BrassLegacyAI.exe'
    if prebuilt is not None:
        shutil.copyfile(bundle/'BrassLegacyAI.exe',exe)
        verify_file(exe,prebuilt['executable']['sha256'],'Copied prebuilt executable changed')
    else:
        lines = ['/nologo','/target:exe','/main:BrassLegacyAI.Program','/langversion:7.3','/nostdlib+',
                 '/deterministic+','/optimize+',f'/out:"{exe}"']
        lines += [f'/reference:"{runtime/f["name"]}"' for f in dependencies]
        lines += [f'"{p}"' for p in sources]
        rsp = out/'build.rsp'; rsp.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
        command = [str(compiler)] if os.name=='nt' else [mono or 'mono',str(compiler)]
        subprocess.run(command+['@'+str(rsp)],check=True,cwd=root,timeout=120,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(reference(root,bank)==ref, 'Search source changed during build; retry')
    # Normalized source hashes were checked above. Bind the exact local bytes
    # only now, allowing Git LF/CRLF conversion without relabelling the binary.
    write_json(runtime/'runtime.json',dict(format='legacy-runtime/0.1.0',player_count=4,rules=payload['rules'],
        protocol=payload['wire'],sources=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in sources],
        files=dependencies+[dict(name=exe.name,sha256=sha(exe))]))
    write_json(out/'search.json',dict(reference=ref,runtime_sha256=sha(runtime/'runtime.json'),origin='prebuilt' if prebuilt else 'compiled',
        strength_qualified=False,note='Current game strategy/search; historical bank is profile ancestry only.'))
    return ref
