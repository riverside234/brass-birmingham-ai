"""Install authored R4 worker beside, never over, the qualified R1 runtime."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

from brass_ai.learning.contracts import digest, read_json, require


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(root):
    root=Path(root).resolve();source=root/'training/runtime/search'
    manifest=read_json(source/'runtime.json');target=root/'artifacts/r4-host'
    require(manifest['extension']=='brass-search-chance/1.0.0','Wrong search host extension')
    copies=[]
    for entry in manifest['files']:
        origin=source/entry['name'] if entry['name']=='BrassHost.exe' else root/'artifacts/r1-host/guarded'/entry['name']
        require(sha(origin)==entry['sha256'],'Missing or changed search dependency: '+str(origin))
        copies.append((origin,target/entry['name']))
    target.mkdir(parents=True,exist_ok=True)
    for origin,destination in copies:
        if destination.exists():require(sha(destination)==sha(origin),'Different R4 runtime; choose a clean experiment checkout')
        else:shutil.copyfile(origin,destination)
    shutil.copyfile(source/'runtime.json',target/'runtime.json')
    return dict(installed=True,directory=str(target),files=len(copies))


def implementation(root):
    root=Path(root)
    paths=sorted((root/'src/brass_ai/searchlab').glob('*.py'))+sorted((root/'src/BrassSearchHost').glob('*.cs'))
    paths += [root/'scripts/build-search-host.py',root/'scripts/r4r5.py',
              root/'scripts/verify-searchlab.py',root/'scripts/smoke-searchlab.py']
    paths += sorted((root/'configs/searchlab').glob('*.json'))
    paths += sorted((root/'tests/SearchLab').glob('*.py'))
    return digest({p.relative_to(root).as_posix():p.read_text(encoding='utf-8-sig').replace('\r\n','\n') for p in paths})


def gate_identity(root,device='cpu',mono=None):
    import importlib.metadata
    import torch
    from brass_ai.step5.league import implementation_id
    require(device in ('cpu','cuda') and (device!='cuda' or torch.cuda.is_available()),'Unavailable experiment device')
    return dict(implementation=implementation(root),runtime=sha(Path(root)/'training/runtime/search/BrassHost.exe'),
        r1_runtime=sha(Path(root)/'artifacts/r1-host/guarded/BrassHost.exe'),step5=implementation_id(),
        openspiel=importlib.metadata.version('open_spiel'),platform=platform.platform(),
        python=sys.version.split()[0],torch=torch.__version__,device=device,
        accelerator=torch.cuda.get_device_name() if device=='cuda' else platform.processor(),
        mono=str(mono or ('default-mono' if platform.system()!='Windows' else 'dotnet-framework')))


def require_gate(root,device='cpu',mono=None):
    report=read_json(Path(root)/f'artifacts/searchlab/correctness-{device}.json')
    require(report['passed'] and report['identity']==gate_identity(root,device,mono),
            'Run scripts/verify-searchlab.py --integration on this host before experiment matches')
    return report
