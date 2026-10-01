"""Install the explicit-chance worker beside the qualified R1 runtime.

The r4-host directory is retained as a transfer-path compatibility detail. It
does not select another search algorithm or require a retired comparison gate.
"""
import hashlib
import json
from pathlib import Path
import shutil

from brass_ai.learning.contracts import read_json, require


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
        if destination.exists():require(sha(destination)==sha(origin),'Different explicit-chance runtime; use a clean checkout')
        else:shutil.copyfile(origin,destination)
    shutil.copyfile(source/'runtime.json',target/'runtime.json')
    return dict(installed=True,directory=str(target),files=len(copies))
