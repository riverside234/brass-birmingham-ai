"""Sealed g1330 AZ_db supplement verification, independent of historical ZIPs."""
import hashlib
import json
from pathlib import Path
import zipfile
from brass_ai.learning.contracts import SOURCE_ROOT,require
from .lab_contract import sha,SOURCE

DIRECTORY='distribution/rust-lab-g1330'
CONTRACT='brass-lab-domino-transfer/1.0.0'


def verify(stage,supplement=None):
    from brass_ai.step5.league import implementation_id
    from .lab_teacher_overlay import source_fingerprint
    path=SOURCE_ROOT/DIRECTORY/'transfer-manifest.json'
    require(stage in ('b025','b03','b04') and (supplement is None or (SOURCE_ROOT/supplement).resolve()==path),
            'Unknown AZ_db supplement/stage')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    require(manifest['contract']==CONTRACT and manifest['base_core']==SOURCE and manifest['method']=='search_v1'
            and manifest['v205_enabled'] is False,'Unknown AZ_db transfer objective')
    receipt=json.loads((path.parent/'bundle.json').read_text())
    archive=SOURCE_ROOT/receipt['path']
    require(archive.parent==path.parent and sha(archive)==receipt['sha256'],'Changed AZ_db archive')
    with zipfile.ZipFile(archive) as z:
        require(json.loads(z.read(DIRECTORY+'/transfer-manifest.json'))==manifest,'AZ_db sidecar differs from archive')
    for name,expected in manifest['files'].items():
        source=(SOURCE_ROOT/name).resolve()
        require(source.is_relative_to(SOURCE_ROOT) and source.is_file()
                and hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,
                'Changed AZ_db source: '+name)
    for name,expected in manifest['archives'].items():
        source=(SOURCE_ROOT/name).resolve()
        require(source.is_relative_to(SOURCE_ROOT) and sha(source)==expected,'Changed original archive: '+name)
    require(implementation_id()==manifest['implementation'] and source_fingerprint()==manifest['producer'],
            'AZ_db checkout differs from reviewed producer')
    return dict(manifest_sha256=sha(path),implementation=manifest['implementation'],producer=manifest['producer'],
                files=len(manifest['files']),milestone=stage,teacher_profile='domino',lab_qualification='Pending')
