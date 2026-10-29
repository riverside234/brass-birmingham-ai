"""Explicit sealed B04 layering for the retained B02.5/B03 qualification tools."""
import hashlib
import json
from pathlib import Path
import zipfile
from brass_ai.learning.contracts import SOURCE_ROOT, require
from .lab_contract import sha

ARCHIVES = {'b025':'b025-search-v1-teacher-phase.zip','b03':'b03-search-v1-admission.zip',
            'b04':'b04-search-v1-teacher.zip'}


def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def verify(milestone, supplement=None):
    from brass_ai.step5.league import implementation_id
    root = SOURCE_ROOT; destination = root/'distribution/rust-lab-b04/transfer-manifest.json'
    if supplement is not None:
        require((root/supplement).resolve() == destination,'Unknown qualification supplement')
    require(milestone in ARCHIVES,'Unknown qualification milestone')
    newest = json.loads(destination.read_text(encoding='utf-8'))
    require(newest['contract'] == 'brass-lab-b04-transfer/1.0.0','Unknown B04 transfer')
    order = ['b025','b03','b04'] if milestone == 'b025' else ['b03','b04'] if milestone == 'b03' else ['b04']
    files = {}
    for m in order:
        name = 'distribution/rust-lab-'+m+'/transfer-manifest.json';path = root/name
        manifest = json.loads(path.read_text(encoding='utf-8'))
        require(manifest['contract'] == 'brass-lab-'+m+'-transfer/1.0.0','Unknown layered transfer')
        archive_name = 'distribution/rust-lab-'+m+'/'+ARCHIVES[m];archive = root/archive_name
        if m != 'b04':
            require(sha(archive) == newest['prior_bundles'][archive_name],'Changed prerequisite archive')
        else:
            receipt = json.loads((archive.parent/'bundle.json').read_text(encoding='utf-8'))
            require(receipt['sha256'] == sha(archive) and receipt['implementation'] == newest['implementation'],
                    'Changed B04 archive')
        with zipfile.ZipFile(archive) as z:
            require(json.loads(z.read(name)) == manifest,'Manifest differs from sealed archive: '+m)
        files.update(manifest['files'])
    for name,expected in files.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.is_file() and text_sha(path) == expected,'Changed layered file: '+name)
    require(implementation_id() == newest['implementation'],'Checkout differs from reviewed B04 source')
    return dict(manifest_sha256=text_sha(destination),implementation=newest['implementation'],files=len(files),milestone=milestone)
