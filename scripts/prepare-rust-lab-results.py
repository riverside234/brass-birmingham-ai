#!/usr/bin/env python3
"""Bundle the sealed B01-B04 inputs with automatic portable result reporting."""
import hashlib
import json
from pathlib import Path
import zipfile

from rust_lab_results import ROOT, sha, write


def main():
    out = ROOT/'distribution/rust-lab-g1326'
    archive = out/'rust-lab-b01-b04-g1326.zip'
    if archive.exists(): raise ValueError('Combined update is already sealed; use a new version')
    log = ROOT/'results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z'
    state = json.loads((log/'state.json').read_text(encoding='utf-8'))
    exported = json.loads((log/'export-manifest.json').read_text(encoding='utf-8'))
    reference = dict(run=exported['run'], generation=state['iteration'], games_done=exported['games_done'],
        checkpoint=state['history'][-1], implementation=state['signature']['implementation'],
        report_directory=log.relative_to(ROOT).as_posix(), state_export_sha256=sha(log/'state.json'),
        report_manifest_sha256=sha(log/'export-manifest.json'), qualification='Pending')
    assert reference['generation'] == 1326
    write(out/'training-reference.json', reference)
    names = ['scripts/rust_lab_results.py', 'scripts/run-rust-lab-qualification.py',
        'scripts/prepare-rust-lab-results.py', 'tests/RustBackendLab/test_result_export.py',
        'distribution/rust-lab-g1326/README.md', 'distribution/rust-lab-g1326/training-reference.json',
        'results/rust-lab-qualification/README.md']
    archives = ['distribution/rust-lab-'+s+'/'+n for s,n in (
        ('b01','b01-search-v1-inputs.zip'), ('b01','b01-search-v1-update.zip'),
        ('b02','b02-search-v1-update.zip'), ('b025','b025-search-v1-teacher-phase.zip'),
        ('b03','b03-search-v1-admission.zip'), ('b04','b04-search-v1-teacher.zip'))]
    b04 = json.loads((ROOT/'distribution/rust-lab-b04/transfer-manifest.json').read_text(encoding='utf-8'))
    for name, expected in b04['prior_bundles'].items():
        if sha(ROOT/name) != expected: raise ValueError('Changed original bundle: '+name)
    b04_receipt = json.loads((ROOT/'distribution/rust-lab-b04/bundle.json').read_text(encoding='utf-8'))
    if sha(ROOT/b04_receipt['path']) != b04_receipt['sha256']: raise ValueError('Changed B04 bundle')
    if sha(ROOT/archives[0]) != '935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76':
        raise ValueError('Changed B01 immutable inputs')
    # Sidecar receipts are needed by B04's original, unchanged integrity checks.
    sidecars = ['distribution/rust-lab-'+s+'/'+n for s in ('b025','b03','b04') for n in ('bundle.json','transfer-manifest.json')]
    payload = {n:(ROOT/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    manifest = dict(contract='brass-lab-results-transfer/1.0.0', training_reference=reference,
        files={n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},
        archives={n:sha(ROOT/n) for n in archives}, lab_qualification='Pending',
        results_directory='results/rust-lab-qualification', extraction_order=['B01','B02','B02.5','B03','B04'])
    for n in archives+sidecars: payload[n]=(ROOT/n).read_bytes()
    write(out/'transfer-manifest.json', manifest)
    payload['distribution/rust-lab-g1326/transfer-manifest.json']=(out/'transfer-manifest.json').read_bytes()
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, content in sorted(payload.items()): z.writestr(name, content)
    receipt = dict(path=archive.relative_to(ROOT).as_posix(), bytes=archive.stat().st_size,
        sha256=sha(archive), entries=len(payload), lab_qualification='Pending')
    write(out/'bundle.json', receipt); print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
