#!/usr/bin/env python3
"""Seal the additive B04 source update; never overwrite a published bundle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Verify layered source coverage without publishing')
    args=parser.parse_args()
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id
    from brass_ai.rust_backend.lab_contract import sha,SOURCE
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    spec=importlib.util.spec_from_file_location('b03_package',ROOT/'scripts/prepare-rust-lab-b03.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    names={
        'src/brass_ai/rust_backend/lab_teacher.py','src/brass_ai/rust_backend/lab_teacher_overlay.py',
        'src/brass_ai/rust_backend/lab_teacher_inference.py','src/brass_ai/rust_backend/lab_transfer.py',
        'src/brass_ai/rust_backend/lab_b04_smoke.py','src/brass_ai/rust_backend/lab_admission.py',
        'src/brass_ai/recovery/inference.py','src/brass_ai/recovery/shared_transport.py',
        'src/brass_ai/recovery/teacher_phase.py','src/brass_ai/recovery/teacher_phase_config.py',
        'src/brass_ai/recovery/teacher_labels.py','src/brass_ai/recovery/league.py',
        'configs/learning/selfplay-resume-sources.json',
        'scripts/qualify-rust-lab-b02.py','scripts/qualify-rust-lab-b025.py','scripts/qualify-rust-lab-b03.py',
        'scripts/qualify-rust-lab-b04.py','scripts/run-rust-lab-b04.sh','scripts/prepare-rust-lab-b04.py',
        'tests/GumbelISMCTS/fixtures/session-v1.json',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/TEACHER_SEARCH_PHASE_REVISION_PLAN.md',
        'ai-research/RUST_BACKEND_B04_REPORT.md','distribution/rust-lab-b04/README.md',
    }|helper.fixture_dependencies('tests/RustBackendLab/test_b04.py')
    names|={p.relative_to(ROOT).as_posix() for p in (ROOT/'lab-native').rglob('*') if p.is_file()
            and p.suffix in ('.rs','.toml','.lock') and 'target' not in p.parts}
    files={name:(ROOT/name).read_bytes().replace(b'\r\n',b'\n') for name in names}
    # Detect an omitted replacement before the immutable archive is created.
    # Read old manifests from the sealed ZIPs, never trust an edited sidecar.
    layered={}
    for milestone,archive in (('b025','b025-search-v1-teacher-phase.zip'),('b03','b03-search-v1-admission.zip')):
        prefix='distribution/rust-lab-'+milestone+'/'
        with zipfile.ZipFile(ROOT/prefix/archive) as z:
            original=json.loads(z.read(prefix+'transfer-manifest.json'))
        layered.update(original['files'])
    layered.update({n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    for name,expected in layered.items():
        require(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,
                'B04 is missing a changed prerequisite file: '+name)
    if args.check_only:
        print(json.dumps(dict(passed=True,files=len(files),layered_files=len(layered)),indent=2))
        return
    implementation=implementation_id();checks=json.loads((ROOT/'artifacts/rust-backend/b04-development/qualification.json').read_text())
    require(checks['passed'] is True and checks['implementation']==implementation and
            checks['producer']==source_fingerprint(),'Complete short development checks on this exact source first')
    for name,expected in checks['logs'].items():
        require(sha(ROOT/'artifacts/rust-backend/b04-development'/name)==expected,'Changed development evidence: '+name)
    native=json.loads((ROOT/'artifacts/rust-backend/b04-development/source-verification.json').read_text())
    require(native['passed'] is True and native['producer']==source_fingerprint()
            and native['core_source']==checks['core_source'] and sha(native['binary']['path'])==native['binary']['sha256'],
            'Native build source/binary verification differs')
    prior=('distribution/rust-lab-b01/b01-search-v1-update.zip','distribution/rust-lab-b02/b02-search-v1-update.zip',
           'distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip','distribution/rust-lab-b03/b03-search-v1-admission.zip')
    manifest=dict(contract='brass-lab-b04-transfer/1.0.0',implementation=implementation,
        files={n:hashlib.sha256(b).hexdigest() for n,b in sorted(files.items())},base_core_source=SOURCE,
        teacher_producer=source_fingerprint(),method='search_v1',v205_enabled=False,b04_lab_qualification='Pending',
        prior_bundles={n:sha(ROOT/n) for n in prior},local_checks=checks,
        extraction_order=['B01','B02','B02.5','B03','B04'],scope='Opt-in native teacher exports/collector and two-table lab shadow qualification; no production training or game deployment')
    out=ROOT/'distribution/rust-lab-b04';out.mkdir(exist_ok=True)
    archive=out/'b04-search-v1-teacher.zip'
    require(not archive.exists() and not (out/'transfer-manifest.json').exists(),'B04 is already sealed; publish a new version')
    encoded=(json.dumps(manifest,indent=2)+'\n').encode()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,b in sorted(files.items()):z.writestr(n,b)
        z.writestr('distribution/rust-lab-b04/transfer-manifest.json',encoded)
    (out/'transfer-manifest.json').write_bytes(encoded)
    receipt=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
                 implementation=implementation,producer=source_fingerprint())
    (out/'bundle.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
