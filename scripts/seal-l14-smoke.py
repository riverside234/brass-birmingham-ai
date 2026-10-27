"""Pin completed short L14 evidence and prove that installed files are intact."""
import ast
import json
from pathlib import Path
import runpy
import zipfile

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'artifacts/rust-backend/20261026-l14-smoke'
OLD=ROOT/'artifacts/rust-backend/20261026-l14-study-v2'
BUILD=ROOT/'artifacts/rust-backend/20261026-l14-candidate-v5'
L13=ROOT/'artifacts/rust-backend/20261026-l13-qualified'
BASE=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))


def main():
    assert not (S/'final-checks.json').exists()
    installed=ROOT/'artifacts/patch-installation.json';pin=read(OLD/'freeze.json')['installed_receipt']
    assert sha(installed)==pin['sha256']
    for entry in read(installed)['files']:
        assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/entry['name'])==entry['sha256'].lower()
    q=read(L13/'qualification.json')
    for entry in q['files']:
        p=(L13/entry['path']).resolve();assert p.is_relative_to(L13) and sha(p)==entry['sha256'].lower()
    for entry in read(L13/'BrassNative/alpha-g890/build.json')['files']:
        assert sha(ROOT/'BrassNative/alpha-g890'/entry['path'])==entry['sha256']
    build=read(BUILD/'build-manifest.json')
    for entry in build['sourcePins']:assert sha(BUILD/'runtime'/entry['path'])==entry['sha256']
    for entry in build['files']:assert sha(BUILD/'Managed'/entry['name'])==entry['sha256']
    assert build['native_source_sha256']==BASE['native_fingerprint']()
    for entry in read(BUILD/'BrassNative/alpha-g890/build.json')['files']:
        assert sha(BUILD/'BrassNative/alpha-g890'/entry['path'])==entry['sha256']
    checks=[BUILD/'unity.json',BUILD/'preserved-runtime.json',OLD/'smoke-v5-contract/validation-events.json',OLD/'smoke-v5-contract/checkpoint-tests.json']
    for p in checks:assert read(p)['passed']
    assert not read(BUILD/'unity.json')['nativeErrors']
    for label in ('external-ort-count','external-torch-count-02','external-ort-timed','external-torch-timed'):
        assert sha(S/label/'external_evaluator.executed.py')==read(S/label/'results.json')['external_source']['sha256']
    assert sha(S/'gpy-count-02/reference.executed.py')==read(S/'gpy-count-02/results.json')['source']['sha256']
    for label in ('gpy-whole-count','gpy-whole-timed-torch'):
        assert sha(S/label/'reference.py')==read(S/label/'results.json')['source']['sha256']
    sources=['scripts/l14-smoke.py','scripts/report-l14-smoke.py','scripts/seal-l14-smoke.py','scripts/benchmark-rust-backend.py',
        'tests/RustBackend/external_evaluator.py','tests/RustBackend/gumbel_session_reference.py',
        'tests/RustBackendLocal/NativeStageTrace.cs','tests/RustBackendLocal/MonoAllocationProbe.c','tests/RustBackendLocal/run-unity.ps1',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/RUST_BACKEND_L14_REPORT.md']
    for name in sources:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf-8'))
    with zipfile.ZipFile(S/'followup-sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    failures=[dict(path=name,reason=reason) for name,reason in [
        ('trace-equivalence-01','Initial smoke runner pointed at qualification outputs rather than captured input directory; corrected runner.'),
        ('trace-equivalence-02','Combined all-case JSON exceeded the existing 256 MiB diagnostic bound; selected eight coverage cases instead, bound unchanged.'),
        ('external-torch-count','Reference model returns a tuple; diagnostic adapter corrected to the five named heads.'),
        ('quality-smoke-01','Training-only balanced profile rejected from evaluation pool; selected the already admitted pottery-flex evaluation profile.'),
        ('native-service-trace.jsonl','Unity allocation callback self-test unavailable; service/IPC timestamps succeeded. Allocation totals remain open.')]]
    result=dict(format='brass-l14-smoke-seal/1.0.0',passed=True,installed=False,full_l14_qualified=False,
        candidate=receipt(BUILD/'build-manifest.json'),assessment=receipt(S/'assessment.json'),
        checks=[receipt(p) for p in checks],installed_receipt=receipt(installed),
        sealed_l13_files_verified=len(q['files']),installed_managed_files_verified=len(read(installed)['files']),
        installed_native_files_verified=len(read(L13/'BrassNative/alpha-g890/build.json')['files']),
        historical_protocol_unchanged=sha(OLD/'protocol.json')==read(OLD/'freeze.json')['protocol']['sha256'],
        failures_preserved=failures,source_archive=receipt(S/'followup-sources.zip'),sources=[receipt(ROOT/name) for name in sources])
    assert result['historical_protocol_unchanged']
    write(S/'final-checks.json',result)
    write(S/'progress.json',dict(status='bounded_followup_complete',full_l14_qualification='open',retained=True,installed=False,
        final_checks=receipt(S/'final-checks.json'),open_gates=read(S/'assessment.json')['open_gates']))
    print('Short follow-up sealed; 529 L13 files, installed package and historical protocol unchanged.')


if __name__=='__main__':main()
