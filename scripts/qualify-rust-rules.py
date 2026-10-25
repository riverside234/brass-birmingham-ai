"""Qualify L05 staged rules with frozen C# fixtures and the offline native toolchain."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
read, write, receipt, sha = (BASE[k] for k in ('read', 'write', 'receipt', 'sha'))


def verify_capture(folder):
    capture = read(folder / 'capture.json')
    for name, digest in capture['files'].items():
        assert sha(folder / name) == digest, f'Capture changed: {name}'
    return capture


def prepare_reference(out):
    """Materialize only admitted sources; add the separately identified native client."""
    baseline = read(BASE['FROZEN'] / 'baseline.json')
    archived = baseline['reference_source']['archive']
    assert sha(ROOT / archived['path']) == archived['sha256']
    reference = out / 'reference-source'
    reference.mkdir()
    with zipfile.ZipFile(ROOT / archived['path']) as archive:
        for item in baseline['reference_source']['files']:
            path = (reference / item['path']).resolve()
            assert path.is_relative_to(reference)
            data = archive.read(item['path'])
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    # The L02 comparison harness was added after the L01 source archive. Its
    # independent qualification receipt pins the additional reference files.
    harness_receipt = BASE['FROZEN'] / 'local/fixtures/qualification.json'
    harness = []
    for name, digest in read(harness_receipt)['harness'].items():
        source = ROOT / name
        target = (reference / name).resolve()
        assert target.is_relative_to(reference) and not target.exists()
        assert sha(source) == digest, 'L02 comparison harness changed: ' + name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        harness.append(dict(source=receipt(source), executed=receipt(target)))
    overlay = []
    for source in sorted((ROOT / 'src/brass_ai/native').glob('*.py'), key=str):
        target = reference / source.relative_to(ROOT)
        assert not target.exists(), 'Candidate cannot overwrite a frozen reference module'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        overlay.append(dict(candidate=receipt(source), executed=receipt(target)))
    write(out / 'python-reference.json', dict(reference_root=reference.relative_to(ROOT).as_posix(),
          archive=archived, source_digest=baseline['reference_source']['digest'],
          reference_files=len(baseline['reference_source']['files']), candidate_overlay=overlay,
          harness_receipt=receipt(harness_receipt), harness_overlay=harness))
    return reference


def preserved_identity(value):
    # Checkout drift is recorded, not executed. The reference archive, executed
    # source copies, model, installed game, saves and upstream inputs stay pinned.
    return {k: v for k, v in value.items() if k != 'provenance_drift'}


def qualify(out, capture, branches):
    assert out.is_relative_to(ROOT / 'artifacts/rust-backend')
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'qualification.json').exists(), 'Completed evidence is immutable'
    write(out / 'progress.json', dict(status='running', milestone='L05'))
    source_identity = BASE['native_fingerprint']()
    reference = prepare_reference(out)
    before = BASE['verify_frozen'](reference_source_root=reference)
    write(out / 'preservation-before.json', before)
    prior_paths = [ROOT / 'artifacts/rust-backend' / folder / name
                   for folder in ('20261024-l03-verified', '20261025-l04-qualified')
                   for name in ('qualification.json', 'final-verification.json', 'implementation.zip', 'executables.zip')]
    prior = [receipt(p) for p in prior_paths]
    frozen, alternatives = verify_capture(capture), verify_capture(branches)
    assert alternatives['roots_sha256'] == frozen['files']['roots.jsonl.gz']
    assert alternatives['exporter_sha256'] == frozen['exporter_sha256']
    assert preserved_identity(frozen['preservation']) == preserved_identity(before)
    write(out / 'capture-receipts.json', dict(games=frozen, alternatives=alternatives))

    with gzip.open(capture / 'roots.jsonl.gz', 'rt', encoding='utf-8') as stream:
        roots = [json.loads(line) for line in stream]
    candidates = [r for r in roots if r['partition'] == 'development'
                  and r['id'].startswith('game-')
                  and r['expected']['state']['context']['phase'] == 'payment'
                  and r['expected']['state']['context']['era'] == 'RailEra']
    complex_row = max(candidates, key=lambda r: (sum('tile' in s for s in r['expected']['state']['slots']),
                                                len(r['expected']['state']['history'])))
    initial = next(r for r in roots if r['id'] == 'game-731001-root-000')
    bench = out / 'bench-inputs'
    bench.mkdir(exist_ok=True)
    for name, row in [('initial', initial), ('complex', complex_row)]:
        write(bench / (name + '.json'), row['expected']['state'])
    write(bench / 'selection.json', dict(initial_root=initial['id'], complex_root=complex_row['id'],
          criterion='Development Rail payment; most occupied slots, then longest history',
          held_out_used=False, inputs=[receipt(bench / (n + '.json')) for n in ('initial', 'complex')]))

    run = BASE['Qualification'](out)
    run.env.pop('PROPTEST_RNG_SEED', None)
    run.env['CARGO_INCREMENTAL'] = '0'
    run.env.update(BRASS_RULES_CAPTURE=str(capture), BRASS_RULES_BRANCHES=str(branches),
                   BRASS_STATE_BENCH_DIR=str(bench))
    run.env['PYTHONPATH'] = os.pathsep.join([str(reference / 'src'),
                           *[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p) != ROOT / 'src']])
    check_imports = '''import hashlib,json,sys,unittest
from pathlib import Path
from brass_ai.learning import reward
from brass_ai.recovery import config
from brass_ai.searchlab import spiel
from brass_ai.native import protocol
from brass_ai.rust_backend import contracts
root=Path(sys.argv[1]).resolve(); result={}
for module in (reward,config,spiel,protocol,contracts):
    path=Path(module.__file__).resolve(); assert path.is_relative_to(root), str(path)
    result[module.__name__]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
loader=unittest.TestLoader(); loader.discover(sys.argv[2]); assert not loader.errors, loader.errors
print(json.dumps(result,indent=2))
'''
    run.run('reference-python-identities', [sys.executable, '-c', check_imports, reference, ROOT / 'tests/RustBackendNative'])
    run.cargo_run('fmt', 'fmt', '--all', '--', '--check')
    run.cargo_run('clippy', 'clippy', '--workspace', '--all-targets', '--locked', '--offline', '--', '-D', 'warnings')
    for profile, flags in [('debug', []), ('release', ['--release'])]:
        run.cargo_run(profile + '-tests', 'test', '--workspace', *flags, '--locked', '--offline')
    from brass_ai.native import NativeWorker
    builds = {}
    for profile, flags in [('debug', []), ('release', ['--release']), ('profiling', ['--profile', 'profiling'])]:
        run.cargo_run(profile + '-build', 'build', '-p', 'brass-ai-worker', *flags, '--locked', '--offline')
        binary = ROOT / 'artifacts/rust-native-target' / BASE['TARGET'] / profile / 'brass-ai-worker.exe'
        with NativeWorker(binary) as worker:
            hello = worker.hello
            assert hello['build']['source_sha256'] == source_identity == BASE['native_fingerprint']()
            assert hello['build']['cargo_lock_sha256'] == sha(ROOT / 'native/Cargo.lock')
            assert hello['target_rules_qualified'] is False
        builds[profile] = dict(binary=receipt(binary), handshake=hello, pe_machine=BASE['machine_type'](binary))
        env = dict(run.env, BRASS_NATIVE_WORKER=str(binary),
                   BRASS_STATE_PARITY_REPORT=str(out / (profile + '-state-parity.json')),
                   BRASS_RULES_PARITY_REPORT=str(out / (profile + '-rules-parity.json')),
                   BRASS_RULES_FAILURE=str(out / (profile + '-first-difference.json')))
        run.run(profile + '-python-tests', [sys.executable, '-m', 'unittest', 'discover', '-s',
                ROOT / 'tests/RustBackendNative', '-v'], env=env, timeout=1200)
    run.cargo_run('rules-criterion', 'bench', '-p', 'brass-ai-core', '--bench', 'rules', '--locked', '--offline')
    shutil.copytree(ROOT / 'artifacts/rust-native-target/criterion/staged_rules', out / 'criterion', dirs_exist_ok=True)

    after = BASE['verify_frozen'](reference_source_root=reference)
    write(out / 'preservation-after.json', after)
    assert preserved_identity(before) == preserved_identity(after)
    for item in read(out / 'python-reference.json')['candidate_overlay']:
        assert receipt(ROOT / item['candidate']['path']) == item['candidate']
        assert receipt(ROOT / item['executed']['path']) == item['executed']
    for item in read(out / 'python-reference.json')['harness_overlay']:
        assert receipt(ROOT / item['source']['path']) == item['source']
        assert receipt(ROOT / item['executed']['path']) == item['executed']
    assert source_identity == BASE['native_fingerprint'](), 'Native sources changed during qualification'
    assert prior == [receipt(p) for p in prior_paths]
    assert frozen == verify_capture(capture) and alternatives == verify_capture(branches)
    paths = list((ROOT / 'native').rglob('*')) + list((ROOT / 'src/brass_ai/native').glob('*.py'))
    paths += list((ROOT / 'tests/RustBackendNative').rglob('*'))
    paths += list((ROOT / 'configs/contracts/rust-backend').glob('*.md'))
    paths += [ROOT / 'scripts' / name for name in (
        'capture-rust-states.py', 'complete-rust-state-capture.py', 'capture-rust-state-regressions.py',
        'prepare-rust-state-catalog.py', 'qualify-rust-state.py', 'qualify-rust-scaffold.py',
        'capture-rust-rules.py', 'capture-rust-rule-branches.py', 'qualify-rust-rules.py')]
    paths = sorted({p for p in paths if p.is_file() and
                    (p.suffix in ('.rs', '.toml', '.lock', '.json', '.py', '.cs', '.md', '.patch') or p.name == 'LICENSE')}, key=str)
    with zipfile.ZipFile(out / 'implementation.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out / 'executables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for profile, build in builds.items():
            archive.write(ROOT / build['binary']['path'], profile + '/brass-ai-worker.exe')
    result = dict(format='brass-native-l05-qualification/1.0.0', status='passed', milestone='L05',
                  completed_utc=datetime.now(timezone.utc).isoformat(), builds=builds, commands=run.commands,
                  sources=[receipt(p) for p in paths], prior_evidence=prior,
                  python_reference=receipt(out / 'python-reference.json'),
                  captures=[receipt(p / 'capture.json') for p in (capture, branches)],
                  archives=[receipt(out / n) for n in ('implementation.zip', 'executables.zip')],
                  sessions=2, threads_per_session=1, roots=frozen['roots'], games=len(frozen['games']),
                  game_transitions=sum(g['steps'] for g in frozen['games']), branch_transitions=alternatives['transitions'],
                  reward_games=20, unexplained_differences=0, staged_rules_qualified=True,
                  target_rules_qualified=False, search_available=False, installed_backend_changed=False,
                  nn_executed=False, matched_search_run=False, matched_search_seconds=4,
                  next_milestone='L06: permitted observations, recall and world sampling')
    write(out / 'qualification.json', result)
    write(out / 'progress.json', dict(status='passed', milestone='L05', next='L06'))
    print('L05 qualification passed:', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--capture', type=Path, default=ROOT / 'artifacts/rust-backend/20261025-l05-capture-v2')
    parser.add_argument('--branches', type=Path, default=ROOT / 'artifacts/rust-backend/20261025-l05-branches')
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        qualify(out, args.capture.resolve(), args.branches.resolve())
    except BaseException as error:
        if out.is_relative_to(ROOT / 'artifacts/rust-backend') and not (out / 'qualification.json').exists():
            write(out / 'progress.json', dict(status='failed', milestone='L05', error=str(error)))
        raise


if __name__ == '__main__':
    main()
