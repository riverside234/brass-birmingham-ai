#!/usr/bin/env python3
"""Seal the B03 source supplement under distribution/ without replacing archives."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def fixture_dependencies(start):
    """Include the shared authored fixtures imported by the qualification suite."""
    search = [ROOT / n for n in ('tests/R2', 'tests/Recovery', 'tests/FeatureEngineering', 'tests/RustBackendLab')]
    pending = [ROOT / start]; result = set()
    while pending:
        path = pending.pop()
        if path in result: continue
        result.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for name in modules:
                if not name or not (name.startswith('test_') or name == 'fixtures'): continue
                candidates = [d / (name + '.py') for d in search]
                target = next((p for p in candidates if p.is_file()), None)
                if target is None: raise ValueError('Missing shared fixture: ' + name)
                pending.append(target)
    return {p.relative_to(ROOT).as_posix() for p in result}


def main():
    from brass_ai.bank import sha
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id
    from brass_ai.rust_backend.lab_contract import SOURCE
    names = {
        'src/brass_ai/rust_backend/lab_admission.py', 'src/brass_ai/rust_backend/lab_continuation.py',
        'src/brass_ai/rust_backend/lab_b03_smoke.py', 'src/brass_ai/rust_backend/lab_families.py',
        'src/brass_ai/rust_backend/lab_b02_smoke.py', 'src/brass_ai/rust_backend/lab_snapshot.py',
        'src/brass_ai/recovery/replay.py', 'src/brass_ai/recovery/ingress.py',
        'src/brass_ai/recovery/local_scoring_targets.py', 'src/brass_ai/recovery/value_buffer.py',
        'src/brass_ai/recovery/resume.py', 'src/brass_ai/recovery/league.py',
        'src/brass_ai/recovery/teacher_phase.py', 'src/brass_ai/step5/league.py',
        'configs/learning/selfplay-resume-sources.json',
        'scripts/qualify-rust-lab-b02.py', 'scripts/qualify-rust-lab-b025.py', 'scripts/qualify-rust-lab-b03.py',
        'scripts/run-rust-lab-b03.sh', 'scripts/restore-rust-lab-b03.py', 'scripts/prepare-rust-lab-b03.py',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md', 'ai-research/RUST_BACKEND_B03_REPORT.md',
        'distribution/rust-lab-b03/README.md',
        'results/joint7-v11-2-grouped-lr-critic-g1320-20261027T075134827445Z/recorded-config.json',
    } | fixture_dependencies('tests/RustBackendLab/test_b03.py')
    files = {name: ((ROOT / name).read_bytes() if name.endswith('.json') else
                   (ROOT / name).read_bytes().replace(b'\r\n', b'\n')) for name in names}
    implementation = implementation_id()
    checks = json.loads((ROOT / 'artifacts/rust-backend/b03-development/qualification.json').read_text(encoding='utf-8'))
    require(checks.get('passed') is True and checks['implementation'] == implementation,
            'Run the bounded development checks on this exact source before packaging')
    prior = ('distribution/rust-lab-b01/b01-search-v1-update.zip',
             'distribution/rust-lab-b02/b02-search-v1-update.zip',
             'distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip')
    manifest = dict(contract='brass-lab-b03-transfer/1.0.0', implementation=implementation,
        files={n: hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest() for n, b in sorted(files.items())},
        shared_core_source=SOURCE, method='search_v1', v205_enabled=False,
        b01_qualification='Pending', b02_qualification='Pending', b025_lab_qualification='Pending',
        b03_lab_qualification='Pending', local_checks=checks, prior_bundles={n: sha(ROOT / n) for n in prior},
        required_checkout='Updated project checkout/private training runtime; extract B01, B02, B02.5, B03 in order before qualification',
        scope='Native replay admission and reviewed joint7 full-state restoration. No new games, optimizer steps, teacher enablement or production adoption.')
    out = ROOT / 'distribution/rust-lab-b03'; out.mkdir(parents=True, exist_ok=True)
    path = out / 'b03-search-v1-admission.zip'
    encoded = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    require(not path.exists() and not (out / 'transfer-manifest.json').exists(),
            'B03 transfer is already sealed; publish a new version')
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files): z.writestr(name, files[name])
        z.writestr('distribution/rust-lab-b03/transfer-manifest.json', encoded)
    (out / 'transfer-manifest.json').write_bytes(encoded)
    receipt = dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                   sha256=sha(path), implementation=implementation)
    (out / 'bundle.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
