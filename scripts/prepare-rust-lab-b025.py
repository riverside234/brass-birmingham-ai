#!/usr/bin/env python3
"""Seal a new B02.5 source-only transfer under distribution/; never overwrite it."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from brass_ai.step5.league import implementation_id
    from brass_ai.learning.contracts import require
    from brass_ai.bank import sha
    from brass_ai.recovery.teacher_phase_config import SOURCE, CONTRACT
    names = [
        'src/brass_ai/recovery/teacher_phase_config.py', 'src/brass_ai/recovery/teacher_labels.py',
        'src/brass_ai/recovery/teacher_buffer.py', 'src/brass_ai/recovery/teacher_term.py',
        'src/brass_ai/recovery/teacher_phase.py', 'src/brass_ai/recovery/config.py',
        'src/brass_ai/recovery/resume.py', 'src/brass_ai/recovery/updates.py',
        'src/brass_ai/recovery/league.py', 'src/brass_ai/recovery/history.py',
        'src/brass_ai/step5/league.py', 'src/brass_ai/rust_backend/lab_contract.py',
        'configs/learning/selfplay-resume-sources.json', 'scripts/configure-teacher-phase.py',
        'scripts/prepare-rust-lab-b025.py', 'scripts/qualify-rust-lab-b025.py', 'scripts/run-rust-lab-b025.sh',
        'tests/RustBackendLab/test_b025.py', 'tests/R2/fixtures.py', 'tests/Recovery/test_teacher_value.py',
        'scripts/reproduce-gumbel-is-mcts.py', 'tests/GumbelISMCTS/fixtures/reference-v1.json',
        'results/joint7-v11-2-grouped-lr-critic-g1320-20261027T075134827445Z/recorded-config.json',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md', 'ai-research/TEACHER_SEARCH_PHASE_REVISION_PLAN.md',
        'ai-research/RUST_BACKEND_B025_REPORT.md', 'distribution/rust-lab-b025/README.md']
    # Config/receipt JSON retains its repository bytes; source/shell text uses LF.
    files = {name: ((ROOT / name).read_bytes() if name.endswith('.json') else
                   (ROOT / name).read_bytes().replace(b'\r\n', b'\n')) for name in names}
    implementation = implementation_id()
    evidence = ROOT / 'artifacts/rust-backend/b025-development/qualification.json'
    checks = json.loads(evidence.read_text(encoding='utf-8'))
    require(checks.get('passed') is True and checks['implementation'] == implementation,
            'Run final local fixture checks for this exact implementation before packaging')
    manifest = dict(contract='brass-lab-b025-transfer/1.0.0', teacher_contract=CONTRACT,
        implementation=implementation, files={n: hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest() for n, b in files.items()},
        shared_core_source=SOURCE, method='search_v1', v205_enabled=False, default_mode='off',
        local_checks=checks, b01_qualification='Pending', b02_qualification='Pending', b025_lab_qualification='Pending',
        required_checkout='Current g1324 source checkout plus the search_v1 B01/B02 updates; see README',
        prior_bundles={name: sha(ROOT / name) for name in (
            'distribution/rust-lab-b01/b01-search-v1-update.zip',
            'distribution/rust-lab-b02/b02-search-v1-update.zip')},
        scope='T1 Python fixtures/config/admission/buffer/PPO/resume only. Live teacher routing remains B04. No model weights or game binaries.')
    out = ROOT / 'distribution/rust-lab-b025'; out.mkdir(parents=True, exist_ok=True)
    path = out / 'b025-search-v1-teacher-phase.zip'
    encoded = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    # All checks precede publication. Existing sealed transfers are immutable.
    require(not path.exists() and not (out / 'transfer-manifest.json').exists(), 'B02.5 transfer already sealed; use a new version')
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files): z.writestr(name, files[name])
        z.writestr('distribution/rust-lab-b025/transfer-manifest.json', encoded)
    (out / 'transfer-manifest.json').write_bytes(encoded)
    receipt = dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path), implementation=implementation)
    (out / 'bundle.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
