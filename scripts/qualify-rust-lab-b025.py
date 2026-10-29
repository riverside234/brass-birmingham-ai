#!/usr/bin/env python3
"""Short B02.5 fixture qualification and optional read-only joint7 adoption check."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEST_PATHS = ['src', 'tests/R2', 'tests/Recovery', 'tests/FeatureEngineering', 'tests/RustBackendLab']
for name in reversed(TEST_PATHS): sys.path.insert(0, str(ROOT / name))


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def verify_transfer(supplement=None):
    if supplement is not None and (ROOT/supplement).resolve() == ROOT/'distribution/rust-lab-b04/transfer-manifest.json':
        from brass_ai.rust_backend.lab_transfer import verify
        return verify('b025',supplement)
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id
    path = ROOT / 'distribution/rust-lab-b025/transfer-manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    require(manifest['contract'] == 'brass-lab-b025-transfer/1.0.0', 'Unknown B02.5 transfer')
    expected_files = dict(manifest['files']); implementation = manifest['implementation']
    if supplement is not None:
        supplement = (ROOT / supplement).resolve()
        require(supplement == ROOT / 'distribution/rust-lab-b03/transfer-manifest.json', 'Unknown B02.5 supplement')
        newer = json.loads(supplement.read_text(encoding='utf-8'))
        require(newer['contract'] == 'brass-lab-b03-transfer/1.0.0', 'Unknown B03 transfer')
        archive_name = 'distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip'
        archive = ROOT / archive_name
        require(hashlib.sha256(archive.read_bytes()).hexdigest() == newer['prior_bundles'][archive_name],
                'B03 supplement has a different B02.5 prerequisite')
        with zipfile.ZipFile(archive) as z:
            require(json.loads(z.read(path.relative_to(ROOT).as_posix())) == manifest,
                    'Original B02.5 manifest differs from its sealed archive')
        expected_files.update(newer['files']); implementation = newer['implementation']
    for name, expected in expected_files.items():
        p = (ROOT / name).resolve()
        require(p.is_relative_to(ROOT) and p.is_file() and source_sha(p) == expected, 'B02.5 file changed: ' + name)
    require(implementation_id() == implementation,
            'Lab checkout source differs from reviewed B02.5; update the checkout, do not rewrite the review hash')
    return dict(manifest_sha256=source_sha(path), implementation=implementation, files=len(expected_files),
                supplement_sha256=source_sha(supplement) if supplement else None)


def inspect_run(run, generation):
    """Load full model/Adam on CPU, but never commit, advance or alter a run."""
    import torch
    from brass_ai.bank import read, sha
    from brass_ai.learning.contracts import require
    from brass_ai.learning.checkpoint import resume_optimizer
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.retention import owned, safe, run_lock, _same_payload
    from brass_ai.recovery.resume import load_resume
    from brass_ai.recovery.teacher_phase_config import adopt
    from brass_ai.recovery.teacher_phase import TeacherPhase
    from brass_ai.step5.league import implementation_id
    _, run = owned(ROOT, run)
    with run_lock(run):
        path = safe(run, run / 'state.json'); before = sha(path); state = read(path)
        require(state['iteration'] == generation, 'Expected generation differs; pause at the committed boundary')
        config = validate_config(adopt(state['signature']['config']))
        require(config['teacher_phase']['mode'] == 'off', 'B02.5 qualification only adopts/checks off; live teachers belong to B04')
        signature = dict(state['signature'], config=config, implementation=implementation_id())
        model, raw, changed = load_resume(ROOT, run, state, signature, device='cpu', reviewed=True)
        optimizer = torch.optim.AdamW(model.parameters())
        resume_optimizer(raw, optimizer, restore_rng=False)
        require(_same_payload(optimizer.state_dict(), raw['optimizer']), 'Adam restore differs')
        TeacherPhase(ROOT, run, config, state, model, raw)
        entry = state['history'][-1]
        require(sha(path) == before and sha(ROOT / entry['path']) == entry['sha256'], 'Run changed during qualification')
        return dict(generation=generation, state_sha256=before, checkpoint=entry, source_transition=changed,
            proposed_mode='off', optimizer_restored=True, run_modified=False,
            scope='Read-only full model/Adam adoption check; PPG/auxiliary optimizer continuation remains B05')


def device_smoke(device):
    import torch
    from brass_ai.learning.contracts import require
    from brass_ai.recovery.value_preservation import TrunkGradients
    from brass_ai.recovery.teacher_term import TeacherTerm
    import test_b025 as fixture
    require(device.startswith('cuda') and torch.cuda.is_available(), 'Requested CUDA device unavailable')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = fixture.model_fixture().to(device)
    rows = [(0, fixture.example(model, seat=s)[0]) for s in range(4)]
    c = fixture.config()['teacher_phase']; c['mode'] = 'on'
    term = TeacherTerm(model, c, rows, seed=1, generation=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    term.apply(model, TrunkGradients(model), 1)
    require(all(p.grad is None for module in (model.outcomes, model.utility_head, model.descriptor_encoder)
                for p in module.parameters()), 'Teacher touched non-policy heads')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
    torch.cuda.synchronize()
    return dict(device=device, rows=4, optimizer_steps=1, loss=term.steps[0]['loss'], passed=True,
                scope='One authored teacher-policy CUDA step; no native games or real training data')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--run', type=Path)
    p.add_argument('--expected-generation', type=int)
    p.add_argument('--device', default='cpu', help='CPU exact fixtures always run; cuda:0 adds one authored device step')
    p.add_argument('--device-child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--transfer-manifest', type=Path, help='Explicit sealed B03 or B04 supplement; verifies original prerequisite archives')
    a = p.parse_args()
    if a.device_child:
        print(json.dumps(device_smoke(a.device), indent=2)); return
    from brass_ai.learning.contracts import project_path, require
    from brass_ai.step5.league import write_json
    require((a.run is None) == (a.expected_generation is None), 'Supply both --run and --expected-generation')
    out = project_path(ROOT, a.output or ('artifacts/rust-backend/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-b025'))
    require(out.is_relative_to(ROOT / 'artifacts') and not out.exists(), 'Use a new evidence directory under artifacts/')
    if a.run:
        run = project_path(ROOT, a.run)
        require(not out.is_relative_to(run) and not run.is_relative_to(out), 'Evidence must be separate from the training run')
    out.mkdir(parents=True)
    report = dict(contract='brass-lab-b025/1.0.0', milestone='B02.5', passed=False, platform=platform.system(),
        method='search_v1', v205_enabled=False, native_teacher_qualification='Pending (B04)',
        b01_qualification='Pending / not evaluated', b02_qualification='Pending / not evaluated',
        scope='T1 Python fixtures only; no full games, no iteration advance, no live teacher enablement')
    began = time.monotonic()
    try:
        report['transfer'] = verify_transfer(a.transfer_manifest)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / path) for path in TEST_PATHS),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command = [sys.executable, '-m', 'unittest', 'test_b025', '-v']
        print('B02.5: short CPU fixture suite', flush=True)
        with (out / 'fixtures.log').open('wb') as log:
            result = subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                stdout=log, stderr=subprocess.STDOUT, timeout=180)
        require(result.returncode == 0, 'B02.5 fixtures failed; see ' + str(out / 'fixtures.log'))
        report['fixtures'] = dict(passed=True, log='fixtures.log', sha256=source_sha(out / 'fixtures.log'), timeout_seconds=180)
        if a.device != 'cpu':
            print('B02.5: one CUDA fixture step', flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__)), '--device-child', '--device', a.device],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
            (out / 'device.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            require(result.returncode == 0, 'CUDA fixture failed; see device.log')
            report['device'] = json.loads(result.stdout)
        if a.run:
            print('B02.5: read-only committed model/Adam adoption check', flush=True)
            report['run'] = inspect_run(run, a.expected_generation)
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['seconds'] = time.monotonic() - began
        write_json(out / 'qualification.json', report)
        print(str(out / 'qualification.json'), flush=True)


if __name__ == '__main__': main()
