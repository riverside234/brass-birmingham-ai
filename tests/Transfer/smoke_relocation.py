"""Native old-archive -> different root -> reviewed-source PPO continuation.

The source ref must match the bounded fixture's recorded implementation. Only
disposable copies are changed/deleted. No access to GitHub or a source machine.
"""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('transfer', ROOT / 'scripts/step5-transfer.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def git_snapshot(ref, destination):
    content = subprocess.check_output(['git', '-C', str(ROOT), 'archive', ref,
        *transfer.CODE_TREES, 'requirements-step5.txt', 'requirements-transfer.txt'])
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        for member in archive:
            if member.isdir():
                continue
            transfer.relative(member.name)
            assert member.isfile()
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--source-ref', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = transfer.below(ROOT, args.run)
    original = (source / 'state.json').read_bytes()
    state = json.loads(original)
    cfg = state['signature']['config']
    assert cfg['games_per_iteration'] <= 2 and cfg['search_every_games'] == 0
    assert not cfg['evaluation_schedule']['enabled'] and not state['signature'].get('imitation_source')
    run_name = source.relative_to(ROOT).as_posix()
    logs = ROOT / 'artifacts/step5-relocation-validation'
    logs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='reloc-native-', dir=ROOT / 'artifacts') as temp:
        area = Path(temp).resolve()
        old = area / 'old-project'
        destination = area / 'new-project'
        print('Prepare exact historical Git code and real stopped checkpoint/runtime', flush=True)
        git_snapshot(args.source_ref, old)
        for path in source.rglob('*'):
            if path.is_file():
                copy_file(path, old / run_name / path.relative_to(source))
        distribution = transfer.read(ROOT / 'training/runtime/distribution.json')
        runtime = set(distribution['private_runtime_files']) | {a['path'] for a in distribution['assets']}
        host = ROOT / cfg['host']
        runtime.add((host / 'runtime.json').relative_to(ROOT).as_posix())
        runtime.update((host / f['name']).relative_to(ROOT).as_posix() for f in transfer.read(host / 'runtime.json')['files'])
        for name in sorted(runtime):
            copy_file(ROOT / name, old / name)
        # Use the actual previous exporter, not a fabricated manifest fingerprint.
        old_spec = importlib.util.spec_from_file_location('historical_transfer', old / 'scripts/step5-transfer.py')
        previous = importlib.util.module_from_spec(old_spec)
        old_spec.loader.exec_module(previous)
        assert previous.source_identity(old) == state['signature']['implementation']
        bundle = area / 'github-assets/migration.tar.zst'
        exported = previous.export(old, run_name, bundle, dependencies='bundled', threads=2)
        manifest = transfer.inspect_bundle(exported['bundle'])
        hashes = {r['path']:r['sha256'] for r in manifest['files']}
        assert old.resolve().parent == area and area.is_relative_to(ROOT / 'artifacts')
        shutil.rmtree(old)
        assert not old.exists()
        print('Restore old archive into a different root, then verify original bytes/source', flush=True)
        restored = transfer.restore(destination, exported['bundle'], relocate=True)
        assert restored['requires_code_update']
        assert transfer.restore(destination, exported['bundle'], verify_only=True)['verified']
        assert all(transfer.sha(destination / name) == expected for name,expected in hashes.items())
        assert (destination / run_name / 'state.json').read_bytes() == original
        # This is the documented second phase: update code only after exact
        # restoration. Do not weaken archive/external dependency verification.
        print('Install current reviewed source, preserving restored checkpoint/state/config', flush=True)
        for prefix in transfer.CODE_TREES:
            for path in (ROOT / prefix).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    copy_file(path, destination / path.relative_to(ROOT))
        assert (destination / run_name / 'state.json').read_bytes() == original
        current = transfer.source_identity(destination)
        assert current == transfer.source_identity(ROOT)
        deps = [str(Path(p).resolve()) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                if p and Path(p).resolve() != ROOT / 'src']
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(destination / 'src'), *deps]), PYTHONDONTWRITEBYTECODE='1')
        base = [sys.executable, str(destination / 'scripts/step5.py'), '--root', str(destination),
                '--config', str(destination / manifest['config']), '--device', 'cpu']
        commands = [('preflight', base + ['preflight']),
                    ('resume', base + ['train', '--run', run_name, '--iterations', '1', '--resume-reviewed-source'])]
        for phase, command in commands:
            print('Relocated project: ' + phase, flush=True)
            result = subprocess.run(command, cwd=destination, env=env, capture_output=True, text=True, timeout=300)
            (logs / f'native-{phase}.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            assert result.returncode == 0, f'{phase} failed: {result.stderr[-5000:]}'
        preflight = transfer.read(destination / 'artifacts/step5/preflight.json')
        assert preflight['passed'] and all(v['optimized'] for v in preflight['throughput_workers'].values())
        resumed = transfer.read(destination / run_name / 'state.json')
        assert resumed['iteration'] == state['iteration'] + 1
        assert resumed['games_done'] == state['games_done'] + cfg['games_per_iteration']
        assert resumed['signature'] == dict(state['signature'], implementation=current)
        transition = resumed['source_transitions'][-1]
        assert transition['from_signature'] == state['signature'] and transition['checkpoint'] == state['history'][-1]
        backup = destination / run_name / transition['backup']
        assert transfer.read(backup) == state
        assert transfer.sha(destination / state['history'][-1]['path']) == state['history'][-1]['sha256']
        assert (source / 'state.json').read_bytes() == original
        report = dict(passed=True, source_implementation=state['signature']['implementation'], implementation=current,
            different_root=True, original_root_removed=True, archived_files_verified=len(hashes),
            state_unchanged_until_resume=True, reviewed_transition=True,
            runtime_files=len(runtime), throughput_workers=preflight['throughput_workers'],
            iterations=[state['iteration'], resumed['iteration']], games=[state['games_done'], resumed['games_done']],
            elapsed_seconds=round(time.monotonic() - started, 2))
        (logs / 'native-smoke-result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
