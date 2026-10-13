"""Recover a bounded, stopped native fixture using only Git files and assets.

Example: python tests/Transfer/smoke_runtime_restore.py --run <fixture-run>
Uses real rules DLLs and the current trainer, in a disposable project directory.
The original run is read only. No GitHub request or upload is made.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('transfer', ROOT / 'scripts/step5-transfer.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Stopped current-source CPU fixture, at most two games/iteration')
    args = parser.parse_args()
    started = time.monotonic()
    source = transfer.below(ROOT, args.run)
    assert not transfer.active(source), 'Stop the fixture before testing'
    original_state = (source / 'state.json').read_bytes()
    state = json.loads(original_state)
    cfg = state['signature']['config']
    assert cfg['games_per_iteration'] <= 2 and cfg['search_every_games'] == 0
    assert not cfg['evaluation_schedule']['enabled'], 'Use a bounded smoke fixture'
    print('Inventory current code, rules runtime and stopped fixture', flush=True)
    data = transfer.inventory(ROOT, source)
    commit, tracked = transfer.git_baseline(ROOT)
    assert commit, 'Run this check from the project checkout'
    run_name = source.relative_to(ROOT).as_posix()
    log_dir = ROOT / 'artifacts/step5-transfer-validation'
    log_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='transfer-native-', dir=ROOT / 'artifacts') as folder:
        area = Path(folder).resolve()
        project = area / 'project'
        print(f'Copy {len(data["files"])} fixture dependencies into a disposable source', flush=True)
        for name, path in data['files'].items():
            target = project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        bundle = area / 'github-assets/selfplay-migration.tar.zst'
        with patch.object(transfer, 'git_baseline', return_value=(commit, tracked)):
            exported = transfer.export(project, run_name, bundle, threads=2)
        print('Archive created; remove the disposable source and recover from Git files/assets', flush=True)
        assert exported['bundled_runtime_files'] == len(data['runtime'])
        manifest = transfer.inspect_bundle(exported['bundle'])
        git_files = {r['path']: (project / r['path']).read_bytes()
                     for r in manifest['files'] if r['external']}
        assert set(git_files).issubset(tracked)
        assert not (set(git_files) & data['runtime'])

        # Remove the disposable source before recovery; retain only the bytes
        # supplied by Git and the Release assets, not a second runtime backup.
        assert project.resolve().parent == area and area.is_relative_to(ROOT / 'artifacts')
        shutil.rmtree(project)
        for name, content in git_files.items():
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        assert not (project / cfg['host']).exists()
        assert not (project / run_name).exists()
        restored = transfer.restore(project, exported['bundle'])
        assert restored['verified']
        assert transfer.restore(project, exported['bundle'], verify_only=True)['verified']
        assert (project / run_name / 'state.json').read_bytes() == original_state
        for name in data['runtime']:
            assert transfer.sha(project / name) == transfer.sha(data['files'][name])
        assert not (project / 'artifacts/step5/preflight.json').exists()

        # Launch the restored entry point with its own source tree. Only the
        # installed third-party Python environment is shared with this machine.
        deps = [str(Path(p).resolve()) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                if p and Path(p).resolve() != ROOT / 'src']
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(project / 'src'), *deps]),
                   PYTHONDONTWRITEBYTECODE='1')
        base = [sys.executable, str(project / 'scripts/step5.py'), '--root', str(project),
                '--config', str(project / manifest['config']), '--device', 'cpu']
        for phase, command in [('preflight', base + ['preflight']),
                               ('resume', base + ['train', '--run', str(project / run_name), '--iterations', '1'])]:
            print(f'Restored project: {phase}', flush=True)
            result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=240)
            (log_dir / f'restored-{phase}.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            assert result.returncode == 0, f'{phase} failed: {result.stderr[-5000:]}'
        preflight = transfer.read(project / 'artifacts/step5/preflight.json')
        assert preflight['passed']
        assert all(v['optimized'] for v in preflight['throughput_workers'].values())
        resumed = transfer.read(project / run_name / 'state.json')
        assert resumed['iteration'] == state['iteration'] + 1
        assert resumed['games_done'] == state['games_done'] + cfg['games_per_iteration']
        assert resumed['signature'] == state['signature']
        assert resumed.get('source_transitions') == state.get('source_transitions')
        assert (source / 'state.json').read_bytes() == original_state
        report = dict(passed=True, implementation=state['signature']['implementation'],
                      runtime_files=exported['bundled_runtime_files'],
                      runtime_logical_bytes=exported['bundled_runtime_bytes'],
                      archive_bytes=exported['compressed_bytes'], parts=exported['parts'],
                      original_source_removed=True, restored_bytes_verified=True,
                      throughput_workers=preflight['throughput_workers'],
                      iterations=[state['iteration'], resumed['iteration']],
                      games=[state['games_done'], resumed['games_done']],
                      elapsed_seconds=round(time.monotonic() - started, 2))
        (log_dir / 'recovery-smoke-result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
