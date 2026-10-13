"""Real CPU workers: old archive -> relocated resume/evaluation -> third root.

The input must be a stopped, bounded test fixture, never a production campaign.
Its files are read-only. Absolute references are constructed in the disposable
fixture BEFORE exporting, with a correctly bound checkpoint and state.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from smoke_relocation import ROOT, copy_file, git_snapshot, transfer


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--source-ref', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.monotonic()
    source = transfer.below(ROOT, args.run)
    assert not transfer.active(source)
    input_hashes = {p.relative_to(source).as_posix():transfer.sha(p) for p in source.rglob('*') if p.is_file()}
    original = transfer.read(source / 'state.json')
    cfg = original['signature']['config']
    assert cfg['games_per_iteration'] <= 2 and cfg['search_every_games'] == 0
    assert not cfg['evaluation_schedule']['enabled'] and not original['signature'].get('imitation_source')
    run_name = source.relative_to(ROOT).as_posix()
    logs = ROOT / 'artifacts/step5-relocation-audit'
    logs.mkdir(exist_ok=True)
    # Keep native quarantine renames within Windows' supported path length.
    # The repository itself already has a long directory name with spaces.
    with tempfile.TemporaryDirectory(prefix='s5-', dir=ROOT / 'artifacts') as temp:
        area = Path(temp).resolve()
        old, second, third = [area / n for n in ('a old', 'b new', 'c new')]
        print('Create historical-source fixture with mixed absolute/relative references', flush=True)
        git_snapshot(args.source_ref, old)
        for name in input_hashes:
            copy_file(source / name, old / run_name / name)
        distribution = transfer.read(ROOT / 'training/runtime/distribution.json')
        runtime = set(distribution['private_runtime_files']) | {e['path'] for e in distribution['assets']}
        host = ROOT / cfg['host']
        runtime.add((host / 'runtime.json').relative_to(ROOT).as_posix())
        runtime.update((host / e['name']).relative_to(ROOT).as_posix() for e in transfer.read(host / 'runtime.json')['files'])
        for name in runtime:
            copy_file(ROOT / name, old / name)
        # Create an independently valid fixture, not altered historical evidence.
        # Everything below is finalized before the first archive is written.
        state = copy.deepcopy(original)
        state['signature']['config']['host'] = str(old / cfg['host'])
        latest = state['history'][-1]['path']
        assert not Path(latest).is_absolute() and '..' not in Path(latest).parts
        assert Path(latest).is_relative_to(Path(run_name)), 'Use an owned project-relative input fixture'
        checkpoint = old / latest
        raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
        raw['training'].update(signature=transfer.digest(state['signature']), fresh_run=str(old / run_name))
        torch.save(raw, checkpoint)
        entries = state['history'] + state.get('resume', []) + state.get('population', []) + [state['evaluation_anchor']]
        if state.get('champion'):
            entries.append(state['champion'])
        for entry in entries:
            if entry['path'] == latest:
                entry.update(path=str(checkpoint), sha256=transfer.sha(checkpoint))
        (old / run_name / 'state.json').write_bytes(transfer.encode(state))
        state_bytes = (old / run_name / 'state.json').read_bytes()
        # Invoke the actual historical exporter in its own interpreter/source.
        deps = [str(Path(p).resolve()) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                if p and Path(p).resolve() != ROOT / 'src']
        def environment(project, forbidden=()):
            return dict(os.environ, PYTHONPATH=os.pathsep.join([str(guard), str(project / 'src'), *deps]),
                        PYTHONDONTWRITEBYTECODE='1', STEP5_FORBIDDEN_ROOTS=json.dumps([str(p) for p in forbidden]))
        guard = area / 'access guard'
        guard.mkdir()
        (guard / 'sitecustomize.py').write_text('''import json, os, sys
roots = [os.path.normcase(os.path.abspath(p)) for p in json.loads(os.environ.get('STEP5_FORBIDDEN_ROOTS', '[]'))]
def check(event, args):
    if event not in ('open','os.mkdir','os.listdir','os.scandir','os.remove','os.rmdir','os.rename','os.link','os.symlink'):
        return
    for value in args[:2 if event in ('os.rename','os.link','os.symlink') else 1]:
        if not isinstance(value, (str, bytes, os.PathLike)):
            continue
        path = os.path.normcase(os.path.abspath(os.fsdecode(value)))
        if any(path == root or path.startswith(root + os.sep) for root in roots):
            raise RuntimeError('Forbidden original-root filesystem access: ' + event + ' ' + path)
sys.addaudithook(check)
''', encoding='utf-8')
        def execute(label, project, command, forbidden=()):
            print(label, flush=True)
            result = subprocess.run(command, cwd=project, env=environment(project, forbidden),
                                    text=True, capture_output=True, timeout=600)
            (logs / (label + '.log')).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            assert result.returncode == 0, label + ': ' + result.stderr[-6000:]
            return result.stdout
        exported = json.loads(execute('old-export', old,
            [sys.executable, str(old / 'scripts/step5-transfer.py'), 'export', '--root', str(old), '--run', run_name,
             '--bundle', str(area / 'old.tar.zst'), '--dependencies', 'bundled', '--threads', '2']))
        archive = transfer.inspect_bundle(exported['bundle'])
        def remove(project):
            assert project.parent == area and area.is_relative_to(ROOT / 'artifacts')
            shutil.rmtree(project)
            assert not project.exists()
        remove(old)
        restored = transfer.restore(second, exported['bundle'], relocate=True)
        assert restored['requires_code_update']
        assert transfer.restore(second, exported['bundle'], verify_only=True)['verified']
        assert all(transfer.sha(second / r['path']) == r['sha256'] for r in archive['files'])
        assert (second / run_name / 'state.json').read_bytes() == state_bytes
        print('Install reviewed destination source after byte-exact restoration', flush=True)
        for prefix in transfer.CODE_TREES:
            for path in (ROOT / prefix).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    copy_file(path, second / path.relative_to(ROOT))
        current = transfer.source_identity(second)
        assert current == transfer.source_identity(ROOT)
        def base(project, config=None):
            return [sys.executable, str(project / 'scripts/step5.py'), '--root', str(project),
                    '--config', str(project / (config or archive['config'])), '--device', 'cpu']
        execute('second-preflight', second, base(second) + ['preflight'], [old])
        execute('second-resume', second, base(second) + ['train', '--run', run_name, '--iterations', '1',
                                                     '--resume-reviewed-source'], [old])
        resumed = transfer.read(second / run_name / 'state.json')
        assert resumed['iteration'] == state['iteration'] + 1 and resumed['games_done'] == state['games_done'] + cfg['games_per_iteration']
        assert resumed['signature'] == dict(state['signature'], implementation=current)
        assert transfer.read(second / run_name / resumed['source_transitions'][-1]['backup']) == state
        assert transfer.sha(second / latest) == state['history'][-1]['sha256']
        evaluation = run_name + '/evaluations/e'
        # This screen tests replay-cache relocation. Select a new evaluation
        # config that retains raw games; never modify archived training config
        # or an existing evaluation spec to keep its otherwise-pruned caches.
        from brass_ai.step5.evaluation import evaluation_config as make_evaluation_config
        evaluation_config_name = run_name + '/evaluation-config.json'
        evaluation_config = copy.deepcopy(make_evaluation_config(resumed['signature']['config'],
            setups=1, include_legacy=False, use_search=False, lineups=['recent-neural']))
        evaluation_config['retention']['enabled'] = False
        (second / evaluation_config_name).write_bytes(transfer.encode(evaluation_config))
        eval_args = ['evaluate', '--candidate', resumed['history'][-1]['path'], '--references',
                     resumed['evaluation_anchor']['path'], '--output', evaluation, '--setups', '1',
                     '--no-include-legacy', '--no-search', '--lineups', 'recent-neural']
        execute('second-evaluate', second, base(second, evaluation_config_name) + eval_args, [old])
        screen = transfer.read(second / evaluation / 'report.json')
        assert screen['games'] == 4 and screen['failed_games'] == 0
        spec_bytes = (second / evaluation / 'spec.json').read_bytes()
        # Full mode keeps actual replay caches so the third-root evaluation can
        # demonstrate reuse of all four games and their unchanged provenance.
        reexported = json.loads(execute('second-export', second,
            [sys.executable, str(second / 'scripts/step5-transfer.py'), 'export', '--root', str(second),
             '--run', run_name, '--bundle', str(area / 'second.tar.zst'), '--mode', 'full',
             '--dependencies', 'bundled', '--threads', '2', '--part-mb', '1'], [old]))
        next_archive = transfer.inspect_bundle(reexported['bundle'])
        # Export's new, exact resume config exists in the archive only. verify
        # validates a restored project, so materialize that file first even at
        # the source root. Every existing file must still match byte-for-byte.
        transfer.restore(second, reexported['bundle'])
        assert transfer.restore(second, reexported['bundle'], verify_only=True)['verified']
        remove(second)
        final = transfer.restore(third, reexported['bundle'], relocate=True)
        assert not final['requires_code_update']
        assert transfer.restore(third, reexported['bundle'], verify_only=True)['verified']
        assert all(transfer.sha(third / r['path']) == r['sha256'] for r in next_archive['files'])
        execute('third-preflight', third, base(third, next_archive['config']) + ['preflight'], [old, second])
        # Use the new absolute host spelling while retaining the historical spec.
        evaluation_config = dict(screen['spec']['config'], host=str(third / cfg['host']))
        evaluation_config_name = run_name + '/evaluation-at-destination.json'
        (third / evaluation_config_name).write_bytes(transfer.encode(evaluation_config))
        execute('third-evaluate', third, base(third, evaluation_config_name) + eval_args, [old, second])
        cached = transfer.read(third / evaluation / 'report.json')
        assert cached['execution']['recent-neural']['cached_games'] == 4
        assert cached['results'] == screen['results'] and (third / evaluation / 'spec.json').read_bytes() == spec_bytes
        execute('third-resume', third, base(third, next_archive['config']) +
                ['train', '--run', run_name, '--iterations', '1'], [old, second])
        continued = transfer.read(third / run_name / 'state.json')
        assert continued['iteration'] == resumed['iteration'] + 1
        assert continued['games_done'] == resumed['games_done'] + cfg['games_per_iteration']
        assert continued['source_transitions'] == resumed['source_transitions']
        assert not old.exists() and not second.exists()
        assert {p.relative_to(source).as_posix():transfer.sha(p) for p in source.rglob('*') if p.is_file()} == input_hashes
        report = dict(passed=True, implementation=current, source_implementation=state['signature']['implementation'],
            archived_files=[len(archive['files']), len(next_archive['files'])], original_roots_removed=True,
            subprocess_old_root_access_guard=True, absolute_host_and_checkpoint=True, historical_files_byte_exact=True,
            fresh_processes=True, preflight_roots=2, evaluation_games=4, relocated_cached_games=4,
            reviewed_transitions=len(continued['source_transitions']),
            iterations=[state['iteration'], resumed['iteration'], continued['iteration']],
            games=[state['games_done'], resumed['games_done'], continued['games_done']],
            reexport_parts=reexported['parts'], source_fixture_unchanged=True, elapsed_seconds=round(time.monotonic()-start, 2))
        (logs / 'native-result.json').write_bytes(transfer.encode(report))
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
