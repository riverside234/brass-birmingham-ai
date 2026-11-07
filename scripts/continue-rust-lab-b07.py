#!/usr/bin/env python3
"""Finish B07 gates, select Rust, then perform exactly one real same-run iteration.

Re-running after G+1 committed verifies that checkpoint; it never starts G+2.
All training settings are copied from the committed signature, not latest.json.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/'scripts'/name)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def stage(state, generation):
    from brass_ai.learning.contracts import require
    value = state.get('native_backend')
    require(state['iteration'] in (generation, generation+1),
            'Run moved outside the requested G-to-G+1 boundary; never rewind it')
    if value is None:
        require(state['iteration'] == generation, 'G+1 committed without Rust adoption')
        return 'adopt'
    require(value['mode'] == 'rust' and value['adopted_generation'] == generation,
            'Existing backend selection differs; inspect the explicit switch receipt')
    return 'train' if state['iteration'] == generation else 'verify'


def imitation_arguments(root, run, state):
    """Carry the committed IL dependency, never infer a new initialization.

    The trainer still loads and validates the complete IL lineage and the saved
    self-play checkpoint. Do not duplicate that model load in this launcher.
    """
    from brass_ai.learning.contracts import project_path, require
    from brass_ai.learning.paths import bound_reference
    from brass_ai.recovery.config import latest_model_recipe
    config = state['signature']['config']
    source = state['signature'].get('imitation_source')
    required = latest_model_recipe(config)
    require(source is not None or not required,
            'Committed latest-model recipe is missing its imitation_source binding')
    if source is None:return []
    require(isinstance(source, dict) and isinstance(source.get('bc_run'), str) and source['bc_run'],
            'Committed imitation_source is missing its bc_run binding')
    require(config.get('imitation', {}).get('manifest') is None,
            'Committed imitation_source conflicts with in-run imitation')
    if required:
        bound_reference(root, source, config.get('imitation_checkpoint'), ('bc_run', 'path', 'manifest'))
    dependency = project_path(root, source['bc_run'])
    require(dependency != project_path(root, run), 'Self-play cannot use itself as its imitation dependency')
    return ['--imitation-run', str(dependency)]


def training_command(root, run, config, generation, mono, *, state):
    return [sys.executable, str(root/'scripts/step5.py'), '--root', str(root),
            '--config', str(config), '--device', 'cuda', '--mono', mono,
            'train', '--run', str(run), '--iterations', '1', '--expected-generation', str(generation),
            *imitation_arguments(root, run, state)]


def command(argv, log, env):
    """Keep the child in our signal group so its existing cleanup runs on Ctrl-C."""
    with Path(log).open('wb') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            code = process.wait()
            if code:raise subprocess.CalledProcessError(code, argv)
        except BaseException:
            if process.poll() is None:
                import psutil
                try:children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:children = []
                try:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):pass
                finally:
                    for child in reversed(children):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    if process.poll() is None:process.kill()
                    process.wait()
            raise


def execute(args, out, report, invoke):
    """Keep orchestration injectable; every mutating child has its own run lock."""
    from brass_ai.bank import read
    from brass_ai.learning.contracts import digest, project_path, require
    from brass_ai.recovery.retention import owned, run_lock
    from brass_ai.rust_backend import lab_backend as backend
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.step5.league import write_json
    _, run = owned(ROOT, args.run)
    require((run/'state.json').is_file(), 'Continue an existing run; no new run is created')

    def inspect():
        with run_lock(run):
            state = read(run/'state.json')
            selected = stage(state, args.expected_generation)
            require(state['signature']['config'].get('teacher_phase', {}).get('mode', 'off') == 'off',
                    'Backend continuation requires teacher off')
            imitation_arguments(ROOT, run, state)
            if selected != 'adopt':backend.validate_state(state)
            return state, selected

    def manage(action, generation, extra=()):
        target = out/action
        invoke(action, [sys.executable, str(ROOT/'scripts/manage-rust-lab-b07.py'), action,
                       '--run', str(run), '--expected-generation', str(generation), '--out', str(target), *extra])
        result = read(target/'qualification.json')
        require(result['passed'], 'B07 '+action+' did not pass')
        report[action] = pin(target/'qualification.json')

    state, selected = inspect()
    from brass_ai.rust_backend.lab_replay_compat import execution
    report['execution_source'] = execution()
    initial_config = digest(state['signature']['config'])
    initial_imitation = digest(state['signature'].get('imitation_source'))
    if selected == 'adopt':
        if args.qualification:
            qualification = project_path(ROOT, args.qualification)
        else:
            target = out/'remaining'
            command = [sys.executable, str(ROOT/'scripts/qualify-rust-lab-b07.py'), '--complete-remaining',
                       '--run', str(run), '--device', 'cuda:0', '--mono', args.mono, '--out', str(target)]
            for flag, value in (('--resume-from', args.resume_qualification), ('--teacher-build', args.teacher_build)):
                if value:command += [flag, str(project_path(ROOT, value))]
            if args.offline:command += ['--offline']
            invoke('remaining', command)
            qualification = target/'qualification.json'
        q = read(qualification)
        require(q.get('adoption_ready') is True, 'B07 remaining gates are not complete')
        report['qualification'] = pin(qualification)
        manage('adopt', args.expected_generation, ['--qualification', str(qualification)])
        state, selected = inspect()
    require(digest(state['signature']['config']) == initial_config, 'Backend adoption changed the training recipe')
    require(digest(state['signature'].get('imitation_source')) == initial_imitation,
            'Backend adoption changed the committed imitation dependency')
    report.update(production_adoption='Passed',backend='rust',teacher='off')
    # Full old-state restoration is read-only; production remains at its current boundary.
    manage('verify-rollback', state['iteration'])
    if selected == 'train':
        config = out/'committed-training-config.json'
        write_json(config, state['signature']['config'])
        argv = training_command(ROOT, run, config, args.expected_generation, args.mono, state=state)
        report.update(training_config=pin(config),starting_checkpoint=state['history'][-1],
                      training_started=True,training_command=argv)
        write_json(out/'qualification.json', report)
        invoke('training', argv)
    manage('verify-next', args.expected_generation+1)
    report.update(passed=True,backend='rust',teacher='off',generation=args.expected_generation+1,
                  production_adoption='Passed',next_normal_commit_restart='Passed',
                  training_iterations_launched=int(selected == 'train'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation', type=int, required=True)
    p.add_argument('--mono', default='mono')
    p.add_argument('--cuda-visible-devices', help='One GPU ordinal or UUID; applied before importing Torch')
    p.add_argument('--qualification', type=Path, help='Already completed adoption-ready B07 qualification.json')
    p.add_argument('--resume-qualification', type=Path, help='Resume only unfinished phases from an interrupted R5 qualification')
    p.add_argument('--teacher-build', type=Path, help='Reuse this exact implementation teacher build')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    if args.qualification and args.resume_qualification:p.error('Choose completed qualification or partial resume')
    if args.cuda_visible_devices is not None:
        if not args.cuda_visible_devices.strip() or ',' in args.cuda_visible_devices:p.error('Select exactly one GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    else:os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',', 1)[0]
    from brass_ai.step5.league import write_json
    from rust_lab_results import export
    out = ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-b07-continue-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    report = dict(milestone='B07-production',passed=False,expected_generation=args.expected_generation,
                  training_started=False,production_adoption='Pending',next_normal_commit_restart='Pending')
    base = module('qualify-rust-lab.py')

    def invoke(name, argv):
        report['active_stage'] = name
        write_json(out/'qualification.json', report)
        print('B07 '+name+'; log: '+str(out/(name+'.log')), flush=True)
        # The real training iteration has its normal workload, not a smoke watchdog.
        command(argv, out/(name+'.log'), base.python_environment())

    try:
        module('qualify-rust-lab-b07.py').verify()
        execute(args, out, report, invoke)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(out/'qualification.json', report)
        print('Git-visible results: '+str(export(out)), flush=True)
        print('Original lab evidence: '+str(out), flush=True)


if __name__ == '__main__':main()
