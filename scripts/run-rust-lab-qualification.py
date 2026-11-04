#!/usr/bin/env python3
"""Run B01-B04 with automatic Git-visible result export, even on failure."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from rust_lab_results import ROOT, export, sha, write

STAGES = ('b01', 'b02', 'b025', 'b03', 'b04')
PACKAGE = ROOT/'distribution/rust-lab-g570'
RUN = 'artifacts/step5/joint7-v11-2-g1290-trained-restart'
CONFIG = 'results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z/recorded-config.json'


def verify_package():
    sys.path.insert(0, str(ROOT/'src'))
    from brass_ai.rust_backend.lab_transfer import verify
    verify('b01', PACKAGE/'transfer-manifest.json')
    manifest = json.loads((PACKAGE/'transfer-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError('Qualification update changed/missing: '+name)
    for name, expected in manifest['archives'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError('Sealed prerequisite changed/missing: '+name)
    return manifest


def run_identity(run, generation=None, checkpoint_hash=None):
    path = run/'state.json'
    data = path.read_bytes(); state = json.loads(data); latest = state['history'][-1]
    actual = state['iteration']
    if (type(actual) is not int or actual < 0 or latest['generation'] != actual
            or (generation is not None and actual != generation)
            or (checkpoint_hash is not None and latest['sha256'] != checkpoint_hash)):
        raise ValueError('Committed generation/checkpoint differs; pause training and supply its exact identity')
    if len(latest['sha256']) != 64 or any(c not in '0123456789abcdef' for c in latest['sha256']):
        raise ValueError('Invalid committed checkpoint hash')
    return dict(run=str(run), generation=actual, checkpoint=latest, state_sha256=hashlib.sha256(data).hexdigest(),
                implementation=state['signature']['implementation'])


def command(stage, args, out, receipts):
    script = 'qualify-rust-lab.py' if stage == 'b01' else 'qualify-rust-lab-'+stage+'.py'
    cmd = [sys.executable, str(ROOT/'scripts'/script)]
    cmd += ['--output' if stage == 'b025' else '--out', str(out), '--device', args.device]
    if args.fixtures_only:
        if stage != 'b025': cmd += ['--fixtures-only']
    else:
        cmd += ['--run', str(args.run)]
        if stage in ('b025', 'b03', 'b04'): cmd += ['--expected-generation', str(args.expected_generation)]
        if stage in ('b03', 'b04'): cmd += ['--checkpoint-sha256', args.checkpoint_sha256]
    if stage == 'b01':
        cmd += ['--profile', str(ROOT/'configs/search/lab-search-v1.json'),
                '--bundle', str(ROOT/'distribution/rust-lab-b01/b01-search-v1-inputs.zip'),
                '--bundle-sha256', '935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76']
    if stage == 'b02': cmd += ['--workers', str(args.b02_workers)]
    if stage in ('b025', 'b03','b04'):
        cmd += ['--transfer-manifest', str(PACKAGE/'transfer-manifest.json')]
    if not args.fixtures_only:
        for needed in {'b02': ('b01',), 'b03': ('b02',), 'b04': ('b02', 'b03')}.get(stage, ()):
            if not receipts.get(needed): raise ValueError(stage+' needs --'+needed+' or --stage all')
            cmd += ['--'+needed, str(receipts[needed])]
    return cmd


def selected_stages(args, receipts):
    if args.stage != 'all':
        return (args.stage,)
    stages = STAGES[STAGES.index(args.start_at or 'b01'):]
    available = {s for s, value in receipts.items() if value is not None}
    for stage in stages:
        for needed in {'b02': ('b01',), 'b03': ('b02',), 'b04': ('b02', 'b03')}.get(stage, ()):
            if needed not in available:
                raise ValueError('Starting at '+stages[0]+' requires --'+needed+' with its original passing receipt')
        available.add(stage)
    return stages


def execute(cmd, log):
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            for line in process.stdout:
                stream.write(line); stream.flush()
                print(line, end='', flush=True)
            return process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == 'posix': process.send_signal(signal.SIGINT)
                else: process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        finally: process.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    parser.add_argument('--start-at', choices=STAGES,
                        help='Resume --stage all at this stage, supplying prior original receipts; every gate still verifies them')
    parser.add_argument('--run', type=Path, default=Path(RUN))
    parser.add_argument('--expected-generation', type=int, help='Optional assertion; otherwise capture the latest committed generation')
    parser.add_argument('--checkpoint-sha256', help='Optional assertion; otherwise capture the latest committed checkpoint hash')
    parser.add_argument('--config', type=Path, default=Path(CONFIG), help='Must match the committed run signature; never adopted by qualification')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--b02-workers', type=int, default=60, help='B02 collection lane ceiling; 1..60')
    for stage in ('b01', 'b02', 'b03'): parser.add_argument('--'+stage, type=Path)
    parser.add_argument('--fixtures-only', action='store_true', help='CPU development only; stage b025, b03 or b04')
    args = parser.parse_args(argv)
    if not 1 <= args.b02_workers <= 60: parser.error('--b02-workers must be 1..60')
    if args.start_at is not None and (args.stage != 'all' or args.fixtures_only):
        parser.error('--start-at requires --stage all and a real lab run')
    if args.fixtures_only and (args.stage not in ('b025', 'b03', 'b04') or args.device != 'cpu'):
        parser.error('--fixtures-only requires --stage b025/b03/b04 --device cpu')
    args.run = (ROOT/args.run).resolve()
    target = 'latest' if args.expected_generation is None else 'g'+str(args.expected_generation)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+target+'-'+args.stage+'-'+uuid.uuid4().hex[:8]
    out = ROOT/'artifacts/rust-backend'/name; out.mkdir(parents=True, exist_ok=False)
    report = dict(contract='brass-lab-runner/1.0.0', passed=False, lab_qualification='Pending',
        requested_generation=args.expected_generation, requested_checkpoint=args.checkpoint_sha256,
        fixtures_only=args.fixtures_only, device=args.device, stages=[])
    receipts = {s: getattr(args, s) for s in ('b01', 'b02', 'b03')}
    code = 1
    try:
        report['bundle'] = dict(manifest_sha256=sha(PACKAGE/'transfer-manifest.json'),
                                training_reference=verify_package()['training_reference'])
        stages = selected_stages(args, receipts)
        report['selected_stages'] = list(stages)
        report['reused_receipts'] = {stage: dict(path=str(path.resolve()),sha256=sha(path))
            for stage,path in receipts.items() if path is not None and stage not in stages}
        identity = None if args.fixtures_only else run_identity(args.run, args.expected_generation, args.checkpoint_sha256)
        report['run'] = identity
        if identity is not None:
            # Resolve once, then assert these exact pins around every stage.
            # Never follow an advancing run or qualify an older checkpoint.
            args.expected_generation = identity['generation']
            args.checkpoint_sha256 = identity['checkpoint']['sha256']
            actual = json.loads((args.run/'state.json').read_text(encoding='utf-8'))['signature']['config']
            requested = json.loads((ROOT/args.config).read_text(encoding='utf-8'))
            if actual != requested:
                raise ValueError('Committed config differs from --config. Supply the matching reviewed current config; qualification never adopts a config or changes a run.')
            report['config'] = dict(path=str(args.config),sha256=sha(ROOT/args.config))
        for stage in stages:
            if identity is not None and run_identity(args.run, args.expected_generation, args.checkpoint_sha256) != identity:
                raise ValueError('Training state changed between qualification stages')
            folder = out/stage
            cmd = command(stage, args, folder, receipts)
            status = dict(stage=stage, argv=cmd, passed=False)
            report['stages'].append(status)
            print('Running '+stage+'; result export will be under results/rust-lab-qualification/', flush=True)
            code = execute(cmd, out/(stage+'-console.log')); status['exit_code'] = code
            receipt = folder/'qualification.json'
            if receipt.exists():
                value = json.loads(receipt.read_text(encoding='utf-8'))
                status.update(passed=code == 0 and value.get('passed') is True,
                    receipt=receipt.relative_to(out).as_posix(), sha256=sha(receipt),
                    lab_qualification=value.get('lab_qualification', value.get('scope', 'See stage receipt')))
                receipts[stage] = receipt
            if identity is not None and run_identity(args.run, args.expected_generation, args.checkpoint_sha256) != identity:
                raise ValueError('Training state changed during qualification')
            if not status['passed']:
                code = code or 2
                break
        report['passed'] = bool(report['stages']) and all(s['passed'] for s in report['stages'])
        if report['passed']:
            code = 0
            report['lab_qualification'] = 'Pending: CPU fixtures only' if args.fixtures_only else 'Selected stages passed; see individual scopes'
    except BaseException as error:
        report['error'] = type(error).__name__+': '+str(error)
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        print(report['error'], file=sys.stderr)
    finally:
        write(out/'runner.json', report)
        destination = export(out)
        print('\nGit-visible results: '+str(destination), flush=True)
        print('Original lab evidence: '+str(out), flush=True)
    return code


if __name__ == '__main__': sys.exit(main())
