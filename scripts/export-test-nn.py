"""Export NN test checkpoints, with read-only utility validation by default.

Checkpoint bytes travel as bounded Git-friendly parts. After pulling, --restore
verifies the bundle and rebuilds checkpoint.pt, which stays ignored by Git.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = 'artifacts/step5/joint7-v11-2-grouped-lr-critic'
FORMAT = 'migration-test-nn/1.0.0'
POINTER = 'migration-test-nn-selection/1.0.0'
CHUNK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def below(root, name):
    root = Path(root).resolve()
    path = Path(name)
    require('..' not in path.parts, f'Parent traversal is not allowed: {name}')
    path = path if path.is_absolute() else root / path
    require(path != root and path.is_relative_to(root), f'Path outside root: {name}')
    for item in (path, *path.parents):
        if item == root:
            break
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                f'Linked paths are not supported: {item}')
    require(path.resolve().is_relative_to(root), f'Resolved path outside root: {name}')
    return path


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def read(path):
    return json.loads(Path(path).read_bytes().decode('utf-8-sig'))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def receipt(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK):
            checksum.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=checksum.hexdigest())


def git_info(root):
    def command(*args):
        try:
            result = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=15)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    return dict(commit=command('rev-parse', 'HEAD'),
                tracked_changes=command('status', '--porcelain=v1', '--untracked-files=no'),
                scope='Exporter checkout; checkpoint source identity remains in state.signature')


def find_run(root, name=None):
    if name:
        return below(root, name)
    preferred = below(root, DEFAULT_RUN)
    if (preferred / 'state.json').is_file():
        return preferred
    candidates = []
    for path in sorted((root / 'artifacts/step5').glob('*/state.json')):
        state = read(below(root, path))
        if state.get('history') or state.get('last'):
            candidates.append(path.parent)
    require(len(candidates) == 1,
            'Cannot choose a unique run; pass --run. Candidates: ' +
            ', '.join(str(p.relative_to(root)) for p in candidates))
    return below(root, candidates[0])


def selections(state, choice):
    labels = ('latest', 'best') if choice == 'both' else (choice,)
    result = {}
    for label in labels:
        if label == 'latest':
            history = state.get('history')
            entry = history[-1] if history else state.get('last')
            reason = 'committed history[-1]' if history else 'committed last'
        elif state.get('best'):
            entry, reason = state['best'], 'recorded best'
        elif state.get('champion'):
            entry, reason = state['champion'], 'accepted champion'
        else:
            active = (state.get('policy_anchor') or {}).get('active') or {}
            entry, reason = active.get('checkpoint'), 'qualified policy anchor (not a global best claim)'
        require(isinstance(entry, dict),
                f'No committed {label} checkpoint. Use --select latest if no best/anchor exists.')
        require(isinstance(entry.get('path'), str) and entry['path'].endswith('.pt'),
                f'{label}: checkpoint must be a recorded .pt file, not a temporary file')
        require(isinstance(entry.get('sha256'), str) and
                re.fullmatch('[0-9a-fA-F]{64}', entry['sha256']), f'{label}: missing checkpoint SHA-256')
        if label == 'latest' and state.get('history') and 'iteration' in state:
            require(entry.get('generation') == state['iteration'],
                    'Latest checkpoint generation differs from committed state.iteration')
        result[label] = dict(entry=entry, reason=reason)
    return result


def checkpoint_path(root, run, entry):
    name = Path(entry['path'])
    # Self-play receipts are project-relative; IL best/last are run-relative.
    base = root if name.is_absolute() or (name.parts and name.parts[0] == 'artifacts') else run
    path = below(root, base / name)
    require(path.is_relative_to(run), 'Checkpoint must belong to the selected run')
    require(path.is_file(), f'Committed checkpoint missing: {path}')
    return path


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(below(bundle, 'manifest.json'))
    require(manifest.get('format') == FORMAT, f'Unknown bundle format: {bundle}')
    require(manifest['checkpoint']['path'] == 'checkpoint.pt', 'Unexpected restored filename')
    checksum = hashlib.sha256()
    size = 0
    names = set()
    require(manifest.get('parts'), 'Bundle has no checkpoint parts')
    for part in manifest['parts']:
        require(part['path'] not in names and
                re.fullmatch(r'checkpoint\.pt\.part\d{4,}', part['path']), 'Invalid/duplicate part name')
        names.add(part['path'])
        path = below(bundle, part['path'])
        individual = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            while block := stream.read(CHUNK):
                individual.update(block)
                checksum.update(block)
                count += len(block)
        require(count == part['bytes'] and individual.hexdigest() == part['sha256'],
                f'Checkpoint part missing/changed: {path}')
        size += count
    require(size == manifest['checkpoint']['bytes'] and
            checksum.hexdigest() == manifest['checkpoint']['sha256'], 'Combined checkpoint hash differs')
    for item in manifest['metadata']:
        path = below(bundle, item['path'])
        require(receipt(path) == {k: item[k] for k in ('bytes', 'sha256')},
                f'Metadata hash differs: {path}')
    require(digest((bundle / 'state.json').read_bytes()) == manifest['state_sha256'], 'State snapshot differs')
    if 'calibration' in manifest:
        proof = manifest['calibration']
        report = read(below(bundle, 'calibration.json'))
        require(proof.get('ready') is True and report.get('ready') is True and
                report['calibration_receipt'].get('ready') is True,
                'Calibrated bundle does not have a passed utility receipt')
        require(proof['request_id'] == report['request_id'] and
                proof['source_checkpoint_sha256'] == report['source_checkpoint_sha256'] ==
                manifest['recorded_checkpoint']['sha256'].lower(), 'Calibration parent/request differs')
        require(manifest['derived_checkpoint']['sha256'] == manifest['checkpoint']['sha256'] and
                manifest['derived_checkpoint']['model_id'] == report['exported_model_id'],
                'Calibrated checkpoint identity differs')
        if proof.get('mode') == 'read-only-validation':
            require(report['completed'] is True and proof['weights_unchanged'] is True and
                    report['calibration_receipt']['steps'] == 0 and
                    report['calibration_receipt']['mode'] == 'read-only-validation' and
                    report['checks']['weights_unchanged'] is True and
                    report['source_model_id'] == report['exported_model_id'] ==
                    manifest['recorded_checkpoint']['model_id'], 'Read-only validation identity differs')
    return manifest


def make_bundle(root, output, run, raw_state, state, entry, source, part_bytes, *,
                extra_files=None, provenance=None):
    run_name = re.sub('[^A-Za-z0-9_.-]', '-', run.name)[:64]
    run_ref = run.relative_to(root).as_posix()
    state_hash = digest(raw_state)
    generation = entry.get('generation', entry.get('epoch', state.get('epoch', 'unknown')))
    name = f'{run_name}-{digest(run_ref.encode())[:8]}-g{generation}-{entry["sha256"][:12]}-{state_hash[:12]}'
    bundle = below(output, name)
    if bundle.exists():
        manifest = verify_bundle(bundle)
        require(manifest['source_run'] == run_ref and manifest['state_sha256'] == state_hash and
                manifest['checkpoint']['sha256'] == entry['sha256'].lower(), 'Existing bundle identity differs')
        if provenance:
            require(all(manifest.get(k) == value for k, value in provenance.items()),
                    'Existing calibrated bundle provenance differs')
        return bundle
    stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=output))
    try:
        parts, checksum, total = [], hashlib.sha256(), 0
        with source.open('rb') as stream:
            while True:
                block = stream.read(min(CHUNK, part_bytes))
                if not block:
                    break
                part = stage / f'checkpoint.pt.part{len(parts) + 1:04d}'
                part_hash, count = hashlib.sha256(), 0
                with part.open('xb') as target:
                    while block:
                        target.write(block)
                        part_hash.update(block)
                        checksum.update(block)
                        count += len(block)
                        total += len(block)
                        if count == part_bytes:
                            break
                        block = stream.read(min(CHUNK, part_bytes - count))
                parts.append(dict(path=part.name, bytes=count, sha256=part_hash.hexdigest()))
        require(total > 0 and checksum.hexdigest() == entry['sha256'].lower(),
                'Checkpoint hash differs from captured committed state; nothing published. '
                'Retry if training retention advanced during copying.')
        (stage / 'state.json').write_bytes(raw_state)
        config = state.get('signature', {}).get('config', state.get('config'))
        require(isinstance(config, dict), 'No effective run config in captured state')
        (stage / 'config.json').write_bytes(encode(config))
        (stage / 'source.json').write_bytes(encode(git_info(root)))
        metadata_names = ['state.json', 'config.json', 'source.json']
        for relative, origin in (extra_files or {}).items():
            require(relative not in metadata_names and relative != 'manifest.json', 'Duplicate bundle metadata')
            target = below(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(below(root, origin), target)
            metadata_names.append(relative)
        manifest = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
                        source_run=run_ref, committed_generation=state.get('iteration'),
                        state_sha256=state_hash, recorded_checkpoint=entry,
                        checkpoint=dict(path='checkpoint.pt', bytes=total, sha256=checksum.hexdigest()),
                        parts=parts, metadata=[dict(path=p, **receipt(stage / p)) for p in metadata_names],
                        config_scope='Source run at snapshot time; checkpoint.model_config and reward '
                                     'are authoritative for an older best/anchor. Not a resume bundle.')
        if provenance:
            manifest.update(provenance)
        (stage / 'manifest.json').write_bytes(encode(manifest))
        verify_bundle(stage)
        require(not bundle.exists(), f'Bundle appeared concurrently; retry: {bundle}')
        stage.rename(bundle)
        return bundle
    finally:
        if stage.exists():
            # Only our newly created staging directory, never an existing bundle/run.
            require(stage.parent.resolve() == output.resolve() and stage.name.startswith('.staging-')
                    and not stage.is_symlink(), 'Unsafe staging cleanup target')
            shutil.rmtree(stage)


def write_pointer(output, label, bundle, entry, reason):
    target = below(output, label + '.json')
    if target.exists():
        old = read(target)
        require(old.get('format') == POINTER and old.get('selection') == label,
                f'Refusing to replace an unrelated file: {target}')
    pointer = dict(format=POINTER, selection=label, selection_reason=reason,
                   bundle=bundle.relative_to(output).as_posix(),
                   generation=entry.get('generation'), model_id=entry.get('model_id'),
                   checkpoint_sha256=entry['sha256'].lower(),
                   manifest_sha256=receipt(bundle / 'manifest.json')['sha256'])
    descriptor, temporary = tempfile.mkstemp(prefix='.' + label + '-', suffix='.tmp', dir=output)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encode(pointer))
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def calibrate_bundle(root, bundle, options, label, part_bytes):
    # Restoration and byte-copy exports keep working without PyTorch.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_test_nn_calibration',
                                                Path(__file__).with_name('test_nn_calibration.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare(root, bundle, options, label, part_bytes=part_bytes)


def export(root, run=None, choice='latest', *, part_bytes=48 * CHUNK, calibration=None):
    root = Path(root).resolve()
    require(0 < part_bytes <= 48 * CHUNK, 'Part size must be 1 byte to 48 MiB')
    run = find_run(root, run)
    raw_state = below(root, run / 'state.json').read_bytes()
    state = json.loads(raw_state.decode('utf-8-sig'))
    selected = selections(state, choice)
    sources = {label: checkpoint_path(root, run, value['entry']) for label, value in selected.items()}
    output = below(root, 'test_nn')
    output.mkdir(exist_ok=True)
    bundles = {label: make_bundle(root, output, run, raw_state, state, value['entry'],
                                 sources[label], part_bytes) for label, value in selected.items()}
    if calibration is not None:
        completed = {}
        for label, bundle in list(bundles.items()):
            if bundle not in completed:
                completed[bundle] = calibrate_bundle(root, bundle, calibration, label, part_bytes)
            bundles[label], derived = completed[bundle]
            selected[label] = dict(entry=derived,
                                   reason=selected[label]['reason'] + '; read-only utility validation; weights unchanged')
    # Publish selections only after every requested checkpoint has passed verification.
    for label, value in selected.items():
        write_pointer(output, label, bundles[label], value['entry'], value['reason'])
    return bundles


def materialize(root, label, *, restore=True):
    output = below(Path(root).resolve(), 'test_nn')
    pointer = read(below(output, label + '.json'))
    require(pointer.get('format') == POINTER and pointer.get('selection') == label,
            'Unknown selection pointer')
    bundle = below(output, pointer['bundle'])
    require(receipt(bundle / 'manifest.json')['sha256'] == pointer['manifest_sha256'],
            'Selection manifest hash differs')
    manifest = verify_bundle(bundle)
    require(manifest['checkpoint']['sha256'] == pointer['checkpoint_sha256'], 'Selection checkpoint differs')
    target = below(bundle, 'checkpoint.pt')
    expected = {k: manifest['checkpoint'][k] for k in ('bytes', 'sha256')}
    if target.exists():
        require(receipt(target) == expected, f'Restored checkpoint changed: {target}')
    elif restore:
        descriptor, temporary = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=bundle)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                for part in manifest['parts']:
                    with below(bundle, part['path']).open('rb') as source:
                        shutil.copyfileobj(source, stream, CHUNK)
            require(receipt(temporary) == expected, 'Restored checkpoint hash differs')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', help='Project-relative run; defaults to the current v11.2 run')
    parser.add_argument('--select', choices=('latest', 'best', 'both'), default='latest')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--restore', action='store_true', help='After Git pull: verify and reconstruct checkpoint.pt')
    mode.add_argument('--verify', action='store_true', help='Verify exported bytes without creating checkpoints')
    parser.add_argument('--part-mib', type=int, default=48, choices=range(1, 49), metavar='1..48')
    parser.add_argument('--no-utility-validation', '--no-calibrate', action='store_true',
                        help='Byte-copy only; does not issue a new utility readiness receipt')
    parser.add_argument('--validation-games', '--calibration-games', type=int, default=4,
                        help='Independent complete games for read-only metrics; no fitting (default: 4)')
    parser.add_argument('--validation-rows-per-game', '--calibration-rows-per-game', type=int, default=128)
    parser.add_argument('--validation-seed', '--calibration-seed', type=int, default=1700000000)
    parser.add_argument('--validation-device', '--calibration-device', choices=('cpu', 'cuda', 'cuda:0'), default='cpu')
    parser.add_argument('--validation-threads', '--calibration-threads', type=int, default=1)
    parser.add_argument('--validation-host', '--calibration-host',
                        help='Existing prepared rules host; no runtime installation is performed')
    args = parser.parse_args()
    try:
        if args.restore or args.verify:
            require(not args.run, '--run is only used when exporting on the lab')
            for label in (('latest', 'best') if args.select == 'both' else (args.select,)):
                path = materialize(args.root, label, restore=args.restore)
                print(f'{label}: {path}' if args.restore else f'{label}: verified')
        else:
            calibration = None if args.no_utility_validation else dict(
                games=args.validation_games, rows_per_game=args.validation_rows_per_game, seed=args.validation_seed,
                device=args.validation_device, threads=args.validation_threads, host=args.validation_host)
            for label, bundle in export(args.root, args.run, args.select,
                                       part_bytes=args.part_mib * CHUNK, calibration=calibration).items():
                print(f'{label}: {bundle}')
            print('Prepared test_nn/. Commit and push it on the lab, then pull locally and run:')
            print(f'python scripts/export-test-nn.py --restore --select {args.select}')
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError, json.JSONDecodeError) as error:
        parser.exit(1, f'export-test-nn: {error}\n')


if __name__ == '__main__':
    main()
