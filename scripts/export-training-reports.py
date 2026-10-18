"""Copy a self-play analysis bundle to results/ without importing the trainer.

Original JSON bytes and run files are preserved. Checkpoints, replay shards,
feature caches and profiler traces are not part of this report-only export.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = 'artifacts/step5/joint7-v11-1-grouped-lr'
FORMAT = 'selfplay-report-export/1.0.0'
REPORT_NAMES = {'state.json', 'retention-history.json', 'report.json', 'spec.json',
                'config.json', 'best.json', 'summary.json', 'comparison.json',
                'validation-comparison.json', 'receipt.json', 'manifest.json'}
SKIP_DIRS = {'.git', '.retention', '.retention-trash', '__pycache__', 'population'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def below(root, name):
    """Resolve only ordinary paths inside this project, including on Windows."""
    path = Path(name)
    require('..' not in path.parts, f'Parent traversal is not allowed: {name}')
    path = path if path.is_absolute() else root / path
    require(path.is_relative_to(root) and path != root, f'Path outside project: {name}')
    for part in (path, *path.parents):
        if part == root:
            break
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                f'Linked path is not supported: {part}')
    require(path.resolve().is_relative_to(root), f'Resolved path outside project: {name}')
    return path


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def report_files(root, directory):
    """Walk report directories without traversing links or bulk artifact trees."""
    directory = below(root, directory)
    if not directory.exists():
        return
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith('.')
                                and d not in SKIP_DIRS and not d.startswith('profile'))
        for name in directories:
            below(root, Path(parent) / name)
        for name in sorted(files):
            if name in REPORT_NAMES or name.endswith(('-report.json', '-receipt.json')):
                yield below(root, Path(parent) / name)


def publish(stage, output):
    """Allow short Windows file-handle contention without overwriting a bundle."""
    for attempt, delay in enumerate((0, .05, .1, .2, .4)):
        if delay:
            time.sleep(delay)
        require(not output.exists(), f'Output appeared during export: {output}')
        try:
            stage.rename(output)
            return
        except PermissionError as error:
            if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 4:
                raise


def export(root, run, output=None, *, latest=2, extras=()):
    root = Path(root).resolve()
    run = below(root, run)
    require(run.is_relative_to(root / 'artifacts'), 'Choose the original run under artifacts/, not a previous export')
    require(latest >= 1, '--latest must be at least 1')
    state_path = below(root, run / 'state.json')
    require(state_path.is_file(), f'Run state does not exist: {state_path}')
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes.decode('utf-8-sig'))
    require(str(state.get('format', '')).startswith('step5-recovery-league/'), 'Expected a self-play run state')
    generation = state['iteration']
    require(type(generation) is int and generation >= 0, 'Invalid committed iteration')
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = f'results/{run.name}-g{generation:04d}-{stamp}'
    output = below(root, output)
    require(output.is_relative_to(root / 'results') and output != root / 'results',
            '--output must name a folder under results/')
    require(not output.exists(), f'Output already exists; choose a new folder: {output}')

    files, missing, warnings = {}, {}, []
    selected_iterations = list(range(max(0, generation - latest), generation))

    def add(path, destination, reason, *, expected=False, document='json'):
        path = below(root, path)
        name = Path(destination).as_posix()
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe bundle path')
        if not path.is_file():
            if expected:
                missing[path.relative_to(root).as_posix()] = reason
            return None
        raw = path.read_bytes()
        # Validate complete documents before publication. Only explicitly
        # selected compact rating outcomes/curves may use these other formats.
        if document == 'json':
            value = json.loads(raw.decode('utf-8-sig'))
        elif document == 'jsonl':
            value = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
        elif document == 'svg':
            import xml.etree.ElementTree as ET
            ET.fromstring(raw); value = None
        else:
            raise ValueError('Unknown report document format')
        entry = dict(source=path.relative_to(root).as_posix(), path=name,
                     bytes=len(raw), sha256=checksum(raw), reason=reason)
        if name in files:
            require(files[name][0] == entry, f'Conflicting export destination: {name}')
        else:
            files[name] = (entry, raw)
        return value

    add(state_path, 'state.json', 'Committed configuration, lineage and progress', expected=True)
    require(files['state.json'][1] == state_bytes, 'Run advanced during export; retry')
    add(run / 'retention-history.json', 'retention-history.json', 'Retained iteration trends', expected=True)
    for path in sorted(run.glob('*.json')):
        if not path.name.startswith('.') and path.name not in (
                'state.json', 'retention-history.json', 'recorded-config.json', 'export-manifest.json'):
            add(path, path.name, 'Run metadata')
    for iteration in selected_iterations:
        directory = run / f'iteration-{iteration:04d}'
        report_path = directory / 'report.json'
        report = add(report_path, report_path.relative_to(run), 'Latest completed iteration', expected=True)
        if report is not None:
            require(report.get('iteration') == iteration, f'Iteration report identity differs: {report_path}')
            if 'iteration_seconds' not in report.get('timings', {}):
                warnings.append(f'{report_path.relative_to(run)} has no final iteration timing; it may still be finalizing')
        for path in report_files(root, directory):
            if path != report_path:
                add(path, path.relative_to(run), 'Related iteration/teaching report')
    for path in report_files(root, run / 'evaluations'):
        add(path, path.relative_to(run), 'Evaluation report or specification')
    for path in report_files(root, run / 'il-kl'):
        add(path, path.relative_to(run), 'Fixed IL-KL probe manifest or initial measurement')
    for path in report_files(root, run / 'ratings'):
        add(path, path.relative_to(run), 'Plackett-Luce strength report or specification')
    archive = state.get('ratings', {}).get('archive')
    if archive is not None:
        path = below(root, archive['path'])
        require(path.is_relative_to(run / 'ratings'), 'Rating archive points outside this run')
        require(checksum(path.read_bytes()) == archive['sha256'], 'Archived rating evidence changed')
        add(path, path.relative_to(run), 'Compact cumulative rating outcomes and diagnostic-panel history')
    for slot in state.get('ratings', {}).get('slots', []):
        for key in ('records', 'panels', 'report', 'spec'):
            reference = slot[key]
            path = below(root, reference['path'])
            require(checksum(path.read_bytes()) == reference['sha256'], 'Committed rating evidence changed')
            if path.relative_to(run).as_posix() not in files:
                add(path, path.relative_to(run), 'Append-only held-out rating outcomes or separate panel results',
                    document='jsonl' if key == 'records' else 'json')
        curve = below(root, Path(slot['report']['path']).parent / 'progress.svg')
        if curve.exists():
            add(curve, curve.relative_to(run), 'Strength progress curve', document='svg')

    # State receipts also reveal evaluations whose files retention has removed.
    gate = state.get('champion_gate') or {}
    receipts = list(state.get('evaluations', [])) + list(gate.get('recent', []))
    accepted = gate.get('accepted_gate')
    if isinstance(accepted, dict):
        receipts.append(accepted)
    for receipt in receipts:
        if not receipt.get('report'):
            continue
        path = below(root, receipt['report'])
        require(path.is_relative_to(run), 'Evaluation receipt points outside this run')
        name = path.relative_to(run).as_posix()
        if name not in files:
            add(path, name, 'Evaluation referenced by state', expected=True)
        if name in files and receipt.get('report_sha256'):
            entry = files[name][0]
            entry['recorded_sha256'] = receipt['report_sha256']
            entry['matches_recorded_sha256'] = entry['sha256'] == receipt['report_sha256']
            if not entry['matches_recorded_sha256']:
                warnings.append(f'{name} differs from its state receipt; inspect retention/pruning provenance')
        spec = path.parent / 'spec.json'
        if spec.relative_to(run).as_posix() not in files:
            add(spec, spec.relative_to(run), 'Evaluation specification', expected=True)

    # Carry IL reports (and ancestor reports), not the large models or dataset.
    visited = set()

    def lineage(origin):
        if not isinstance(origin, dict) or not origin.get('bc_run'):
            return
        directory = below(root, origin['bc_run'])
        require(directory.is_relative_to(root / 'artifacts'), 'IL lineage must be under artifacts/')
        if directory in visited:
            return
        visited.add(directory)
        parent_state = None
        for name in ('state.json', 'report.json'):
            path = directory / name
            value = add(path, Path('lineage') / path.relative_to(root), 'Imitation lineage report', expected=True)
            if name == 'state.json':
                parent_state = value
        if parent_state:
            lineage(parent_state.get('signature', {}).get('parent_imitation'))

    lineage(state.get('origin') or state.get('signature', {}).get('imitation_source'))
    for extra in extras:
        path = below(root, extra)
        require(path.exists(), f'Extra report path does not exist: {path}')
        for item in report_files(root, path) if path.is_dir() else [path]:
            require(item.suffix == '.json', '--extra accepts JSON reports or report directories')
            add(item, Path('related') / item.relative_to(root), 'Explicit related report')

    manifest = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        run=run.relative_to(root).as_posix(), committed_generation=generation,
        games_done=state.get('games_done'), implementation=state.get('signature', {}).get('implementation'),
        latest_iterations=selected_iterations, run_lock_present=(run / '.active-run.json').exists(),
        files=[entry for entry, _ in files.values()],
        missing=[dict(source=path, reason=reason) for path, reason in sorted(missing.items())],
        warnings=sorted(set(warnings)),
        generated=['recorded-config.json', 'export-manifest.json'],
        scope='Analysis reports only; not a resumable checkpoint backup. Missing/pruned evidence is not reconstructed.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.report-export-', dir=output.parent) as temporary:
        require(Path(temporary).resolve().is_relative_to((root / 'results').resolve()), 'Temporary export escaped results/')
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        for entry, raw in files.values():
            destination = stage / entry['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        (stage / 'recorded-config.json').write_bytes(encode(state.get('signature', {}).get('config', {})))
        (stage / 'export-manifest.json').write_bytes(encode(manifest))
        for entry, raw in files.values():
            require(below(root, entry['source']).read_bytes() == raw,
                    f"Report changed during export; retry: {entry['source']}")
        publish(stage, output)
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root; defaults to this script\'s project')
    parser.add_argument('--run', default=DEFAULT_RUN, help=f'Self-play run under artifacts/ (default: {DEFAULT_RUN})')
    parser.add_argument('--output', help='New results/ folder; default includes run, generation and UTC timestamp')
    parser.add_argument('--latest', type=int, default=2, help='Number of latest committed iteration reports (default: 2)')
    parser.add_argument('--extra', action='append', default=[], help='Additional JSON report or directory; repeat as needed')
    args = parser.parse_args()
    try:
        output, manifest = export(args.root, args.run, args.output, latest=args.latest, extras=args.extra)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Export failed: {error}\n')
    print(f'Copied {len(manifest["files"])} reports/metadata files to {output}')
    print(f'Committed generation {manifest["committed_generation"]}; iterations {manifest["latest_iterations"]}')
    print(f'Missing files: {len(manifest["missing"])}; warnings: {len(manifest["warnings"])}. See export-manifest.json.')


if __name__ == '__main__':
    main()
