"""Copy a self-play analysis bundle to results/ without importing the trainer.

Original JSON bytes and run files are preserved. Checkpoints, replay shards,
feature caches and profiler traces are not part of this report-only export.
With --continue, unchanged reports remain in earlier bundles and the new
manifest records their verified locations. Rating slots include only the latest
committed generation; the cumulative archive is still included when present.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = 'artifacts/step5/joint7-v11-1-grouped-lr'
FORMAT = 'selfplay-report-export/1.1.0'
PREVIOUS_FORMATS = {'selfplay-report-export/1.0.0', FORMAT}
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


def report_files(root, directory, *, skip_dirs=()):
    """Walk report directories without traversing links or bulk artifact trees."""
    directory = below(root, directory)
    if not directory.exists():
        return
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith('.')
                                and d not in SKIP_DIRS and d not in skip_dirs
                                and not d.startswith('profile'))
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


def previous_files(root, run, generation, warnings):
    """Index physical files in older same-run exports, including legacy bundles.

    Reused entries are deliberately not followed: indexing their original
    physical bundle keeps references flat and avoids trusting reference chains.
    Actual bytes and safe paths are checked only when a file could be reused.
    """
    index = {}
    for path in sorted((root / 'results').rglob('export-manifest.json'), reverse=True):
        if any(part.startswith('.') for part in path.relative_to(root / 'results').parts):
            continue
        try:
            path = below(root, path)
            manifest = json.loads(path.read_bytes().decode('utf-8-sig'))
            if not isinstance(manifest, dict) or manifest.get('format') not in PREVIOUS_FORMATS:
                continue
            if manifest.get('run') != run.relative_to(root).as_posix():
                continue
            committed = manifest.get('committed_generation')
            if type(committed) is not int or not 0 <= committed <= generation:
                continue
            for entry in manifest.get('files', []):
                if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str)
                                                        for k in ('source', 'path', 'sha256')):
                    continue
                key = entry['source'], entry['path'], entry['sha256']
                index.setdefault(key, []).append(path.parent)
        except (ValueError, OSError, TypeError) as error:
            warnings.append(f'Ignored previous export {path.relative_to(root)}: {error}')
    return index


def incremental_files(root, run, generation, files, warnings):
    index = previous_files(root, run, generation, warnings)
    copied, reused, dependencies = {}, [], []
    for name, (entry, raw) in files.items():
        for directory in index.get((entry['source'], name, entry['sha256']), []):
            try:
                prior = below(root, directory / name)
                require(prior.is_relative_to(directory), 'Previous report escaped its export')
                require(prior.read_bytes() == raw, 'Previous report is missing or changed')
            except (ValueError, OSError):
                warnings.append(f'Previous report unavailable or changed; copying {directory.relative_to(root) / name}')
                continue
            reused.append(dict(entry, previous_export=directory.relative_to(root).as_posix(),
                               previous_path=name))
            dependencies.append((prior, raw))
            break
        else:
            copied[name] = entry, raw
    return copied, reused, dependencies


def export(root, run, output=None, *, latest=2, extras=(), continue_export=False):
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
    retained=add(run / 'retention-history.json', 'retention-history.json', 'Retained iteration trends', expected=True) or {}
    # Cleanup can be interrupted after a new generation commits. An older
    # report may then be outside --latest but not yet in compact history.
    # Copy that real report rather than silently leaving a gap in the bundle.
    uncompacted=[]
    for directory in sorted(run.glob('iteration-*')):
        match=re.fullmatch(r'iteration-([0-9]+)',directory.name)
        if match is None:continue
        iteration=int(match[1]);relative=directory.relative_to(run)/'report.json'
        if iteration>=generation or iteration in selected_iterations or relative.as_posix() in retained:continue
        report=add(directory/'report.json',relative,'Committed report not yet in compact history')
        if report is not None:
            require(report.get('iteration')==iteration,f'Iteration report identity differs: {directory}')
            uncompacted.append(iteration)
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
    # Old slot folders may still exist after retention or alongside a pending
    # evaluation. Only the latest slot committed in state belongs in this export.
    slots = state.get('ratings', {}).get('slots', [])
    require(isinstance(slots, list) and all(isinstance(slot, dict) and
            type(slot.get('generation')) is int and 0 <= slot['generation'] <= generation
            for slot in slots), 'Invalid committed rating slots')
    latest_slot = max(slots, key=lambda slot: slot['generation'], default=None)
    for path in report_files(root, run / 'ratings', skip_dirs={'slots'}):
        add(path, path.relative_to(run), 'Plackett-Luce strength report or specification')
    archive = state.get('ratings', {}).get('archive')
    if archive is not None:
        path = below(root, archive['path'])
        require(path.is_relative_to(run / 'ratings'), 'Rating archive points outside this run')
        require(checksum(path.read_bytes()) == archive['sha256'], 'Archived rating evidence changed')
        add(path, path.relative_to(run), 'Compact cumulative rating outcomes and diagnostic-panel history')
    for slot in [latest_slot] if latest_slot is not None else []:
        slot_directory = below(root, Path(slot['report']['path']).parent)
        require(slot_directory.parent == run / 'ratings' / 'slots' and
                re.fullmatch(r'generation-0*' + str(slot['generation']), slot_directory.name),
                'Latest rating slot directory differs from its committed generation')
        for path in report_files(root, slot_directory):
            add(path, path.relative_to(run), 'Latest committed rating slot report or specification')
        for key in ('records', 'panels', 'report', 'spec'):
            reference = slot[key]
            path = below(root, reference['path'])
            require(path.is_relative_to(slot_directory), 'Rating evidence points outside its slot')
            require(checksum(path.read_bytes()) == reference['sha256'], 'Committed rating evidence changed')
            if path.relative_to(run).as_posix() not in files:
                add(path, path.relative_to(run), 'Append-only held-out rating outcomes or separate panel results',
                    document='jsonl' if key == 'records' else 'json')
        curve = below(root, Path(slot['report']['path']).parent / 'progress.svg')
        if curve.exists():
            add(curve, curve.relative_to(run), 'Strength progress curve', document='svg')
    for comparison in state.get('temperature_ratings', []):
        reference=comparison['report'];path=below(root,reference['path'])
        require(path.is_relative_to(run/'ratings') and checksum(path.read_bytes())==reference['sha256'],
                'Temperature rating report changed or outside this run')
        if path.relative_to(run).as_posix() not in files:
            add(path,path.relative_to(run),'Paired deployment temperature comparison')
        evidence=json.loads(path.read_text(encoding='utf-8'))
        for key in ('spec','records'):
            item=evidence[key];source=below(root,item['path'])
            require(source.is_relative_to(run/'ratings') and checksum(source.read_bytes())==item['sha256'],
                    'Temperature comparison evidence changed')
            if source.relative_to(run).as_posix() not in files:
                add(source,source.relative_to(run),'Paired temperature rating evidence',
                    document='jsonl' if key=='records' else 'json')

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
        upgrade_path = directory / 'upgrade.json'
        if upgrade_path.is_file():
            upgrade = add(upgrade_path, Path('lineage') / upgrade_path.relative_to(root),
                          'Completed IL critic upgrade receipt', expected=True)
            require(upgrade.get('contract') == 'completed-il-critic-upgrade/1.0.0',
                    'Unrecognized IL upgrade lineage contract')
            lineage(upgrade.get('parent'))
            return
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

    copied, reused, dependencies = (incremental_files(root, run, generation, files, warnings)
                                     if continue_export else (files, [], []))
    manifest = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        run=run.relative_to(root).as_posix(), committed_generation=generation,
        games_done=state.get('games_done'), implementation=state.get('signature', {}).get('implementation'),
        latest_iterations=selected_iterations, uncompacted_iterations=uncompacted,
        incremental=continue_export, reused_files=reused,
        rating_slot_generation=latest_slot['generation'] if latest_slot is not None else None,
        run_lock_present=(run / '.active-run.json').exists(),
        files=[entry for entry, _ in copied.values()],
        missing=[dict(source=path, reason=reason) for path, reason in sorted(missing.items())],
        warnings=sorted(set(warnings)),
        generated=['recorded-config.json', 'export-manifest.json'],
        scope='Analysis reports only; not a resumable checkpoint backup. Missing/pruned evidence is not reconstructed. '
              'Incremental exports require the previous bundles named in reused_files; references point to original physical files.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.report-export-', dir=output.parent) as temporary:
        require(Path(temporary).resolve().is_relative_to((root / 'results').resolve()), 'Temporary export escaped results/')
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        for entry, raw in copied.values():
            destination = stage / entry['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        (stage / 'recorded-config.json').write_bytes(encode(state.get('signature', {}).get('config', {})))
        (stage / 'export-manifest.json').write_bytes(encode(manifest))
        for entry, raw in files.values():
            require(below(root, entry['source']).read_bytes() == raw,
                    f"Report changed during export; retry: {entry['source']}")
        for path, raw in dependencies:
            require(below(root, path).read_bytes() == raw,
                    f'Previous report changed during export; retry: {path.relative_to(root)}')
        publish(stage, output)
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root; defaults to this script\'s project')
    parser.add_argument('--run', default=DEFAULT_RUN, help=f'Self-play run under artifacts/ (default: {DEFAULT_RUN})')
    parser.add_argument('--output', help='New results/ folder; default includes run, generation and UTC timestamp')
    parser.add_argument('--latest', type=int, default=2, help='Number of latest committed iteration reports (default: 2)')
    parser.add_argument('--continue', dest='continue_export', action='store_true',
                        help='Skip unchanged files already in previous same-run exports; record their locations in the manifest')
    parser.add_argument('--extra', action='append', default=[], help='Additional JSON report or directory; repeat as needed')
    args = parser.parse_args()
    try:
        output, manifest = export(args.root, args.run, args.output, latest=args.latest, extras=args.extra,
                                  continue_export=args.continue_export)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Export failed: {error}\n')
    print(f'Copied {len(manifest["files"])} reports/metadata files to {output}')
    print(f'Reused {len(manifest["reused_files"])} unchanged files; latest rating slot: {manifest["rating_slot_generation"]}')
    if manifest['reused_files']:
        print('Keep the previous bundles referenced in export-manifest.json (reused_files).')
    print(f'Committed generation {manifest["committed_generation"]}; iterations {manifest["latest_iterations"]}')
    print(f'Missing files: {len(manifest["missing"])}; warnings: {len(manifest["warnings"])}. See export-manifest.json.')


if __name__ == '__main__':
    main()
