"""Copy a self-play analysis bundle to results/ without importing the trainer.

Run files are preserved. Exported retention history is whitespace-minified and
gzip-compressed without changing any JSON tokens; VP charts are redrawn from history,
and other reports preserve original bytes.
The manifest records both source and exported hashes for transformed history.
Checkpoints, replay shards,
feature caches and profiler traces are not part of this report-only export.
With --continue, unchanged reports remain in earlier bundles and the new
manifest records their verified locations. Detailed reports and teacher
calibration/collection diagnostics use the latest committed iterations only.
External training console logs and launcher failure evidence are not selected.
Small performance and profiling documents (names such as timing-report.json,
lane-startup.json or PROFILE.md) are selected wherever the latest iterations,
evaluations and rating slots keep them, without listing each new file name.
Raw profiler traces, files over 16 MiB and unreadable files are never copied;
the first two are listed in the manifest's missing entries.
Rating slots include only the latest
committed generation; the cumulative archive is still included when present.
When the run's cumulative self-play VP history lacks generations that earlier
exports of the same run (same initial checkpoint) still hold, the exported
ratings/vp-history.json and latest progress_vp.svg are rebuilt from both and
marked as derived in the manifest. The run's own files are never edited.
The latest VP chart is redrawn from valid cumulative VP history using the current renderer.
Teacher VP history and progress_teacher_vp.svg are included when present.
The default run is the current g1190 restart (artifacts/step5/joint7-v11-2-g1190-restart-3); pass --run for another.
A branch-restart run reports the fresh-start seed recorded by its source run (checked by SHA-256), and its first --continue
export copies everything because earlier exports belong to the source run.
The CLI publishes to the independent ai-training-results repository by default.
Fetch/push use the configured results URL directly, independently of origin.
Git URL rewrites must preserve that repository's identity. Authentication may
come from the existing private .env.results-git credential file.
Use --no-push to export locally without Git or network operations.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
import sys

# Keep direct CLI and importlib-based report tests independent of trainer imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_git


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-3'
BRANCH_RESTART = 'rust-branch-restart/1.0.0'
FORMAT = 'selfplay-report-export/1.4.0'
PREVIOUS_FORMATS = {'selfplay-report-export/1.0.0', 'selfplay-report-export/1.1.0',
                    'selfplay-report-export/1.2.0', 'selfplay-report-export/1.3.0', FORMAT}
COMPACT_JSON = 'json-whitespace-only/1.0.0'
GZIP_JSON = 'gzip-json-whitespace-only/1.0.0'
VP_MERGE = 'selfplay-vp-history-merged/1.0.0'
VP_RENDER = 'selfplay-vp-chart-rendered/1.0.0'
DERIVED = (COMPACT_JSON, GZIP_JSON, VP_MERGE, VP_RENDER)
REPORT_NAMES = {'state.json', 'retention-history.json', 'report.json', 'spec.json',
                'config.json', 'best.json', 'summary.json', 'comparison.json',
                'validation-comparison.json', 'receipt.json', 'manifest.json',
                'collection.json', 'calibration.json'}
SKIP_DIRS = {'.git', '.retention', '.retention-trash', '__pycache__', 'population'}
# Performance documents are recognised by name, so a newly added profiler or
# timing report needs no change here. Torch/TensorBoard traces are bulk data.
PERFORMANCE_NAME = re.compile(r'(?:^|[-_.])(?:profile|profiling|profiler|perf|performance|timing|timings|startup|'
                              r'throughput|telemetry|benchmark|latency|bottleneck)(?:[-_.]|$)', re.IGNORECASE)
PERFORMANCE_DOCUMENTS = {'.json': 'json', '.jsonl': 'jsonl', '.md': 'text', '.txt': 'text', '.csv': 'text'}
TRACE_NAME = re.compile(r'\.trace\.json$|\.pb$|\.tfevents', re.IGNORECASE)
PERFORMANCE_MAX_BYTES = 16 * 1024 * 1024


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


def json_bytes(path):
    """Decode a JSON file, accepting .json.gz when the plain path is absent."""
    path = Path(path)
    if not path.is_file() and path.suffix == '.json':
        path = path.with_suffix('.json.gz')
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == '.gz' else raw


def load_json(path):
    """Read old plain reports or new gzip reports using either file name."""
    return json.loads(json_bytes(path).decode('utf-8-sig'))


def compress_history(raw):
    # An empty filename and fixed mtime make identical JSON reusable across
    # exports. GzipFile also avoids platform-dependent header bytes.
    output = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=9, mtime=0) as stream:
        stream.write(raw)
    return output.getvalue()


def compact_json(raw):
    """Minify already validated UTF-8 JSON, preserving exact strings/numbers.

    Do not round-trip through dumps: that can round long decimal tokens or
    collapse duplicate keys. Whitespace inside strings and every escape remain.
    """
    if raw.startswith(b'\xef\xbb\xbf'):raw=raw[3:]
    result=bytearray();quoted=escaped=False
    for byte in raw:
        if quoted:
            result.append(byte)
            if escaped:escaped=False
            elif byte==92:escaped=True
            elif byte==34:quoted=False
        elif byte==34:
            quoted=True;result.append(byte)
        elif byte not in (9,10,13,32):result.append(byte)
    result.append(10)
    return bytes(result)


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


def performance_kind(name):
    """Document type of a small performance report, or None when it is not one.

    The file name itself must say so (timing, startup, profile, ...); a generic
    name such as report.json inside a profiler folder stays unselected, and raw
    traces never count.
    """
    kind = PERFORMANCE_DOCUMENTS.get(Path(name).suffix.lower())
    if kind is None or TRACE_NAME.search(name):
        return None
    return kind if PERFORMANCE_NAME.search(Path(name).stem) else None


def performance_files(root, directory):
    """Yield (path, document type) for performance reports below a directory.

    Links and bulk artifact trees are skipped exactly as in report_files(), but
    profile-* folders are entered: only files whose names say so are yielded.
    """
    directory = below(root, directory)
    if not directory.exists():
        return
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith('.') and d not in SKIP_DIRS)
        for name in directories:
            below(root, Path(parent) / name)
        for name in sorted(files):
            kind = performance_kind(name)
            if kind is not None:
                yield below(root, Path(parent) / name), kind


def ranges(values):
    """[[first, last], ...] for sorted integers; compact manifest evidence."""
    result = []
    for value in sorted(values):
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def vp_plot_module():
    """Import the VP chart helpers lazily; the exporter itself needs no trainer packages."""
    source = str(ROOT / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    from brass_ai.recovery import vp_plot
    return vp_plot


def earlier_vp_rows(vp, directory, initial, limit):
    """Exact self-play VP rows one earlier same-run bundle holds, or None for another lineage.

    Only its small cumulative cache, iteration reports and recent state are read.
    Its large retention history is not parsed: later caches already contain it.
    """
    saved = load_json(directory / 'state.json')
    if vp._initial(saved) != initial:
        return None
    committed = saved['iteration']
    cap = min(limit, committed)
    rows = {}
    cache = directory / vp.HISTORY_PATH
    if cache.is_file():
        history = vp._read(cache)
        if history.get('format') == vp.FORMAT and history.get('initial_checkpoint') == initial:
            for item in history['iterations']:
                generation, totals = item['generation'], vp._totals({'health_by_composition': {'selfplay': item}})
                if type(generation) is int and 1 <= generation <= cap and totals is not None:
                    vp._merge(rows, dict(generation=generation, **totals))

    def add(report, iteration=None):
        index = report.get('iteration')
        require(type(index) is int and (iteration is None or index == iteration), 'Iteration identity differs in VP history')
        totals = vp._totals(report) if 0 <= index < cap else None
        if totals is not None:
            vp._merge(rows, dict(generation=index + 1, **totals))

    for path in sorted(directory.glob('iteration-*/report.json')):
        match = re.fullmatch(r'iteration-(\d+)', path.parent.name)
        if match and int(match[1]) < cap:
            add(load_json(path), int(match[1]))
    for report in saved.get('recent_training', []):
        add(report)
    return committed, rows


def merged_vp_history(root, state, history, bundles, warnings):
    """Run VP history plus generations earlier same-run exports still hold.

    The run's own rows are authoritative. A bundle whose rows conflict with
    them, or with a newer bundle's rows, is skipped whole and named in a warning.
    Returns (merged history, added generations, contributing bundles) or None
    when nothing is missing, so an intact run keeps its exact original bytes.
    """
    vp = vp_plot_module()
    initial = vp._initial(state)
    if not initial or history.get('format') != vp.FORMAT or history.get('initial_checkpoint') != initial:
        return None
    limit = history['committed_generation']
    require(type(limit) is int and limit >= 0, 'Invalid VP history committed generation')
    own = {}
    for item in history['iterations']:
        generation, totals = item['generation'], vp._totals({'health_by_composition': {'selfplay': item}})
        if type(generation) is int and generation >= 1 and totals is not None:
            vp._merge(own, dict(generation=generation, **totals))
    candidates = []
    for directory in bundles:
        try:
            candidates.append((directory, *earlier_vp_rows(vp, directory, initial, limit)))
        except (ValueError, OSError, KeyError, TypeError) as error:
            warnings.append(f'Skipped earlier VP history in {directory.relative_to(root)}: {error}')
    combined, sources = dict(own), []
    # Newest bundle first: it is the better-informed record when two overlap.
    for directory, committed, rows in sorted((c for c in candidates if c[2]), key=lambda c: (c[1], c[0].name), reverse=True):
        trial = dict(combined)
        try:
            for generation in sorted(rows):
                vp._merge(trial, rows[generation])
        except ValueError as error:
            warnings.append(f'Skipped earlier VP history in {directory.relative_to(root)}: {error}')
            continue
        gained = sorted(set(trial) - set(combined))
        if gained:
            sources.append(dict(export=directory.relative_to(root).as_posix(), committed_generation=committed,
                                added_generations=len(gained)))
        combined = trial
    added = sorted(set(combined) - set(own))
    if not added:
        return None
    merged = dict(history, iterations=[combined[g] for g in sorted(combined)])
    return merged, added, sources


def vp_history_bytes(history):
    """The same compact encoding the trainer uses for ratings/vp-history.json."""
    return (json.dumps(history, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def vp_chart_bytes(history, generation):
    vp = vp_plot_module()
    with tempfile.TemporaryDirectory(prefix='.vp-chart-') as temporary:
        output = Path(temporary) / 'progress_vp.svg'
        vp.render(history, output, generation=generation)
        return output.read_bytes()


def context_files(root,run,state):
    """Committed context evidence/visuals only; no raw replays or model tensors."""
    saved=state.get('context_evaluation')
    if saved is None:return
    base=below(root,run/'ratings/contexts')
    index=below(root,base/'index.json')
    if index.is_file():yield index,'json'
    def verified(ref):
        path=below(root,ref['path'])
        require(path.is_relative_to(base) and checksum(path.read_bytes())==ref['sha256'],
                'Context report receipt changed or outside its owned directory')
        return path
    for entry in saved['panels']:
        generation=entry['generation'];directory=base/f'generation-{generation:06d}'
        for key in ('spec','report','games','summaries','status_receipt','final_cost','visualization','visualization_failure'):
            if key not in entry:continue
            path=verified(entry[key]);require(path.parent==directory,'Context receipt generation differs')
            yield path,'jsonl' if path.suffix=='.jsonl' else 'json'
        if entry['status']!='complete':continue
        report=load_json(verified(entry['report']));require(report['generation']==generation,'Context displayed generation differs')
        for ref in report.get('representatives',[]):yield verified(ref),'json'
        markdown=below(root,directory/'report.md')
        if markdown.is_file():yield markdown,'text'
        v=entry.get('visualization')
        if v is not None:
            visual=load_json(verified(v))
            require(visual['generation']==generation and visual['report']==entry['report'],'Context visualization report mismatch')
            if visual['status']=='complete':
                if visual.get('cost'):
                    require(visual['cost']==entry.get('final_cost'),'Context visual timing receipt differs')
                svg=verified(visual['svg'])
                require(svg.parent==directory and f'DOMiNO context evaluation g{generation}' in svg.read_text(encoding='utf-8'),
                    'Context SVG displayed generation differs')
                yield svg,'svg'
    pending=saved.get('pending')
    if pending:
        yield verified(pending['spec']),'json'
        status=below(root,base/f'generation-{pending["generation"]:06d}'/'status.json')
        if status.exists():yield status,'json'
    latest=saved.get('latest_visualization')
    if latest:
        alias=verified(latest['svg']);source=verified(latest['source'])
        require(alias.read_bytes()==source.read_bytes() and f'DOMiNO context evaluation g{latest["generation"]}' in alias.read_text(encoding='utf-8'),
            'Latest context SVG does not match its committed generation')
        yield alias,'svg'


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
            manifest = load_json(path)
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
                # Match exported bytes, not source_sha256: compact history
                # may be identical even if the source's formatting changed.
                key = entry['source'], entry['path'], entry['sha256']
                index.setdefault(key, []).append(path.parent)
        except (ValueError, OSError, TypeError) as error:
            warnings.append(f'Ignored previous export {path.relative_to(root)}: {error}')
    return index


def incremental_files(root,run,generation,files,warnings,*,index=None):
    if index is None:index=previous_files(root,run,generation,warnings)
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


def branch_source(root,run):
    """The run a branch restart was prepared from (its signed config keeps that run's paths), else this run."""
    receipt=below(root,run/'branch-restart.json')
    if not receipt.is_file():return run
    binding=load_json(receipt).get('binding')
    require(isinstance(binding,dict) and binding.get('contract')==BRANCH_RESTART and
            binding.get('destination_run')==run.relative_to(root).as_posix() and isinstance(binding.get('source_run'),str),
            'Branch restart receipt changed or belongs to another run')
    source=below(root,binding['source_run'])
    require(source.is_relative_to(root/'artifacts') and source!=run,'Branch restart source must be another run under artifacts/')
    return source


def seed_report_source(root,run,state,seed,previous,warnings,bindings):
    """Resolve only exact original seed JSON, never seed weights or replays.

    Compact transfers can omit this generation-zero-only document. A committed
    run can still be reported; recover immutable bytes when available, otherwise
    let the normal missing-evidence manifest describe the absent provenance.
    """
    path=below(root,seed['manifest']);expected=seed['sha256']
    origins=[run]
    while True:
        origin=branch_source(root,origins[-1])
        if origin==origins[-1]:break
        require(origin not in origins,'Circular branch restart provenance')
        origins.append(origin)
    require(path in (origin/'seed/manifest.json' for origin in origins),
        'Fresh-start seed manifest changed or belongs to another run')
    if path.is_file():
        require(checksum(path.read_bytes())==expected,'Fresh-start seed manifest changed or belongs to another run')
        return path,None
    transfer=state.get('machine_transfer') or {};rollback=below(root,'.training-rollback')
    if transfer.get('archive_id') and rollback.is_dir():
        for journal in sorted(rollback.glob('restore-*.json')):
            journal=below(root,journal);raw=journal.read_bytes();record=json.loads(raw.decode('utf-8-sig'))
            if record.get('archive_id')!=transfer['archive_id'] or record.get('run')!=run.relative_to(root).as_posix():continue
            identity=checksum(json.dumps({k:v for k,v in record.items() if k!='id'},
                sort_keys=True,separators=(',',':'),allow_nan=False).encode())
            require(record.get('contract')=='step5-inplace-export-revision/1.0.0' and record.get('id')==identity and
                record.get('restore_generation')==transfer.get('origin_generation'),'Rollback report provenance binding changed')
            backup=below(root,record['backup']);require(backup.parent==rollback,'Rollback report backup escaped its directory')
            source_state=below(root,backup/'state.json')
            if not source_state.is_file():
                warnings.append(f'Rollback seed provenance is unavailable: {source_state.relative_to(root)}');continue
            saved_raw=source_state.read_bytes()
            require(checksum(saved_raw)==record['state_sha256'],'Rollback report state changed')
            saved=json.loads(saved_raw.decode('utf-8-sig'))
            require(saved.get('signature',{}).get('config',{}).get('fresh_selfplay_seed')==seed,
                    'Rollback seed receipt differs from the active run')
            candidate=below(root,backup/'seed/manifest.json')
            if candidate.is_file():
                require(checksum(candidate.read_bytes())==expected,'Rollback seed manifest changed')
                bindings.extend([(journal,raw),(source_state,saved_raw)])
                return candidate,'rollback-backup'
    # Only physical reports from verified same-run export manifests are eligible.
    # The state-pinned seed SHA, not file naming/age or a synthesized summary,
    # establishes the identity. Stable ordering keeps incremental reuse flat.
    candidates=set()
    indexes=[previous()]
    # Successive restarts can retain the original ancestor's seed receipt.
    indexes.extend(previous_files(root,origin,state['iteration'],warnings) for origin in origins[1:])
    for index in indexes:
        for (_,name,h),directories in index.items():
            if name=='fresh-start.json' and h==expected:
                candidates.update(directory/name for directory in directories)
    for candidate in sorted(candidates):
        candidate=below(root,candidate)
        if candidate.is_file() and checksum(candidate.read_bytes())==expected:return candidate,'previous-report'
        warnings.append(f'Previous seed report unavailable or changed: {candidate.relative_to(root)}')
    return None,None


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
    provenance_bindings=[];previous_index=None
    def previous():
        nonlocal previous_index
        if previous_index is None:previous_index=previous_files(root,run,generation,warnings)
        return previous_index
    selected_iterations = list(range(max(0, generation - latest), generation))

    def add_performance(directory, reason):
        """Small performance reports only; one unreadable or oversized file never blocks the export."""
        for path, kind in performance_files(root, directory):
            source = path.relative_to(root).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                continue  # Removed while the walk was running.
            if size > PERFORMANCE_MAX_BYTES:
                missing[source] = f'Performance file is over the {PERFORMANCE_MAX_BYTES // 2**20} MiB export limit; not copied'
                continue
            try:
                add(path, path.relative_to(run), reason, document=kind)
            except ValueError as error:
                warnings.append(f'Skipped unreadable performance file {source}: {error}')

    def add(path, destination, reason, *, expected=False, document='json'):
        path = below(root, path)
        name = Path(destination).as_posix()
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe bundle path')
        if name == 'retention-history.json' and path == run/'retention-history.json' and not path.is_file():
            compressed = below(root, path.with_suffix('.json.gz'))
            if compressed.is_file():
                path = compressed
        if not path.is_file():
            if expected:
                missing[path.relative_to(root).as_posix()] = reason
            return None
        source_raw = path.read_bytes()
        raw = gzip.decompress(source_raw) if document == 'json' and path.suffix == '.gz' else source_raw
        # Validate complete documents before publication. Only explicitly
        # selected compact rating outcomes/curves may use these other formats.
        if document == 'json':
            value = json.loads(raw.decode('utf-8-sig'))
        elif document == 'jsonl':
            value = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
        elif document == 'svg':
            import xml.etree.ElementTree as ET
            ET.fromstring(raw); value = None
        elif document == 'text':
            raw.decode('utf-8'); value = None
        else:
            raise ValueError('Unknown report document format')
        compact_history=(document=='json' and name=='retention-history.json'
                         and path in (run/'retention-history.json',run/'retention-history.json.gz'))
        if compact_history:
            uncompressed=compact_json(raw)
            raw=compress_history(uncompressed)
            name+='.gz'
        entry = dict(source=path.relative_to(root).as_posix(), path=name,
                     bytes=len(raw), sha256=checksum(raw), reason=reason)
        if compact_history:
            entry.update(transform=GZIP_JSON,source_bytes=len(source_raw),
                         source_sha256=checksum(source_raw),
                         uncompressed_bytes=len(uncompressed),uncompressed_sha256=checksum(uncompressed))
        if name in files:
            previous, previous_raw = files[name]
            # A report may serve both the active anchor and the latest rating
            # slot. Selection reasons describe its uses, not file identity.
            require(all(previous[key] == entry[key] for key in ('source', 'path', 'bytes', 'sha256'))
                    and previous.get('source_sha256')==entry.get('source_sha256')
                    and previous_raw == raw, f'Conflicting export destination: {name}')
        else:
            files[name] = (entry, raw)
        return value

    add(state_path, 'state.json', 'Committed configuration, lineage and progress', expected=True)
    require(files['state.json'][1] == state_bytes, 'Run advanced during export; retry')
    seed=state.get('signature',{}).get('config',{}).get('fresh_selfplay_seed')
    if seed is not None:
        path,recovered=seed_report_source(root,run,state,seed,previous,warnings,provenance_bindings)
        if path is None:
            add(seed['manifest'],'fresh-start.json','Original seed provenance omitted/pruned; its exact receipt remains in state.json',expected=True)
            warnings.append('Original seed manifest is absent; no seed document or tensors were reconstructed. See state.json fresh_selfplay_seed receipt.')
        else:
            add(path,'fresh-start.json','Weights-only source and imported history cutoff; no model tensors',expected=True)
            require(files['fresh-start.json'][0]['sha256']==seed['sha256'],'Fresh-start seed changed during report selection')
            if recovered:
                files['fresh-start.json'][0].update(recovered_from=recovered,recorded_source=seed['manifest'])
    transfer=state.get('machine_transfer') or {}
    for key in ('instruction','original_state'):
        if key not in transfer:continue
        ref=transfer[key];path=below(root,ref['path'])
        require(path.is_relative_to(run),'Machine-transfer report evidence is outside this run')
        if path.is_file():require(checksum(path.read_bytes())==ref['sha256'],'Machine-transfer report evidence changed')
        value=add(path,path.relative_to(run),'Checkpoint-bound reward revision and fresh-buffer adoption evidence',expected=True)
        if value is not None:require(files[path.relative_to(run).as_posix()][0]['sha256']==ref['sha256'],
                                   'Machine-transfer report evidence changed during selection')
    domino=state.get('domino_adoption')
    if domino is not None:
        add(run / f'domino-adoption-g{domino["generation"]:06d}.json', 'domino-adoption.json',
            'DOMiNO transition and archived PPG receipts; no cache arrays', expected=True)
    local_scoring_receipt=None
    if state.get('signature',{}).get('config',{}).get('local_scoring_auxiliary') is not None:
        # Small training contracts/manifests only: never copy label arrays or
        # the multi-GB PPG cache. Per-iteration metrics already live in reports.
        path=below(root,'training/runtime/local-scoring/ordinary/runtime.json')
        add(path,'contracts/local-scoring-runtime.json','Separately admitted scoring recorder',expected=True)
        receipt=state.get('local_scoring_adoption')
        if receipt is not None:
            compact={k:receipt[k] for k in ('contract','origin','config_sha256','checkpoint','old_schedule',
                'old_epoch_control','old_cache_receipts_sha256','new_origin','next_phase','cleanup_complete','cleanup')}
            if 'old_cache_metadata' in receipt:compact['old_cache_metadata']=receipt['old_cache_metadata']
            local_scoring_receipt=compact
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
        add_performance(directory, 'Latest iteration performance/profiling report')
    for path in report_files(root, run / 'evaluations'):
        add(path, path.relative_to(run), 'Evaluation report or specification')
    add_performance(run / 'evaluations', 'Evaluation performance/profiling report')
    for child in sorted(run.glob('*')):
        # Run-level profiler folders such as profile-collection/ contribute only named performance reports.
        if child.is_dir() and not child.name.startswith('.') and (child.name.lower().startswith('profile')
                                                                   or PERFORMANCE_NAME.search(child.name)):
            add_performance(child, 'Run-level performance/profiling report')
    vp_history=run/'ratings'/'vp-history.json'
    if vp_history.is_file():
        add(vp_history,vp_history.relative_to(run),'Cumulative ordinary self-play learner VP totals')
    teacher_vp_history=run/'ratings'/'teacher-vp-history.json'
    if teacher_vp_history.is_file():
        add(teacher_vp_history,teacher_vp_history.relative_to(run),'Cumulative teacher VP totals by exploration group')
    for path in report_files(root, run / 'il-kl'):
        add(path, path.relative_to(run), 'Fixed IL-KL probe manifest or initial measurement')
    for path in report_files(root,run/'policy-anchor'):
        add(path,path.relative_to(run),'Qualified KL reference transition report')
    for path in report_files(root,run/'exploration'):
        add(path,path.relative_to(run),'Exploration configuration transition report')
    active_anchor=(state.get('policy_anchor') or {}).get('active')
    if active_anchor:
        for reference in [active_anchor['cache'],active_anchor['fit']['report'],active_anchor['fit']['spec']]:
            path=below(root,reference['path'])
            require(path.is_relative_to(run) and checksum(path.read_bytes())==reference['sha256'],
                'Active policy-anchor report evidence changed or outside this run')
            add(path,path.relative_to(run),'Active KL reference evidence or probability-cache manifest')
    # Old slot folders may still exist after retention or alongside a pending
    # evaluation. Only the latest slot committed in state belongs in this export.
    slots = state.get('ratings', {}).get('slots', [])
    require(isinstance(slots, list) and all(isinstance(slot, dict) and
            type(slot.get('generation')) is int and 0 <= slot['generation'] <= generation
            for slot in slots), 'Invalid committed rating slots')
    latest_slot = max(slots, key=lambda slot: slot['generation'], default=None)
    for path in report_files(root, run / 'ratings', skip_dirs={'slots','contexts'}):
        add(path, path.relative_to(run), 'Plackett-Luce strength report or specification')
    for path,document in context_files(root,run,state):
        add(path,path.relative_to(run),'Report-only DOMiNO context evidence or visualization',document=document)
    archive = state.get('ratings', {}).get('archive')
    if archive is not None:
        path = below(root, archive['path'])
        require(path.is_relative_to(run / 'ratings'), 'Rating archive points outside this run')
        require(checksum(path.read_bytes()) == archive['sha256'], 'Archived rating evidence changed')
        add(path, path.relative_to(run), 'Compact cumulative rating outcomes and diagnostic-panel history')
    # Retired-panel evidence remains useful until it moves into the compact
    # archive. Export its committed slot even when it is no longer the latest.
    triggers=state.get('ratings',{}).get('panel_gates',{}).get('retired',{}).values()
    needed={t['generation'] for t in triggers if t is not None}
    coverage=state.get('ratings',{}).get('coverage',{}).get('activated')
    # A manually authorized trained restart has no measured panel evidence.
    # Its override remains in state.json; do not expect or invent a rating slot.
    if coverage:needed.update(t['generation'] for t in coverage.get('panels',{}).values())
    if latest_slot is not None:needed.add(latest_slot['generation'])
    for slot in [s for s in slots if s['generation'] in needed]:
        slot_directory = below(root, Path(slot['report']['path']).parent)
        require(slot_directory.parent == run / 'ratings' / 'slots' and
                re.fullmatch(r'generation-0*' + str(slot['generation'])+r'(?:-source-[0-9a-f]{16})?', slot_directory.name),
                'Latest rating slot directory differs from its committed generation')
        for path in report_files(root, slot_directory):
            add(path, path.relative_to(run), 'Latest committed rating slot report or specification')
        add_performance(slot_directory, 'Latest committed rating slot performance/profiling report')
        for key in ('records', 'panels', 'report', 'spec'):
            reference = slot[key]
            path = below(root, reference['path'])
            require(path.is_relative_to(slot_directory), 'Rating evidence points outside its slot')
            require(checksum(path.read_bytes()) == reference['sha256'], 'Committed rating evidence changed')
            if path.relative_to(run).as_posix() not in files:
                add(path, path.relative_to(run), 'Append-only held-out rating outcomes or separate panel results',
                    document='jsonl' if key == 'records' else 'json')
        for name,reason in (('progress.svg','Strength progress curve'),
                            ('progress_vp.svg','Ten-iteration self-play learner VP curve'),
                            ('progress_teacher_vp.svg','Teacher VP by cotton exploration and overall')):
            curve = below(root, Path(slot['report']['path']).parent / name)
            if curve.exists():
                add(curve, curve.relative_to(run), reason, document='svg')
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
    for receipt in state.get('teacher_phase',{}).get('buffer',{}).get('receipts',[]):
        shard=below(root,receipt['path'],) if receipt['path'].startswith('artifacts/') else run/receipt['path']
        sidecar=Path(str(shard)+'.receipt.json')
        add(sidecar,sidecar.relative_to(run),'Sealed teacher contexts/evaluator/fit provenance; excludes label replay',expected=True)

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

    lineage(state.get('origin'))
    lineage(state.get('signature', {}).get('imitation_source'))
    for extra in extras:
        path = below(root, extra)
        require(path.exists(), f'Extra report path does not exist: {path}')
        for item in report_files(root, path) if path.is_dir() else [path]:
            require(item.suffix == '.json', '--extra accepts JSON reports or report directories')
            add(item, Path('related') / item.relative_to(root), 'Explicit related report')

    vp_merge = None
    history_name = vp_history.relative_to(run).as_posix()
    if history_name in files:
        # A restore can leave the run's cumulative VP cache short while earlier
        # exports still hold the missing generations. Rebuild only the exported copies.
        try:
            bundles = sorted({d for directories in previous().values() for d in directories
                              if (d / 'state.json').is_file()})
            history = json.loads(files[history_name][1].decode('utf-8-sig'))
            merged = merged_vp_history(root, state, history, bundles, warnings)
            vp = vp_plot_module()
            valid_history = (vp._initial(state) and history.get('format') == vp.FORMAT
                             and history.get('initial_checkpoint') == vp._initial(state))
            chart = None
            if valid_history and latest_slot is not None:
                chart = (below(root, Path(latest_slot['report']['path']).parent) / 'progress_vp.svg').relative_to(run).as_posix()
                chart = chart if chart in files else None
            if merged is not None:
                history, added, sources = merged
            svg = vp_chart_bytes(history, latest_slot['generation']) if chart is not None else None
        except (ImportError, ValueError, OSError, KeyError, TypeError) as error:
            warnings.append(f'VP history was not rebuilt from earlier exports: {error}')
        else:
            def derive(name, raw, reason, transform=VP_MERGE):
                entry = files[name][0]
                files[name] = (dict(entry, bytes=len(raw), sha256=checksum(raw), reason=reason, transform=transform,
                                    source_bytes=entry['bytes'], source_sha256=entry['sha256']), raw)
            if merged is not None:
                derive(history_name, vp_history_bytes(history),
                       'Cumulative self-play VP totals completed from earlier exports of this run')
                if svg is not None:
                    derive(chart, svg, 'Self-play learner VP curve redrawn from the completed history')
                vp_merge = dict(contract=VP_MERGE, run_generations=len(history['iterations']) - len(added),
                                added_generations=len(added), added_ranges=ranges(added), sources=sources,
                                chart=chart)
                warnings.append(f'Run ratings/vp-history.json lacked {len(added)} generations held by earlier exports; '
                                f'the exported history and chart combine both (see vp_history_merge)')
            elif svg is not None:
                derive(chart, svg, 'Self-play learner VP curve redrawn with missing history groups omitted', VP_RENDER)
    copied, reused, dependencies = (incremental_files(root,run,generation,files,warnings,index=previous())
                                     if continue_export else (files, [], []))
    manifest = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        run=run.relative_to(root).as_posix(), committed_generation=generation,
        games_done=state.get('games_done'), implementation=state.get('signature', {}).get('implementation'),
        latest_iterations=selected_iterations, uncompacted_iterations=uncompacted,
        incremental=continue_export, reused_files=reused,
        rating_slot_generation=latest_slot['generation'] if latest_slot is not None else None,
        run_lock_present=(run / '.active-run.json').exists(),
        vp_history_merge=vp_merge,
        files=[entry for entry, _ in copied.values()],
        missing=[dict(source=path, reason=reason) for path, reason in sorted(missing.items())],
        warnings=sorted(set(warnings)),
        generated=['recorded-config.json', 'export-manifest.json']+(['local-scoring-adoption.json'] if local_scoring_receipt is not None else []),
        scope='Analysis reports only; not a resumable checkpoint backup. Missing/pruned evidence is not reconstructed. '
              'Retention history is gzip-compressed whitespace-minified JSON (.json.gz); load_json accepts both formats. '
              'Its source_bytes/source_sha256 describe the original '
              'and bytes/sha256 describe the exported file. VP history/chart transforms also record source hashes. Other reports preserve original bytes. '
              'Latest teacher collection/calibration reports are included; external console/failure logs are excluded. '
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
        if local_scoring_receipt is not None:(stage/'local-scoring-adoption.json').write_bytes(encode(local_scoring_receipt))
        (stage / 'export-manifest.json').write_bytes(encode(manifest))
        for entry, raw in files.values():
            current=below(root, entry['source']).read_bytes()
            matches=(len(current)==entry['source_bytes'] and checksum(current)==entry['source_sha256']
                     if entry.get('transform') in DERIVED else current==raw)
            require(matches,
                    f"Report changed during export; retry: {entry['source']}")
        for path, raw in dependencies:
            require(below(root, path).read_bytes() == raw,
                    f'Previous report changed during export; retry: {path.relative_to(root)}')
        for path,raw in provenance_bindings:
            require(below(root,path).read_bytes()==raw,
                    f'Report provenance changed during export; retry: {path.relative_to(root)}')
        publish(stage, output)
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root; defaults to this script\'s project')
    parser.add_argument('--run', default=DEFAULT_RUN, help=f'Self-play run under artifacts/ (default: {DEFAULT_RUN})')
    parser.add_argument('--output', help='New results/ folder; default includes run, generation and UTC timestamp')
    parser.add_argument('--latest', type=int, default=2, help='Number of latest detailed committed iterations (default: 2)')
    parser.add_argument('--continue', dest='continue_export', action='store_true',
                        help='Reuse unchanged same-run exported files with verified locations; keep the --latest detail limit')
    parser.add_argument('--extra', action='append', default=[], help='Additional JSON report or directory; repeat as needed')
    parser.add_argument('--no-push', action='store_true', help='Local export only; do not initialize/fetch/commit/push results Git')
    parser.add_argument('--push-only', action='store_true', help='Retry committed results uploads without exporting again')
    parser.add_argument('--results-remote', default=results_git.REMOTE, help='Independent results repository URL')
    args = parser.parse_args()
    try:
        require(not (args.no_push and args.push_only), '--push-only cannot be combined with --no-push')
        if args.push_only:
            with results_git.transaction(args.root, args.results_remote) as directory:
                commit = results_git.push(args.root, directory, args.results_remote)
                print(f'Uploaded results commit {commit} to {results_git.display(args.results_remote)}')
            return
        def collect():
            return export(args.root, args.run, args.output, latest=args.latest, extras=args.extra,
                          continue_export=args.continue_export)
        if args.no_push:
            output, manifest = collect()
        else:
            with results_git.transaction(args.root, args.results_remote) as directory:
                output, manifest = collect()
                commit = results_git.publish(args.root, directory, output, manifest, args.results_remote)
                print(f'Uploaded results commit {commit} to {results_git.display(args.results_remote)}')
    except (ValueError, OSError, EOFError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f'Export failed: {error}\n')
    print(f'Copied {len(manifest["files"])} reports/metadata files to {output}')
    print(f'Reused {len(manifest["reused_files"])} unchanged files; latest rating slot: {manifest["rating_slot_generation"]}')
    if manifest['reused_files']:
        print('Keep the previous bundles referenced in export-manifest.json (reused_files).')
    print(f'Committed generation {manifest["committed_generation"]}; iterations {manifest["latest_iterations"]}')
    print(f'Missing files: {len(manifest["missing"])}; warnings: {len(manifest["warnings"])}. See export-manifest.json.')


if __name__ == '__main__':
    main()
