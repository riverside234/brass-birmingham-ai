"""Process orchestration for the unchanged, fully audited raw FE converter.

Keep scheduling outside feature_engineering: worker counts do not change the
feature contract or invalidate model checkpoints/encoded data.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
import io
import multiprocessing
import os
from pathlib import Path
import time

from brass_ai.bank import sha
from brass_ai.learning.contracts import compatibility, project_path, read_json, require


MAX_WORKERS = 48


def available_cpus():
    """Respect process affinity and a lab scheduler's per-task allocation."""
    count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    if hasattr(os, 'sched_getaffinity'):
        try:
            count = min(count, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    allocation = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if allocation.isdecimal() and int(allocation) > 0:
        count = min(count, int(allocation))
    return max(1, count)


def worker_count(workers, shards):
    require(workers is None or type(workers) is int and 1 <= workers <= MAX_WORKERS,
            f'Conversion workers must be an integer from 1 to {MAX_WORKERS}')
    require(type(shards) is int and shards > 0, 'Empty demonstration manifest')
    return min(workers or MAX_WORKERS, available_cpus(), shards)


def _worker_init():
    # A separate tensor thread pool per process would oversubscribe the CPU.
    from brass_ai.recovery.demonstrations import _loader_init
    _loader_init()


def _convert_shard(task):
    """Reuse all existing validation; transfer only audited files and metadata."""
    from brass_ai.feature_engineering.data import convert as serial_convert
    from brass_ai.recovery.replay import _cache_paths
    from brass_ai.step5.league import write_json

    root, output, index, name, selected, manifest, mono = task
    root = Path(root)
    out = project_path(root, output)
    work = out / 'work'
    source = work / f'source-{index:06d}.json'
    temporary = work / f'converted-{index:06d}'
    started = time.monotonic()
    write_json(source, dict(manifest, entries=selected))
    # Child converters report a one-shard progress line. The coordinator reports
    # real dataset progress instead of interleaved "1/1" messages from workers.
    with redirect_stdout(io.StringIO()):
        receipt = serial_convert(root, source, temporary, mono=mono)
    result = read_json(temporary / 'manifest.json')
    entries = result['entries']
    old = project_path(root, entries[0]['path'])
    require(all(e['path'] == entries[0]['path'] for e in entries),
            'Worker converted more than one source shard')
    target = out / 'raw' / f'shard-{index:06d}.jsonl.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    for before, after in zip((old, *_cache_paths(old)), (target, *_cache_paths(target))):
        require(not after.exists(), 'Converted shard already exists')
        before.rename(after)
    for entry in entries:
        entry['path'] = target.relative_to(root).as_posix()
    # These are only this worker's new, temporary metadata/files, never inputs.
    for path in (temporary / 'manifest.json', temporary / 'conversion.json', source):
        path.unlink()
    old.parent.rmdir()
    temporary.rmdir()
    return dict(entries=entries, inputs=receipt['inputs'], preserved=receipt['preserved'],
                source=name, pid=os.getpid(), seconds=time.monotonic() - started)


def convert(root, manifest_path, output, *, mono=None, workers=None):
    """Audit independent replay shards concurrently; commit in source order.

    Workers own separate rules hosts and output files. No feature arrays or raw
    replay records cross process boundaries. The final manifest is parent-owned
    and is never published after a worker/duplicate/source-integrity failure.
    """
    from brass_ai.feature_engineering.data import raw_compatible
    from brass_ai.recovery.demonstrations import _manifest
    from brass_ai.step5.league import write_json

    root = Path(root).resolve()
    source = project_path(root, manifest_path)
    out = project_path(root, output)
    require(not out.exists(), 'Conversion requires a new output directory; old files are preserved')
    source_sha256 = sha(source)
    manifest = read_json(source)
    require(manifest['format'] == 'bc-dataset/1.0.0'
            and manifest['label_kind'] == 'qualified-v3-choice/1.0.0'
            and manifest['context'] == 'neutral-predecision/1.0.0',
            'Unsupported demonstration manifest')
    raw_compatible(manifest['compatibility'])
    grouped = {}
    for entry in manifest['entries']:
        require(entry['transitions_verified'], 'Source demonstrations were not audited')
        grouped.setdefault(entry['path'], []).append(entry)
    width = worker_count(workers, len(grouped))
    # Do not copy the entire dataset manifest to each worker/task.
    header = {k: v for k, v in manifest.items() if k not in ('entries', 'excluded')}
    tasks = [(str(root), str(out), i, name, selected, header, mono)
             for i, (name, selected) in enumerate(grouped.items())]
    parts = [None] * len(tasks)
    started = time.monotonic()
    out.mkdir(parents=True)
    print(f'feature conversion: {len(tasks)} shards, {width} CPU workers '
          f'(maximum {MAX_WORKERS}); full native audits enabled', flush=True)
    failing = None
    try:
        if width > 1:
            executor = ProcessPoolExecutor(max_workers=width,
                mp_context=multiprocessing.get_context('spawn'), initializer=_worker_init)
            pending = {}
            remaining = iter(enumerate(tasks))

            def submit():
                item = next(remaining, None)
                if item is not None:
                    index, task = item
                    pending[executor.submit(_convert_shard, task)] = index

            try:
                # Bound queued work as well as live workers and IPC payloads.
                for _ in range(min(len(tasks), 2 * width)):
                    submit()
                completed = 0
                while pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        index = pending.pop(future)
                        failing = tasks[index][3]
                        parts[index] = future.result()
                        completed += 1
                        print(f'feature conversion: {completed}/{len(tasks)} shards audited; '
                              f'{time.monotonic()-started:.1f}s elapsed', flush=True)
                        submit()
            finally:
                # Let live audits close their private native hosts, cancel tasks
                # still queued, and never leave workers running after failure.
                executor.shutdown(wait=True, cancel_futures=True)
        else:
            for index, task in enumerate(tasks):
                failing = task[3]
                parts[index] = _convert_shard(task)
                print(f'feature conversion: {index+1}/{len(tasks)} shards audited; '
                      f'{time.monotonic()-started:.1f}s elapsed', flush=True)
        entries, originals, seen, identities, groups = [], [], set(), set(), {}
        for part in parts:
            failing = part['source']
            for entry in part['entries']:
                require(entry['semantic_sha256'] not in seen, 'Duplicate semantic source game')
                require(entry['game_id'] not in identities, 'Duplicate source game ID')
                require(groups.setdefault(entry['setup_group'], entry['split']) == entry['split'],
                        'Setup family crosses partitions')
                seen.add(entry['semantic_sha256'])
                identities.add(entry['game_id'])
                entries.append(entry)
            originals.extend(part['inputs'])
        require(sha(source) == source_sha256, 'Source manifest changed during conversion')
        (out / 'work').rmdir()
        execution = dict(workers=width, requested_workers=workers, maximum_workers=MAX_WORKERS,
            available_cpus=available_cpus(), shards=len(tasks),
            worker_pids=sorted({p['pid'] for p in parts}), seconds=time.monotonic()-started)
        result = _manifest(root, out/'manifest.json', entries, [], manifest['teachers'],
                           manifest.get('calibration'), width)
        receipt = dict(format='heuristic-raw-to-features2/1.0.0',
            source_manifest=source.relative_to(root).as_posix(), source_manifest_sha256=source_sha256,
            feature_version=compatibility()['feature_version'], compatibility=compatibility(),
            inputs=originals, games=len(entries), partitions=result['partitions'],
            native_transitions_verified=True, preserved=parts[0]['preserved'],
            neural_or_search_label_reuse=False, execution=execution)
        write_json(out/'conversion.json', receipt)
        return receipt
    except BaseException as error:
        # The final manifest must not make partial results eligible for training.
        (out/'manifest.json').unlink(missing_ok=True)
        write_json(out/'failure.json', dict(source=failing, error=str(error), committed=False,
            remedy='Inspect the failed shard and rerun into a new output directory; source files are unchanged.'))
        if isinstance(error, Exception):
            raise ValueError(f'Feature conversion failed for {failing}: {error}') from error
        raise
