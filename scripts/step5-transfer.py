"""Byte-preserving transfer of a stopped Step 5 run.

Default: Git code plus bundled private runtime, in multithreaded tar.zst
(backports.zstd on Python < 3.14). Publish through GitHub Release assets.
Legacy ZIP export/restore remains available with only the standard library.

The transport itself stays outside the training fingerprint. The shared path
resolver is source-bound and older trainers need a reviewed source transition.
No checkpoints or state signatures are rewritten and no training is launched.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import socket
import stat
import subprocess
import sys
import tarfile
import uuid
import zipfile
from bisect import bisect_right

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brass_ai.learning.paths import (RELOCATION_FILE, confined, make_relocation, read_relocation,
    reference_suffix, resolve_project_path, root_key, source_roots, validate_roots)

FORMAT = 'step5-private-transfer/1.1.0'
OLD_FORMAT = 'step5-private-transfer/1.0.0'
PARTS_FORMAT = 'step5-transfer-parts/1.0.0'
SOURCE_RECIPE = 3  # 1: Python; 2: hosts/build inputs; 3: bank/legacy path resolvers.
CURRENT_SOURCE_RECIPE = 4  # Shared complete training inventory, including Rust.
RELOCATION_VERSION = 'step5-project-relocation/1.1.0'
SKIP = {'__pycache__', '.git', '.active-run.json', '.run-locks', 'target'}
CODE_TREES = ('src', 'scripts', 'configs', 'training/opponents', 'training/runtime', 'lab-native')
CHUNK = 1024 * 1024
ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(temporary, path):
    """Atomic publication that cannot overwrite a racing destination file."""
    if os.name == 'nt':
        temporary.rename(path)  # Windows rename fails if the destination exists.
    else:
        os.link(temporary, path)
        temporary.unlink()


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name, 'Invalid portable path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name
            and all(x not in ('.', '..') and not x.endswith((' ', '.')) for x in p.parts), 'Unsafe archive path: ' + name)
    require(not any(x.casefold() in ('.git', '.aws', '.ssh', '.codex', '.agents') for x in p.parts), 'Private configuration path forbidden')
    require(not any(any(c in x for c in '<>"|?*') or x.split('.')[0].upper() in
                    {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
                    for x in p.parts), 'Non-portable archive path: ' + name)
    return p


def below(root, name):
    root = Path(root).resolve()
    p = resolve_project_path(root, name)
    require(p != root, 'Expected a path below project root: ' + str(name))
    return p


def source_identity(root, *, recipe=None):
    """The current implementation_id recipe, without importing torch/runtimes."""
    if recipe is None:
        recipe = (CURRENT_SOURCE_RECIPE if (Path(root)/'src/brass_ai/learning/source_identity.py').is_file()
                  else SOURCE_RECIPE)
    require(recipe in (1, 2, 3, 4), 'Unsupported implementation fingerprint recipe')
    if recipe == 4:
        from brass_ai.learning.source_identity import implementation_id
        return implementation_id(root)
    root = Path(root)
    base = root / 'src/brass_ai'
    paths = [p for group in ('step5', 'recovery', 'search', 'learning') for p in (base / group).glob('*.py')]
    paths += [base / p for p in ('learning/runner.py', 'learning/batching.py', 'searchlab/spiel.py', 'beliefs.py', 'runtime_contracts.py')]
    if recipe >= 2:
        paths += [base / 'host.py']
        paths += sorted((root / 'src/BrassTrainingHost').glob('*.cs'))
        paths += [root / 'scripts/build-training-host.py', root / 'scripts/build-search-host.py']
        paths += sorted((root / 'training/runtime/throughput').glob('*/runtime.json'))
    if recipe >= 3:
        paths += [base / 'bank.py', base / 'legacy.py']
    paths += sorted((root / 'configs/contracts/masked-ismcts').glob('*.json'))
    paths += [root / 'configs/references/masked-ismcts.lock.json']
    return digest({p.relative_to(root).as_posix(): p.read_text(encoding='utf-8').replace('\r\n', '\n') for p in sorted(paths)})


def active(run):
    path = run / '.active-run.json'
    if not path.exists():
        return False
    lock = read(path)
    if lock['host'] != socket.gethostname():
        return True
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, lock['pid'])
        if not handle:
            return ctypes.get_last_error() == 5
        kernel.CloseHandle(ctypes.c_void_p(handle))
        return True
    try:
        os.kill(lock['pid'], 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def lease(run):
    require(not active(run), 'Run is active; stop training before transfer: ' + str(run))
    run.mkdir(parents=True, exist_ok=True)
    lock = run / '.active-run.json'
    if lock.exists():
        lock.unlink()  # only a verified dead local process
    token = uuid.uuid4().hex
    with lock.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(), token=token), f)
    try:
        yield
    finally:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()


def inventory(root, run, *, mode='compact'):
    """Keep live continuation dependencies; full mode also retains old artifacts."""
    require(mode in ('compact', 'full'), 'Unknown transfer mode')
    root = Path(root).resolve()
    run = below(root, run)
    require(run.is_relative_to(root / 'artifacts'), 'Expected an owned artifacts run')
    state = read(run / 'state.json')
    require(state.get('format') == 'step5-recovery-league/2.2.0', 'Unsupported self-play state')
    native = state.get('native_backend') is not None
    require(not native or mode == 'compact', 'Use compact mode for native transfer: replay buffers and original IL caches must be omitted')
    recipe = CURRENT_SOURCE_RECIPE if native else SOURCE_RECIPE
    actual_source = source_identity(root, recipe=recipe)
    qualified_source = actual_source
    if native and state['signature']['implementation'] != actual_source:
        review = read(root/'configs/learning/b07-replay-location-review.json')
        if review['contract'] == 'brass-b07-replay-location-review/1.0.0' and review['execution_implementation'] == actual_source:
            qualified_source = review['qualified_implementation']
    require(state['signature']['implementation'] in (actual_source, qualified_source),
            'Source differs from saved run. Complete the supported reviewed source continuation on the old machine '
            'before export, or use matching source. Transfer does not authorize a source upgrade.')
    pending = run / f"iteration-{state['iteration']:04d}"
    require(not pending.exists() or (pending.is_dir() and not any(pending.iterdir())),
            'Uncommitted iteration: finish/recover it on the source machine before export')
    require(not (run / f"generation-{state['iteration'] + 1:04d}.pt").exists(), 'Uncommitted checkpoint; recover before export')
    files = {}
    runs = {run}
    hosts = {below(root, state['signature']['config']['host'])}
    def add(path, expected=None, *, rating_summary=False):
        path = below(root, path)
        if native and path.is_relative_to(root/'artifacts'):
            if rating_summary:
                # Only the typed rating importer below may request this. Keep
                # the exact state receipt: no arbitrary JSONL/replay exception.
                committed={below(root,s['records']['path']):s['records']['sha256']
                    for s in state.get('ratings',{}).get('slots',[])}
                require(path in committed and expected==committed[path] and
                    path.name=='games.jsonl' and path.parent.parent==run/'ratings/slots',
                    'Rating summary is outside its committed slot receipt')
            require('.prepared' not in path.relative_to(root).parts and
                    (not path.name.endswith(('.jsonl.gz','.jsonl')) or rating_summary),
                'Replay/preparation data must not enter this transfer: '+str(path))
        if native and path.is_relative_to(run):
            parts = path.relative_to(run).parts
            require(parts[0] not in ('.ppg-eight-cache', '.ppg-value-cache', 'teacher-labels', 'search-values')
                    and parts[:2] != ('il-kl', '.prepared'), 'Replay/preparation cache must not enter this transfer: '+str(path))
        require(path.is_file(), 'Missing dependency: ' + str(path))
        name = path.relative_to(root).as_posix()
        relative(name)
        if expected:
            require(sha(path) == expected, 'Dependency hash differs: ' + name)
        files[name] = path
    def tree(path):
        path = below(root, path)
        require(path.is_dir(), 'Missing directory: ' + str(path))
        for base, dirs, names in os.walk(path, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP)
            for d in dirs:
                below(root, Path(base) / d)
            for name in sorted(names):
                if name not in SKIP and not name.endswith(('.pyc', '.tmp')):
                    add(Path(base) / name)
    for prefix in CODE_TREES:
        tree(prefix)
    for path in root.glob('requirements*.txt'):
        add(path)
    if mode == 'full':
        tree(run)
    else:
        add(run / 'state.json')
        for name in ('best.json', 'best.pt', 'pins.json'):
            if (run / name).is_file():
                add(run / name)
        # Diagnostics/evidence stay readable without copying consumed PPO and
        # teacher replay. Unfinished evaluations are different: keep their cache
        # so continuation cannot silently recollect a different gate.
        latest_report = run / f"iteration-{state['iteration'] - 1:04d}" / 'report.json'
        if latest_report.is_file():
            add(latest_report)
        if (run / 'pins.json').is_file():
            for name in read(run / 'pins.json')['paths']:
                add(name)
    # A pending screen can select a different host or models outside this run.
    # Full mode includes its local files, but still needs these dependencies.
    for folder in sorted((run / 'evaluations').glob('*')):
        if not folder.is_dir():
            continue
        if (folder / 'spec.json').is_file() and not (folder / 'report.json').is_file():
            tree(folder)
            spec = read(folder / 'spec.json')
            if spec.get('config', {}).get('host'):
                hosts.add(below(root, spec['config']['host']))
            def references(value):
                if isinstance(value, dict):
                    if isinstance(value.get('path'), str) and value.get('sha256'):
                        add(value['path'], value['sha256'])
                    for child in value.values():
                        references(child)
                elif isinstance(value, list):
                    for child in value:
                        references(child)
            references(spec)
        else:
            for name in ('spec.json', 'report.json'):
                if (folder / name).is_file():
                    add(folder / name)
    live = state.get('resume', []) + state.get('population', []) + [state['history'][-1], state['evaluation_anchor']]
    live += [e for e in state.get('quality_pool', {}).get('members', []) if 'path' in e]
    # update_population also considers historical checkpoints still on disk.
    # Keep any such candidate not already represented in the live weight pool;
    # omitting it would silently narrow the next iteration's opponent selection.
    pooled = {e['model_id'] for e in state.get('population', [])}
    live += [e for e in state['history'] if e['model_id'] not in pooled and below(root, e['path']).is_file()]
    if state.get('champion'):
        live.append(state['champion'])
    for entry in live:
        require(below(root, entry['path']).is_relative_to(run), 'Live checkpoint outside owned run')
        add(entry['path'], entry['sha256'])
    gate = state.get('champion_gate', {}).get('accepted_gate')
    if gate:
        add(gate['report'], gate['report_sha256'])
        add(below(root, gate['report']).parent / 'spec.json')
    distribution = read(root / 'training/runtime/distribution.json')
    runtime = {below(root, name).relative_to(root).as_posix() for name in distribution['private_runtime_files']}
    for name, expected in distribution['private_runtime_files'].items():
        add(name, expected)
    for entry in distribution['assets']:
        add(entry['path'], entry['sha256'])
    # Runtime metadata/receipts are preserved byte-for-byte. No new parity
    # receipt travels: preflight must run on the destination computer.
    for host in sorted(hosts):
        add(host / 'runtime.json')
        runtime.add((host / 'runtime.json').relative_to(root).as_posix())
        for entry in read(host / 'runtime.json')['files']:
            require(len(relative(entry['name']).parts) == 1, 'Unsafe runtime filename: ' + entry['name'])
            add(host / entry['name'], entry['sha256'])
            runtime.add((host / entry['name']).relative_to(root).as_posix())
    # LegacyWorker checks path/sha256 source pins at startup; additive hosts
    # check LF-normalized authored_sources. These are live dependencies even
    # when a custom host keeps its sources outside the usual code trees.
    metadata_paths = {root / n for n in runtime if n.endswith('/runtime.json')}
    metadata_paths.update((root / 'training/runtime/throughput').glob('*/runtime.json'))
    for metadata_path in sorted(metadata_paths):
        metadata = read(metadata_path)
        for entry in metadata.get('sources', []):
            if 'path' in entry:
                add(entry['path'], entry['sha256'])
        for name, expected in metadata.get('authored_sources', {}).items():
            path = below(root, name)
            add(path)
            require(hashlib.sha256(path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode()).hexdigest() == expected,
                    'Runtime authored source hash differs: ' + name)
    environment = {name for name in files if any(name.startswith(p + '/') for p in CODE_TREES)
                   or name.startswith('requirements') or name in runtime
                   or any(below(root, name).is_relative_to(host) for host in hosts)}
    manifests = set()
    def dataset(name, expected=None, replay=False):
        path = below(root, name)
        add(path, expected)
        manifests.add(path.relative_to(root).as_posix())
        if native:
            # Explicit pins cannot smuggle original IL shards or feature
            # sidecars into an otherwise fixed-probe-only transfer.
            for entry in read(path).get('entries',[]):
                source=below(root,entry['path']).relative_to(root).as_posix()
                require(not any(n==source or n.startswith(source+'.features') for n in files),
                    'Original IL replay/cache is pinned; remove the pin before fixed-probe-only transfer')
        if replay:
            data = read(path)
            require(data['format'] == 'bc-dataset/1.0.0', 'Unsupported rehearsal dataset')
            for entry in data['entries']:
                if entry['split'] == 'train':
                    add(entry['path'], entry['sha256'])
            calibration = data.get('calibration') or {}
            if calibration.get('budget', {}).get('calibration'):
                add(calibration['budget']['calibration'], calibration['sha256'])
    seen = set()
    def imitation(origin, *, legacy_upgrade=False):
        folder = below(root, origin['bc_run'])
        require(folder not in seen and folder.is_relative_to(root / 'artifacts'), 'Cyclic/invalid IL lineage')
        seen.add(folder); runs.add(folder)
        require(not active(folder), 'IL source is active')
        # completed_imitation() dispatches critic-upgraded IL through its
        # immutable receipt BEFORE looking for a BC state/report. Such a run
        # intentionally contains only upgrade.json + candidate.pt; its parent
        # owns the completed BC state/report and original dataset provenance.
        upgrade = folder / 'upgrade.json'
        if upgrade.is_file():
            require(not legacy_upgrade, 'An upgraded IL cannot be upgraded again')
            add(upgrade)
            receipt = read(upgrade)
            require(receipt.get('contract') == 'completed-il-critic-upgrade/1.0.0'
                    and receipt.get('compatibility') == state['signature'].get('compatibility')
                    and isinstance(receipt.get('compatibility'), dict)
                    and origin.get('bc_signature') == digest(receipt),
                    'IL upgrade receipt/source binding differs')
            candidate = receipt['candidate']
            selected = below(root, candidate['path'])
            require(selected.parent == folder and selected == below(root, origin['path'])
                    and all(origin[k] == candidate[k] for k in ('sha256', 'model_id')),
                    'Upgraded IL candidate identity/ownership differs')
            add(selected, candidate['sha256'])
            parent = receipt['parent']
            require(below(root, origin['manifest']) == below(root, parent['manifest'])
                    and origin['manifest_sha256'] == parent['manifest_sha256'],
                    'Upgraded IL dataset lineage differs')
            dataset(origin['manifest'], origin['manifest_sha256'])
            if mode == 'full': tree(folder)
            imitation(parent, legacy_upgrade=True)
            return
        if mode == 'full':
            tree(folder)
        else:
            add(folder / 'state.json')
            add(folder / 'report.json')
        s = read(folder / 'state.json')
        require((folder / 'report.json').is_file(), 'Missing completed IL report')
        add(origin['path'], origin['sha256'])
        if s.get('best'):
            selected = resolve_project_path(root, s['best']['path'], base=folder)
            require(selected.is_relative_to(folder), 'IL checkpoint outside its owned run')
            add(selected, s['best']['sha256'])
        dataset(origin['manifest'], origin['manifest_sha256'])
        parent = s['signature'].get('parent_imitation')
        if parent:
            imitation(parent, legacy_upgrade=legacy_upgrade)
    origin = state['signature'].get('imitation_source')
    if origin:
        imitation(origin)
    cfg = state['signature']['config']
    rehearsal = bool(cfg.get('rehearsal_manifest') and cfg['rehearsal_weight'] and state['games_done'] < cfg['rehearsal_until'])
    require(not native or not rehearsal, 'Fixed-probe-only transfer requires original IL rehearsal to be disabled through the training config')
    for name in {cfg.get('rehearsal_manifest'), cfg.get('imitation', {}).get('manifest')} - {None}:
        dataset(name, replay=rehearsal and name == cfg.get('rehearsal_manifest'))
    # In-run imitation also binds its source manifest after rehearsal expires.
    if state.get('imitation', {}).get('manifest'):
        dataset(state['imitation']['manifest'], state['imitation']['manifest_sha256'])
    rebuild = None
    if native:
        from step5_transfer_runtime import native_inventory
        rebuild = native_inventory(root, run, state, add=add, read=read, require=require, runtime=runtime)
        environment.update(runtime)
    return dict(root=root, run=run, runs=sorted(runs), state=state, files=files,
                rehearsal_included=rehearsal, datasets=sorted(manifests), environment=environment, runtime=runtime,
                source_recipe=recipe, execution_implementation=actual_source, buffer_rebuild=rebuild)


def summary(manifest):
    return {k: manifest[k] for k in ('format', 'source_root', 'run', 'config', 'iteration', 'games_done', 'rehearsal_included')}


def git_baseline(root,*,candidates=None,expected=None):
    """Only byte-reproducible unchanged files may arrive through Git clone."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)
    try:
        require(Path(os.fsdecode(git('rev-parse', '--show-toplevel')).strip()).resolve() == root, 'Project is not this Git root')
        commit = git('rev-parse', 'HEAD').decode().strip()
        tracked = {os.fsdecode(p).replace('\\', '/') for p in git('ls-files', '-z').split(b'\0') if p}
        changed = {os.fsdecode(p).replace('\\', '/') for p in git('diff', '--name-only', '-z', 'HEAD', '--').split(b'\0') if p}
        from step5_transfer_restore import reproducible_git_files
        unchanged=tracked-changed
        if candidates is not None:unchanged.intersection_update(candidates)
        return commit,reproducible_git_files(root,commit,unchanged,expected=expected)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None, set()  # Repo mode cannot omit files without a Git baseline.


def zstd_backend():
    try:
        from compression import zstd
    except ImportError:
        try:
            from backports import zstd
        except ImportError as error:
            raise ValueError('Zstandard support is missing. Run: python -m pip install -r requirements-transfer.txt '
                             '(or python -m pip install backports.zstd). ZIP export: --format zip') from error
    return zstd


def available_cpus():
    """Respect process affinity and a lab scheduler's per-task allocation."""
    counts = [getattr(os, 'process_cpu_count', os.cpu_count)() or 1]
    if hasattr(os, 'sched_getaffinity'):
        counts.append(len(os.sched_getaffinity(0)))
    allocated = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if allocated.isdecimal() and int(allocated) > 0:
        counts.append(int(allocated))
    return max(1, min(counts))


@contextmanager
def create_archive(path, archive_format, level, threads):
    with ExitStack() as stack:
        if archive_format == 'zip':
            archive = stack.enter_context(zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED,
                                                          compresslevel=6, allowZip64=True))
        else:
            zstd = zstd_backend()
            params = zstd.CompressionParameter
            compressed = stack.enter_context(zstd.ZstdFile(path, 'xb', options={
                params.compression_level: level, params.nb_workers: threads if threads > 1 else 0,
                params.checksum_flag: 1}))
            archive = stack.enter_context(tarfile.open(fileobj=compressed, mode='w|', bufsize=CHUNK))
        yield archive


def add_member(archive, name, source, size):
    if isinstance(archive, zipfile.ZipFile):
        if isinstance(source, Path):
            archive.write(source, name)
        else:
            archive.writestr(name, source)
    else:
        info = tarfile.TarInfo(name)
        info.size = size
        info.mode = 0o600
        with source.open('rb') if isinstance(source, Path) else io.BytesIO(source) as stream:
            archive.addfile(info, stream)


def export(root, run, bundle, *, dry_run=False, mode='compact', dependencies='repo', part_mb=1900,
           archive_format='zstd', zstd_level=3, threads=0):
    require(dependencies in ('repo', 'bundled', 'external'), 'Unknown dependency mode')
    require(type(part_mb) is int and 0 <= part_mb <= 2000, 'Part size must be 0 (unsplit) or 1–2000 MB')
    require(archive_format in ('zstd', 'zip'), 'Unknown archive format')
    require(type(zstd_level) is int and 1 <= zstd_level <= 22, 'Zstd level must be 1–22')
    require(type(threads) is int and threads >= 0, 'Threads must be 0 (auto) or a positive integer')
    compression_threads = (threads or available_cpus()) if archive_format == 'zstd' else 1
    if archive_format == 'zstd' and not dry_run:
        zstd = zstd_backend()  # Fail before acquiring run leases or writing an archive.
        worker_limit = zstd.CompressionParameter.nb_workers.bounds()[1]
        require(compression_threads <= max(1, worker_limit), 'Too many compression threads for this zstd build')
    root = Path(root).resolve()
    run = below(root, run)
    require(not active(run), 'Run is active; wait for training to stop')
    with ExitStack() as leases:
        if not dry_run:
            leases.enter_context(lease(run))
        data = inventory(root, run, mode=mode)
        for folder in data['runs']:
            if folder != run and not dry_run:
                leases.enter_context(lease(folder))
        state_bytes = (run / 'state.json').read_bytes()
        state_hash = hashlib.sha256(state_bytes).hexdigest()
        cfg_name = f'artifacts/step5-transfer/resume-{state_hash[:16]}.json'
        target_config=data['state']['signature']['config']
        if data['buffer_rebuild'] is not None:
            from brass_ai.learning.post_transfer_reward import eligible,revise,REVISION
            if eligible(target_config):
                target_config=revise(target_config)
                data['buffer_rebuild']['reward_revision']=REVISION
        cfg = encode(target_config)
        files = dict(data['files'])
        require(cfg_name not in files, 'Transfer config collides with an existing dependency')
        files[cfg_name] = cfg
        records = [dict(path=name, sha256=sha(source) if isinstance(source, Path) else hashlib.sha256(source).hexdigest(),
                        bytes=source.stat().st_size if isinstance(source, Path) else len(source),
                        executable=bool(isinstance(source, Path) and source.stat().st_mode & 0o111),
                        external=False) for name, source in sorted(files.items())]
        candidates=data['environment']-data['runtime']
        expected={r['path']:r for r in records}
        commit, tracked = git_baseline(root,candidates=candidates,expected=expected) if dependencies in ('repo', 'external') else (None, set())
        for record in records:
            name=record['path']
            record['external']=name in data['environment'] and (dependencies=='external' or
                (name in tracked and name not in data['runtime']))
        manifest = dict(format=FORMAT, source_root=str(root), run=run.relative_to(root).as_posix(),
            runs=[p.relative_to(root).as_posix() for p in data['runs']], iteration=data['state']['iteration'],
            games_done=data['state']['games_done'], implementation=data['execution_implementation'],
            source_recipe=data['source_recipe'], path_roots=source_roots(root), relocation_version=RELOCATION_VERSION,
            state_sha256=state_hash, config=cfg_name, imitation_run=(below(root, data['state']['signature']['imitation_source']['bc_run']).relative_to(root).as_posix()
                if data['state']['signature'].get('imitation_source') else None),
            rehearsal_included=data['rehearsal_included'], mode=mode, dependencies=dependencies, git_commit=commit, files=records)
        if data['buffer_rebuild'] is not None:
            manifest['buffer_rebuild'] = data['buffer_rebuild']
        manifest['id'] = digest(manifest)
        included = [r for r in records if not r['external']]
        bundled_runtime = [r for r in included if r['path'] in data['runtime']]
        unique = {r['sha256']: r for r in included}
        sizes = {}
        for record in included:
            category = 'environment' if record['path'] in data['environment'] else 'selfplay' if record['path'].startswith(manifest['run'] + '/') else 'IL_and_data'
            sizes[category] = sizes.get(category, 0) + record['bytes']
        result = dict(summary(manifest), mode=mode, dependencies=dependencies, git_commit=commit, files=len(included),
            external_files=len(records) - len(included), bytes=sum(r['bytes'] for r in included),
            runtime_files=len(data['runtime']), bundled_runtime_files=len(bundled_runtime),
            bundled_runtime_bytes=sum(r['bytes'] for r in bundled_runtime),
            unique_payload_bytes=sum(r['bytes'] for r in unique.values()), category_bytes=sizes,
            largest_files=sorted(included, key=lambda r: (-r['bytes'], r['path']))[:10],
            part_mb=part_mb, dry_run=dry_run, archive_format=archive_format,
            zstd_level=zstd_level if archive_format == 'zstd' else None, compression_threads=compression_threads)
        if data['buffer_rebuild'] is not None: result['buffer_rebuild'] = data['buffer_rebuild']
        if dry_run:
            return result
        target = Path(bundle).absolute()
        require(not target.exists() and not Path(str(target) + '.parts.json').exists()
                and not any(target.is_relative_to(p) for p in data['runs']), 'Use a new bundle path outside source runs')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
        try:
            with create_archive(temporary, archive_format, zstd_level, compression_threads) as archive:
                manifest_bytes = encode(manifest)
                add_member(archive, 'transfer.json', manifest_bytes, len(manifest_bytes))
                written = set()
                for index, record in enumerate(records, 1):
                    source = files[record['path']]
                    if not record['external'] and record['sha256'] not in written:
                        payload = payload_name(manifest, record)
                        add_member(archive, payload, source, record['bytes'])
                        written.add(record['sha256'])
                    if isinstance(source, Path):
                        require(sha(source) == record['sha256'], 'Source changed during export: ' + record['path'])
                    if index == len(records) or index % max(1, len(records) // 5) == 0:
                        print(f'Transfer export: {index}/{len(records)} files', file=sys.stderr, flush=True)
            require((run / 'state.json').read_bytes() == state_bytes, 'Run changed during export')
            require(source_identity(root, recipe=manifest['source_recipe']) == manifest['implementation'], 'Training source changed during export')
            inspect_bundle(temporary)
            compressed_bytes = temporary.stat().st_size
            require(not target.exists(), 'Bundle target appeared during export')
            if part_mb:
                target, part_count = split_bundle(temporary, target, part_mb * 1_000_000)
            else:
                temporary.rename(target)
                part_count = 0
        finally:
            if temporary.exists():
                temporary.unlink()
        return dict(result, bundle=str(target), bundle_sha256=sha(target), parts=part_count, compressed_bytes=compressed_bytes)


def payload_name(manifest, record):
    return 'project/' + record['path'] if manifest['format'] == OLD_FORMAT else 'objects/' + record['sha256']


def split_bundle(source, target, limit):
    """Publish the small index last. Leave no duplicate monolithic archive."""
    target = Path(target)
    index = Path(str(target) + '.parts.json')
    count = (source.stat().st_size + limit - 1) // limit
    paths = [Path(str(target) + f'.part{i + 1:04d}') for i in range(count)]
    require(not any(p.exists() for p in paths + [index]), 'Part/index target exists; choose a new bundle name')
    created = []
    try:
        parts = []
        with source.open('rb') as stream:
            for path in paths:
                with path.open('xb') as out:
                    created.append(path)
                    remaining = limit
                    while remaining and (block := stream.read(min(remaining, 1024 * 1024))):
                        out.write(block); remaining -= len(block)
                parts.append(dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path)))
        data = dict(format=PARTS_FORMAT, bytes=source.stat().st_size, sha256=sha(source), parts=parts)
        data['id'] = digest(data)
        with index.open('xb') as out:
            created.append(index); out.write(encode(data))
        return index, count
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


class PartsReader(io.RawIOBase):
    """Seek across validated parts without making another large joined archive."""
    def __init__(self, index):
        super().__init__()
        index = Path(index)
        require(index.stat().st_size < 32_000_000, 'Oversized parts index')
        data = read(index)
        require(data['format'] == PARTS_FORMAT and data['id'] == digest({k: v for k, v in data.items() if k != 'id'}), 'Invalid parts index')
        self.paths = []; self.ends = []; total = 0; seen = set(); whole = hashlib.sha256()
        for part in data['parts']:
            name = part['name']; relative(name)
            require('/' not in name and name.casefold() not in seen, 'Invalid/duplicate part name')
            seen.add(name.casefold())
            path = below(index.parent, name)
            require(path.is_file() and path.stat().st_size == part['bytes'], 'Missing/truncated transfer part: ' + name)
            h = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    h.update(block); whole.update(block)
            require(h.hexdigest() == part['sha256'], 'Corrupted transfer part: ' + name)
            total += part['bytes']; self.paths.append(path); self.ends.append(total)
        require(total > 0 and total == data['bytes'] and whole.hexdigest() == data['sha256'], 'Joined parts differ')
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        require(whence in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END), 'Invalid seek origin')
        position = offset + (0 if whence == os.SEEK_SET else self.position if whence == os.SEEK_CUR else self.ends[-1])
        require(position >= 0, 'Negative archive seek')
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.ends[-1] - self.position)
        remaining = remaining if size is None or size < 0 else min(size, remaining)
        blocks = []
        while remaining:
            i = bisect_right(self.ends, self.position)
            start = self.ends[i - 1] if i else 0
            count = min(remaining, self.ends[i] - self.position)
            with self.paths[i].open('rb') as stream:
                stream.seek(self.position - start); block = stream.read(count)
            require(len(block) == count, 'Transfer part changed while reading')
            blocks.append(block); self.position += count; remaining -= count
        return b''.join(blocks)


@contextmanager
def open_bundle(bundle):
    """Yield an iterator of (name, size, stream); consume streams in order.

    Detect by bytes, so old ZIPs and either kind of parts index need no flags.
    Tar payloads are never extracted or buffered as a complete intermediate file.
    """
    path = Path(bundle)
    if not path.exists() and Path(str(path) + '.parts.json').exists():
        path = Path(str(path) + '.parts.json')
    with ExitStack() as stack:
        source = stack.enter_context(PartsReader(path) if path.name.endswith('.parts.json') else path.open('rb'))
        magic = source.read(4)
        source.seek(0)
        if magic == ZSTD_MAGIC:
            decoded = stack.enter_context(zstd_backend().ZstdFile(source, 'rb'))
            archive = stack.enter_context(tarfile.open(fileobj=decoded, mode='r|', bufsize=CHUNK))

            def members():
                for info in archive:
                    relative(info.name)
                    require(info.isfile() and not info.issparse() and info.size >= 0, 'Invalid archive member')
                    with archive.extractfile(info) as stream:
                        yield info.name, info.size, stream
                # Tar stops at its first zero end marker, possibly before zstd's
                # final checksum. Drain buffered padding AND the decoder to EOF
                # to reject a truncated frame, corrupt footer or appended data.
                while block := archive.fileobj.read(CHUNK):
                    require(not block.strip(b'\0'), 'Unexpected data after tar end')
        else:
            archive = stack.enter_context(zipfile.ZipFile(source))

            def members():
                for info in archive.infolist():
                    relative(info.filename)
                    require(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16), 'Invalid archive member')
                    with archive.open(info) as stream:
                        yield info.filename, info.file_size, stream
        yield members()


def inspect_bundle(bundle):
    actual = {}; seen = set(); manifest = None
    with open_bundle(bundle) as members:
        for name, size, stream in members:
            require(name.casefold() not in seen, 'Duplicate/case-colliding archive members')
            seen.add(name.casefold())
            if name == 'transfer.json':
                require(size < 32_000_000, 'Oversized transfer manifest')
                manifest = json.loads(stream.read())
            else:
                actual[name] = (size, hashlib.file_digest(stream, 'sha256').hexdigest())
    require(manifest is not None, 'Missing transfer manifest')
    require(manifest['format'] in (FORMAT, OLD_FORMAT) and manifest['id'] == digest({k: v for k, v in manifest.items() if k != 'id'}), 'Invalid transfer manifest')
    require(manifest.get('source_recipe', 1) in (1, 2, 3, 4), 'Unsupported implementation fingerprint recipe')
    roots = validate_roots(manifest.get('path_roots', [manifest['source_root']]))
    require(roots and root_key(roots[0]) == root_key(manifest['source_root']), 'Transfer source root binding differs')
    if manifest.get('imitation_run'):
        suffix = reference_suffix(roots, manifest['imitation_run'])
        relative('/'.join(suffix) if suffix is not None else manifest['imitation_run'])
    files = manifest['files']
    names = [r['path'] for r in files]
    require(len(names) == len({n.casefold() for n in names}), 'Duplicate/case-colliding transfer paths')
    folded = {n.casefold() for n in names}
    for name in names:
        parts = relative(name).parts
        # Only the marker is created/deleted by transfer's lease. Old full
        # archives may contain inert OS guard files; preserve their bytes like
        # any other file, with normal hash/path checks. New exports omit them.
        require(not any(p.casefold() == '.active-run.json' for p in parts),
                'Archive contains reserved run lease/control path: ' + name)
        require(not any('/'.join(parts[:i]).casefold() in folded for i in range(1, len(parts))),
                'Transfer file collides with a required directory: ' + name)
    payloads = {payload_name(manifest, r) for r in files if not r.get('external', False)}
    require(set(actual) == payloads, 'Unexpected archive members')
    for record in files:
        relative(record['path'])
        require(record['path'].casefold() != RELOCATION_FILE.casefold(), 'Archive cannot supply relocation metadata')
        require(type(record.get('external', False)) is bool and type(record['bytes']) is int and record['bytes'] >= 0, 'Invalid file metadata')
        require(type(record.get('executable',False)) is bool, 'Invalid executable-file metadata')
        require(len(record['sha256']) == 64 and all(c in '0123456789abcdef' for c in record['sha256']), 'Invalid file digest')
        if record.get('external', False):
            require(manifest['format'] == FORMAT, 'Old transfer cannot omit external files')
            continue
        payload = payload_name(manifest, record)
        require(actual[payload][0] == record['bytes'], 'Invalid archive member size')
        require(actual[payload][1] == record['sha256'], 'Corrupted transfer file: ' + record['path'])
    for name in [manifest['run'], manifest['config'], *manifest['runs']]:
        relative(name)
    require(manifest['run'] in manifest['runs'] and all(n.startswith('artifacts/') for n in manifest['runs']), 'Invalid run roots')
    by_name = {r['path']: r for r in files}
    if manifest.get('buffer_rebuild') is not None:
        expected=dict(contract='fresh-buffers/1.0.0',origin_generation=manifest['iteration'],
            auxiliary_buffer_iterations=8,il_probe_positions=2048)
        if 'reward_revision' in manifest['buffer_rebuild']:
            from brass_ai.learning.post_transfer_reward import REVISION
            expected['reward_revision']=REVISION
        require(manifest.get('source_recipe')==4 and manifest['buffer_rebuild']==expected,
            'Unknown buffer-rebuild transfer policy')
        prefix=manifest['run']+'/'
        forbidden=('.ppg-eight-cache/','.ppg-value-cache/','teacher-labels/','search-values/','il-kl/.prepared/')
        require(not any(n.startswith(prefix+p) for n in names for p in forbidden), 'Replay cache in fresh-buffer archive')
        require(prefix+'machine-transfer-resume.json' not in by_name, 'Old transfer instruction cannot be replayed into a new transfer')
        require(all(prefix+'il-kl/'+name in by_name for name in ('manifest.json','probe.npz')), 'Missing fixed IL probe in fresh-buffer archive')
    require(manifest['config'] in by_name, 'Missing saved resume config')
    require(not any(n.casefold() in folded for n in manifest['runs']), 'Run root collides with a transfer file')
    require(by_name[manifest['run'] + '/state.json']['sha256'] == manifest['state_sha256'], 'State binding differs')
    return manifest


def commands(root, manifest, *, iterations=20, device='cuda'):
    root = Path(root)
    base = ['python', str(root / 'scripts/step5.py'), '--root', str(root), '--config', str(root / manifest['config']), '--device', device]
    resume = base + ['train', '--run', str(root / manifest['run']), '--iterations', str(iterations)]
    if manifest.get('buffer_rebuild') is not None:
        resume += ['--expected-generation', str(manifest['iteration']), '--resume-reviewed-source']
    if manifest['imitation_run']:
        resume += ['--imitation-run', str(below(root, manifest['imitation_run']))]
    result=dict(preflight=shlex.join(base + ['preflight']), resume=shlex.join(resume))
    if manifest.get('buffer_rebuild',{}).get('reward_revision'):
        result['prepare_reward']=shlex.join(['python',str(root/'scripts/prepare-rust-post-transfer-reward.py'),
            '--run',str(root/manifest['run']),'--expected-generation',str(manifest['iteration'])])
    return result


def restore(root, bundle, *, verify_only=False, relocate=False, strict_repo_files=False,use_checkout_code=False):
    require(not (strict_repo_files and use_checkout_code),'--strict-repo-files and --use-checkout-code are mutually exclusive')
    root = Path(root).resolve()
    manifest = inspect_bundle(bundle)  # verify all archive bytes BEFORE filesystem writes
    original_state_present = (root/manifest['run']/'state.json').is_file()
    existing = read_relocation(root)
    same_root = root_key(str(root)) == root_key(manifest['source_root'])
    metadata = make_relocation(root, manifest.get('path_roots', [manifest['source_root']]), manifest['id'])
    metadata_bytes = encode(metadata)
    # Match the shared runtime reader's limit before creating any files. A
    # manifest's individually valid roots can still exceed that combined limit.
    require(len(metadata_bytes) < 65536, 'Oversized relocation metadata; historical root mappings exceed 64 KiB')
    needs_mapping = bool(metadata['source_roots'])
    # Re-exporting an already relocated project produces a new archive ID.
    # Verification at that source must retain its original sidecar receipt;
    # the identical mapping and all archive file hashes are still checked.
    matching_metadata = existing == metadata or (same_root and existing is not None and
                                                existing['source_roots'] == metadata['source_roots'])
    require(not (verify_only and needs_mapping and existing is None),
            'Restore has not completed at destination --root '+str(root)+
            ': missing relocation metadata. Fix the restore failure and rerun restore --relocate successfully before verify. '+
            'The archive records its original root '+manifest['source_root']+'; a different destination --root is supported.')
    require(same_root or relocate or existing == metadata,
            'Destination differs from the original absolute project root: ' + manifest['source_root'] +
            '. Use --relocate to map historical paths without rewriting signed evidence.')
    require(existing is None or matching_metadata, 'Conflicting destination relocation metadata; nothing overwritten')
    runs = [below(root, n) for n in manifest['runs']]
    require(all(not p.exists() or p.is_dir() for p in runs), 'Conflicting destination run directory')
    require(all(not active(p) for p in runs), 'Destination run is active')
    from step5_transfer_restore import RestorePlan
    plan=RestorePlan(root,manifest,below=below,sha=sha,require=require,read=read,encode=encode,publish=publish,
        verify_only=verify_only,strict=strict_repo_files,original_state_present=original_state_present,
        use_checkout_code=use_checkout_code)
    if not verify_only:
        with ExitStack() as leases:
            for run in runs:
                leases.enter_context(lease(run))
            plan.prepare()  # All clone backups become durable before any replacement.
            def install(record,source):
                path=plan.target(record);path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name('.transfer-'+uuid.uuid4().hex+'.partial')
                try:
                    with temporary.open('xb') as target:
                        while block:=source.read(CHUNK):target.write(block)
                    require(sha(temporary)==record['sha256'],'Restored file hash differs')
                    plan.install(record,temporary)
                finally:temporary.unlink(missing_ok=True)
                return path
            for record in manifest['files']:
                if record['path'] in plan.external:
                    install(record,io.BytesIO(plan.external[record['path']]))
            needed = {}
            for record in manifest['files']:
                if not record.get('external',False) and (not plan.target(record).exists() or record['path'] in plan.replacements):
                    needed.setdefault(payload_name(manifest, record), []).append(record)
            with open_bundle(bundle) as members:
                for name, size, stream in members:
                    first = None
                    for record in needed.pop(name, []):
                        # Decode a deduplicated object once; copy subsequent
                        # paths independently, preserving existing no-hardlink semantics.
                        with ExitStack() as copying:
                            source=stream if first is None else copying.enter_context(first.open('rb'))
                            first=install(record,source)
            require(not needed, 'Missing archive payloads during restore')
        # Native Rust workers launch directly on Linux. Archive payload bytes
        # alone are insufficient if restoration loses their executable bit.
        for record in manifest['files']:
            if record.get('executable',False) and record['path'] not in plan.checkout:
                path=plan.target(record)
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
    if verify_only and os.name!='nt':
        require(all(r['path'] in plan.checkout or not r.get('executable',False) or plan.target(r).stat().st_mode & stat.S_IXUSR
            for r in manifest['files']), 'Restored native executable permission is missing')
    require(all(plan.matches(r) for r in manifest['files']),
            'Restored file changed before final verification')
    # Archives written before the throughput hosts used the previous recipe.
    # Verify their original identity; never relabel it as the current source.
    require(source_identity(root, recipe=manifest.get('source_recipe', 1)) == manifest['implementation'],
            'Destination source identity differs'+('; --use-checkout-code skips per-file archive matches, '
            'but training source must still be compatible with this checkpoint' if use_checkout_code else ''))
    if needs_mapping and existing is None:
        path = confined(root, (RELOCATION_FILE,))
        temporary = path.with_name('.transfer-' + uuid.uuid4().hex + '.partial')
        try:
            with temporary.open('xb') as out:
                out.write(metadata_bytes)
            # No replacement: concurrent/unrelated destination metadata must win.
            publish(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if manifest.get('buffer_rebuild') is not None and not verify_only:
        from brass_ai.recovery.transfer_resume import install_instruction
        # In-place archive verification on the source must never reset its
        # buffers. A new destination receives a checkpoint-bound instruction.
        if not same_root or not plan.identity['original_state_present']:
            install_instruction(root, manifest)
    if manifest.get('buffer_rebuild') is not None and verify_only and not same_root:
        from brass_ai.recovery.transfer_resume import instruction
        marker = instruction(root, root/manifest['run'])
        require(marker is not None and marker['archive_id'] == manifest['id'], 'Missing/different fresh-buffer transfer instruction')
    return dict(summary(manifest), verified=True, destination_root=str(root), relocated=needs_mapping,
                code_verification='checkout' if use_checkout_code else 'archive',
                repository_reconciliation=plan.report(),
                imitation_run=(below(root, manifest['imitation_run']).relative_to(root).as_posix()
                               if manifest['imitation_run'] else None),
                requires_code_update=needs_mapping and (manifest.get('source_recipe', 1) < 3 or
                                                        manifest.get('relocation_version') != RELOCATION_VERSION),
                **commands(root, manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    out = sub.add_parser('export')
    out.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    out.add_argument('--run', required=True)
    out.add_argument('--bundle', required=True, type=Path)
    out.add_argument('--dry-run', action='store_true')
    out.add_argument('--mode', choices=('compact', 'full'), default='compact', help='Default: live checkpoints/evidence; omit consumed replays, caches, trash and unused IL epochs')
    out.add_argument('--dependencies', choices=('repo', 'bundled', 'external'), default='repo', help='Default repo: unchanged code from Git, game runtime included for Release assets. Bundled: also include code. External: legacy mode requiring separately installed code and runtime')
    out.add_argument('--format', choices=('zstd', 'zip'), default='zstd', dest='archive_format', help='Default zstd: streaming .tar.zst; zip: legacy DEFLATE format')
    out.add_argument('--zstd-level', type=int, default=3, help='Zstd compression level 1–22 (default 3)')
    out.add_argument('--threads', type=int, default=0, help='Zstd compression threads: 0=available CPUs/SLURM allocation (default), 1=single thread')
    out.add_argument('--part-mb', type=int, default=1900, help='Decimal MB: default 1900 for Release assets; use 100 for Git push; 0 for one archive; maximum 2000')
    for name in ('restore', 'verify'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--root', required=True, type=Path)
        cmd.add_argument('--bundle', required=True, type=Path)
        cmd.add_argument('--relocate', action='store_true', help='Use --root as a new project directory; preserve signed files and persist confined old-root mappings')
        code=cmd.add_mutually_exclusive_group()
        code.add_argument('--strict-repo-files',action='store_true',help='Reject differing Git-managed files instead of backing them up and restoring archive-pinned bytes')
        code.add_argument('--use-checkout-code',action='store_true',help='Keep existing Git-tracked text/code/config files without archive byte matches; retain checkpoint/data and training compatibility checks')
    a = parser.parse_args()
    if a.command == 'export':
        result = export(a.root, a.run, a.bundle, dry_run=a.dry_run, mode=a.mode, dependencies=a.dependencies, part_mb=a.part_mb,
                        archive_format=a.archive_format, zstd_level=a.zstd_level, threads=a.threads)
    else:
        result = restore(a.root, a.bundle, verify_only=a.command == 'verify', relocate=a.relocate,
            strict_repo_files=a.strict_repo_files,use_checkout_code=a.use_checkout_code)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
