"""Private, byte-preserving transfer of a stopped Step 5 run (stdlib only).

Keep this outside the source-bound learning packages: adding a transport tool
must not invalidate a live PPO run's implementation identity. No checkpoints or
state signatures are rewritten, and no game/training process is launched.
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
import uuid
import zipfile
from bisect import bisect_right


FORMAT = 'step5-private-transfer/1.1.0'
OLD_FORMAT = 'step5-private-transfer/1.0.0'
PARTS_FORMAT = 'step5-transfer-parts/1.0.0'
SKIP = {'__pycache__', '.git', '.active-run.json'}
CODE_TREES = ('src', 'scripts', 'configs', 'training/opponents', 'training/runtime')


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


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name, 'Invalid portable path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name
            and all(x not in ('.', '..') and not x.endswith((' ', '.')) for x in p.parts), 'Unsafe archive path: ' + name)
    require(not any(x in ('.git', '.aws', '.ssh', '.codex', '.agents') for x in p.parts), 'Private configuration path forbidden')
    return p


def below(root, name):
    root = Path(root).resolve()
    p = Path(name)
    p = p if p.is_absolute() else root / p
    require(p.is_relative_to(root) and p != root, 'Path outside project root: ' + str(name))
    for item in (p, *p.parents):
        if item == root:
            break
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                    'Linked transfer path: ' + str(item))
    require(p.resolve().is_relative_to(root), 'Resolved path outside project')
    return p


def source_identity(root):
    """The current implementation_id recipe, without importing torch/runtimes."""
    root = Path(root)
    base = root / 'src/brass_ai'
    paths = [p for group in ('step5', 'recovery', 'search', 'learning') for p in (base / group).glob('*.py')]
    paths += [base / p for p in ('learning/runner.py', 'learning/batching.py', 'searchlab/spiel.py', 'beliefs.py', 'runtime_contracts.py')]
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
    require(source_identity(root) == state['signature']['implementation'],
            'Source differs from saved run. Use the matching source; migration does not authorize a source upgrade.')
    pending = run / f"iteration-{state['iteration']:04d}"
    require(not pending.exists() or (pending.is_dir() and not any(pending.iterdir())),
            'Uncommitted iteration: finish/recover it on the source machine before export')
    require(not (run / f"generation-{state['iteration'] + 1:04d}.pt").exists(), 'Uncommitted checkpoint; recover before export')
    files = {}
    runs = {run}
    def add(path, expected=None):
        path = below(root, path)
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
        for folder in sorted((run / 'evaluations').glob('*')):
            if not folder.is_dir():
                continue
            if (folder / 'spec.json').is_file() and not (folder / 'report.json').is_file():
                tree(folder)
                def references(value):
                    if isinstance(value, dict):
                        if isinstance(value.get('path'), str) and value.get('sha256'):
                            path = below(root, value['path'])
                            if path.is_relative_to(run):
                                add(path, value['sha256'])
                        for child in value.values():
                            references(child)
                    elif isinstance(value, list):
                        for child in value:
                            references(child)
                references(read(folder / 'spec.json'))
            else:
                for name in ('spec.json', 'report.json'):
                    if (folder / name).is_file():
                        add(folder / name)
        if (run / 'pins.json').is_file():
            for name in read(run / 'pins.json')['paths']:
                add(name)
    live = state.get('resume', []) + state.get('population', []) + [state['history'][-1], state['evaluation_anchor']]
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
    for name, expected in distribution['private_runtime_files'].items():
        add(name, expected)
    for entry in distribution['assets']:
        add(entry['path'], entry['sha256'])
    # Runtime metadata/receipts are preserved byte-for-byte. No new parity
    # receipt travels: preflight must run on the destination computer.
    host = below(root, state['signature']['config']['host'])
    add(host / 'runtime.json')
    for entry in read(host / 'runtime.json')['files']:
        add(host / entry['name'], entry['sha256'])
    environment = {name for name in files if any(name.startswith(p + '/') for p in CODE_TREES)
                   or name.startswith('requirements') or name in distribution['private_runtime_files']
                   or below(root, name).is_relative_to(host)}
    manifests = set()
    def dataset(name, expected=None, replay=False):
        path = below(root, name)
        add(path, expected)
        manifests.add(path.relative_to(root).as_posix())
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
    def imitation(origin):
        folder = below(root, origin['bc_run'])
        require(folder not in seen and folder.is_relative_to(root / 'artifacts'), 'Cyclic/invalid IL lineage')
        seen.add(folder); runs.add(folder)
        require(not active(folder), 'IL source is active')
        if mode == 'full':
            tree(folder)
        else:
            add(folder / 'state.json')
            add(folder / 'report.json')
        s = read(folder / 'state.json')
        require((folder / 'report.json').is_file(), 'Missing completed IL report')
        add(origin['path'], origin['sha256'])
        dataset(origin['manifest'], origin['manifest_sha256'])
        parent = s['signature'].get('parent_imitation')
        if parent:
            imitation(parent)
    origin = state['signature'].get('imitation_source')
    if origin:
        imitation(origin)
    cfg = state['signature']['config']
    rehearsal = bool(cfg.get('rehearsal_manifest') and cfg['rehearsal_weight'] and state['games_done'] < cfg['rehearsal_until'])
    for name in {cfg.get('rehearsal_manifest'), cfg.get('imitation', {}).get('manifest')} - {None}:
        dataset(name, replay=rehearsal and name == cfg.get('rehearsal_manifest'))
    # In-run imitation also binds its source manifest after rehearsal expires.
    if state.get('imitation', {}).get('manifest'):
        dataset(state['imitation']['manifest'], state['imitation']['manifest_sha256'])
    return dict(root=root, run=run, runs=sorted(runs), state=state, files=files,
                rehearsal_included=rehearsal, datasets=sorted(manifests), environment=environment)


def summary(manifest):
    return {k: manifest[k] for k in ('format', 'source_root', 'run', 'iteration', 'games_done', 'rehearsal_included')}


def git_baseline(root):
    """Only unchanged tracked files are expected to arrive through Git clone."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)
    try:
        require(Path(os.fsdecode(git('rev-parse', '--show-toplevel')).strip()).resolve() == root, 'Project is not this Git root')
        commit = git('rev-parse', 'HEAD').decode().strip()
        tracked = {os.fsdecode(p).replace('\\', '/') for p in git('ls-files', '-z').split(b'\0') if p}
        changed = {os.fsdecode(p).replace('\\', '/') for p in git('diff', '--name-only', '-z', 'HEAD', '--').split(b'\0') if p}
        return commit, tracked - changed
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None, set()  # Repo mode cannot omit files without a Git baseline.


def export(root, run, bundle, *, dry_run=False, mode='compact', dependencies='external', part_mb=100):
    require(dependencies in ('repo', 'bundled', 'external'), 'Unknown dependency mode')
    require(type(part_mb) is int and 0 <= part_mb <= 100, 'Part size must be 0 (unsplit) or 1–100 MB')
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
        cfg = encode(data['state']['signature']['config'])
        files = dict(data['files'])
        require(cfg_name not in files, 'Transfer config collides with an existing dependency')
        files[cfg_name] = cfg
        commit, tracked = git_baseline(root) if dependencies in ('repo', 'external') else (None, set())
        records = [dict(path=name, sha256=sha(source) if isinstance(source, Path) else hashlib.sha256(source).hexdigest(),
                        bytes=source.stat().st_size if isinstance(source, Path) else len(source),
                        external=name in data['environment'] and (dependencies == 'external' or name in tracked)) for name, source in sorted(files.items())]
        manifest = dict(format=FORMAT, source_root=str(root), run=run.relative_to(root).as_posix(),
            runs=[p.relative_to(root).as_posix() for p in data['runs']], iteration=data['state']['iteration'],
            games_done=data['state']['games_done'], implementation=data['state']['signature']['implementation'],
            state_sha256=state_hash, config=cfg_name, imitation_run=data['state']['signature'].get('imitation_source', {}).get('bc_run'),
            rehearsal_included=data['rehearsal_included'], mode=mode, dependencies=dependencies, git_commit=commit, files=records)
        manifest['id'] = digest(manifest)
        included = [r for r in records if not r['external']]
        unique = {r['sha256']: r for r in included}
        sizes = {}
        for record in included:
            category = 'environment' if record['path'] in data['environment'] else 'selfplay' if record['path'].startswith(manifest['run'] + '/') else 'IL_and_data'
            sizes[category] = sizes.get(category, 0) + record['bytes']
        result = dict(summary(manifest), mode=mode, dependencies=dependencies, git_commit=commit, files=len(included),
            external_files=len(records) - len(included), bytes=sum(r['bytes'] for r in included),
            unique_payload_bytes=sum(r['bytes'] for r in unique.values()), category_bytes=sizes,
            largest_files=sorted(included, key=lambda r: (-r['bytes'], r['path']))[:10],
            part_mb=part_mb, dry_run=dry_run)
        if dry_run:
            return result
        target = Path(bundle).absolute()
        require(not target.exists() and not Path(str(target) + '.parts.json').exists()
                and not any(target.is_relative_to(p) for p in data['runs']), 'Use a new bundle path outside source runs')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
        try:
            with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
                archive.writestr('transfer.json', encode(manifest))
                written = set()
                for index, record in enumerate(records, 1):
                    source = files[record['path']]
                    if not record['external'] and record['sha256'] not in written:
                        payload = payload_name(manifest, record)
                        if isinstance(source, Path):
                            archive.write(source, payload)
                        else:
                            archive.writestr(payload, source)
                        written.add(record['sha256'])
                    if isinstance(source, Path):
                        require(sha(source) == record['sha256'], 'Source changed during export: ' + record['path'])
                    if index == len(records) or index % max(1, len(records) // 5) == 0:
                        print(f'Transfer export: {index}/{len(records)} files', flush=True)
            require((run / 'state.json').read_bytes() == state_bytes, 'Run changed during export')
            require(source_identity(root) == manifest['implementation'], 'Training source changed during export')
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
    """Publish the small index last. Leave no duplicate monolithic ZIP."""
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
    """Seek across validated parts without making another large joined ZIP."""
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
    path = Path(bundle)
    if not path.exists() and Path(str(path) + '.parts.json').exists():
        path = Path(str(path) + '.parts.json')
    with ExitStack() as stack:
        source = stack.enter_context(PartsReader(path)) if path.name.endswith('.parts.json') else path
        yield stack.enter_context(zipfile.ZipFile(source))


def inspect_bundle(bundle):
    with open_bundle(bundle) as archive:
        require(archive.getinfo('transfer.json').file_size < 32_000_000, 'Oversized transfer manifest')
        manifest = json.loads(archive.read('transfer.json'))
        require(manifest['format'] in (FORMAT, OLD_FORMAT) and manifest['id'] == digest({k: v for k, v in manifest.items() if k != 'id'}), 'Invalid transfer manifest')
        files = manifest['files']
        names = [r['path'] for r in files]
        require(len(names) == len({n.casefold() for n in names}), 'Duplicate/case-colliding transfer paths')
        payloads = {payload_name(manifest, r) for r in files if not r.get('external', False)}
        require(len(archive.namelist()) == len(payloads) + 1 and set(archive.namelist()) == {'transfer.json', *payloads}, 'Unexpected archive members')
        checked = {}
        for record in files:
            relative(record['path'])
            require(type(record.get('external', False)) is bool and type(record['bytes']) is int and record['bytes'] >= 0, 'Invalid file metadata')
            require(len(record['sha256']) == 64 and all(c in '0123456789abcdef' for c in record['sha256']), 'Invalid file digest')
            if record.get('external', False):
                require(manifest['format'] == FORMAT, 'Old transfer cannot omit external files')
                continue
            payload = payload_name(manifest, record)
            if payload in checked:
                require(checked[payload] == record['bytes'], 'Inconsistent duplicate payload size')
                continue
            info = archive.getinfo(payload)
            require(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16) and info.file_size == record['bytes'], 'Invalid archive member')
            with archive.open(info) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256'], 'Corrupted transfer file: ' + record['path'])
            checked[payload] = record['bytes']
        for name in [manifest['run'], manifest['config'], *manifest['runs']]:
            relative(name)
        require(manifest['run'] in manifest['runs'] and all(n.startswith('artifacts/') for n in manifest['runs']), 'Invalid run roots')
        by_name = {r['path']: r for r in files}
        require(by_name[manifest['run'] + '/state.json']['sha256'] == manifest['state_sha256'], 'State binding differs')
        return manifest


def commands(root, manifest, *, iterations=20, device='cuda'):
    root = Path(root)
    base = ['python', str(root / 'scripts/step5.py'), '--root', str(root), '--config', str(root / manifest['config']), '--device', device]
    resume = base + ['train', '--run', str(root / manifest['run']), '--iterations', str(iterations)]
    if manifest['imitation_run']:
        resume += ['--imitation-run', str(root / manifest['imitation_run'])]
    return dict(preflight=shlex.join(base + ['preflight']), resume=shlex.join(resume))


def restore(root, bundle, *, verify_only=False):
    root = Path(root).resolve()
    manifest = inspect_bundle(bundle)  # verify all archive bytes BEFORE filesystem writes
    require(str(root) == manifest['source_root'],
            'Preserve the original absolute project root on the new machine: ' + manifest['source_root'] +
            '. Stored reports/configs may contain absolute paths; this tool never rewrites signed evidence.')
    runs = [below(root, n) for n in manifest['runs']]
    require(all(not active(p) for p in runs), 'Destination run is active')
    for record in manifest['files']:
        path = below(root, record['path'])
        if path.exists():
            require(path.is_file() and sha(path) == record['sha256'], 'Conflicting destination; nothing overwritten: ' + record['path'])
        else:
            require(not record.get('external', False), 'Missing external dependency; sync matching code/runtime first: ' + record['path'])
            require(not verify_only, 'Missing restored file: ' + record['path'])
    if not verify_only:
        with ExitStack() as leases:
            for run in runs:
                leases.enter_context(lease(run))
            with open_bundle(bundle) as archive:
                for record in manifest['files']:
                    path = below(root, record['path'])
                    if path.exists():
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
                    try:
                        with archive.open(payload_name(manifest, record)) as source, temporary.open('xb') as target:
                            while block := source.read(1024 * 1024):
                                target.write(block)
                        require(sha(temporary) == record['sha256'], 'Restored file hash differs')
                        require(not path.exists(), 'Destination appeared during restore')
                        temporary.rename(path)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
    require(source_identity(root) == manifest['implementation'], 'Destination source identity differs')
    return dict(summary(manifest), verified=True, **commands(root, manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    out = sub.add_parser('export')
    out.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    out.add_argument('--run', required=True)
    out.add_argument('--bundle', required=True, type=Path)
    out.add_argument('--dry-run', action='store_true')
    out.add_argument('--mode', choices=('compact', 'full'), default='compact', help='Default: live checkpoints/evidence; omit consumed replays, caches, trash and unused IL epochs')
    out.add_argument('--dependencies', choices=('external', 'repo', 'bundled'), default='external', help='Default external: code from Git and private runtime from SSH, both hash-checked. Repo/bundled include private runtime and are for private file transfer only')
    out.add_argument('--part-mb', type=int, default=100, help='100 decimal MB by default (below 100 MiB); 0 for one ZIP, 1–100 for bounded parts')
    for name in ('restore', 'verify'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--root', required=True, type=Path)
        cmd.add_argument('--bundle', required=True, type=Path)
    a = parser.parse_args()
    if a.command == 'export':
        result = export(a.root, a.run, a.bundle, dry_run=a.dry_run, mode=a.mode, dependencies=a.dependencies, part_mb=a.part_mb)
    else:
        result = restore(a.root, a.bundle, verify_only=a.command == 'verify')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
