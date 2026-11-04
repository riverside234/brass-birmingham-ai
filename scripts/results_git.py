"""Publish analysis bundles to their own repository, never the code index."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


REMOTE = 'https://github.com/riverside234/ai-training-results.git'


def require(value, message):
    if not value:
        raise ValueError(message)


def git(directory, *args, data=None, allowed=(0,)):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_OBJECT_DIRECTORY',
                 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_COMMON_DIR'):
        env.pop(name, None)
    result = subprocess.run(['git', '-c', 'core.longpaths=true', '--literal-pathspecs', '-C', str(directory), *args],
                            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, timeout=300)
    if result.returncode not in allowed:
        message = result.stderr.decode('utf-8', 'replace').strip()
        raise ValueError(f'Results Git {args[0]} failed: {message}')
    return result


def identity(remote):
    value = remote.rstrip('/')
    for prefix in ('https://github.com/', 'ssh://git@github.com/', 'git@github.com:'):
        if value.startswith(prefix):
            return 'github.com/' + value[len(prefix):].removesuffix('.git').lower()
    return value.removesuffix('.git')


def head(directory, ref='HEAD'):
    result = git(directory, 'rev-parse', '--verify', ref, allowed=(0, 128))
    return result.stdout.decode().strip() if result.returncode == 0 else None


def verify(root, remote=REMOTE):
    root = Path(root).resolve()
    directory = root / 'results'
    require(directory.is_dir() and not directory.is_symlink(), 'results/ must be an ordinary directory')
    # Explicit .git prevents Git from walking up to the main repository.
    require((directory / '.git').is_dir() and not (directory / '.git').is_symlink(),
            'results/ must have its own .git directory; a parent repository or submodule is not accepted')
    top = Path(git(directory, 'rev-parse', '--show-toplevel').stdout.decode().strip()).resolve()
    require(top == directory.resolve(), 'Refusing to publish the parent code repository')
    origin = git(directory, 'remote', 'get-url', 'origin').stdout.decode().strip()
    push_origin = git(directory, 'remote', 'get-url', '--push', 'origin').stdout.decode().strip()
    require(identity(origin) == identity(push_origin) == identity(remote), 'Unexpected results origin/push URL')
    branch = git(directory, 'symbolic-ref', '--quiet', '--short', 'HEAD').stdout.decode().strip()
    require(branch == 'main', 'Results repository must use the main branch')
    parent = git(root, 'rev-parse', '--show-toplevel', allowed=(0, 128))
    if parent.returncode == 0 and Path(parent.stdout.decode().strip()).resolve() == root:
        require(not git(root, 'ls-files', '-z', '--', 'results').stdout,
                'The code repository still tracks results/. Adopt the results-history migration first; use --no-push for local export')
    return directory


def prepare(root, remote=REMOTE):
    """Download previous bundles before incremental export; preserve local files."""
    root = Path(root).resolve()
    directory = root / 'results'
    parent = git(root, 'rev-parse', '--show-toplevel', allowed=(0, 128))
    if parent.returncode == 0 and Path(parent.stdout.decode().strip()).resolve() == root:
        require(not git(root, 'ls-files', '-z', '--', 'results').stdout,
                'The code repository still tracks results/. Adopt the results-history migration first; use --no-push for local export')
    require(not directory.is_symlink(), 'Linked results/ is not supported')
    directory.mkdir(exist_ok=True)
    if not (directory / '.git').exists():
        artifacts = root / 'artifacts'
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='results-git-', dir=artifacts) as temp:
            clone = Path(temp) / 'checkout'
            git(root, 'clone', '--no-checkout', '--', remote, str(clone))
            git(clone, 'config', 'core.autocrlf', 'false')
            git(clone, 'config', 'core.longpaths', 'true')
            if head(clone) is None:
                git(clone, 'symbolic-ref', 'HEAD', 'refs/heads/main')
            else:
                require(git(clone, 'symbolic-ref', '--short', 'HEAD').stdout.strip() == b'main',
                        'Results remote default branch must be main')
                git(clone, 'read-tree', 'HEAD')
            # The existing report files are not moved, deleted or overwritten.
            shutil.move(str(clone / '.git'), str(directory / '.git'))
        if head(directory):
            missing = git(directory, 'ls-files', '--deleted', '-z').stdout
            if missing:
                git(directory, 'checkout-index', '-z', '--stdin', data=missing)
    if head(directory) is None and not (directory / '.gitattributes').exists():
        (directory / '.gitattributes').write_bytes(b'* -text\n')
    return verify(root, remote)


@contextmanager
def transaction(root, remote=REMOTE):
    directory = prepare(root, remote)
    lock = directory / '.git' / 'report-export.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError('Another report export is active; if it was interrupted, inspect/remove results/.git/report-export.lock') from error
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()) + '\n')
        require(not git(directory, 'diff', '--cached', '--name-only', '-z').stdout,
                'Results index has staged changes; finish that commit before automatic export')
        refs = git(directory, 'ls-remote', '--heads', 'origin', 'refs/heads/main').stdout
        if refs:
            git(directory, 'fetch', 'origin', 'refs/heads/main:refs/remotes/origin/main')
            remote_head = head(directory, 'refs/remotes/origin/main')
            local = head(directory)
            require(local is not None, 'Results HEAD is missing')
            if local != remote_head:
                behind = git(directory, 'merge-base', '--is-ancestor', local, remote_head, allowed=(0, 1))
                if behind.returncode == 0:
                    require(not git(directory, 'status', '--porcelain').stdout,
                            'Remote results advanced while local files changed; reconcile them before exporting')
                    git(directory, 'merge', '--ff-only', remote_head)
                else:
                    ahead = git(directory, 'merge-base', '--is-ancestor', remote_head, local, allowed=(0, 1))
                    require(ahead.returncode == 0, 'Results history diverged; automatic publication never force-pushes')
        yield directory
    finally:
        lock.unlink(missing_ok=True)


def publish(root, directory, output, manifest, remote=REMOTE):
    """Stage only the new export and its verified incremental dependencies."""
    verify(root, remote)
    root, directory, output = Path(root).resolve(), Path(directory).resolve(), Path(output).resolve()
    require(output.is_relative_to(directory) and output != directory, 'Export is outside the results repository')
    paths, visited = set(), set()
    def bundle(base, receipt):
        base = Path(base).resolve()
        require(base.is_relative_to(directory) and base != directory, 'Incremental dependency is outside results/')
        if base in visited:
            return
        visited.add(base)
        require(str(receipt.get('format', '')).startswith('selfplay-report-export/'), 'Invalid dependency manifest')
        def include(name, expected=None):
            path = base / name
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                    and path.resolve().is_relative_to(base) and not path.is_symlink() and path.is_file(),
                    'Missing/unsafe report publication path')
            if expected is not None:
                require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                        'Report changed before Git publication: ' + str(path))
            paths.add(path.relative_to(directory).as_posix())
        for entry in receipt['files']:
            include(entry['path'], entry['sha256'])
        for name in receipt.get('generated', ['recorded-config.json', 'export-manifest.json']):
            include(name)
        include('export-manifest.json')
        for entry in receipt.get('reused_files', []):
            previous = root / entry['previous_export']
            require(previous.resolve().is_relative_to(directory) and previous.resolve() != directory,
                    'Incremental dependency is outside results/')
            previous_manifest = previous / 'export-manifest.json'
            bundle(previous, json.loads(previous_manifest.read_text(encoding='utf-8-sig')))
    bundle(output, manifest)
    if (directory / '.gitattributes').is_file() and not git(directory, 'ls-files', '--', '.gitattributes').stdout:
        require((directory / '.gitattributes').read_bytes() == b'* -text\n', 'Unexpected pending results attributes')
        paths.add('.gitattributes')
    git(directory, 'add', '--pathspec-from-file=-', '--pathspec-file-nul',
        data=b'\0'.join(path.encode('utf-8') for path in sorted(paths)) + b'\0')
    if git(directory, 'diff', '--cached', '--quiet', allowed=(0, 1)).returncode:
        git(directory, 'commit', '-m', f'Export {Path(manifest["run"]).name} g{manifest["committed_generation"]:04d}')
    # No force/rebase/reset. A rejected push leaves the commit available to retry.
    git(directory, 'push', '--set-upstream', 'origin', 'HEAD:refs/heads/main')
    return head(directory)
