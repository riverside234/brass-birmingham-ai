"""Publish analysis bundles to their own repository, never the code index."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit
import results_git_auth


REMOTE = 'https://github.com/riverside234/ai-training-results.git'


def require(value, message):
    if not value:
        raise ValueError(message)


def git(directory, *args, data=None, allowed=(0,), auth_root=None, auth_remote=None):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_OBJECT_DIRECTORY',
                 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_COMMON_DIR'):
        env.pop(name, None)
    authentication = results_git_auth.git_options(auth_root, auth_remote, env) if auth_root is not None else []
    literal = [] if args[0] == 'check-ignore' else ['--literal-pathspecs']
    result = subprocess.run(['git', '-c', 'core.longpaths=true', *authentication,
                             *literal, '-C', str(directory), *args],
                            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, timeout=300)
    if result.returncode not in allowed:
        message = result.stderr.decode('utf-8', 'replace').strip()
        if 'could not read Username' in message or 'Authentication failed' in message:
            message += ('\nConfigure credentials once with: python scripts/results_git_auth.py configure'
                        '\nTo retry an already committed upload: python scripts/export-training-reports.py --push-only')
        raise ValueError(f'Results Git {args[0]} failed: {message}')
    return result


def identity(remote):
    """Compare GitHub repositories independently of transport and credentials.

    Git expands insteadOf/pushInsteadOf in get-url, so authenticated HTTPS URLs
    can appear even when origin was configured with a clean URL. Never include
    those URLs in validation errors. Non-GitHub/local remotes retain their exact
    identity; --results-remote still explicitly selects an alternate repository.
    """
    value = remote.rstrip('/')
    scp = re.fullmatch(r'git@github\.com:(.+)', value, flags=re.IGNORECASE)
    if scp:
        return 'github.com/' + scp[1].lower().removesuffix('.git')
    try:
        url = urlsplit(value)
        github = (url.hostname == 'github.com' and not url.query and not url.fragment
                  and ((url.scheme == 'https' and url.port in (None, 443))
                       or (url.scheme == 'ssh' and url.username == 'git'
                           and url.password is None and url.port in (None, 22))))
        if github:
            return 'github.com/' + url.path.removeprefix('/').lower().removesuffix('.git')
    except ValueError:
        # Malformed URLs cannot gain the canonical identity of the expected URL.
        pass
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
    origins = git(directory, 'remote', 'get-url', '--all', 'origin').stdout.decode().splitlines()
    push_origins = git(directory, 'remote', 'get-url', '--push', '--all', 'origin').stdout.decode().splitlines()
    expected = identity(remote)
    mismatch = [kind for kind, urls in (('fetch', origins), ('push', push_origins))
                if not urls or any(identity(url) != expected for url in urls)]
    require(not mismatch,
            'Unexpected results origin/push URL: ' + '/'.join(mismatch) +
            ' does not identify the selected results repository. Correct results/ origin, '
            'or use --results-remote URL for an intentional alternate repository. '
            'Equivalent GitHub HTTPS/SSH URLs, including HTTPS credentials, are accepted.')
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
            git(root, 'clone', '--no-checkout', '--', remote, str(clone), auth_root=root, auth_remote=remote)
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
        refs = git(directory, 'ls-remote', '--heads', 'origin', 'refs/heads/main', auth_root=root, auth_remote=remote).stdout
        if refs:
            git(directory, 'fetch', 'origin', 'refs/heads/main:refs/remotes/origin/main', auth_root=root, auth_remote=remote)
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
    paths, metadata = set(), set()
    def include(base, name, expected=None):
        base = Path(base).resolve()
        require(base.is_relative_to(directory) and base != directory, 'Incremental dependency is outside results/')
        path = base / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and '.git' not in Path(name).parts
                and path.resolve().is_relative_to(base) and not path.is_symlink() and path.is_file(),
                'Missing/unsafe report publication path')
        if expected is not None:
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    'Report changed before Git publication: ' + str(path))
        paths.add(path.relative_to(directory).as_posix())

    require(str(manifest.get('format', '')).startswith('selfplay-report-export/'), 'Invalid export manifest')
    for entry in manifest['files']:
        include(output, entry['path'], entry['sha256'])
    for name in manifest.get('generated', ['recorded-config.json', 'export-manifest.json']):
        include(output, name)
    include(output, 'export-manifest.json')
    for entry in manifest.get('reused_files', []):
        previous = root / entry['previous_export']
        # Export receipts point directly to physical files, not to a dependency
        # chain. Only this exact reused file is part of the new publication.
        # Revalidating/re-staging every file in its old bundle would block on
        # unrelated stale reports, or silently upload historical local edits.
        include(previous, entry['previous_path'], entry['sha256'])
        previous_manifest = previous / 'export-manifest.json'
        if previous_manifest not in metadata and previous_manifest.is_file():
            metadata.add(previous_manifest)
            relative = previous_manifest.relative_to(directory).as_posix()
            tracked = git(directory, 'ls-files', '--error-unmatch', '--', relative, allowed=(0, 1))
            if tracked.returncode:
                # An unpublished local cache needs its existing index so a
                # fresh clone can discover the supplied physical file again.
                include(previous, 'export-manifest.json')
    if (directory / '.gitattributes').is_file() and not git(directory, 'ls-files', '--', '.gitattributes').stdout:
        require((directory / '.gitattributes').read_bytes() == b'* -text\n', 'Unexpected pending results attributes')
        paths.add('.gitattributes')
    git(directory, 'add', '--pathspec-from-file=-', '--pathspec-file-nul',
        data=b'\0'.join(path.encode('utf-8') for path in sorted(paths)) + b'\0')
    if git(directory, 'diff', '--cached', '--quiet', allowed=(0, 1)).returncode:
        git(directory, 'commit', '-m', f'Export {Path(manifest["run"]).name} g{manifest["committed_generation"]:04d}')
    return push(root, directory, remote)


def push(root, directory, remote=REMOTE):
    verify(root, remote)
    require(head(directory), 'There are no committed reports to upload')
    # No force/rebase/reset. A rejected push leaves the commit available to retry.
    git(directory, 'push', '--set-upstream', 'origin', 'HEAD:refs/heads/main', auth_root=root, auth_remote=remote)
    return head(directory)
