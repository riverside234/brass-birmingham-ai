"""Private dotenv credentials, exposed only through Git's credential protocol."""
import argparse
import getpass
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
from urllib.parse import quote, unquote, urlsplit


NAME = '.env.results-git'
KEY = 'RESULTS_GIT_URL'
ENV = 'RESULTS_GIT_AUTH_FILE'
REPOSITORY = 'riverside234/ai-training-results'
REMOTE = f'https://github.com/{REPOSITORY}.git'


def require(value, message):
    if not value:
        raise ValueError(message)


def parse_url(value):
    # Errors deliberately never include the supplied URL, username or token.
    try:
        url = urlsplit(value)
        require(url.scheme == 'https' and url.hostname == 'github.com'
                and url.port is None and not url.query and not url.fragment
                and url.path.rstrip('/').removesuffix('.git').lstrip('/') == REPOSITORY,
                'Invalid results credential URL; it must name the HTTPS results repository')
        username, token = unquote(url.username or ''), unquote(url.password or '')
        require(username and token and not any(c.isspace() or c == '\0' for c in username + token),
                'Results credential URL must contain a username and token without whitespace')
    except (ValueError, TypeError):
        raise ValueError('Invalid results credential URL; check the private dotenv file') from None
    return username, token


def read(path):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), 'Results credential file must be a regular file')
    if os.name != 'nt':
        require(not stat.S_IMODE(info.st_mode) & 0o077, 'Run chmod 600 on .env.results-git before uploading')
    require(info.st_size <= 8192, 'Results credential file is too large')
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:]
        key, separator, value = line.partition('=')
        require(separator and key.strip() == KEY and KEY not in values, 'Invalid results credential dotenv entry')
        value = value.strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        values[key.strip()] = value
    require(KEY in values, 'Missing RESULTS_GIT_URL in .env.results-git')
    return parse_url(values[KEY])


def configure(root, username, token):
    from results_git import git
    root = Path(root).resolve()
    path = root / NAME
    require(not path.is_symlink(), 'Refusing a linked results credential file')
    require(git(root, 'check-ignore', '--quiet', '--no-index', '--', NAME, allowed=(0, 1)).returncode == 0,
            'Add /.env.results-git to the code repository .gitignore first')
    require(git(root, 'ls-files', '--error-unmatch', '--', NAME, allowed=(0, 1)).returncode == 1,
            'Credential file is already tracked; refuse to store a secret in it')
    username, token = username.strip(), token.strip()
    value = f'https://{quote(username, safe="")}:{quote(token, safe="")}@github.com/{REPOSITORY}.git'
    parse_url(value)
    descriptor, temporary = tempfile.mkstemp(prefix=NAME + '-', dir=root)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(KEY + '=' + value + '\n')
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def git_options(root, remote, env):
    path = Path(root).resolve() / NAME
    # A token for this repository must never be sent to another host/repository
    # or affect local/SSH operations. Existing Git credential helpers still work
    # when the optional file is absent.
    if not isinstance(remote, str) or remote.rstrip('/') not in (REMOTE, REMOTE.removesuffix('.git')) or not path.exists():
        return []
    read(path)  # Validate privately before starting the network operation.
    helper = '!' + shlex.join([Path(sys.executable).as_posix(), Path(__file__).resolve().as_posix()])
    env[ENV] = str(path)
    env['GIT_TRACE_REDACT'] = '1'
    return ['-c', 'credential.helper=', '-c', 'credential.helper=' + helper,
            '-c', 'credential.useHttpPath=true']


def helper(action, stream, output):
    if action != 'get':
        return
    fields = {}
    for line in stream:
        if not line.strip():
            break
        key, separator, value = line.rstrip('\n').partition('=')
        if separator:
            fields[key] = value
    if fields.get('protocol') != 'https' or fields.get('host') != 'github.com' \
            or fields.get('path', '').rstrip('/').removesuffix('.git') != REPOSITORY:
        return
    path = os.environ.get(ENV)
    require(path and Path(path).name == NAME, 'Missing private results credential file')
    username, token = read(path)
    # This stdout is a pipe consumed by Git, never an export log or report.
    output.write(f'username={username}\npassword={token}\n\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('configure', 'get', 'store', 'erase'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        if args.action == 'configure':
            require(sys.stdin.isatty(), 'Configure credentials in an interactive terminal; the token is not accepted as an argument')
            username = input('GitHub username [riverside234]: ').strip() or 'riverside234'
            token = getpass.getpass('GitHub personal access token (hidden): ')
            configure(args.root, username, token)
            print('Saved ignored .env.results-git (mode 600). Remote URLs are unchanged.')
        else:
            helper(args.action, sys.stdin, sys.stdout)
    except (ValueError, OSError):
        # Never interpolate parser/file errors which could contain a secret.
        print('Results credentials unavailable or invalid; use configure and check file permissions.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
