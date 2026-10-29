"""B01 admission and immutable input verification; no training state mutation."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import struct
import zipfile

from brass_ai.learning.contracts import digest, read_json, require

METHOD = 'search_v2_focus_only'
FOCUS = 'adaptive-focus/2.0.0'
COMPOSITE = 'brass-composite-plan/2.0.0'
SOURCE = 'fadb7af2524eb84c960c24cf2c846ceef1de3a8ba07871f59abb1a707625f994'
ARCHIVE = 'dba778c92e4716d4bc3f5e9eae8ed05a6e40b88963dc6612846cef66b39da777'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def admit(profile):
    expected = dict(contract='brass-lab-cache-only/1.0.0', method=METHOD,
                    focus_policy=FOCUS, composite_contract=COMPOSITE, v205_enabled=False,
                    native_source_sha256=SOURCE, source_archive_sha256=ARCHIVE)
    require(type(profile) is dict and set(profile) == set(expected), 'Invalid lab backend profile fields')
    require(profile == expected and profile['v205_enabled'] is False,
            'Lab profile requires search_v2_focus_only with V205 disabled')
    return digest(profile)


def admit_search_request(profile, request):
    """Required at the future B04 dispatch boundary; never fill a missing method."""
    admit(profile)
    require(request.get('method') == METHOD, 'Missing/unsupported lab search method; V205 is disabled')
    require(not any(k in request for k in ('reuse', 'reuse_policy', 'v205_enabled')),
            'Subtree-reuse overrides are not admitted')
    return request


def native_fingerprint(directory):
    root = Path(directory)
    value = hashlib.sha256()
    for path in sorted(root.rglob('*'), key=lambda p: p.relative_to(root).parts):
        if not path.is_file() or path.suffix not in ('.rs', '.toml', '.lock', '.json'):
            continue
        if any(p in ('target', '.git', '__pycache__') for p in path.relative_to(root).parts):
            continue
        name, data = path.relative_to(root).as_posix().encode(), path.read_bytes()
        value.update(struct.pack('<Q', len(name)) + name + struct.pack('<Q', len(data)) + data)
    return value.hexdigest()


def extract_new(archive, destination):
    """Bounded extraction into a new directory; no traversal, links or overwrites."""
    destination = Path(destination).resolve()
    require(not destination.exists(), 'Extraction destination already exists')
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        require(len(infos) <= 10000 and sum(i.file_size for i in infos) <= 256 * 1024**2,
                'Archive exceeds B01 bounds')
        names = set()
        for entry in infos:
            name = PurePosixPath(entry.filename)
            require(entry.orig_filename == entry.filename and '\\' not in entry.orig_filename,
                    'Ambiguous archive path')
            require(not name.is_absolute() and '..' not in name.parts and '\\' not in entry.filename
                    and ':' not in entry.filename and name.parts, 'Unsafe archive path')
            require(entry.filename not in names, 'Duplicate archive member')
            names.add(entry.filename)
            require((entry.external_attr >> 16) & 0o170000 != 0o120000, 'Archive symlink')
        destination.mkdir(parents=True)
        for entry in infos:
            path = destination / entry.filename
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(source.read(entry))


def verify_inputs(directory, profile):
    admit(profile)
    directory = Path(directory)
    manifest = read_json(directory / 'inputs.json')
    require(manifest['contract'] == 'brass-lab-b01-inputs/1.0.0', 'Unknown B01 inputs')
    for name, expected in manifest['files'].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), 'Input pin escapes bundle')
        require(path.is_file() and sha(path) == expected, 'B01 input changed: ' + name)
    require(sha(directory / 'source.zip') == ARCHIVE, 'Not the selected local source archive')
    build = read_json(directory / 'candidate.json')
    deployment = read_json(directory / 'deployment.json')
    smoke = read_json(directory / 'installed-smoke.json')
    acceptance = read_json(directory / 'local-acceptance.json')
    require(build['native_source_sha256'] == SOURCE, 'Selected source fingerprint differs')
    require(deployment['passed'] is True and deployment['method'] == METHOD
            and deployment['v205Enabled'] is False, 'Cache-only installation not accepted')
    require(sha(directory / 'installed-smoke.json') == deployment['smokeVerificationSha256'].lower(),
            'Installed smoke is not bound to deployment')
    require(smoke['passed'] is True and smoke['searchMethod'] == METHOD and smoke['v205Enabled'] is False
            and smoke['candidateSha256'].lower() == sha(directory / 'candidate.json'), 'Installed candidate mismatch')
    require(acceptance.get('passed') is True, 'Part 1 acceptance did not pass')
    return dict(profile_id=admit(profile), inputs=pin(directory / 'inputs.json'),
                archive=pin(directory / 'source.zip'), native_source_sha256=SOURCE,
                scope='local evidence verified; Linux/CUDA qualification still required')
