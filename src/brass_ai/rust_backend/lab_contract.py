"""B01 admission and immutable input verification; no training state mutation."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import struct
import zipfile

from brass_ai.learning.contracts import digest, read_json, require

METHOD = 'search_v1'
ALGORITHM = 'brass-gumbel-is-mcts/count-v1'
SESSION = 'brass-gumbel-is-mcts/session-v1'
B01_CONTRACT = 'brass-lab-b01/2.0.0'
SOURCE = 'fa0b3f7b74e9ed638c8b6f96d3f23b0910b4e5df9387ff8df58d5472a8014dd2'
ARCHIVE = '6dc6e230b96a4c57bd14d85fb994c8d0bd624b9b504d83647ff8d157842179c7'
POLICIES = ('policy.json', 'policy-k8.json', 'policy-t15.json', 'policy-k8-t15.json')
INPUT_FILES = frozenset(('source.zip', 'candidate.json', 'deployment.json', 'installed-smoke.json',
    'local-acceptance.json', 'local-qualification.json', 'local-native-build.json', 'fixtures.json',
    'vendor-LICENSE', 'vendor-UPSTREAM.json', 'vendor-upstream-local.patch', *POLICIES))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def admit(profile):
    expected = dict(contract='brass-lab-search-v1/1.0.0', method=METHOD,
                    algorithm=ALGORITHM, session_contract=SESSION, v205_enabled=False,
                    native_source_sha256=SOURCE, source_archive_sha256=ARCHIVE)
    require(type(profile) is dict and set(profile) == set(expected), 'Invalid lab backend profile fields')
    require(profile == expected and profile['v205_enabled'] is False,
            'Lab profile requires explicit search_v1 with search_v2/V205 disabled')
    return digest(profile)


def admit_search_request(profile, request):
    """Required at the future B04 dispatch boundary; never fill a missing method."""
    admit(profile)
    require(request.get('method') == METHOD, 'Missing/unsupported lab search method; V205 is disabled')
    require(not any(k in request for k in ('reuse', 'reuse_policy', 'v205_enabled', 'focus_policy',
                                         'composite_contract', 'search_revision')),
            'Focus/composite/subtree-reuse overrides are not admitted')
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
    require(manifest['contract'] == 'brass-lab-b01-inputs/2.0.0'
            and manifest.get('method') == METHOD and manifest.get('profile_id') == admit(profile),
            'Unknown B01 inputs/profile')
    require(set(manifest['files']) == INPUT_FILES, 'Incomplete/unexpected B01 input inventory')
    for name, expected in manifest['files'].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), 'Input pin escapes bundle')
        require(path.is_file() and sha(path) == expected, 'B01 input changed: ' + name)
    require(sha(directory / 'source.zip') == ARCHIVE, 'Not the selected local source archive')
    build = read_json(directory / 'candidate.json')
    deployment = read_json(directory / 'deployment.json')
    smoke = read_json(directory / 'installed-smoke.json')
    acceptance = read_json(directory / 'local-acceptance.json')
    qualification = read_json(directory / 'local-qualification.json')
    native = read_json(directory / 'local-native-build.json')
    require(build['native_source_sha256'] == deployment['source_sha256'] == native['source_sha256'] == SOURCE,
            'Selected source fingerprint differs')
    require(build['version'] == deployment['version'] == smoke['version'] == qualification['version'] == '1.4.42-rust-local'
            and build['search_method'] == deployment['method'] == qualification['search_method'] == smoke['searchMethod'] == METHOD
            and all(x is False for x in (build['v205_enabled'], deployment['v205_enabled'],
                                        qualification['v205_enabled'], smoke['v205Enabled']))
            and all(x['passed'] is True for x in (deployment, qualification, smoke)),
            'Local search_v1 installation not accepted')
    require(sha(directory / 'candidate.json') == deployment['build_sha256'] == qualification['build_manifest_sha256']
            and sha(directory / 'local-qualification.json') == deployment['qualification_sha256'] == smoke['qualificationSha256'].lower()
            and sha(directory / 'installed-smoke.json') == deployment['installed_smoke_sha256'],
            'Installed smoke/qualification is not bound to deployment')
    native_pins = {p['path']: p['sha256'] for p in build['nativeFiles']}
    tested = {p['path']: p['sha256'].lower() for p in smoke['testedFiles']}
    expected_tested = {'BrassNative/' + p['path']: p['sha256'] for p in build['nativeFiles']}
    expected_tested.update({'BrassBirminghamUnity_Data/Managed/' + p['name']: p['sha256'] for p in build['files']})
    require(tested == expected_tested and native_pins['alpha-g890/native-source.zip'] == ARCHIVE
            and sha(directory / 'local-native-build.json') == build['native_build_sha256'] == native_pins['alpha-g890/build.json'],
            'Installed package/source identity differs')
    policy_pins = {p['path']: p['sha256'] for p in native['files']}
    for name in POLICIES:
        policy = read_json(directory / name)
        require(sha(directory / name) == policy_pins[name] == native_pins['alpha-g890/' + name]
                and policy['method'] == METHOD, 'Installed policy is not qualified search_v1: ' + name)
    require(acceptance.get('passed') is True, 'Part 1 acceptance did not pass')
    return dict(profile_id=admit(profile), inputs=pin(directory / 'inputs.json'),
                archive=pin(directory / 'source.zip'), native_source_sha256=SOURCE,
                scope='local evidence verified; Linux/CUDA qualification still required')
