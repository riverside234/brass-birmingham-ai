"""Verify the task-only restart transfer; qualify its latest committed state."""
import hashlib
import json
from pathlib import Path
import zipfile
from brass_ai.learning.contracts import SOURCE_ROOT, require
from .lab_contract import sha, SOURCE

DIRECTORY = 'distribution/rust-lab-g570'
CONTRACT = 'brass-lab-task-transfer/2.0.0'
RETIRED = (
    'src/brass_ai/recovery/teacher_diversity.py',
    'src/brass_ai/recovery/domino_search_fit.py',
    'src/brass_ai/rust_backend/lab_diversity_transfer.py',
    'src/brass_ai/rust_backend/lab_domino_normalization.py',
    'src/brass_ai/rust_backend/lab_domino_overlay.py',
    'src/brass_ai/rust_backend/lab_intrinsic.py',
    'lab-native/core-overlay/domino.rs',
    'scripts/prepare-rust-lab-diversity.py',
    'tests/RustBackendLab/test_teacher_diversity.py',
)


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def implementation_inventory(sources):
    return {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in sources.items()}


def check_producer(manifest, sources, producer):
    """Bind the complete fingerprint input set, including newly added modules."""
    from brass_ai.learning.contracts import digest
    actual = digest(sources)
    expected_files = manifest['implementation_files']
    current_files = implementation_inventory(sources)
    missing = sorted(set(expected_files) - set(current_files))
    added = sorted(set(current_files) - set(expected_files))
    changed = sorted(n for n in set(expected_files) & set(current_files)
                     if expected_files[n] != current_files[n])
    if actual != manifest['implementation'] or producer != manifest['producer'] or missing or added or changed:
        details = [f"implementation expected {manifest['implementation']}, actual {actual}",
                   f"native producer expected {manifest['producer']}, actual {producer}"]
        for kind, names in (('added', added), ('missing', missing), ('changed', changed)):
            if names:
                details.append(f'{kind} ({len(names)}): '+', '.join(names[:12])+(' ...' if len(names)>12 else ''))
        raise ValueError('Checkout differs from reviewed producer; '+'; '.join(details)+
                         '. Use the matching current checkout and g570 bundle; do not overlay an older source ZIP.')
    return actual


def verify(milestone, supplement=None):
    from brass_ai.step5.league import implementation_sources
    from .lab_teacher_overlay import source_fingerprint
    path = SOURCE_ROOT/DIRECTORY/'transfer-manifest.json'
    require(milestone in ('b01','b02','b025','b03','b04') and
            (supplement is None or (SOURCE_ROOT/supplement).resolve() == path), 'Unknown lab transfer/stage')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    require(manifest['contract'] == CONTRACT and manifest['base_core'] == SOURCE
            and manifest['method'] == 'search_v1' and manifest['v205_enabled'] is False
            and manifest['domino_enabled'] is False and manifest['az_db_enabled'] is False,
            'Unknown task-only transfer')
    require(manifest['retired_files'] == list(RETIRED), 'Retired-source inventory differs')
    for name in RETIRED:
        require(not (SOURCE_ROOT/name).exists(), 'Retired AZ_DB source remains; use the current checkout: '+name)
    receipt = json.loads((path.parent/'bundle.json').read_text(encoding='utf-8'))
    archive = (SOURCE_ROOT/receipt['path']).resolve()
    require(archive.parent == path.parent and sha(archive) == receipt['sha256'], 'Changed lab archive')
    with zipfile.ZipFile(archive) as z:
        require(json.loads(z.read(DIRECTORY+'/transfer-manifest.json')) == manifest, 'Archive/sidecar differs')
    check_producer(manifest, implementation_sources(), source_fingerprint())
    for name, expected in manifest['files'].items():
        source = (SOURCE_ROOT/name).resolve()
        require(source.is_relative_to(SOURCE_ROOT) and source.is_file() and text_sha(source) == expected,
                'Changed lab source: '+name)
    for name, expected in manifest['archives'].items():
        source = (SOURCE_ROOT/name).resolve()
        require(source.is_relative_to(SOURCE_ROOT) and sha(source) == expected, 'Changed original archive: '+name)
    return dict(manifest_sha256=sha(path),implementation=manifest['implementation'],producer=manifest['producer'],
                files=len(manifest['files']),milestone=milestone,teacher_profile='task-only',lab_qualification='Pending')
