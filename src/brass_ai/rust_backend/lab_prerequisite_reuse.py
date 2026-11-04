"""Exact reviewed reuse of r7 collection evidence for the B03 CUDA check repair.

No ignore-source flag: only the sealed repair's enumerated old/new file hashes
are accepted. Every other executable/runtime input and original receipt stays
subject to the existing qualification checks.
"""
from pathlib import Path
import hashlib

from brass_ai.learning.contracts import SOURCE_ROOT, read_json, require
from .lab_contract import pin

DIRECTORY = 'distribution/rust-lab-g570'
REVIEW = DIRECTORY+'/prerequisite-reuse.json'
ALLOWED = frozenset((
    'configs/learning/selfplay-resume-sources.json',
    'src/brass_ai/rust_backend/lab_continuation.py',
    'src/brass_ai/rust_backend/lab_b03_smoke.py',
    'src/brass_ai/rust_backend/lab_prerequisite_reuse.py',
))


def reviewed():
    from brass_ai.step5.league import implementation_id
    from .lab_teacher_overlay import source_fingerprint
    manifest = read_json(SOURCE_ROOT/DIRECTORY/'transfer-manifest.json')
    raw = (SOURCE_ROOT/REVIEW).read_bytes()
    require(hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == manifest['files'].get(REVIEW),
            'Prerequisite reuse review changed or is not sealed')
    value = read_json(SOURCE_ROOT/REVIEW)
    require(value['contract'] == 'brass-lab-prerequisite-reuse/1.0.0'
            and value['to_implementation'] == manifest['implementation'] == implementation_id()
            and value['producer'] == manifest['producer'] == source_fingerprint()
            and value['scope'] == ['b01', 'b02', 'b025'], 'Unreviewed prerequisite reuse')
    require(set(value['changes']) == ALLOWED, 'Prerequisite review exceeds the diagnostic repair')
    return value


def verify_delta(previous, current, review):
    changed = {name for name in previous.keys() | current.keys() if previous.get(name) != current.get(name)}
    require(changed == set(review['changes']) == ALLOWED,
            'B01 executable sources/runtime changed outside the reviewed B03 repair')
    for name, expected in review['changes'].items():
        require(previous.get(name) == expected['before'] and current.get(name) == expected['after'],
                'B01 prerequisite source differs from exact reviewed revision: '+name)


def verify_b02(path, implementation):
    value = reviewed()
    require(implementation == value['from_implementation']
            and pin(Path(path)) == value['receipts']['b02'],
            'B02 evidence does not match the exact reviewed collection receipt')
    return dict(review=pin(SOURCE_ROOT/REVIEW), original_b02=value['receipts']['b02'],
                from_implementation=implementation, to_implementation=value['to_implementation'])
