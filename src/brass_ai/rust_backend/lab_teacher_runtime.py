"""Pinned verification-only adapter update; original search qualification survives."""
import hashlib
from pathlib import Path

from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from .lab_contract import pin

POLICY='sampled-unbounded-verification/1'
SELECTOR='artifacts/rust-backend/teacher-verification/build.json'


def fingerprint():
    from .lab_perf_runtime import FILES
    files=(*FILES,'src/teacher.rs','src/bin/teacher.rs','core-overlay/teacher.rs')
    return digest({name:hashlib.sha256((SOURCE_ROOT/'lab-native'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                   for name in files})


def resolve(root,original):
    from .lab_backend import checked_pin
    from .lab_replay_compat import accepts,reviewed_validator
    from .lab_teacher_overlay import source_fingerprint
    path=Path(root)/SELECTOR
    require(path.is_file(),'Prepare the unbounded teacher verification adapter with scripts/prepare-rust-teacher-verification.py')
    receipt=pin(path);value=checked_pin(receipt)
    require(value['contract']=='brass-lab-b07-teacher-build/1.0.0' and
        value.get('verification_policy')==POLICY and value.get('verification_source')==fingerprint() and
        value.get('original_teacher_receipt')==original['receipt'] and value['producer']==source_fingerprint() and
        (accepts(value['implementation']) or reviewed_validator(value['implementation'])) and value['core_source']==original['core_source'] and
        value['bootstrap']==original['bootstrap'] and value['wire_regressions_passed'] is True,
        'Teacher verification runtime/source/original binding changed; rerun prepare-rust-teacher-verification.py')
    for item in (value['binary'],value['tests'],value['build_log'],value['bootstrap']['binary'],*value['bootstrap']['dependencies']):
        require(pin(item['path'])==item,'Teacher verification runtime/evidence changed: '+item['path'])
    return dict(value,receipt=receipt)
