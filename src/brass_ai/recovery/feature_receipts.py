"""Exact reviewed feature-source equivalence, independent of execution ID.

Only validation/storage code changed. Keep byte-identical Features-4 caches;
unknown encoder dependencies still get a new identity and must be rebuilt.
"""
from functools import lru_cache
import hashlib
from brass_ai.bank import read
from brass_ai.learning.contracts import CONFIG_DIR,SOURCE_ROOT,compatibility,digest

PATHS=('recovery/encoding.py','recovery/inputs.py','recovery/replay.py','recovery/il_replay_storage.py','runtime_contracts.py')


@lru_cache(maxsize=1)
def review():
    path=CONFIG_DIR/'collector-performance-teacher-reuse.json'
    if not path.is_file():return None
    value=read(path).get('derived_features')
    actual={name:hashlib.sha256((SOURCE_ROOT/'src/brass_ai'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in PATHS}
    return value if value is not None and value['current_files']==actual and value['compatibility_sha256']==digest(compatibility()) else None


def versions(current):
    value=review()
    return {current,*value['previous_encoders']} if value else {current}
