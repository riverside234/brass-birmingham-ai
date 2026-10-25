import os
from pathlib import Path

from brass_ai.learning.contracts import SOURCE_ROOT, read_json
from brass_ai.rust_backend.fixtures import read_gzip


def evidence():
    return SOURCE_ROOT/os.environ.get('BRASS_RUST_RUN','artifacts/rust-backend/20261024-g890-l01-l02')


def root_fixture():
    directory=evidence()/'local/fixtures'
    manifest=read_json(directory/'corpus.json')
    for entry in manifest['roots']:
        value=read_gzip(directory/entry['path'])
        if value['projection']['context']['phase']=='action' and value['projection']['chance'] is None:
            return value
    raise AssertionError('No ordinary decision fixture')
