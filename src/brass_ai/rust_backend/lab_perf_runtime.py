"""Additive ordinary wire upgrade; original adoption/teacher receipts survive."""
import hashlib
from pathlib import Path
from brass_ai.learning.contracts import SOURCE_ROOT, CONFIG_DIR, digest, require

CONTRACT='brass-collector-performance-build/1'
FILES=('Cargo.toml','Cargo.lock','src/main.rs','src/ordinary_delta.rs','src/lib.rs','src/collection_wire.rs','src/scoring.rs')


def predecessor(item,original):
    from brass_ai.bank import read
    from .lab_backend import check_build
    from .lab_contract import pin
    value=check_build(item)
    allowed=read(CONFIG_DIR/'collector-performance-teacher-reuse.json').get('ordinary_predecessors',[])
    require(value.get('contract')==CONTRACT and value.get('ordinary_source') in allowed and
            value['parent_build_sha256']==original['_receipt_sha256'] and value['bootstrap']==original['bootstrap'] and
            value['b01']==original['b01'] and value.get('wire_regressions_passed') is True,
            'Unknown previous collector source/parent')
    for name in ('tests','build_log','performance_tests'):
        require(pin(value[name]['path'])==value[name],'Previous collector build evidence changed')
    return value


def fingerprint():
    return digest({name:hashlib.sha256((SOURCE_ROOT/'lab-native'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                   for name in FILES})


def resolve(root, original):
    from .lab_backend import check_build
    from .lab_contract import pin
    path=Path(root)/'artifacts/rust-backend/collector-performance/build.json'
    if not path.exists():return original
    candidate=check_build(pin(path))
    require(candidate['contract']==CONTRACT and candidate['ordinary_source']==fingerprint() and
            candidate['parent_build_sha256']==original['_receipt_sha256'] and
            candidate['bootstrap']==original['bootstrap'] and candidate['b01']==original['b01'] and
            candidate['performance_wire']=='ordinary-collector-fast/1' and candidate['wire_regressions_passed'] is True,
            'Collector performance runtime/source/parent differs; rebuild with prepare-rust-collector-performance.py')
    for name in ('tests','build_log','performance_tests'):
        require(pin(candidate[name]['path'])==candidate[name],'Collector performance build evidence changed')
    previous=candidate.get('previous_runtimes',[])
    require(type(previous) is list and len(previous)<=8,'Previous collector history exceeds bound')
    identities=[original['_receipt_sha256']]
    for item in previous:
        value=predecessor(item,original);identities.append(value['_receipt_sha256'])
    return dict(candidate,_previous_build_sha256=original['_receipt_sha256'],
                _previous_build_sha256s=tuple(dict.fromkeys(identities)))
