"""Reuse only byte-bound passing B07 phases at the same committed boundary."""
import gzip
import hashlib
import json
from pathlib import Path

from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, digest, require
from brass_ai.rust_backend.lab_contract import pin
from brass_ai.rust_backend.lab_snapshot import verify_unchanged
from brass_ai.step5.league import implementation_id

DIRECTORY = ROOT / 'distribution/rust-lab-b07-g1250'
PHASES = ('canary-1', 'canary-2', 'continuation')


def workflow_id():
    names = ('qualify-rust-lab-b07.py', 'rust_lab_b07_completion.py',
             'rust_lab_b07_smoke.py', 'qualify-rust-lab-b06.py')
    return digest({n: (ROOT/'scripts'/n).read_text(encoding='utf-8') for n in names})


def boundary(value, snapshot):
    require(value['implementation'] == implementation_id(), 'Completion source differs')
    require(value['generation'] == snapshot['generation'] and
            value['checkpoint_sha256'] == snapshot['checkpoint']['sha256'] and
            value['config_sha256'] == snapshot['config_sha256'],
            'Completion requires the qualified committed checkpoint/config')


def ordinary(reference, snapshot):
    """R4's reviewed ordinary result is unchanged; original binaries stay pinned."""
    reuse = reference['completed_ordinary']
    values = {}
    for name, item in reuse['evidence'].items():
        path = DIRECTORY/'evidence'/item['file']
        require(path.parent == DIRECTORY/'evidence', 'Unsafe completion evidence path')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == item['sha256'], 'Completed ordinary evidence changed')
        values[name] = json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)
    q, checked, original = values['qualification'], values['ordinary'], values['snapshot']
    boundary(q, snapshot)
    require(q['milestone'] == 'B07' and q['passed'] is True and q['ordinary'] == checked and
            checked['passed'] is True and checked['matched']['equal'] is True and
            checked['performance']['useful_accepted_work_gain'] is True,
            'Completed ordinary correctness/useful performance is missing')
    require(original['state_file'] == snapshot['state_file'] and
            original['signature_sha256'] == snapshot['signature_sha256'],
            'Committed state changed since ordinary qualification')
    verify_unchanged(original)
    require(pin(q['ordinary_build']['path']) == q['ordinary_build'], 'Qualified ordinary build changed')
    return q, checked


def resume(path, snapshot):
    """Failed later phases do not invalidate earlier successful, sealed phases."""
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    boundary(value, snapshot)
    require(value.get('milestone') == 'B07' and value.get('completion_workflow') == workflow_id(),
            'Resume only this exact B07 completion workflow')
    require(value['completion_state'] == snapshot['state_file'], 'Resume committed state changed')
    result = {}
    for phase, item in value.get('completed_phases', {}).items():
        require(phase in PHASES and pin(item['path']) == item, 'Completed phase evidence changed')
        detail = json.loads(Path(item['path']).read_text(encoding='utf-8'))
        require(detail.get('passed') is True, 'Cannot reuse a failed phase')
        if phase.startswith('canary-'):
            require(detail['pass_index'] == int(phase[-1]) and detail['tables'] == 2 and
                    detail['blend_weight'] == 0 and detail['checkpoint_reloaded'] and detail['buffer_reloaded'],
                    'Canary phase scope differs')
        result[phase] = detail
    return value, result
