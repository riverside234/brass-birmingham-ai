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
COMPLETION_CHANGES = {
    'src/brass_ai/recovery/search_value_config.py',
    'src/brass_ai/recovery/search_value_phase.py',
    'src/brass_ai/recovery/teacher_phase.py',
    'src/brass_ai/rust_backend/lab_teacher.py',
    'src/brass_ai/rust_backend/lab_teacher_finalization.py',
    'lab-native/src/teacher.rs', 'lab-native/tests/teacher.rs'}


def workflow_id():
    names = ('qualify-rust-lab-b07.py', 'rust_lab_b07_completion.py',
             'rust_lab_b07_smoke.py', 'qualify-rust-lab-b06.py')
    return digest({n: (ROOT/'scripts'/n).read_text(encoding='utf-8') for n in names})


def reviewed_source(reference=None):
    reference = reference or json.loads((DIRECTORY/'reference.json').read_text())
    review = reference.get('completion_source_review')
    require(isinstance(review,dict) and review['to_implementation'] == implementation_id()
            and set(review['changes']) == COMPLETION_CHANGES, 'Unreviewed completion source')
    for name, change in review['changes'].items():
        require(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == change['after'],
                'Completion source review changed')
    return review


def reuse_build(value, kind):
    """Rebind only the unchanged ordinary executable, never an old teacher."""
    if value['implementation'] == implementation_id() and 'source_reuse' not in value:return value
    reference=json.loads((DIRECTORY/'reference.json').read_text());review=reviewed_source(reference)
    if kind == 'teacher' and 'teacher_checkpoint_reuse' in reference:
        entry=reference['teacher_checkpoint_reuse'];change=reference['checkpoint_source_review']
        require(change['to_implementation']==implementation_id() and
                set(change['changes'])=={'src/brass_ai/recovery/search_value_phase.py'}, 'Unreviewed checkpoint source')
        for name,item in change['changes'].items():
            require(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==item['after'],
                    'Checkpoint source review changed')
        require(value['implementation'] in (change['from_implementation'],implementation_id()), 'Unreviewed teacher source')
        original=entry['original_receipt']
        require(pin(original['path'])==original, 'Original teacher build receipt changed')
        saved=json.loads(Path(original['path']).read_text())
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        require(saved['implementation']==change['from_implementation'] and saved['producer']==source_fingerprint(),
                'Teacher native inputs changed')
        rebound=dict(saved,implementation=implementation_id(),source_reuse=dict(original_receipt=original,
            from_implementation=saved['implementation'],review=digest(change),native_bytes_unchanged=True,
            scope='R7 plain checkpoint labels only; exact R6 teacher executable and tests'))
        require(value==saved or value==rebound, 'Reused teacher differs from original receipt')
        return rebound
    require(kind == 'ordinary', 'Changed teacher deadline requires a new teacher build')
    require(value['implementation'] in (review['from_implementation'],implementation_id()), 'Unreviewed build source')
    entry=reference['build_reuse'][kind];original=entry['original_receipt']
    require(pin(original['path'])==original, 'Original reused build receipt changed')
    saved=json.loads(Path(original['path']).read_text())
    require(saved['implementation']==review['from_implementation'] and
            saved['producer']==entry['from_producer'] and entry['ordinary_compile_inputs_unchanged'] is True,
            'Unreviewed ordinary build source')
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    rebound=dict(saved,implementation=implementation_id(),producer=source_fingerprint(),
        source_reuse=dict(original_receipt=original,from_implementation=saved['implementation'],
            from_producer=saved['producer'],review=digest(review),native_bytes_unchanged=True,
            scope='Ordinary inputs unchanged; teacher-only module/test deadline rebuilt separately'))
    require(value==saved or value==rebound, 'Reused build differs from original receipt')
    return rebound


def resumable_teacher(saved_build, reused):
    """Old failed canaries may supply context but cannot supply an old deadline."""
    if saved_build is None:
        require(not reused, 'Completed phases lost their teacher build')
        return None
    require(pin(saved_build['path'])==saved_build, 'Resumed teacher build changed')
    value=json.loads(Path(saved_build['path']).read_text())
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    current=value.get('implementation')==implementation_id() and value.get('producer')==source_fingerprint()
    require(current or not reused, 'Cannot reuse phases with a different teacher build')
    return saved_build['path'] if current else None


def boundary(value, snapshot, *, implementation=None):
    require(value['implementation'] == (implementation or implementation_id()), 'Completion source differs')
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
    previous = reviewed_source(reference)['from_implementation'] if q['implementation']!=implementation_id() else None
    boundary(q, snapshot, implementation=previous)
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
    previous=None;legacy=False
    if value.get('completion_workflow') != workflow_id():
        reference=json.loads((DIRECTORY/'reference.json').read_text())
        legacy=pin(path)['sha256']==reference.get('previous_completion',{}).get('sha256')
        require(legacy and not value.get('completed_phases'), 'Resume only this exact B07 completion workflow')
        previous=reference.get('previous_completion',{}).get('implementation') or reviewed_source(reference)['from_implementation']
    boundary(value, snapshot,implementation=previous)
    require(value.get('milestone') == 'B07' and (legacy or value.get('completion_workflow') == workflow_id()),
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
