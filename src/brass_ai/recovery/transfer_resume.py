"""Explicit, byte-preserving machine transfer with fresh replay buffers.

The archive restores the original checkpoint/state. A separate hash-bound
instruction authorizes resetting only buffer references, once at that boundary.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid

from brass_ai.learning.contracts import digest, project_path, require

CONTRACT = 'step5-fresh-buffer-resume/1.0.0'
MARKER = 'machine-transfer-resume.json'
KEY = 'machine_transfer'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _publish(path, data):
    if path.exists():
        require(path.read_bytes() == data, 'Conflicting machine-transfer instruction')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name('.transfer-'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('xb') as stream: stream.write(data)
        if os.name=='nt':temporary.rename(path)
        else:os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def install_instruction(root, manifest):
    run = project_path(root, manifest['run'])
    state = _read(run/'state.json')
    policy=dict(contract='fresh-buffers/1.0.0',origin_generation=state['iteration'],auxiliary_buffer_iterations=8,il_probe_positions=2048)
    revision=manifest['buffer_rebuild'].get('reward_revision')
    if revision is not None:
        from brass_ai.learning.post_transfer_reward import REVISION,revise
        require(revision==REVISION and _read(project_path(root,manifest['config']))==revise(state['signature']['config']),
            'Post-transfer reward config differs')
        policy['reward_revision']=revision
    require(manifest['buffer_rebuild']==policy,'Unknown buffer-rebuild instruction')
    require(_sha(run/'state.json') == manifest['state_sha256'], 'Transfer source state changed')
    if not (run/MARKER).exists():
        for name in ('.ppg-eight-cache','.ppg-value-cache','teacher-labels','search-values'):
            folder=project_path(root,run/name)
            require(not folder.exists() or (folder.is_dir() and not any(folder.iterdir())),
                'Destination has existing replay buffers; restore into a clean run directory')
    backup = 'machine-transfer/state-'+manifest['state_sha256']+'.json'
    value = dict(contract=CONTRACT, archive_id=manifest['id'], run=manifest['run'],
        origin_generation=state['iteration'], checkpoint=state['history'][-1],
        source_signature=digest(state['signature']), original_state=backup,
        original_state_sha256=manifest['state_sha256'], auxiliary_buffer_iterations=8,
        previous=state.get(KEY))
    if revision is not None:value['reward_revision']=revision
    value['id'] = digest(value)
    _publish(project_path(root, run/backup), (run/'state.json').read_bytes())
    _publish(project_path(root, run/MARKER), (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
    return value


def instruction(root, run):
    path = project_path(root, run/MARKER)
    if not path.exists(): return None
    value = _read(path)
    require(value.get('contract') == CONTRACT and value['id'] == digest({k:v for k,v in value.items() if k != 'id'})
        and project_path(root, value['run']) == run and value['auxiliary_buffer_iterations'] == 8,
        'Machine-transfer instruction binding changed')
    require(_sha(project_path(root, run/value['original_state'])) == value['original_state_sha256'],
        'Original machine-transfer state changed')
    original = _read(project_path(root, run/value['original_state']))
    require(original['iteration'] == value['origin_generation'] and original['history'][-1] == value['checkpoint']
        and digest(original['signature']) == value['source_signature'] and original.get(KEY) == value['previous'],
        'Machine-transfer source/checkpoint binding differs')
    if value.get('reward_revision') is not None:
        from brass_ai.learning.post_transfer_reward import REVISION,eligible
        require(value['reward_revision']==REVISION and eligible(original['signature']['config']),
                'Unknown post-transfer reward revision')
    return value


def prepare(root, run, state, raw):
    value = instruction(root, run)
    if value is None: return None, False
    record = dict(contract=CONTRACT, instruction_id=value['id'], archive_id=value['archive_id'],
        origin_generation=value['origin_generation'], checkpoint=value['checkpoint'],
        auxiliary_buffer_iterations=8,
        instruction=dict(path=(run/MARKER).relative_to(root).as_posix(),sha256=_sha(run/MARKER)),
        original_state=dict(path=(run/value['original_state']).relative_to(root).as_posix(),sha256=value['original_state_sha256']))
    if value.get('reward_revision'):
        from brass_ai.learning.post_transfer_reward import specification,checkpoint_specification
        original=_read(project_path(root,run/value['original_state']))
        record.update(reward_revision=value['reward_revision'],reward_before=checkpoint_specification(original),
            reward_after=specification(True))
    generation = state['iteration']; origin = value['origin_generation']
    require(generation >= origin and value['checkpoint'] in state['history'], 'Transfer checkpoint lineage differs')
    prior = state.get(KEY); saved = raw['training'].get(KEY)
    if generation == origin:
        require(state['history'][-1] == value['checkpoint'] and prior in (value['previous'], record)
            and saved == value['previous'], 'Fresh-buffer reset is outside its original checkpoint boundary')
        state[KEY] = record
        return record, True
    require(prior == saved == record, 'Committed machine-transfer receipt missing/different')
    return record, False


def ppg_payload(raw, record, reset):
    if not reset: return raw
    saved = raw['training'].get('actor_value_phases')
    require(saved is not None and 'rolling_buffer' in saved and 'cache_origin' in saved,
        'Fresh-buffer transfer requires the current disk-backed PPG recipe')
    # Preserve the actual optimizer/RNG containers. Only replace replay metadata.
    phase = dict(saved, rolling_buffer=[], cache_origin=record['origin_generation'])
    return dict(raw, training=dict(raw['training'], actor_value_phases=phase))


def search_payload(saved, record):
    if record is None or saved is None: return saved
    value = dict(saved, buffer=dict(saved['buffer'], receipts=[], retired=[]), label_ages=[])
    value['stage'] = dict(saved['stage'], buffer_rebuild=dict(origin_generation=record['origin_generation'],
        required_ages=8, ready_generation=None, instruction_id=record['instruction_id']))
    if record.get('reward_revision'):
        value['reward_id']=digest(record['reward_after']);value['buffer']=dict(value['buffer'],reward_id=value['reward_id'])
    return value
