"""Explicit epoch-cap extensions; the original IL recipe stays immutable."""
from copy import deepcopy
from itertools import product

from brass_ai.bank import read
from brass_ai.learning.contracts import CONFIG_DIR, compatibility, digest, project_path, require
from brass_ai.step5.league import implementation_id


CONTRACT = 'imitation-epoch-extension/1.0.0'


def source_review():
    review = read(CONFIG_DIR/'imitation-extension-sources.json')
    require(review['format'] == CONTRACT, 'Invalid imitation extension source review')
    if (review['target_implementation'] != implementation_id() or
            review['learning_compatibility'] != digest(compatibility())):
        return None
    return review


def source_allowed(previous, current):
    if previous == current:
        return True
    review = source_review()
    return bool(review and current == review['target_implementation'] and
                previous in review['from_implementations'])


def epoch_limit(state):
    """Replay the additive audit trail without rewriting old checkpoint signatures."""
    limit = state['signature']['epochs']
    require(type(limit) is int and limit > 0, 'Invalid original IL epoch limit')
    extensions = state.get('epoch_extensions', [])
    require(type(extensions) is list, 'Invalid IL extension history')
    for entry in extensions:
        require(set(entry) == {'contract', 'from_epochs', 'to_epochs', 'at_epoch',
                              'checkpoint_sha256', 'implementation'} and
                entry['contract'] == CONTRACT and entry['from_epochs'] == limit and
                type(entry['to_epochs']) is int and entry['to_epochs'] > limit and
                type(entry['at_epoch']) is int and entry['at_epoch'] == limit and
                entry['at_epoch'] <= state['epoch'] and
                all(type(entry[k]) is str and len(entry[k]) == 64 and
                    all(c in '0123456789abcdef' for c in entry[k])
                    for k in ('checkpoint_sha256', 'implementation')),
                'Invalid IL extension history')
        limit = entry['to_epochs']
    require(type(state['epoch']) is int and 0 <= state['epoch'] <= limit,
            'IL progress exceeds its epoch limit')
    return limit


def resume_state(root, state, requested, extend_to_epochs):
    original = state['signature']
    require({k:v for k,v in original.items() if k not in ('manifest_path', 'implementation')} ==
            {k:v for k,v in requested.items() if k not in ('manifest_path', 'implementation')} and
            project_path(root, original['manifest_path']) == project_path(root, requested['manifest_path']),
            'BC run settings/source changed; retain the original --epochs and use --extend-to-epochs')
    extensions = state.get('epoch_extensions', [])
    previous = extensions[-1]['implementation'] if extensions else original['implementation']
    require(previous == requested['implementation'] or
            ((extend_to_epochs is not None or extensions) and
             source_allowed(previous, requested['implementation'])),
            'BC run settings/source changed; exact IL extension requires a pinned source review')
    limit = epoch_limit(state)
    if extend_to_epochs is None:
        return state
    require(type(extend_to_epochs) is int and extend_to_epochs > 0,
            'The IL extension target must be a positive total epoch count')
    if extensions and extend_to_epochs == limit:
        return state  # Repeating an interrupted/completed extension is idempotent.
    require(extend_to_epochs > limit, 'The IL extension must increase the total epoch limit')
    require(not state['stopped'], 'IL stopped by patience; an epoch extension does not reset early stopping')
    require(state['epoch'] == limit, 'Finish the existing IL epoch limit before extending it')
    state = deepcopy(state)
    state.setdefault('epoch_extensions', []).append(dict(contract=CONTRACT,
        from_epochs=limit, to_epochs=extend_to_epochs, at_epoch=state['epoch'],
        checkpoint_sha256=state['last']['sha256'], implementation=requested['implementation']))
    return state


def checkpoint_extension(state, step):
    # A checkpoint at the extension boundary was saved before that extension.
    entries = [e for e in state.get('epoch_extensions', []) if e['at_epoch'] < step]
    return digest(entries) if entries else None


def validate_checkpoint(root, out, state, raw):
    metadata = raw['training']; signature = state['signature']
    require(raw['step'] == state['epoch'] and metadata.get('signature') == digest(signature) and
            metadata.get('initialization') == signature['initialization'] and
            metadata.get('fresh_run') and project_path(root, metadata['fresh_run']) == out and
            raw['model_config'] == signature['architecture'] and
            metadata.get('epoch_extension') == checkpoint_extension(state, raw['step']),
            'BC epoch/checkpoint/extension mismatch')
    if raw['step'] > 0:
        require(metadata.get('manifest_sha256') == signature['manifest_sha256'] and
                metadata.get('parent') == state['parent'], 'BC checkpoint lineage differs')
    if state.get('epoch_extensions'):
        previous = state['epoch_extensions'][-1]
        if raw['step'] == previous['at_epoch']:
            require(state['last']['sha256'] == previous['checkpoint_sha256'],
                    'IL extension starting checkpoint changed')


def equivalent_preparations(format, replay):
    """Exact reviewed old cache keys; raw data and output hashes are still checked."""
    review = source_review()
    if review is None:
        return []
    sources = review['prepared_source_hashes']
    names = sorted(sources)
    contracts = compatibility()
    return [digest(dict(format=format, compatibility=contracts, replay=replay,
                        sources=dict(zip(names, hashes))))
            for hashes in product(*(sources[name] for name in names))]
