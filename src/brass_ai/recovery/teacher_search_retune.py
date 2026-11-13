"""Change CPU teacher search at a committed boundary and refill its buffers."""
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import digest, require
from brass_ai.step5.league import write_json
from . import adaptive_teacher_beta as adaptive
from . import training_revision_g1491 as g1491
from .retention import owned, run_lock, safe
from .search_value_iteration import stamp
from .teacher_phase_config import buffer_identity
from .utility_width_upgrade import archive_pending


def configure(config, *, split_c_scale=False):
    from brass_ai.rust_backend.lab_teacher_activation import is_on
    from .config import validate_config

    require(is_on(config) and g1491.variant_of(config) == g1491.SEARCH_REFRESH,
            'Retuning needs the refreshed teacher-on recipe')
    if split_c_scale:
        require(g1491.count_search(config) == g1491.RETUNED_SEARCH and
                g1491.teacher_search_head_weight(config) is not None,
                'Cotton-specific scales need the retuned standard/cotton table mix')
    phase = config['teacher_phase']
    profile = g1491.COTTON_SPLIT_SEARCH if split_c_scale else g1491.RETUNED_SEARCH
    revision = dict(phase[g1491.KEY], count_search=dict(profile))
    search = dict(phase['search_value'], max_epochs=1)
    result = dict(config, teacher_phase=dict(phase, search_value=search, **{g1491.KEY: revision}))
    if split_c_scale:
        result = g1491.configure_teacher_games(result, g1491.teacher_search_head_weight(config))
    validate_config(result)
    return result


def prepare(root, run, *, expected_generation, split_c_scale=False):
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_replay_compat import execution
    from .resume import load_resume

    root, run = owned(root, run)
    with run_lock(run), torch.random.fork_rng(devices=[]):
        state = read(run/'state.json')
        generation = state['iteration']
        require(generation == expected_generation, 'Run advanced; use the committed generation in state.json')
        before = state['signature']['config']
        profile = g1491.count_search(before)
        if profile == g1491.COTTON_SPLIT_SEARCH or (profile is not None and not split_c_scale):
            return state
        config = configure(before, split_c_scale=split_c_scale)
        model, raw, _ = load_resume(root, run, state, state['signature'])
        del model
        validate_state(state, raw)
        training = raw['training']
        teacher = training['teacher_phase']
        search = training['search_value']
        require(teacher == state['teacher_phase'] and search['generation'] == generation and
                stamp(search) == state['search_value'], 'Teacher/search-value checkpoint state differs')
        stage = search['stage']
        allowed_stages = ('A', 'B', 'C') if split_c_scale else ('A',)
        require(stage['stage'] in allowed_stages, 'The committed search-head stage does not support this retune')

        previous = state['history'][-1]
        profile = g1491.count_search(config)
        tag = 'teacher-k8-c020-cotton010' if split_c_scale else 'teacher-k8'
        directory = safe(run, run/'configuration-history'/f'{tag}-{generation:04d}')
        backup = directory/'state-before.json'
        labels, values = teacher['buffer'], search['buffer']
        receipt = dict(generation=generation, previous_checkpoint=previous, count_search=profile,
                       retired_teacher_shards=len(labels['receipts']), retired_value_shards=len(values['receipts']),
                       backup=backup.relative_to(run).as_posix())
        weight = dict(teacher['weight'], outside_streak=0)
        weight[adaptive.REMEASUREMENT] = dict(requested_generation=generation, previous_beta=weight['beta'])
        weight.pop(adaptive.KEY, None)
        teacher = dict(teacher, weight=weight, buffer=dict(labels, identity=buffer_identity(config), receipts=[],
                       retired=[*labels['retired'], *labels['receipts']]))
        rebuild = dict(origin_generation=generation, required_ages=8, ready_generation=None, instruction_id=digest(receipt))
        search = dict(search, stage=dict(stage, buffer_rebuild=rebuild), label_ages=[],
                      buffer=dict(values, receipts=[], retired=[*values['retired'], *values['receipts']]))

        signature = dict(state['signature'], config=config)
        backend = dict(state['native_backend'], config_sha256=digest(config))
        checkpoint = dict(raw, training=dict(training, signature=digest(signature), native_backend=backend,
                          native_backend_execution=execution(), teacher_phase=teacher, search_value=search))

        if backup.exists():
            require(read(backup) == state, 'Teacher search migration backup differs from committed state')
        else:
            write_json(backup, state)
        output = safe(run, run/f'generation-{generation:04d}-{tag}.pt')
        temporary = safe(run, output.with_suffix('.pt.tmp'))
        try:
            torch.save(checkpoint, temporary)
            verified_model, verified_raw = load_checkpoint(temporary)
            del verified_model, verified_raw
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        entry = dict(previous, path=output.relative_to(root).as_posix(), sha256=sha(output))
        migration = 'teacher_search_scale_split' if split_c_scale else 'teacher_search_retune'
        updated = dict(state, signature=signature, native_backend=backend, teacher_phase=teacher,
                       search_value=stamp(search), history=[*state['history'][:-1], entry], **{migration: receipt})
        updated.pop('native_backend_switch', None)
        if 'resume' in state:
            updated['resume'] = [entry if item == previous else item for item in state['resume']]
        validate_state(updated, checkpoint)
        archive_pending(run, directory, state)
        write_json(run/'state.json', updated)
        long_scale = (f'standard C_scale={profile["c_scale"]:.2f}, cotton C_scale={profile["cotton_c_scale"]:.2f}'
                      if split_c_scale else f'C_scale={profile["c_scale"]}')
        print(f'g{generation}: long K={profile["full_k"]}, {long_scale}, '
              f'short C_scale={profile["fast_c_scale"]}. Teacher-on beta remeasurement scheduled. '
              f'Search value stays in Stage {stage["stage"]} while the buffers refill. '
              'Search-value fitting is capped at one epoch.', flush=True)
        return updated
