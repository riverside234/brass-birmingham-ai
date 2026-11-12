"""Adopt the new CPU teacher search at a committed Stage A boundary."""
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


def configure(config):
    from brass_ai.rust_backend.lab_teacher_activation import is_on
    from .config import validate_config

    require(is_on(config) and g1491.variant_of(config) == g1491.SEARCH_REFRESH,
            'Retuning needs the refreshed teacher-on recipe')
    phase = config['teacher_phase']
    revision = dict(phase[g1491.KEY], count_search=dict(g1491.RETUNED_SEARCH))
    search = dict(phase['search_value'], max_epochs=1)
    result = dict(config, teacher_phase=dict(phase, search_value=search, **{g1491.KEY: revision}))
    validate_config(result)
    return result


def prepare(root, run, *, expected_generation):
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_replay_compat import execution
    from .resume import load_resume

    root, run = owned(root, run)
    with run_lock(run), torch.random.fork_rng(devices=[]):
        state = read(run/'state.json')
        generation = state['iteration']
        require(generation == expected_generation, 'Run advanced; use the committed generation in state.json')
        before = state['signature']['config']
        if g1491.count_search(before) is not None:
            return state
        config = configure(before)
        _, raw, _ = load_resume(root, run, state, state['signature'])
        validate_state(state, raw)
        training = raw['training']
        teacher = training['teacher_phase']
        search = training['search_value']
        require(teacher == state['teacher_phase'] and search['generation'] == generation and
                stamp(search) == state['search_value'] and search['stage']['stage'] == 'A',
                'Retuning needs the committed teacher state and search head in Stage A')

        previous = state['history'][-1]
        profile = g1491.count_search(config)
        directory = safe(run, run/'configuration-history'/f'teacher-k8-{generation:04d}')
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
        search = dict(search, stage=dict(search['stage'], buffer_rebuild=rebuild), label_ages=[],
                      buffer=dict(values, receipts=[], retired=[*values['retired'], *values['receipts']]))

        signature = dict(state['signature'], config=config)
        backend = dict(state['native_backend'], config_sha256=digest(config))
        checkpoint = dict(raw, training=dict(training, signature=digest(signature), native_backend=backend,
                          native_backend_execution=execution(), teacher_phase=teacher, search_value=search))

        if backup.exists():
            require(read(backup) == state, 'Teacher search migration backup differs from committed state')
        else:
            write_json(backup, state)
        output = safe(run, run/f'generation-{generation:04d}-teacher-k8.pt')
        temporary = safe(run, output.with_suffix('.pt.tmp'))
        try:
            torch.save(checkpoint, temporary)
            verified, _ = load_checkpoint(temporary)
            del verified, _
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        entry = dict(previous, path=output.relative_to(root).as_posix(), sha256=sha(output))
        updated = dict(state, signature=signature, native_backend=backend, teacher_phase=teacher,
                       search_value=stamp(search), history=[*state['history'][:-1], entry], teacher_search_retune=receipt)
        updated.pop('native_backend_switch', None)
        if 'resume' in state:
            updated['resume'] = [entry if item == previous else item for item in state['resume']]
        validate_state(updated, checkpoint)
        archive_pending(run, directory, state)
        write_json(run/'state.json', updated)
        print(f'g{generation}: long K={profile["full_k"]}, C_scale={profile["c_scale"]}, '
              f'short C_scale={profile["fast_c_scale"]}. Teacher-on beta remeasurement scheduled. '
              f'Stage A paused at {search["stage"]["stage_a_iterations"]} while the search buffers refill. '
              'Search-value fitting is capped at one epoch.', flush=True)
        return updated
