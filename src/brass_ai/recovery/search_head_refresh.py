"""Replace the search head and adopt the next g1190 teacher-on recipe in place."""
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import digest, require
from brass_ai.learning.search_value import marker, width
from brass_ai.step5.league import write_json
from . import adaptive_teacher_beta as adaptive
from .rating_coverage import G1290_BASELINE
from . import search_value_schedule as schedule
from . import training_revision_g1491 as g1491
from .retention import owned, run_lock, safe
from .search_value_iteration import stamp
from .search_value_training import Companion, StageMachine
from .utility_width_upgrade import archive_pending

KEY = 'search_head_refresh'


def configure(config, *, adaptive_beta=False):
    from brass_ai.rust_backend.lab_teacher_activation import is_on
    from .config import validate_config

    phase = config['teacher_phase']
    options = phase['search_value']
    require(is_on(config) and g1491.variant_of(config) == g1491.WIDE_RESTART and options['head']['hidden'] == 256,
            'Search-head refresh requires the wide g1190 teacher-on recipe')
    search = dict(options, stage='A', head=dict(options['head'], hidden=32),
                  blend=dict(options['blend'], w=0.), trunk=dict(options['trunk'], scale=0.),
                  target=dict(options['target'], **{'lambda': .8}),
                  schedule=schedule.refreshed_schedule(schedule.stage_a_iterations(options)))
    updated = dict(phase, search_value=search, skip_utility_qualification=True,
                   term=dict(phase['term'], rows_per_step=g1491.VARIANTS[g1491.SEARCH_REFRESH]['rows_per_step']),
                   **{g1491.KEY: g1491.block(variant=g1491.SEARCH_REFRESH)})
    if adaptive_beta:
        updated[adaptive.KEY] = adaptive.settings()
    coverage = dict(G1290_BASELINE, seed_reference=dict(G1290_BASELINE['seed_reference']))
    result = dict(config, teacher_phase=updated, rating_coverage=coverage)
    validate_config(result)
    require(is_on(result), 'The refreshed recipe must keep the teacher on')
    return result


def refresh_buffer(run, buffer, generation):
    # Retire old training returns: their .8 bootstrap would use predictions from
    # the removed head. The realized outcomes of dedicated held-outs still apply.
    active = buffer['receipts']
    retired = buffer['retired']
    def recent(r):
        return (r['split'] == 'heldout' and r['reward_id'] == buffer['reward_id'] and
                0 < generation - r['generation'] < schedule.HELDOUT_ITERATIONS)

    kept = {r['file']: r for r in active if recent(r)}
    directory = run / 'search-values'
    for receipt in retired:
        if recent(receipt) and safe(run, directory / receipt['file']).is_file():
            kept.setdefault(receipt['file'], receipt)
    old = {r['file']: r for r in [*retired, *active] if r['file'] not in kept}
    return dict(buffer, receipts=list(kept.values()), retired=list(old.values()))


def prepare(root, run, *, expected_generation, adaptive_beta=False):
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_replay_compat import execution
    from .resume import load_resume
    from .teacher_phase_config import buffer_identity

    root, run = owned(root, run)
    with run_lock(run), torch.random.fork_rng(devices=[]):
        state = read(run / 'state.json')
        generation = state['iteration']
        require(generation == expected_generation, 'Run advanced; use the committed generation in state.json')
        before = state['signature']['config']
        if KEY in state:
            require(schedule.dedicated_holdout(before['teacher_phase']['search_value']), 'Refresh receipt differs from recipe')
            return state
        config = configure(before, adaptive_beta=adaptive_beta)
        model, raw, _ = load_resume(root, run, state, state['signature'])
        validate_state(state, raw)
        training = dict(raw['training'])
        saved = training['search_value']
        require(saved['generation'] == generation and stamp(saved) == state['search_value'] and width(model) == 256,
                'The current search head does not match the committed checkpoint')
        require(training['teacher_phase'] == state['teacher_phase'], 'Teacher checkpoint/state differs')

        del model.search_value_head
        companion = Companion(model, hidden=32)
        options = config['teacher_phase']['search_value']
        stage = StageMachine(options=options).state
        stage['generation'] = generation - 1
        search = dict(saved, stage=stage, companion=companion.snapshot(), label_ages=[],
                      buffer=refresh_buffer(run, saved['buffer'], generation))

        teacher = state['teacher_phase']
        weight = dict(teacher['weight'], **{adaptive.REMEASUREMENT: dict(requested_generation=generation)})
        weight.pop(adaptive.KEY, None)
        labels = teacher['buffer']
        teacher = dict(teacher, weight=weight, buffer=dict(labels, identity=buffer_identity(config), receipts=[],
                        retired=[*labels['retired'], *labels['receipts']]))
        backend = dict(state['native_backend'], config_sha256=digest(config))
        backend['teacher'] = dict(backend['teacher'], stage_schedule=digest(options['schedule']))
        signature = dict(state['signature'], config=config)
        training.update(signature=digest(signature), teacher_phase=teacher, search_value=search,
                        search_value_head=marker(model), native_backend=backend, native_backend_execution=execution(),
                        actor_value_phases=dict(training['actor_value_phases'], beta_clone=.25))
        checkpoint = dict(raw, training=training, model_id=model_id(model),
                          state_dict={k: v.detach() for k, v in model.state_dict().items()})

        previous = state['history'][-1]
        directory = safe(run, run / 'configuration-history' / f'search32-{generation:04d}')
        backup = directory / 'state-before.json'
        if backup.exists():
            require(read(backup) == state, 'Search-head migration backup differs from committed state')
        else:
            write_json(backup, state)
        output = safe(run, run / f'generation-{generation:04d}-search32.pt')
        temporary = output.with_suffix('.pt.tmp')
        try:
            torch.save(checkpoint, temporary)
            verified, _ = load_checkpoint(temporary)
            require(width(verified) == 32, 'Search-head checkpoint conversion failed')
            del verified
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        entry = dict(path=output.relative_to(root).as_posix(), sha256=sha(output),
                     model_id=checkpoint['model_id'], generation=generation)
        receipt = dict(generation=generation, previous_checkpoint=previous, backup=backup.relative_to(run).as_posix())
        updated = dict(state, signature=signature, native_backend=backend, teacher_phase=teacher, search_value=stamp(search),
                       history=[*state['history'][:-1], entry], **{KEY: receipt})
        updated.pop('native_backend_switch', None)
        if 'resume' in state:
            updated['resume'] = [entry if r == previous else r for r in state['resume']]
        validate_state(updated, checkpoint)
        archived = archive_pending(run, directory, state)
        write_json(run / 'state.json', updated)
        print(f'g{generation}: fresh 32-wide search head, Stage A0/{options["schedule"]["stage_a_iterations"]}, w=0, lambda=.8. '
              f'PPG beta reset to .25. Archived {len(archived)} pending path(s).', flush=True)
        return updated
