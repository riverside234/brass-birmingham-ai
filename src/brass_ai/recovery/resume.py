"""Explicit same-run continuation across a pinned, reviewed source transition."""
import copy

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import CONFIG_DIR, digest, project_path, require
from brass_ai.learning.model import ModelConfig
from brass_ai.step5.league import write_json
from .retention import safe


def check_signature(state, signature, reviewed=False):
    require(state['format'] == 'step5-recovery-league/2.2.0', 'Unsupported self-play resume format')
    previous = state['signature']
    if previous == signature:
        return False
    require(reviewed, 'Run configuration/source changed. For the reviewed selfplay-v2 source only, '
            'repeat with --resume-reviewed-source; otherwise use a fresh directory')
    unchanged = lambda s: {k: v for k, v in s.items() if k != 'implementation'}
    require(unchanged(previous) == unchanged(signature),
            'Reviewed resume requires unchanged config, data/IL, opponents and learning contracts')
    approvals = read(CONFIG_DIR / 'selfplay-resume-sources.json')
    require(approvals['format'] == 'selfplay-resume-sources/1.0.0', 'Unsupported source review receipt')
    require(any(e['from_implementation'] == previous['implementation']
                and e['to_implementation'] == signature['implementation']
                and e['learning_compatibility'] == digest(signature['compatibility'])
                for e in approvals['transitions']), 'This self-play source transition has not been reviewed')
    require(state['iteration'] > 0, 'Reviewed continuation requires a committed PPO generation')
    return True


def load_resume(root, run, state, signature, *, device='cpu', reviewed=False):
    """Read/verify everything before altering state or starting collection."""
    changed = check_signature(state, signature, reviewed)
    for entry in state['history']:
        safe(run, project_path(root, entry['path']))
    entry = state['history'][-1]
    path = safe(run, project_path(root, entry['path']))
    require(sha(path) == entry['sha256'], 'Snapshot changed')
    model, raw = load_checkpoint(path, device=device)
    require(model.config == ModelConfig.preset(signature['config']['model_size'])
            and raw['training'].get('fresh_run') == run.relative_to(root).as_posix(),
            "Resume must use this run's residual checkpoint")
    require(raw['model_id'] == entry['model_id'] and raw['step'] == entry['generation'] == state['iteration'],
            'Resume checkpoint generation/model identity differs')
    bound = raw['training'].get('signature') == digest(state['signature'])
    # A restart after the atomic source transition, before its first new PPO
    # commit, still has the original checkpoint. Accept only that exact binding.
    if not bound:
        bound = any(t['checkpoint'] == entry and t['to_signature'] == digest(state['signature'])
                    and raw['training'].get('signature') == digest(t['from_signature'])
                    for t in state.get('source_transitions', []))
    initial = (state['iteration'] == 0 and raw['training'].get('mode') == 'step5-fresh-residual-initial')
    require(bound or initial, 'Checkpoint is not bound to this run signature')
    if changed:
        # An old partial collection has a different provenance digest. Never mix
        # it into the updated collection or discard it behind the user's back.
        pending = safe(run, run / f"iteration-{state['iteration']:04d}")
        require(not pending.exists() or (pending.is_dir() and not any(pending.iterdir())),
                f'Uncommitted iteration at {pending}; finish/recover it with the previous source before upgrading')
        output = safe(run, run / f"generation-{state['iteration'] + 1:04d}.pt")
        require(not output.exists(), f'Uncommitted checkpoint at {output}; recover it before upgrading')
        # Old full history may already be retired; only the live population and
        # evaluation anchor must still be present. Latest snapshot checked above.
        verified = {entry['path']}
        for frozen in state.get('population', []) + [state['evaluation_anchor']]:
            if frozen['path'] in verified:
                continue
            path = safe(run, project_path(root, frozen['path']))
            require(sha(path) == frozen['sha256'], 'Live opponent/evaluation checkpoint changed')
            verified.add(frozen['path'])
    return model, raw, changed


def commit_transition(run, state, signature):
    """Call only after restoring Adam and RNG successfully, under the run lock."""
    previous = copy.deepcopy(state)
    backup = safe(run, run / f"state-before-source-{digest(previous['signature'])[:16]}-{state['iteration']:04d}.json")
    if backup.exists():
        require(read(backup) == previous, 'Source transition backup differs')
    else:
        write_json(backup, previous)
    updated = copy.deepcopy(state)
    updated.setdefault('source_transitions', []).append(dict(
        from_signature=previous['signature'], to_signature=digest(signature),
        checkpoint=copy.deepcopy(state['history'][-1]), iteration=state['iteration'],
        games_done=state['games_done'], backup=backup.name))
    updated['signature'] = signature
    write_json(safe(run, run / 'state.json'), updated)
    print(f"Reviewed same-run continuation: generation {state['iteration']}, {state['games_done']} games; "
          'checkpoint, Adam/RNG, population and schedule counters preserved', flush=True)
    return updated
