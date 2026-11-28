"""g6670: remove the retired rating checkpoints that are stored in another run (the user's request of 2026-11-28).

This run (joint7-v11-2-g4970-rail025-soft8-beta0.02) branched from joint7-v11-2-g1190-restart-3 at g4970 and copied
its rating registry, so 407 of the checkpoints it retires at g6670 (generation-000010 ... generation-004970) are
weights files in the parent run, which stopped at g5177. This run's retention never deletes another run's files,
and the parent's own state.json still names them in its rating registry, so they would stay on disk.

Once, after the first committed iteration of the g6670 revision (league.py, before the scheduled cleanup), for each
other run that stores retired checkpoints:

1. plan: the retired names whose entry in that run's own registry is the same checkpoint (name, generation, model
   id, stored in that run). The plan, with each weights path, is written to this run's receipt first
   (.retention/parent-rating-cleanup-g6670.json; not a control file, so it pins nothing), so an interrupted pass
   resumes with the same files.
2. under that run's lock (refused while it trains: the pass stays pending and is retried after the next commit),
   back up its state.json once (configuration-history/g6670-rating-cleanup/state-before.json) and retire the
   planned names from its registry (ratings.retire_named: generation and model id kept, no weights path).
3. scan every control file of every run (retention.references: all state.json, unfinished or formal specs, pins
   and the manifests they name) and delete only the planned files that nothing still pins. A pinned file stays,
   with the reason in the receipt (that run's opponent pool, its unarchived rating slots' specs, ...).

Nothing else in that run changes: its other rated checkpoints, its pool, checkpoints, replays and evidence.
"""
import os
import time

from brass_ai.bank import read
from brass_ai.learning.contracts import project_path, require
from brass_ai.step5.league import write_json

CONTRACT = 'parent-rating-cleanup-g6670/1.0.0'
RECEIPT = 'parent-rating-cleanup-g6670.json'
BACKUP = ('configuration-history', 'g6670-rating-cleanup', 'state-before.json')


def pending(root, run, rating):
    """{run directory: [names]} of the retired rated checkpoints stored in runs other than `run`."""
    own = run.relative_to(root).as_posix()
    groups = {}
    for name, entry in sorted(rating.get('retired_agents', {}).items()):
        where = entry.get('stored_in')
        if where is not None and where != own: groups.setdefault(where, []).append(name)
    return groups


def _same(entry, retired, where):
    from .ratings import stored_in
    checkpoint = (entry or {}).get('checkpoint') or {}
    return ((entry or {}).get('kind') == 'model' and checkpoint.get('model_id') == retired['model_id']
            and checkpoint.get('generation') == retired['generation'] and 'path' in checkpoint
            and stored_in(checkpoint['path']) == where)


def cleanup(root, run, state, *, generation):
    """Run (or resume) the once-only removal; returns a small summary for the iteration report, or None when it was
    already done or there is nothing to remove."""
    from .ratings import retire_named
    from .retention import active, owned, references, run_lock, safe
    root, run = owned(root, run)
    path = safe(run, run / '.retention' / RECEIPT)
    receipt = read(path) if path.exists() else None
    if receipt is not None:
        require(receipt.get('contract') == CONTRACT, 'Unknown parent rating cleanup receipt')
        if receipt['status'] == 'done': return None
    retired = state['ratings'].get('retired_agents', {})
    groups = pending(root, run, state['ratings'])
    if receipt is None:
        if not groups: return None
        receipt = dict(contract=CONTRACT, run=run.relative_to(root).as_posix(), first_generation=generation, status='planned',
                       parents={where: dict(status='planned', names=len(names)) for where, names in groups.items()})
    began = time.perf_counter()
    for where, names in sorted(groups.items()):
        entry = receipt['parents'].setdefault(where, dict(status='planned', names=len(names)))
        if entry['status'] == 'done': continue
        parent = project_path(root, where)
        state_path = parent / 'state.json'
        if not state_path.is_file():
            entry.update(status='done', reason='no state.json in that run; nothing removed without its registry')
            continue
        _, parent = owned(root, parent)
        if active(parent):
            entry.update(status='pending', reason='that run is active; retried after the next commit')
            continue
        with run_lock(parent):
            other = read(state_path)
            require(str(other.get('format', '')).startswith('step5-recovery-league/') and 'ratings' in other,
                    'The parent rating cleanup expects a Step 5 league run with ratings')
            agents = other['ratings']['agents']
            if 'planned' not in entry:
                planned = {n: agents[n]['checkpoint']['path'] for n in names if _same(agents.get(n), retired[n], where)}
                entry.update(planned=planned, skipped=sorted(set(names) - set(planned)))
                write_json(path, receipt)  # the plan first, so a resumed pass removes the same files
            planned = entry['planned']
            backup = safe(parent, parent.joinpath(*BACKUP))
            if not backup.exists(): write_json(backup, other, compact=True)
            entry['backup'] = backup.relative_to(root).as_posix()
            now = retire_named(other['ratings'], list(planned), retired_by=receipt['run'], retired_generation=generation)
            if now: write_json(state_path, other, compact=True)
            entry['retired_from_registry'] = len(now) + entry.get('retired_from_registry', 0)
            pins = {os.path.normcase(str(p)): reason for p, reason in references(root, parent).items()}
            removed = list(entry.get('removed', [])); kept = {}; missing = []
            for name, relative in sorted(planned.items()):
                if any(r['name'] == name for r in removed): continue
                file = safe(parent, project_path(root, relative))
                reason = pins.get(os.path.normcase(str(file)))
                if reason is not None: kept[name] = reason; continue
                if not file.is_file(): missing.append(name); continue
                size = file.stat().st_size; file.unlink()
                removed.append(dict(name=name, path=relative, bytes=size))
            entry.update(status='done', removed=removed, removed_bytes=sum(r['bytes'] for r in removed),
                         kept_pinned=kept, missing=missing)
        write_json(path, receipt)
    receipt['status'] = 'done' if all(e['status'] == 'done' for e in receipt['parents'].values()) else 'pending'
    write_json(path, receipt)
    return dict(contract=CONTRACT, status=receipt['status'], seconds=time.perf_counter() - began,
                parents={where: dict(status=e['status'], reason=e.get('reason'), planned=len(e.get('planned', {})),
                                     removed=len(e.get('removed', [])), removed_bytes=e.get('removed_bytes', 0),
                                     kept_pinned=len(e.get('kept_pinned', {})), missing=len(e.get('missing', [])))
                         for where, e in receipt['parents'].items()},
                receipt=path.relative_to(root).as_posix())
