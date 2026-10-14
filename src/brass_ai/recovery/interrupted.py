"""Explicit, byte-preserving retirement of a collection that never committed."""
import uuid

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.learning.contracts import digest, project_path, require
from brass_ai.step5.league import implementation_id, write_json
from .resume import check_signature, load_resume
from .retention import owned, run_lock, safe


def archive_uncommitted_iteration(root, run, *, expected_iteration, apply=False):
    """Preserve partial evidence so a reviewed fix can retry the same generation.

    Never adopt a partial update, alter state/signatures, or remove checkpoints.
    Both dry runs and application validate the last committed checkpoint under
    the ordinary run lock. The regular trainer still performs its full resume.
    """
    root, run = owned(root, run)
    require(run.is_dir(), 'Run does not exist')
    require(type(expected_iteration) is int and expected_iteration > 0,
            'Expected a positive committed iteration')
    with run_lock(run):
        state_path = safe(run, run/'state.json')
        state = read(state_path)
        require(state['format'] == 'step5-recovery-league/2.2.0', 'Unsupported self-play recovery format')
        require(state['iteration'] == expected_iteration,
                'Committed iteration differs; inspect state.json before recovery')
        target = dict(state['signature'], implementation=implementation_id())
        check_signature(state, target, reviewed=True)
        # Validate against the stored signature; source adoption happens only
        # later, through train --resume-reviewed-source with the original recipe.
        model, raw, _ = load_resume(root, run, state, state['signature'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=state['signature']['config']['learning_rate'])
        resume_optimizer(raw, optimizer, restore_rng=False)
        for entry in state.get('population', []) + [state['evaluation_anchor']] + (
                [state['champion']] if state.get('champion') else []):
            require(sha(safe(run, project_path(root, entry['path']))) == entry['sha256'],
                    'Live opponent/evaluation checkpoint changed')
        output = safe(run, run/f'generation-{expected_iteration+1:04d}.pt')
        require(not output.exists() and not output.with_name(output.name+'.tmp').exists(),
                'Uncommitted checkpoint exists; checkpoint commit recovery is required')
        pending = safe(run, run/f'iteration-{expected_iteration:04d}')
        require(not pending.exists() or pending.is_dir(), 'Uncommitted iteration is not a directory')
        require(not (pending/'report.json').exists(), 'Iteration report exists; inspect its commit before recovery')
        result = dict(format='step5-interrupted-iteration/1.0.0', run=str(run),
                      iteration=expected_iteration, games_done=state['games_done'],
                      checkpoint=state['history'][-1], state_sha256=sha(state_path),
                      source=pending.relative_to(run).as_posix(), dry_run=not apply)
        if not pending.exists() or not any(pending.iterdir()):
            return dict(result, status='nothing-to-archive')
        # Check every path before descending; reject links/junctions and keep
        # exact hashes for raw replay, derived caches and partial worker output.
        files = []; directories = [pending]
        while directories:
            for child in sorted(directories.pop().iterdir()):
                path = safe(run, child)
                if path.is_dir():
                    directories.append(path)
                else:
                    require(path.is_file(), 'Unsupported partial iteration file')
                    files.append(dict(path=path.relative_to(pending).as_posix(),
                                      bytes=path.stat().st_size, sha256=sha(path)))
        result.update(files=sorted(files, key=lambda e: e['path']),
                      preserved_bytes=sum(e['bytes'] for e in files), status='ready-to-archive')
        if not apply:
            return result
        parent = safe(run, run/'.interrupted-iterations')
        parent.mkdir(exist_ok=True)
        archive = safe(run, parent/f'{pending.name}-{digest(state)[:16]}-{uuid.uuid4().hex[:8]}')
        archive.mkdir()
        destination = safe(run, archive/pending.name)
        require(not destination.exists(), 'Recovery destination already exists')
        result.update(archive=archive.relative_to(run).as_posix(), status='moving')
        (archive/'state-before.json').write_bytes(state_path.read_bytes())
        write_json(archive/'manifest.json', result)
        # Same-filesystem rename preserves every byte, including incomplete
        # files. A crash after this step leaves the manifest and evidence here.
        pending.rename(destination)
        result['status'] = 'archived'
        write_json(archive/'manifest.json', result)
        return result
