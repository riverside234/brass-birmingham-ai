"""Write a reviewed teacher config at a locked committed generation; no run edits."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def generate(root, run_path, output, expected_generation, mode, share_target=None, profile='legacy', checkpoint_sha256=None):
    from brass_ai.bank import read, sha
    from brass_ai.learning.contracts import project_path, require
    from brass_ai.recovery.retention import owned, safe, run_lock
    from brass_ai.recovery.teacher_phase_config import adopt
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import check_signature
    from brass_ai.step5.league import implementation_id, write_json
    root, run = owned(root, project_path(root, run_path)); output = project_path(root, output)
    require(output.is_relative_to(root / 'configs/step5') and output.suffix == '.json', 'Write config under configs/step5')
    with run_lock(run):
        state = read(safe(run, run / 'state.json'))
        require(state['iteration'] == expected_generation, 'Stale expected generation; do not rewind state.json')
        entry = state['history'][-1]; checkpoint = safe(run, project_path(root, entry['path']))
        require(entry['generation'] == expected_generation and checkpoint.is_file() and sha(checkpoint) == entry['sha256'],
                'Committed full checkpoint missing/changed')
        require(checkpoint_sha256 is None or entry['sha256']==checkpoint_sha256,'Expected checkpoint hash differs')
        pending = safe(run, run / f'iteration-{expected_generation:04d}')
        require(not pending.exists() or (pending.is_dir() and not any(pending.iterdir())), 'Uncommitted iteration; recover before configuring')
        require(not (run / f'generation-{expected_generation + 1:04d}.pt').exists(), 'Uncommitted checkpoint')
        before = state['signature']['config']
        require('teacher_phase' in before or mode == 'off', 'One-time teacher adoption must use --mode off')
        config = adopt(before) if 'teacher_phase' not in before else copy.deepcopy(before)
        if profile == 'domino':
            from brass_ai.recovery.teacher_diversity import adopt as adopt_diversity,enabled
            require(enabled(before) or mode=='off','Explicit diversity adoption must use --mode off')
            config=copy.deepcopy(before) if enabled(before) else adopt_diversity(before)
        config['teacher_phase']['mode'] = mode
        if share_target is not None: config['teacher_phase']['term']['share_target'] = share_target
        validate_config(config)
        check_signature(state, dict(state['signature'], config=config, implementation=implementation_id()), reviewed=True)
        require(not output.exists() or read(output) == config, 'Existing output differs; choose a new --output')
        write_json(output, config)
        return dict(config=output.relative_to(root).as_posix(), generation=expected_generation,
            checkpoint_sha256=entry['sha256'], mode=mode, run_modified=False,
            scope='Config only. B05/B07 production activation remains gated; use the isolated B04 qualifier for shadow collection.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--run', required=True)
    p.add_argument('--expected-generation', type=int, required=True)
    p.add_argument('--mode', choices=('off', 'shadow', 'on'), required=True)
    p.add_argument('--share-target', type=float)
    p.add_argument('--profile', choices=('legacy','domino'), default='legacy')
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.root.resolve(), a.run, a.output, a.expected_generation, a.mode, a.share_target,
        a.profile,a.checkpoint_sha256), indent=2))


if __name__ == '__main__': main()
