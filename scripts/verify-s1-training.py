"""Audit the selected rule-based teacher and optionally the fresh NN pipeline."""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actor-only',action='store_true')
    args=parser.parse_args()
    import torch
    from brass_ai.bank import read
    from brass_ai.learning.contracts import require
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.demonstrations import generate, audit, load_dataset
    from brass_ai.recovery.legacy_search import legacy_spec
    from brass_ai.step5.league import write_json
    torch.set_num_threads(2)
    config = read(ROOT / 'configs/step5/recovery-smoke.json')
    config['legacy_budget'].update(role='lab', search_workers=4, work_multiplier=1)
    validate_config(config)
    out = ROOT / 'artifacts/s1-training-checks' / str(time.time_ns())
    demos = generate(ROOT, out / 'demonstrations', 1, config['execution'],
                     seed=17219000, partition='train', budget=config['legacy_budget'])
    paths = list((out / 'demonstrations').glob('*.jsonl.gz'))
    manifest = audit(ROOT, paths, out / 'dataset.json', budget=config['legacy_budget'])
    data, _ = load_dataset(ROOT, out / 'dataset.json', 'train')
    backend=read(ROOT/'training/opponents/pool.json')['backend']
    require(manifest['calibration']['format'] == ('direct-v3-s1-budget/1.0.0' if backend=='S1' else 'direct-masked-budget/1.0.0') and
            len(manifest['entries']) == 1 and data, 'Audited demonstration missing')
    require(all(row[0]['teacher'] is None and row[0]['behavior']['kind'] == 'external'
                for row in data), 'Heuristic decisions were relabelled as neural search targets')
    holdout = next(iter(read(ROOT / 'training/opponents/pool.json')['evaluation']))
    try:
        legacy_spec(ROOT, holdout, config['legacy_budget'], 'training')
    except ValueError:
        pass
    else:
        raise AssertionError('Held-out Pottery identity admitted to training')
    require(legacy_spec(ROOT, holdout, config['legacy_budget'], 'evaluation')['kind'] == 'lab-legacy',
            'Held-out heuristic evaluation dispatch failed')
    if args.actor_only:
        result=dict(passed=True,backend=backend,demonstration_games=1,audited_rows=len(data),holdout_rejected=True)
        write_json(out/'report.json',result)
        print(dict(result,report=str(out/'report.json')),flush=True)
        return
    # Existing compact integration checks fresh imitation/PPO/resume/evaluation,
    # actual batched Masked ISMCTS visits and a separate distillation update.
    sys.path.insert(0, str(ROOT / 'tests/MaskedISMCTS'))
    from integration import run
    neural = run(ROOT)
    result = dict(passed=True, demonstration_games=demos['games'], audited_rows=len(data),
                  holdout_rejected=True, backend=backend, legacy_search_workers=config['legacy_budget']['search_workers'], neural=neural)
    write_json(out / 'report.json', result)
    print(dict(passed=True, report=str(out / 'report.json'), audited_rows=len(data),
               neural_completed_search=neural['completed_search'],
               neural_distilled=neural['distilled']), flush=True)


if __name__ == '__main__':
    main()
