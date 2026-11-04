"""Merge a portable VP cache into the same run's cumulative report history."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def backfill(run, cache):
    from brass_ai.recovery import vp_plot as vp
    run, cache = Path(run).resolve(), Path(cache).resolve()
    history = vp.collect(run)
    imported = json.loads(cache.read_text(encoding='utf-8-sig'))
    if (imported.get('format') != vp.FORMAT or
            not history['initial_checkpoint'] or
            imported.get('initial_checkpoint') != history['initial_checkpoint']):
        raise ValueError('VP backfill belongs to another run lineage or format')
    rows = {r['generation']: r for r in history['iterations']}
    before = len(rows)
    limit = history['committed_generation']
    for item in imported['iterations']:
        generation = item['generation']
        if type(generation) is not int or generation < 1:
            raise ValueError('Invalid VP backfill generation')
        if generation > min(limit, imported['committed_generation']):
            continue
        totals = vp._totals({'health_by_composition': {'selfplay': item}})
        if totals is not None:
            vp._merge(rows, dict(generation=generation, **totals))
    history['iterations'] = [rows[g] for g in sorted(rows)]
    # Refuse to publish a cache against a state that advanced during collection.
    # This is a derived report only; model/optimizer/state files are never edited.
    if json.loads((run/'state.json').read_text(encoding='utf-8-sig'))['iteration'] != limit:
        raise ValueError('Run advanced during VP backfill; stop at a committed boundary and retry')
    output = run/vp.HISTORY_PATH
    vp.save_history(history, output)
    return dict(path=str(output), committed_generation=limit, added_iterations=len(rows)-before,
                total_iterations=len(rows), first_generation=min(rows) if rows else None,
                last_generation=max(rows) if rows else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Stopped training run or complete exported bundle')
    parser.add_argument('--cache', type=Path, default=ROOT/'ai-research/vp-history/joint7-v11-2-g1290-trained-restart/vp-history.json')
    args = parser.parse_args()
    print(json.dumps(backfill(args.run, args.cache), indent=2))


if __name__ == '__main__':
    main()
