"""Small real-host parallel parity/search check, not a strength tournament."""
import argparse
from pathlib import Path
import sys
import time
import uuid

root_source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_source / 'src'))

import torch
from brass_ai.bank import read
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import LAB_ROOT, digest, project_path, require
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import validate_game
from brass_ai.learning.runner import RandomPolicy
from brass_ai.step5.collector import collect_games
from brass_ai.step5.league import collect_search_games, implementation_id, seeds_for, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--mono')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--with-search', action='store_true')
    args = parser.parse_args()
    require(args.root.is_absolute(), 'Use an absolute project root')
    root = args.root.resolve()
    torch.set_num_threads(2)
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    directory = project_path(root, 'artifacts/step5/scaling-checks/' + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    host = root / 'artifacts/r1-host/guarded'
    seeds, cursor = seeds_for(858001, 2)
    traces, timings = [], {}
    for parallel in (False, True):
        policy = RandomPolicy()
        began = time.monotonic()
        games = collect_games(host, seeds, [policy] * 4, workers=2, sampling_seed=2721,
                              parallel_hosts=parallel, mono=args.mono)
        timings['parallel' if parallel else 'serial'] = time.monotonic() - began
        for game in games:
            validate_game(game)
        traces.append([dict(group=g['setup_group'], result=g['result'],
            records=[{key: r[key] for key in ('view', 'choices', 'chosen', 'seat', 'style', 'intent', 'behavior')}
                     for r in g['records']]) for g in sorted(games, key=lambda g: g['setup_group'])])
        print(('parallel' if parallel else 'serial') + ': two complete games validated', flush=True)
    require(traces[0] == traces[1], 'Parallel collection changed semantic trajectories or seat routing')
    report = dict(passed=True, parity_games_per_mode=2, trajectory_sha256=digest(traces[0]),
                  seconds=timings, platform=sys.platform, device=args.device, implementation=implementation_id())
    if args.with_search:
        config = read(root / 'configs/step5/lab.json')
        config.update(search_games=2, search_workers=2)
        config['search'].update(proposals=2, finalists=2, pilot_samples=1, confirmation_samples=2,
                                seconds=1.0, max_searches_per_seat=1)
        torch.manual_seed(1183)
        model = FourSeatNet(ModelConfig.preset('small')).to(args.device)
        before = model_id(model)
        search_seeds, _ = seeds_for(cursor, 2)
        games, telemetry, workers = collect_search_games(root, directory, host, search_seeds, model, config,
            dict(implementation=implementation_id()), 0, device=args.device, mono=args.mono)
        for game in games:
            validate_game(game)
        require(len(games) == 2 and len({g['setup_group'] for g in games}) == 2 and workers == 2,
                'Search collection lost/duplicated games or ignored workers')
        require(model_id(model) == before, 'Concurrent teachers mutated the learner')
        report['search'] = dict(games=len(games), workers=workers, telemetry=telemetry, learner_unchanged=True,
                               note='One-second search smoke; incomplete searches produce no invented labels.')
    write_json(directory / 'report.json', report)
    print('Passed: ' + str(directory / 'report.json'))


if __name__ == '__main__':
    main()
