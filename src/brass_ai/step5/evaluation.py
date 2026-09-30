"""Seat-rotated, setup-clustered neural/legacy comparisons kept out of training."""
from __future__ import annotations

from contextlib import ExitStack
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import statistics
from threading import Event
import time

import torch

from brass_ai.bank import read, sha, require
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest, project_path
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.learning.replay import read_shards, write_shard, terminal_targets
from brass_ai.qualification import grouped_interval
from .collector import collect_games
from .league import implementation_id, pottery_metrics, seeds_for, write_json, validate_config
from .policy import NeuralPolicy, INDUSTRIES
from .search import SearchPolicy, SearchConfig


def outcome(game, candidate_id, seed):
    require(game['opponents'].count(candidate_id) == 1, 'Evaluation requires one candidate seat')
    seat = game['opponents'].index(candidate_id)
    targets = terminal_targets(game['result'])
    rivals = [v for s, v in enumerate(targets['vp']) if s != seat]
    return dict(seed=seed, seat=seat, win=targets['win_share'][seat],
                rank=targets['placement'][seat], vp=targets['vp'][seat],
                margin=targets['vp'][seat] - statistics.mean(rivals),
                margin_to_best=targets['vp'][seat] - max(rivals))


def summarize_outcomes(rows):
    require(rows, 'No complete evaluation games')
    return dict(games=len(rows), setups=len({r['seed'] for r in rows}),
                win_share=statistics.mean(r['win'] for r in rows),
                win_share_setup_bootstrap_95=grouped_interval(rows, 'win'),
                mean_rank=statistics.mean(r['rank'] for r in rows),
                mean_vp=statistics.mean(r['vp'] for r in rows),
                mean_margin=statistics.mean(r['margin'] for r in rows),
                mean_margin_to_best=statistics.mean(r['margin_to_best'] for r in rows),
                seat_counts=dict(Counter(str(r['seat']) for r in rows)))


def validate_evaluation_config(config):
    expected = {'format', 'players', 'host', 'setups', 'seed', 'partition', 'workers', 'batch_size',
                'max_candidate_cells', 'threads', 'parallel_hosts', 'legacy_workers', 'search_workers',
                'include_legacy', 'use_search', 'focus', 'search'}
    if config.get('format') == 'step5-evaluation-config/1.1.0':
        from brass_ai.recovery.config import execution_config
        expected |= {'execution','panel'}
        execution_config(config['execution'])
        require(config['panel'] in ('development','full'),'Invalid evaluation panel')
        ex=config['execution']
        require(config['workers']==ex['processes']*ex['hosts_per_process'] and
            config['legacy_workers']==config['search_workers']==ex['heavy_processes'] and
            config['batch_size']==ex['batch_size'] and config['max_candidate_cells']==ex['max_candidate_cells'],
            'Evaluation aliases differ from central execution config')
        require(config['parallel_hosts'],'Schema 1.1.0 uses private spawned parallel hosts')
    require(set(config) == expected, 'Unknown or missing evaluation config field')
    require(config['format'] in ('step5-evaluation-config/1.0.0','step5-evaluation-config/1.1.0') and config['players'] == 4,
            'Evaluation requires the four-player config')
    for key in ('setups', 'workers', 'batch_size', 'max_candidate_cells', 'threads', 'legacy_workers', 'search_workers'):
        require(type(config[key]) is int and config[key] > 0, 'Invalid evaluation ' + key)
    for key in ('parallel_hosts', 'include_legacy', 'use_search'):
        require(type(config[key]) is bool, 'Invalid evaluation ' + key)
    require(type(config['seed']) is int and 0 <= config['seed'] < 2**31, 'Invalid evaluation seed')
    require(config['partition'] in ('validation', 'test'), 'Evaluation must use held-out setups')
    require(config['focus'] in ('neutral',) + INDUSTRIES, 'Evaluation needs a fixed neural focus')
    SearchConfig(**config['search']).validate()
    return config


def evaluation_config(config, **overrides):
    if config.get('format','').startswith('step5-evaluation-config/'):
        result = dict(config)
    elif config.get('format') == 'step5-training/2.0.0':
        result = read(SOURCE_ROOT / 'configs/step5/development-evaluation.json')
        result.update(host=config['host'],threads=config['threads'],search=config['search'],
            execution=config['execution'])
        ex=config['execution']
        result.update(workers=ex['processes']*ex['hosts_per_process'],legacy_workers=ex['heavy_processes'],
            search_workers=ex['heavy_processes'],batch_size=ex['batch_size'],max_candidate_cells=ex['max_candidate_cells'])
    else:
        # Explicit older --config lab/smoke paths retain their small defaults.
        # The CLI now chooses evaluation.json when no path is supplied.
        validate_config(config)
        result = read(SOURCE_ROOT / 'configs/step5/previous-evaluation.json')
        for key in ('host', 'workers', 'batch_size', 'threads', 'search'):
            result[key] = config[key]
        result.update(setups=5, include_legacy=False, max_candidate_cells=config.get('max_candidate_cells', 8192),
                      parallel_hosts=config.get('parallel_hosts', False),
                      search_workers=config.get('search_workers', 1), legacy_workers=config['workers'])
    result.update({key: value for key, value in overrides.items() if value is not None})
    return validate_evaluation_config(result)


def evaluation_jobs(seeds):
    return [dict(seed=seed, seat=seat, sampling_seed=seed * 7 + seat) for seed in seeds for seat in range(4)]


def collect_evaluation_batch(path, host, jobs, policies, config, spec_id, lineup, *, mono=None):
    """Explicit setup/rotation pairs; scheduling never selects a different seat."""
    provenance = dict(producer='step5-evaluation/1.1.0', purpose='evaluation', spec_id=spec_id,
                      lineup=lineup, jobs=jobs, policies=[p.policy_id for p in policies])
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            require(json.loads(next(stream))['provenance'] == provenance, 'Cached evaluation batch differs')
        games = read_shards([path])
    else:
        games = collect_games(host, [j['seed'] for j in jobs], policies,
            rotations=[j['seat'] for j in jobs], sampling_seeds=[j['sampling_seed'] for j in jobs],
            workers=config['workers'], parallel_hosts=config['parallel_hosts'], mono=mono)
        # Verify assignment/completeness before publishing any reusable shard.
        check_evaluation_batch(games, jobs, policies[0].policy_id, config['partition'])
        write_shard(path, games, provenance)
    check_evaluation_batch(games, jobs, policies[0].policy_id, config['partition'])
    return games


def check_evaluation_batch(games, jobs, candidate_id, partition):
    expected = {(digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=j['seed'])), j['seat']) for j in jobs}
    actual = []
    require(len(expected) == len(jobs), 'Duplicate evaluation job')
    for game in games:
        require(game['split'] == partition, 'Evaluation setup crossed partition')
        require(game['opponents'].count(candidate_id) == 1, 'Evaluation requires one candidate seat')
        actual.append((game['setup_group'], game['opponents'].index(candidate_id)))
    require(len(actual) == len(jobs) and set(actual) == expected, 'Missing, duplicate or misrotated evaluation games')


def evaluate(root, output, candidate_path, references, config, *, setups=None, seed=None,
             partition=None, include_legacy=None, device='cpu', mono=None,
             search=None, focus=None):
    if config.get('format') in ('step5-evaluation-config/1.1.0','step5-training/2.0.0'):
        from brass_ai.recovery.evaluation import evaluate as recovery_evaluate
        return recovery_evaluate(root,output,candidate_path,references,config,setups=setups,seed=seed,
            partition=partition,include_legacy=include_legacy,device=device,mono=mono,search=search,focus=focus)
    root = Path(root).resolve(); directory = project_path(root, output)
    config = evaluation_config(config, setups=setups, seed=seed, partition=partition,
                               include_legacy=include_legacy, use_search=search, focus=focus)
    setups, seed, partition = config['setups'], config['seed'], config['partition']
    search, focus, include_legacy = config['use_search'], config['focus'], config['include_legacy']
    torch.set_num_threads(config['threads'])
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    require(setups > 0 and partition in ('validation', 'test') and references,
            'Held-out partition and frozen neural references required')
    candidate_file = project_path(root, candidate_path)
    model, candidate_raw = load_checkpoint(candidate_file, device=device)
    seeds, _ = seeds_for(seed, setups, partition)
    host = project_path(root, config['host'])
    reference_models = [load_checkpoint(project_path(root, p), device=device) for p in references]
    neural_ids = [raw['model_id'] for _, raw in reference_models]
    require(candidate_raw['model_id'] not in neural_ids, 'Candidate cannot be its own frozen reference')
    lineups = [('recent-neural', ['neural:0'] * 3),
               ('mixed-neural', [f'neural:{i % len(references)}' for i in range(3)])]
    if len(set(neural_ids)) == 1:
        lineups = lineups[:1]
    pool = read(root / 'training/opponents/pool.json')
    if include_legacy:
        training_ids, held_ids = sorted(pool['training']), sorted(pool['evaluation'])
        require(training_ids and held_ids, 'Qualified legacy pools missing')
        lineups += [('mixed-v3', ['legacy-training:' + training_ids[i % len(training_ids)] for i in range(3)]),
                    ('heldout-pottery-v3', ['legacy-evaluation:' + held_ids[0]] * 3)]
    spec = dict(format='step5-evaluation/1.1.0', purpose='evaluation', partition=partition,
                candidate=str(Path(candidate_path).as_posix()), candidate_sha256=sha(candidate_file),
                candidate_id=candidate_raw['model_id'], references=[dict(path=str(Path(p).as_posix()), sha256=sha(project_path(root, p)),
                    model_id=r['model_id'], training_mode=r['training'].get('mode')) for p, (_, r) in zip(references, reference_models)],
                seeds=seeds, lineups=[dict(name=n, opponents=o) for n, o in lineups],
                rotations=4, search=search, search_config=config['search'] if search else None, focus=focus,
                execution_config=config, implementation=implementation_id(), host_sha256=sha(host / 'BrassHost.exe'),
                legacy_pool=pool if include_legacy else None)
    if (directory / 'spec.json').exists():
        require(read(directory / 'spec.json') == spec, 'Evaluation spec changed')
    else:
        write_json(directory / 'spec.json', spec)
    results, raw_evidence, execution = {}, {}, {}
    inference = dict(batch_size=config['batch_size'], max_candidate_cells=config['max_candidate_cells'])
    jobs = evaluation_jobs(seeds)
    spec_id = digest(spec)
    began = time.monotonic()

    def policies_for(identities, stack):
        if search:
            candidate = SearchPolicy(model, host, device, config=SearchConfig(**config['search']), mono=mono, focus=focus)
            stack.callback(candidate.close)
        else:
            candidate = NeuralPolicy(model, device, exploration=0, focus=focus, **inference)
        opponents, shared_neural = [], {}
        for identity in identities:
            if identity.startswith('neural:'):
                if identity not in shared_neural:
                    shared_neural[identity] = NeuralPolicy(reference_models[int(identity.split(':')[1])][0], device,
                                                          exploration=0, focus='neutral', **inference)
                opponent = shared_neural[identity]
            else:
                kind, policy = identity.split(':', 1)
                opponent = LegacyPolicy(root, policy, purpose='evaluation' if kind == 'legacy-evaluation' else 'training', mono=mono)
                stack.callback(opponent.close)
            opponents.append(opponent)
        return [candidate] + opponents

    seed_by_group = {digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=s)): s for s in seeds}
    for name, identities in lineups:
        lineup_began = time.monotonic()
        has_legacy = any(not i.startswith('neural:') for i in identities)
        independent = search or has_legacy
        width = min(len(jobs), config['workers'], config['search_workers'] if search else config['workers'],
                    config['legacy_workers'] if has_legacy else config['workers'])
        games, candidate_id = [], None
        if not independent:
            with ExitStack() as stack:
                policies = policies_for(identities, stack)
                candidate_id = policies[0].policy_id
                for start in range(0, len(jobs), width):
                    path = directory / f'{name}-batch-{start:06d}.jsonl.gz'
                    games.extend(collect_evaluation_batch(path, host, jobs[start:start + width], policies,
                                                          config, spec_id, name, mono=mono))
                    raw_evidence[path.name] = sha(path)
                    print(f'evaluation {name}: {len(games)}/{len(jobs)} games; up to {width} concurrent', flush=True)
        else:
            # Wall-clock search/legacy policies need their OWN game loop and
            # workers: waiting behind other batched games must not spend their
            # turn budgets. No search tree, private context or C# worker is shared.
            stop = Event()

            def lane(items):
                lane_games, evidence = [], {}
                try:
                    with ExitStack() as stack:
                        policies = policies_for(identities, stack)
                        local = dict(config, workers=1, parallel_hosts=False)
                        for job in items:
                            require(not stop.is_set(), 'Another evaluation worker failed; stopping')
                            path = directory / f'{name}-{job["seed"]}-seat-{job["seat"]}.jsonl.gz'
                            lane_games.extend(collect_evaluation_batch(path, host, [job], policies, local, spec_id, name, mono=mono))
                            evidence[path.name] = sha(path)
                            print(f'evaluation {name}: setup {job["seed"]}, seat {job["seat"]} complete', flush=True)
                        return lane_games, evidence, policies[0].policy_id
                except BaseException:
                    stop.set()
                    raise

            with ThreadPoolExecutor(max_workers=width) as executor:
                try:
                    for batch, evidence, identity in executor.map(lane, [jobs[i::width] for i in range(width)]):
                        games.extend(batch); raw_evidence.update(evidence)
                        require(candidate_id is None or candidate_id == identity, 'Candidate changed between workers')
                        candidate_id = identity
                except BaseException:
                    stop.set()
                    raise
        check_evaluation_batch(games, jobs, candidate_id, partition)
        rows = sorted((outcome(g, candidate_id, seed_by_group[g['setup_group']]) for g in games),
                      key=lambda r: (r['seed'], r['seat']))
        results[name] = dict(**summarize_outcomes(rows), outcomes=rows, diversity=pottery_metrics(games, {candidate_id}))
        seconds = time.monotonic() - lineup_began
        execution[name] = dict(mode='independent-game-workers' if independent else 'batched-neural',
                               workers=width, games=len(games), seconds=seconds, games_per_second=len(games) / max(seconds, 1e-9))
    neural_results = [v for k, v in results.items() if k.endswith('neural')]
    trained_reference_count = len({r['model_id'] for r in spec['references']
        if r['training_mode'] in ('step5-population-ppo', 'step5-reptile')})
    eligible = (partition == 'validation' and setups >= 30 and trained_reference_count >= 2 and
                all(v['win_share_setup_bootstrap_95'][0] > .25 for v in neural_results) and
                not any(v['diversity']['pottery_warning'] for v in results.values()))
    report = dict(format='step5-evaluation-report/1.0.0', spec=spec,
                  candidate_sha256=spec['candidate_sha256'], results=results,
                  promotion_eligible=eligible, failed_games=0, raw_record_sha256=raw_evidence,
                  execution=execution, elapsed_seconds=time.monotonic() - began,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
                  limitation='Cached games inflate resume throughput. Small or legacy-only comparisons do not establish broad strength. Search comparisons must use equal whole-turn budgets. No human-play claim.')
    write_json(directory / 'report.json', report)
    return report


def promote(root, checkpoint, report_path, output):
    root = Path(root).resolve(); report = read(project_path(root, report_path))
    require(report['format'] == 'step5-evaluation-report/1.0.0' and report['promotion_eligible'],
            'Candidate has not passed neural-population promotion gates')
    require(sha(project_path(root, checkpoint)) == report['candidate_sha256'], 'Evaluation was for other weights')
    record = dict(format='step5-promoted/1.0.0', checkpoint=checkpoint,
                  checkpoint_sha256=report['candidate_sha256'], report=report_path,
                  report_sha256=sha(project_path(root, report_path)),
                  scope='Neural population validation only; laptop release and human playtests remain separate')
    write_json(project_path(root, output), record)
    return record
