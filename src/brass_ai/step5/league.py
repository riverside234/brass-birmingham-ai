"""Resumable self-play-first population trainer; immutable historical snapshots."""
from __future__ import annotations

from contextlib import ExitStack
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time

import torch

from brass_ai.bank import admission, read, sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, digest, project_path, require
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import read_shards, write_shard, split_for
from brass_ai.learning.runner import match_report
from .collector import collect_games
from .policy import NeuralPolicy, INDUSTRIES
from .search import SearchPolicy, SearchConfig
from .updates import ppo_update, distill_update


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def implementation_id():
    # Source-bound runs cannot silently resume with changed learning/search code.
    base = Path(__file__).parent.parent
    paths = list((base / 'step5').glob('*.py')) + [base / 'learning/runner.py', base / 'learning/batching.py']
    return digest({p.relative_to(base).as_posix(): p.read_text(encoding='utf-8').replace('\r\n', '\n')
                   for p in sorted(paths)})


def validate_config(config):
    expected = {'format', 'players', 'model_size', 'host', 'games_per_iteration', 'workers', 'batch_size',
                'threads', 'learning_rate', 'ppo_epochs', 'exploration', 'entropy_weight', 'seed',
                'legacy_every', 'search_start', 'search_every', 'search_games', 'distill_steps', 'distill_weight', 'search'}
    if config.get('format') == 'step5-training/1.1.0':
        expected |= {'minibatch_size', 'max_candidate_cells', 'parallel_hosts', 'group_tables', 'search_workers'}
        for key in ('minibatch_size', 'max_candidate_cells', 'search_workers'):
            require(type(config.get(key)) is int and config[key] > 0, 'Invalid ' + key)
        for key in ('parallel_hosts', 'group_tables'):
            require(type(config.get(key)) is bool, 'Invalid ' + key)
    require(set(config) == expected, 'Unknown or missing Step 5 config field')
    require(config.get('format') in ('step5-training/1.0.0', 'step5-training/1.1.0') and config.get('players') == 4,
            'Step 5 requires the four-player config')
    for key in ('games_per_iteration', 'workers', 'batch_size', 'threads', 'ppo_epochs', 'search_games', 'distill_steps'):
        require(type(config[key]) is int and config[key] > 0, 'Invalid ' + key)
    require(config['model_size'] in ('small', 'medium', 'large'), 'Unknown model size')
    require(type(config['seed']) is int and 0 <= config['seed'] < 2**31, 'Invalid setup seed')
    require(0 < config['learning_rate'] <= .01 and 0 <= config['exploration'] <= .5 and
            0 <= config['entropy_weight'] <= .2 and 0 < config['distill_weight'] <= 1, 'Invalid optimizer settings')
    require(type(config['legacy_every']) is int and (config['legacy_every'] == 0 or config['legacy_every'] >= 20),
            'Legacy tables are capped at 5%; set legacy_every=0 for pure neural training')
    require(type(config['search_every']) is int and config['search_every'] >= 0 and config['search_start'] >= 0,
            'Invalid search schedule')
    SearchConfig(**config['search']).validate()
    return config


def collection_batches(games_done, count, workers, legacy_every, group_tables=False):
    """Preserve exact table quotas, but optionally fill batches across the cycle.

    Consecutive scheduling tops out at 14 self-play / 5 history / 1 legacy
    games. Bucketing a frozen iteration removes that cap without stale actors.
    """
    groups = []
    if group_tables:
        buckets = {}
        for index in range(count):
            buckets.setdefault(table_kind(games_done + index, legacy_every), []).append(index)
        groups = list(buckets.items())
    else:
        for index in range(count):
            kind = table_kind(games_done + index, legacy_every)
            if not groups or groups[-1][0] != kind:
                groups.append((kind, []))
            groups[-1][1].append(index)
    for kind, indices in groups:
        for start in range(0, len(indices), workers):
            yield kind, indices[start:start + workers]


def seeds_for(start, count, partition='train'):
    seeds = []
    while len(seeds) < count:
        require(start < 2**31, 'Setup seed range exhausted')
        group = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=start))
        if split_for(group) == partition:
            seeds.append(start)
        start += 1
    return seeds, start


def table_kind(index, legacy_every=20):
    if legacy_every and index % legacy_every == legacy_every - 1:
        return 'legacy'
    return 'historical' if index % 20 >= 14 else 'selfplay'


def pottery_metrics(games, learner_ids=None):
    counts, opportunities, focused = Counter(), Counter(), Counter()
    contexts, seen = Counter(), set()
    for game in games:
        for row in game['records']:
            if learner_ids and row['behavior']['policy_id'] not in learner_ids:
                continue
            trajectory = (game['game_id'], row['seat'])
            if trajectory not in seen:
                seen.add(trajectory)
                context = next((name for i, name in enumerate(INDUSTRIES) if row['intent'][i] > .5), 'neutral')
                contexts[context] += 1
            available = {c.get('industry') for c in row['choices'] if c['kind'] == 'build'}
            opportunities.update(available)
            selected = row['choices'][row['chosen']]
            if selected['kind'] == 'build':
                counts[selected['industry']] += 1
                if row['intent'][INDUSTRIES.index(selected['industry'])] > .5:
                    focused[selected['industry']] += 1
    return dict(builds={i: counts[i] for i in INDUSTRIES},
                legal_build_opportunities={i: opportunities[i] for i in INDUSTRIES},
                builds_matching_assigned_focus={i: focused[i] for i in INDUSTRIES},
                seat_trajectory_contexts=dict(contexts),
                pottery_warning=bool(opportunities['Pottery'] >= 30 and counts['Pottery'] == 0),
                interpretation='Opportunity and chosen-build counts, not a mandate to force an industry. Pottery Flex identity is held out, never Pottery actions.')


def checked_snapshot(root, entry, device):
    path = project_path(root, entry['path'])
    require(sha(path) == entry['sha256'], 'Frozen neural snapshot changed')
    # Loading a model must not perturb the learner's training RNG.
    with torch.random.fork_rng(devices=[]):
        model, raw = load_checkpoint(path, device=device)
    require(raw['model_id'] == entry['model_id'], 'Snapshot identity differs')
    return model


def collect_cached(root, path, host, seeds, policies, config, provenance, *, mono=None):
    if path.exists():
        import gzip
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            header = json.loads(next(stream))
        require(header['provenance'] == provenance, 'Cached collection has different opponents/seeds/settings')
        return read_shards([path])
    games = collect_games(host, seeds, policies, workers=config['workers'],
                          sampling_seed=provenance['sampling_seed'], mono=mono,
                          parallel_hosts=config.get('parallel_hosts', False))
    write_shard(path, games, provenance)
    return games


def collect_search_games(root, directory, host, seeds, model, config, signature, iteration, *, device, mono):
    """Each search worker owns its simulator, neural copies and private contexts."""
    workers = min(config.get('search_workers', 1), len(seeds))

    def collect(job):
        index, chunk = job
        search = SearchPolicy(model, host, device, config=SearchConfig(**config['search']), mono=mono)
        try:
            provenance = dict(producer='step5/1.1.0', purpose='training', kind='neural-search',
                seeds=chunk, sampling_seed=config['seed'] + iteration * 17011 + index,
                policies=[search.policy_id] * 4, implementation=signature['implementation'])
            local = dict(config, workers=1 if workers > 1 else config['workers'],
                         parallel_hosts=False if workers > 1 else config.get('parallel_hosts', False))
            name = 'search.jsonl.gz' if workers == 1 else f'search-{index:04d}.jsonl.gz'
            games = collect_cached(root, directory / name, host, chunk, [search] * 4, local, provenance, mono=mono)
            return games, dict(search.telemetry)
        finally:
            search.close()

    jobs = [(index, seeds[index::workers]) for index in range(workers)]
    if workers == 1:
        results = [collect(jobs[0])]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(collect, jobs))
    games, telemetry = [], Counter()
    for batch, counts in results:
        games.extend(batch); telemetry.update(counts)
    return games, dict(telemetry), workers


def train(root, run_path, config, *, iterations=1, device='cpu', initial=None, mono=None, from_run=None):
    root = Path(root).resolve(); run = project_path(root, run_path)
    config = validate_config(config)
    require(iterations > 0, 'Positive iteration count required')
    require(not (initial and from_run), 'Choose --initial or --from-run, not both')
    torch.set_num_threads(config['threads'])
    host = project_path(root, config['host'])
    signature = dict(config=config, compatibility=compatibility(), implementation=implementation_id())
    # The public evaluation Pottery Flex is excluded by admission, not by action masks.
    pool = read(root / 'training/opponents/pool.json')
    qualified = sorted(pool['training']) if config['legacy_every'] else []
    for identity in qualified:
        admission(root, identity, 'training')
        require(not identity.endswith('/pottery-flex-v3'), 'Held-out opponent entered training')
    signature['legacy_pool'] = {p: pool['training'][p] for p in qualified}
    state_path = run / 'state.json'
    if state_path.exists():
        state = read(state_path)
        require(state['signature'] == signature, 'Run configuration/source/pool changed; start a new run')
        require(initial is None and from_run is None, 'Omit --initial/--from-run while resuming')
        entry = state['history'][-1]
        model = checked_snapshot(root, entry, device)
        raw = load_checkpoint(project_path(root, entry['path']), device=device)[1]
    else:
        require(not run.exists() or not any(run.iterdir()), 'New run directory must be empty')
        run.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config['seed'])
        meta_tasks = []
        parent_state = None
        origin = None
        if from_run:
            source = project_path(root, from_run)
            require(source != run, 'Fork into a different run directory')
            parent_state = read(source / 'state.json')
            require(parent_state['format'] == 'step5-league/1.0.0' and
                    parent_state['signature']['compatibility'] == compatibility(), 'Source run is incompatible')
            require(parent_state['signature']['config']['host'] == config['host'], 'Source rules host changed')
            parent = parent_state['history'][-1]
            model = checked_snapshot(root, parent, device)
            parent_raw = load_checkpoint(project_path(root, parent['path']), device=device)[1]
            require(model.config == ModelConfig.preset(config['model_size']), 'Source model size differs from config')
            meta_tasks = parent_state.get('meta_tasks', [])
            origin = dict(run=source.relative_to(root).as_posix(), iteration=parent_state['iteration'],
                          checkpoint=parent, signature=digest(parent_state['signature']))
        elif initial:
            model, initial_raw = load_checkpoint(project_path(root, initial), device=device)
            meta_tasks = initial_raw['training'].get('meta_tasks', [])
        else:
            model = FourSeatNet(ModelConfig.preset(config['model_size'])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        if parent_state is not None:
            resume_optimizer(parent_raw, optimizer)
            # Keep Adam moments/RNG, while honoring the new run's declared LR.
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate']
        path = run / 'generation-0000.pt'
        identity = save_checkpoint(path, model, optimizer, training=dict(mode='step5-initial',
            signature=digest(signature), origin=origin, meta_tasks=meta_tasks))
        entry = dict(path=path.relative_to(root).as_posix(), sha256=sha(path), model_id=identity, generation=0)
        state = dict(format='step5-league/1.0.0', signature=signature, iteration=0,
                     next_seed=parent_state['next_seed'] if parent_state is not None else config['seed'],
                     games_done=parent_state['games_done'] if parent_state is not None else 0,
                     history=(parent_state['history'][:-1] if parent_state is not None else []) + [entry],
                     strength='unmeasured', meta_tasks=meta_tasks, origin=origin)
        write_json(state_path, state)
        raw = load_checkpoint(path, device=device)[1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    resume_optimizer(raw, optimizer)
    summaries = []
    for _ in range(iterations):
        iteration = state['iteration']
        began = time.monotonic()
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        directory = run / f'iteration-{iteration:04d}'
        inference = dict(batch_size=config['batch_size'], max_candidate_cells=config.get('max_candidate_cells', 8192))
        learner = NeuralPolicy(model, device, exploration=config['exploration'], **inference)
        all_games, composition = [], Counter()
        next_seed = state['next_seed']
        index = 0
        for kind, slots in collection_batches(state['games_done'], config['games_per_iteration'], config['workers'],
                                              config['legacy_every'], config.get('group_tables', False)):
            global_index = state['games_done'] + slots[0]
            count = len(slots)
            seeds, next_seed = seeds_for(next_seed, count)
            rng = random.Random(digest(dict(seed=config['seed'], iteration=iteration, index=index)))
            with ExitStack() as stack:
                if kind == 'selfplay':
                    policies = [learner] * 4
                elif kind == 'historical':
                    # Half recent, half whole history; older snapshots remain available.
                    candidates = state['history'][-8:] if rng.random() < .5 else state['history']
                    opponents = [NeuralPolicy(checked_snapshot(root, rng.choice(candidates), device), device,
                                               exploration=.05, **inference) for _ in range(3)]
                    policies = [learner] + opponents
                else:
                    require(qualified, 'Legacy table scheduled but no qualified training opponents')
                    opponents = []
                    for _seat in range(3):
                        opponent = LegacyPolicy(root, rng.choice(qualified), purpose='training', mono=mono)
                        stack.callback(opponent.close); opponents.append(opponent)
                    policies = [learner] + opponents
                offset = global_index % 4
                policies = policies[offset:] + policies[:offset]
                provenance = dict(producer='step5/1.0.0', purpose='training', kind=kind, seeds=seeds,
                    sampling_seed=config['seed'] + iteration * 10007 + index,
                    policies=[p.policy_id for p in policies], implementation=signature['implementation'], slots=slots)
                games = collect_cached(root, directory / f'actor-{index:04d}.jsonl.gz', host, seeds,
                                       policies, config, provenance, mono=mono)
            all_games.extend(games); composition[kind] += count; index += count
            print(json.dumps(dict(iteration=iteration, collected=index, of=config['games_per_iteration'], kind=kind,
                                  concurrent_games=count)), flush=True)
        collected_at = time.monotonic()
        minibatch = config.get('minibatch_size', config['batch_size'])
        update = ppo_update(model, optimizer, all_games, learner, epochs=config['ppo_epochs'],
                            batch_size=minibatch, entropy_weight=config['entropy_weight'],
                            max_candidate_cells=config.get('max_candidate_cells'))
        updated_at = time.monotonic()
        search_report = dict(updated=False, reason='Search schedule not due')
        if config['search_every'] and iteration >= config['search_start'] and iteration % config['search_every'] == 0:
            seeds, next_seed = seeds_for(next_seed, config['search_games'])
            print(json.dumps(dict(iteration=iteration, phase='search', games=len(seeds),
                                  search_workers=min(config.get('search_workers', 1), len(seeds)))), flush=True)
            # Independent complete games/trees. The learner is frozen until ALL
            # search workers finish; only then are confirmed labels distilled.
            searched, telemetry, search_workers = collect_search_games(root, directory, host, seeds, model,
                config, signature, iteration, device=device, mono=mono)
            search_report = distill_update(model, optimizer, searched, steps=config['distill_steps'],
                batch_size=minibatch, weight=config['distill_weight'], max_candidate_cells=config.get('max_candidate_cells'))
            search_report.update(telemetry_this_process=telemetry, collection_workers=search_workers)
        searched_at = time.monotonic()
        timings = dict(collection_seconds=collected_at - began, ppo_seconds=updated_at - collected_at,
                       search_and_distill_seconds=searched_at - updated_at)
        throughput = dict(regular_games_per_second=len(all_games) / max(timings['collection_seconds'], 1e-9),
                          learner_inference=dict(learner.evaluator.telemetry))
        memory = None
        if torch.device(device).type == 'cuda':
            memory = dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        summary = dict(iteration=iteration, composition=dict(composition), actor=update,
                       search=search_report, matches=match_report(all_games),
                       diversity=pottery_metrics(all_games, {learner.policy_id}),
                       elapsed_seconds=time.monotonic() - began, timings=timings, throughput=throughput,
                       cuda_memory=memory, strength='unmeasured')
        output = run / f'generation-{iteration+1:04d}.pt'
        identity = save_checkpoint(output, model, optimizer, step=iteration+1,
            training=dict(mode='step5-population-ppo', signature=digest(signature), iteration=iteration,
                          parent=state['history'][-1]['model_id'], evidence=summary, meta_tasks=state.get('meta_tasks', [])))
        entry = dict(path=output.relative_to(root).as_posix(), sha256=sha(output), model_id=identity, generation=iteration+1)
        write_json(directory / 'report.json', summary)
        state.update(iteration=iteration+1, next_seed=next_seed, games_done=state['games_done']+config['games_per_iteration'],
                     history=state['history']+[entry])
        write_json(state_path, state)  # Atomic commit after complete replay/update/checkpoint.
        summaries.append(summary)
        print(json.dumps(dict(iteration=iteration+1, checkpoint=entry['path'], model_id=identity,
                              composition=summary['composition'], pottery=summary['diversity']['builds']['Pottery'])), flush=True)
    return dict(run=str(run), completed_iteration=state['iteration'], checkpoint=state['history'][-1], summaries=summaries)
