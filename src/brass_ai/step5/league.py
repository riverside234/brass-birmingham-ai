"""Resumable self-play-first population trainer; immutable historical snapshots."""
from __future__ import annotations

from contextlib import ExitStack
from collections import Counter
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
from brass_ai.learning.runner import collect_games, match_report
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
    return digest({p.name: p.read_text(encoding='utf-8').replace('\r\n', '\n')
                   for p in sorted(Path(__file__).parent.glob('*.py'))})


def validate_config(config):
    expected = {'format', 'players', 'model_size', 'host', 'games_per_iteration', 'workers', 'batch_size',
                'threads', 'learning_rate', 'ppo_epochs', 'exploration', 'entropy_weight', 'seed',
                'legacy_every', 'search_start', 'search_every', 'search_games', 'distill_steps', 'distill_weight', 'search'}
    require(set(config) == expected, 'Unknown or missing Step 5 config field')
    require(config.get('format') == 'step5-training/1.0.0' and config.get('players') == 4,
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
                          sampling_seed=provenance['sampling_seed'], mono=mono)
    write_shard(path, games, provenance)
    return games


def train(root, run_path, config, *, iterations=1, device='cpu', initial=None, mono=None):
    root = Path(root).resolve(); run = project_path(root, run_path)
    config = validate_config(config)
    require(iterations > 0, 'Positive iteration count required')
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
        require(initial is None, 'Do not change initialization while resuming')
        entry = state['history'][-1]
        model = checked_snapshot(root, entry, device)
        raw = load_checkpoint(project_path(root, entry['path']), device=device)[1]
    else:
        require(not run.exists() or not any(run.iterdir()), 'New run directory must be empty')
        run.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config['seed'])
        meta_tasks = []
        if initial:
            model, initial_raw = load_checkpoint(project_path(root, initial), device=device)
            meta_tasks = initial_raw['training'].get('meta_tasks', [])
        else:
            model = FourSeatNet(ModelConfig.preset(config['model_size'])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        path = run / 'generation-0000.pt'
        identity = save_checkpoint(path, model, optimizer, training=dict(mode='step5-initial', signature=digest(signature)))
        entry = dict(path=path.relative_to(root).as_posix(), sha256=sha(path), model_id=identity, generation=0)
        state = dict(format='step5-league/1.0.0', signature=signature, iteration=0,
                     next_seed=config['seed'], games_done=0, history=[entry], strength='unmeasured', meta_tasks=meta_tasks)
        write_json(state_path, state)
        raw = load_checkpoint(path, device=device)[1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    resume_optimizer(raw, optimizer)
    summaries = []
    for _ in range(iterations):
        iteration = state['iteration']
        began = time.monotonic()
        directory = run / f'iteration-{iteration:04d}'
        learner = NeuralPolicy(model, device, exploration=config['exploration'], batch_size=config['batch_size'])
        all_games, composition = [], Counter()
        next_seed = state['next_seed']
        index = 0
        while index < config['games_per_iteration']:
            global_index = state['games_done'] + index
            kind = table_kind(global_index, config['legacy_every'])
            count = 1
            while count < config['workers'] and index + count < config['games_per_iteration'] and table_kind(global_index + count, config['legacy_every']) == kind:
                count += 1
            seeds, next_seed = seeds_for(next_seed, count)
            rng = random.Random(digest(dict(seed=config['seed'], iteration=iteration, index=index)))
            with ExitStack() as stack:
                if kind == 'selfplay':
                    policies = [learner] * 4
                elif kind == 'historical':
                    # Half recent, half whole history; older snapshots remain available.
                    candidates = state['history'][-8:] if rng.random() < .5 else state['history']
                    opponents = [NeuralPolicy(checked_snapshot(root, rng.choice(candidates), device), device,
                                               exploration=.05) for _ in range(3)]
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
                    policies=[p.policy_id for p in policies], implementation=signature['implementation'])
                games = collect_cached(root, directory / f'actor-{index:04d}.jsonl.gz', host, seeds,
                                       policies, config, provenance, mono=mono)
            all_games.extend(games); composition[kind] += count; index += count
            print(json.dumps(dict(iteration=iteration, collected=index, of=config['games_per_iteration'], kind=kind)), flush=True)
        update = ppo_update(model, optimizer, all_games, learner, epochs=config['ppo_epochs'],
                            batch_size=config['batch_size'], entropy_weight=config['entropy_weight'])
        search_report = dict(updated=False, reason='Search schedule not due')
        if config['search_every'] and iteration >= config['search_start'] and iteration % config['search_every'] == 0:
            seeds, next_seed = seeds_for(next_seed, config['search_games'])
            search = SearchPolicy(model, host, device, config=SearchConfig(**config['search']), mono=mono)
            try:
                # All four neural searches share no team objective. This is additional
                # neural self-play data, never imitation of the weak legacy bank.
                provenance = dict(producer='step5/1.0.0', purpose='training', kind='neural-search',
                    seeds=seeds, sampling_seed=config['seed'] + iteration * 17011,
                    policies=[search.policy_id] * 4, implementation=signature['implementation'])
                searched = collect_cached(root, directory / 'search.jsonl.gz', host, seeds, [search] * 4,
                                          config, provenance, mono=mono)
                search_report = distill_update(model, optimizer, searched, steps=config['distill_steps'],
                    batch_size=config['batch_size'], weight=config['distill_weight'])
                search_report['telemetry_this_process'] = dict(search.telemetry)
            finally:
                search.close()
        summary = dict(iteration=iteration, composition=dict(composition), actor=update,
                       search=search_report, matches=match_report(all_games),
                       diversity=pottery_metrics(all_games, {learner.policy_id}),
                       elapsed_seconds=time.monotonic() - began, strength='unmeasured')
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
