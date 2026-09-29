"""First-order Reptile experiment over frozen opponent-lineup RL tasks.

This learns an initialization for between-game gradient adaptation, not a
recurrent RL^2 agent, MAML second derivatives or inference-time mind reading.
Support and query games have disjoint setups. Query results never backpropagate.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
from brass_ai.learning.contracts import digest, project_path, require
from .league import collect_cached, implementation_id, seeds_for, write_json
from .policy import NeuralPolicy
from .updates import ppo_update, reptile_step
from .evaluation import outcome, summarize_outcomes


def adapt_task(root, directory, initialization, opponent_models, config, *, seed,
               inner_steps=2, support_games=4, query_games=4, device='cpu', mono=None,
               purpose='meta-training'):
    require(inner_steps >= 2 and support_games > 0 and query_games > 0,
            'Reptile requires at least two inner updates and nonempty support/query games')
    adapted = copy.deepcopy(initialization)
    inference = dict(batch_size=config['batch_size'], max_candidate_cells=config.get('max_candidate_cells', 8192))
    opponents = [NeuralPolicy(m, device, exploration=.05, focus='neutral', **inference) for m in opponent_models]
    require(len(opponents) == 3, 'Task requires three independently motivated opponents')
    task_id = digest([p.evaluator.model_id for p in opponents])
    host = project_path(root, config['host'])
    support_seeds, next_support = seeds_for(seed, inner_steps * support_games, 'train')
    query_seeds, next_query = seeds_for(seed, query_games, 'validation')
    require(not set(support_seeds) & set(query_seeds), 'Support/query setup leakage')

    def query(model, label):
        policy = NeuralPolicy(model, device, exploration=0, focus='neutral', **inference)
        table = [policy] + opponents
        provenance = dict(producer='step5/1.0.0', purpose='evaluation', role=label,
                          task=task_id, seeds=query_seeds, sampling_seed=seed + 11,
                          policies=[p.policy_id for p in table])
        games = collect_cached(root, directory / (label + '.jsonl.gz'), host, query_seeds,
                               table, config, provenance, mono=mono)
        seed_by_group = {digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=s)): s for s in query_seeds}
        rows = [outcome(g, policy.policy_id, seed_by_group[g['setup_group']]) for g in games]
        return summarize_outcomes(rows)

    before = query(initialization, 'query-before')
    optimizer = torch.optim.AdamW(adapted.parameters(), lr=config['learning_rate'])
    updates = []
    for inner in range(inner_steps):
        policy = NeuralPolicy(adapted, device, exploration=config['exploration'], focus='neutral', **inference)
        seeds = support_seeds[inner * support_games:(inner+1) * support_games]
        table = [policy] + opponents
        # Meta-evaluation support is disposable adaptation data and cannot be
        # imported into the main trainer despite using the train setup partition.
        provenance = dict(producer='step5/1.0.0', purpose='training' if purpose == 'meta-training' else 'evaluation',
                          role='support', task=task_id, seeds=seeds, sampling_seed=seed + 100 + inner,
                          policies=[p.policy_id for p in table])
        games = collect_cached(root, directory / f'support-{inner}.jsonl.gz', host, seeds,
                               table, config, provenance, mono=mono)
        updates.append(ppo_update(adapted, optimizer, games, policy, epochs=config['ppo_epochs'],
                                  batch_size=config.get('minibatch_size', config['batch_size']),
                                  entropy_weight=config['entropy_weight'],
                                  max_candidate_cells=config.get('max_candidate_cells')))
    after = query(adapted, 'query-after')
    report = dict(task_id=task_id, opponent_ids=[p.evaluator.model_id for p in opponents],
                  support_seeds=support_seeds, query_seeds=query_seeds, inner_updates=updates,
                  before=before, after=after, query_win_share_change=after['win_share']-before['win_share'],
                  query_used_for_gradient=False, purpose=purpose,
                  next_seed=max(next_support, next_query))
    write_json(directory / 'report.json', report)
    return adapted, report


def meta_train(root, run_path, initial, references, config, *, iterations=1, tasks=3,
               inner_steps=2, support_games=4, query_games=4, fraction=.1,
               device='cpu', mono=None):
    root = Path(root).resolve(); run = project_path(root, run_path)
    require(iterations > 0 and tasks > 0 and references and 0 < fraction <= 1, 'Invalid meta run')
    ref_models = [load_checkpoint(project_path(root, p), device=device)[0] for p in references]
    signature = dict(config=config, initial_sha256=sha(project_path(root, initial)),
        references={p: sha(project_path(root, p)) for p in references}, tasks=tasks, inner_steps=inner_steps,
        support_games=support_games, query_games=query_games, fraction=fraction, implementation=implementation_id())
    state_path = run / 'state.json'
    if state_path.exists():
        state = read(state_path); require(state['signature'] == signature, 'Meta run signature changed')
        path = project_path(root, state['checkpoint'])
        require(sha(path) == state['checkpoint_sha256'], 'Meta checkpoint changed')
        model = load_checkpoint(path, device=device)[0]
    else:
        require(not run.exists() or not any(run.iterdir()), 'New meta directory must be empty')
        model, initial_raw = load_checkpoint(project_path(root, initial), device=device)
        state = dict(format='step5-meta/1.0.0', signature=signature, iteration=0,
                     next_seed=config['seed'] + 2000000, tasks_seen=initial_raw['training'].get('meta_tasks', []))
    for _ in range(iterations):
        iteration = state['iteration']
        adapted, reports = [], []
        next_seed = state['next_seed']
        for task in range(tasks):
            torch.manual_seed(config['seed'] + iteration * 7919 + task)
            rng = random.Random(config['seed'] + iteration * 3571 + task)
            opponents = [rng.choice(ref_models) for _ in range(3)]
            seed = next_seed
            trained, report = adapt_task(root, run / f'outer-{iteration:04d}' / f'task-{task:03d}',
                model, opponents, config, seed=seed, inner_steps=inner_steps, support_games=support_games,
                query_games=query_games, device=device, mono=mono)
            adapted.append(trained); reports.append(report)
            next_seed = report['next_seed']
        reptile_step(model, adapted, fraction)
        task_ids = sorted(set(state['tasks_seen']) | {r['task_id'] for r in reports})
        path = run / f'meta-generation-{iteration+1:04d}.pt'
        save_checkpoint(path, model, step=iteration+1, training=dict(mode='step5-reptile',
            signature=digest(signature), meta_tasks=task_ids, reports=reports, strength='unmeasured'))
        state.update(iteration=iteration+1, next_seed=next_seed,
                     tasks_seen=task_ids, checkpoint=path.relative_to(root).as_posix(), checkpoint_sha256=sha(path))
        write_json(state_path, state)
    return state


def meta_evaluate(root, output, candidate, references, config, *, seed=6510000,
                  inner_steps=2, support_games=4, query_games=4, device='cpu', mono=None):
    root = Path(root).resolve(); directory = project_path(root, output)
    require(len(references) == 3, 'Meta evaluation requires a concrete three-opponent task')
    model, raw = load_checkpoint(project_path(root, candidate), device=device)
    pairs = [load_checkpoint(project_path(root, p), device=device) for p in references]
    task_id = digest([r['model_id'] for _, r in pairs])
    require(task_id not in raw['training'].get('meta_tasks', []), 'Meta-test lineup was used in meta training')
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _, report = adapt_task(root, directory, model, [m for m, _ in pairs], config, seed=seed,
        inner_steps=inner_steps, support_games=support_games, query_games=query_games,
        device=device, mono=mono, purpose='meta-evaluation')
    require(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()), 'Evaluation mutated initialization')
    report.update(initialization_unchanged=True, candidate_sha256=sha(project_path(root, candidate)),
                  adapted_weights_saved=False, held_out_lineup=True,
                  limitation='Small support/query samples do not establish adaptation benefit; compare many held-out tasks.')
    write_json(directory / 'report.json', report)
    return report
