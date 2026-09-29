"""Small real-engine search/privacy/distillation check; no strength tournament."""
from pathlib import Path
import random
import time

import torch

from brass_ai.beliefs import sample_world_setup
from brass_ai.host import RulesHost
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT, read_json, require
from brass_ai.learning.replay import validate_game, write_shard
from brass_ai.learning.runner import collect_games
from brass_ai.step5.league import seeds_for, write_json
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.step5.search import SearchPolicy, SearchConfig
from brass_ai.step5.updates import distill_update


def run(root, mono=None):
    root = Path(root); directory = root / 'artifacts/step5/integration'
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model, _ = load_checkpoint(root / 'training/models/bootstrap/r2-fresh-medium.pt')
    host_dir = root / 'artifacts/r1-host/guarded'
    composition = read_json(SOURCE_ROOT / 'configs/contracts/v0.2.0/card-composition.json')
    config = SearchConfig(proposals=2, finalists=2, pilot_samples=1, confirmation_samples=2,
                          seconds=180, max_searches_per_seat=1)
    with RulesHost(host_dir, mono=mono) as live, RulesHost(host_dir, mono=mono) as hidden:
        context = live.reset(855001)
        view, choices = live.observe(context['actor']), live.choices()
        worlds = [sample_world_setup(view, composition, seed) for seed in (81, 82)]
        require(worlds[0]['cards'] != worlds[1]['cards'], 'Counterfactual worlds did not differ')
        observations = []
        for world in worlds:
            hidden.reset(1, world)
            observations.append(hidden.observe(view['seat']))
            require(hidden.choices() == choices, 'Counterfactual legal choices changed')
        require(observations[0] == observations[1], 'Root can see sampled hidden cards')
        outputs = []
        print('Step 5: fixed-work complete-turn search on two hidden counterfactuals', flush=True)
        for observation in observations:
            search = SearchPolicy(model, host_dir, config=config, mono=mono, focus='neutral')
            try:
                outputs.append(search.search(observation, choices, random.Random(7081), time.monotonic() + 180))
            finally:
                search.close()
        require(outputs[0] == outputs[1], 'Fixed-work neural search used hidden-state differences')
        for identity in outputs[0][0]:
            legal = live.choices()
            selected = next(c for c in legal if c['id'] == identity)
            context = live.apply(context, selected)
        require(context['actor'] != view['seat'], 'Opening plan did not complete its action')
    print('Step 5: complete four-player game with sparse neural-search labels', flush=True)
    search = SearchPolicy(model, host_dir, config=config, mono=mono, focus='Pottery')
    plain = NeuralPolicy(model, exploration=.1)
    try:
        seed = seeds_for(855100, 1)[0][0]
        games = collect_games(host_dir, [seed], [search, plain, plain, plain], workers=1,
                              sampling_seed=92761, mono=mono)
        validate_game(games[0])
        shard = directory / 'searched-game.jsonl.gz'
        require(not shard.exists(), 'Integration output already exists; preserve it and use a new test directory')
        write_shard(shard, games, dict(producer='step5-integration', purpose='training', setup_seed=seed))
        before = model_id(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
        update = distill_update(model, optimizer, games, steps=2, batch_size=24)
        require(update['updated'] and model_id(model) != before, 'Real search labels did not train weights')
        path = directory / 'distilled-smoke.pt'
        save_checkpoint(path, model, step=2, training=dict(mode='step5-search-smoke', strength='unmeasured'))
        report = dict(passed=True, counterfactual_searches=2, complete_games=1,
                      root_plan_choices=len(outputs[0][0]), confirmed_candidates=len(outputs[0][1]),
                      search_telemetry=dict(search.telemetry), distillation=update,
                      model_changed=model_id(model) != before, result=games[0]['result'])
        write_json(directory / 'report.json', report)
        return report
    finally:
        search.close()
