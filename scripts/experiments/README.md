# Experiments

Each experiment has a folder for its entry points and private helpers.
See [legacy](../legacy/README.md) for the historical SearchLab smoke and
verification commands.

## C-scale selection

- [policy-c-scale/short-search.py](policy-c-scale/short-search.py)
- [policy-c-scale/long-search.py](policy-c-scale/long-search.py)
- [policy-c-scale/_study.py](policy-c-scale/_study.py)

## PPO opening temperature

- [ppo-opening-temperature/compare-ppo-opening-temperatures.py](ppo-opening-temperature/compare-ppo-opening-temperatures.py)

## PFSP sampling simulation

- [pfsp/pfsp_sim.py](pfsp/pfsp_sim.py)

## Search value correction

- [search-value/evaluate-rust-search-value.py](search-value/evaluate-rust-search-value.py)

## IL head-to-head

- [il-head-to-head/il_head_to_head.py](il-head-to-head/il_head_to_head.py)

## Legacy style

- [legacy-style/run-legacy.py](legacy-style/run-legacy.py)
- [legacy-style/diagnose-legacy.py](legacy-style/diagnose-legacy.py)
- [legacy-style/diagnose-tactical-diversity.py](legacy-style/diagnose-tactical-diversity.py)

## Legacy economy

- [legacy-economy/diagnose-economy-v3.py](legacy-economy/diagnose-economy-v3.py)
- [legacy-economy/smoke-economy-v3.py](legacy-economy/smoke-economy-v3.py)
- [legacy-economy/smoke-era-v3.py](legacy-economy/smoke-era-v3.py)
- [legacy-economy/smoke-industry-v3.py](legacy-economy/smoke-industry-v3.py)
- [legacy-economy/smoke-network-v4.py](legacy-economy/smoke-network-v4.py)

## Masked ISMCTS probes

- [masked-ismcts/masked-search.py](masked-ismcts/masked-search.py)
- [masked-ismcts/masked-strength-check.py](masked-ismcts/masked-strength-check.py)

## R4/R5 SearchLab

- [searchlab/r4r5.py](searchlab/r4r5.py)

## Personality pilot

- [personality/personality.py](personality/personality.py)
- [personality/personality_search.py](personality/personality_search.py)
- [personality/personality_repertoire.py](personality/personality_repertoire.py)
- [personality/personality_deployment.py](personality/personality_deployment.py)
- [personality/personality_difficulty.py](personality/personality_difficulty.py)

## Independent Gumbel references

- [reproduce-gumbel-is-mcts.py](../reproduce-gumbel-is-mcts.py) — Gumbel reference lock.
- [reproduce-gumbel-coordinator.py](gumbel-is-mcts/reproduce-gumbel-coordinator.py) — Qualified coordinator reference.
- [reproduce-gumbel-sessions.py](gumbel-is-mcts/reproduce-gumbel-sessions.py) — Qualified session reference.

The original count-mode oracle retains its locked path; the session and
coordinator references live together in the experiment folder.

## Related training and installation tools
- [export-test-nn.py](../training/checkpoints/export-test-nn.py) — Frozen Rust baseline and NN package builder.
- [test_nn_calibration.py](../training/checkpoints/test_nn_calibration.py) — Sibling helper used by the checkpoint exporter.
- [personality_laptop.py](../local-game/personality-laptop/personality_laptop.py) — Local install and rollback entry point.
