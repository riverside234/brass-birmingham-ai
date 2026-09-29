# Step 5 scaling checks

29 September 2026. These checks exercise the larger configuration and parallel paths; they do not qualify a stronger model or establish throughput on the lab GPU.

## Local evidence

| Check | Result |
| --- | --- |
| Existing R2 model, replay, batching, checkpoint and ONNX tests | 24 passed on Windows CPU / PyTorch 2.8.0+cpu. |
| Step 5 distribution, learning and scaling tests | 29 passed, including six focused scaling tests. |
| Frozen v3 compatibility | Existing bank and admission checks pass; the pinned `learning/runner.py` bytes are preserved. |
| Ordinary parallel collection | Two real four-player setups completed once serially and once with two concurrent hosts. Permitted observations, legal choices, selected actions, seat routing, behavior probabilities and terminal results match exactly. |
| Parallel search collection | Two independent short-search games completed with two search workers; every replay validated and the learner's weights remained unchanged. |
| Forking a trained run | Weights, Adam state, RNG, historical references and setup cursor retained; source state unchanged; child resumes successfully. This uses tiny authored trajectories with actual gradient/checkpoint operations. |
| Memory splitting | PPO and teacher-distillation updates agree within numerical tolerance with and without candidate-based microbatch splitting; no decisions or covered-teacher weights are dropped. |

Real-host evidence is stored locally in [the scaling report](../artifacts/step5/scaling-checks/001dd8222cc0/report.json). It is generated data, not a required file in a Git clone. The command to reproduce it is in [the scaling guide](STEP5_SCALING.md#focused-checks-and-limits).

The two-game serial pass took 48.61 seconds and the parallel pass 48.08 seconds. This tiny Windows CPU sample establishes semantic parity and **does not demonstrate a meaningful speedup**. The 32-host Linux/CUDA workload requires a fresh lab measurement.

Search was deliberately limited to one second and at most one attempt per seat. All eight attempts were incomplete, so no confirmed teacher labels were invented. This checks worker isolation, legal fallback and full-game completion. It does not measure teacher quality at the main preset's 30-second budget. Synthetic confirmed targets exercise the distillation gradient path separately.

The tested trainer implementation ID is `d03257e216c8738d5553c9d10bf3db902d56f6e9d1335d48f3c3ed055c67c48a`. Raw parity trajectory digest: `6079277593ba89e8ea36e61acfa380884a20d0345db5f4b818bc2242e435e0f8`.

## Lab measurements still needed

Run the first two fresh scaled iterations, then a cycle including scheduled search. Compare ordinary games/second, learner decisions, update time, confirmed-teacher counts and peak allocated CUDA memory. The preset increases workload and removes coordinator concurrency limits; it does not promise to occupy all 90 GB of GPU memory or all 64 logical CPUs. Python encoding/validation, transport, rules work and replay serialization may remain bottlenecks.

No qualification tournament or GPU training was run on this Windows CPU checkout. Current Step 5 weights remain the foundation for the later personality stage after self-training finishes.
