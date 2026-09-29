# R2 four-player learning scaffold

R2 adds a Daiyu-based model and batching/training pipeline over R1's declared
`0.2.0` interface. It initializes **new weights**, uses one player's permitted view
per prediction, and returns results for all four stable seat identities. This is
the foundation for the later search and opponent-population work. It does not
install a replacement opponent in Unity or establish stronger play.

The lab project root defaults to **`/data/not_backed_up/yxu209/ai-training/`**.
All host, replay, model and output paths resolve under that root. An explicit
absolute `--project-root` override is available for laptop checks. Relative path
arguments are relative to that root, never the shell's working directory. Source
and versioned contract files are loaded from the installed checkout.

## Model and permitted features

| Preset | Parameters | FP32 weight bytes, approximately |
| --- | ---: | ---: |
| small | 473,229 | 1.89 MB |
| medium, default | 1,897,389 | 7.59 MB |
| large | 7,797,901 | 31.19 MB |

These are parameter sizes, excluding the inference runtime, metadata, activations
and search memory. The medium preset implements the planned approximately 2M
starting experiment. Model size is not an advertised difficulty level.

The encoder produces 3,207 state fields and 184 fields per legal candidate.
Separate state/candidate encoders feed a shared choice scorer. Other heads predict
four first-place shares, four competition ranks and four final VP estimates.
First-place shares sum to one; placements allow tied ranks such as `[1,1,3,4]`.
All heads are derived from **one** allowed observation, not four private hands.
Internal seats run clockwise from the observer; exported outputs are restored to
stable IDs `0,1,2,3`, independent of spending-based turn order.

Features include both eras, all decision phases, board tiles/links/merchants,
markets, the observer's cards, public resources, opponents' hand counts, face-up
discards, publicly acquired wild cards, pending payments and continuation state.
Scout cards retain multiplicity. Legal choices are semantic R1 choices, including
individual resource payments, optional continuations, bonuses and liquidation.

The initial encoder intentionally compresses mats to counts by level plus the
top two tiles, and public history to counts by seat, era and choice kind. It is
not a lossless information-state encoding or a learned belief model. Four style
inputs and eight intent fields are reserved and validated in `[-1,1]`; current
model/random collection supplies zeros. R3's external legacy adapter supplies
permitted style/intent context in the same fields. Distinct neural personalities
are not trained by the R2 scaffold alone.

Unknown/private fields fail R1 validation. Engine objects, full checkpoints,
actual opponent hands, the true deck, game IDs, engine RNG and setup seeds never
enter feature tensors. Revision and choice IDs are used for routing only.

## Install on the lab

Use Python 3.12. The following selects the pinned PyTorch 2.8 CUDA 12.8 wheel;
the actual lab GPU/driver still needs a local qualification run. This installation
option is listed in [PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/).

```sh
python3.12 -m venv /data/not_backed_up/yxu209/ai-training/.venv
/data/not_backed_up/yxu209/ai-training/.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
/data/not_backed_up/yxu209/ai-training/.venv/bin/python -m pip install -r /data/not_backed_up/yxu209/ai-training/requirements-r2.txt
/data/not_backed_up/yxu209/ai-training/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/verify-r2.py
```

For CPU-only installation, use `https://download.pytorch.org/whl/cpu` for the first
pip command. `--device cuda` fails if CUDA is unavailable; `auto` explicitly
selects CPU when CUDA is unavailable. Nothing in these commands contacts the lab
server on your behalf, copies game files, or publishes generated data.

The Python model/trainer and unit/export tests run without the game. New match
collection requires your R1 host and its fingerprinted dependencies at
`/data/not_backed_up/yxu209/ai-training/artifacts/r1-host/guarded/`.
They are not included in Git. The host's Linux/Mono path is experimental: R1 was
qualified on Windows .NET Framework and actual Unity Mono. Until Linux parity
passes, collect with the qualified Windows host, privately transfer the R2
replay shards under the lab root, and train those shards on the server. R2 does
not claim to have tested the lab, CUDA, or Intel graphics.

## Runnable commands

Initialize fresh default-sized weights:

```sh
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/r2.py --device cuda init --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/fresh.pt
```

Collect complete four-player games with the qualified host. Omit `--checkpoint`
to use the diagnostic legal-random policy. `--workers` creates separate host
processes; neural requests from concurrent games are batched. No unsafe shared
engine clones or simultaneous calls on one host are used.

```sh
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/r2.py --seed 2100 collect --games 8 --workers 4 --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/fresh.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/r2/games-2100.jsonl.gz
```

Train the outcome heads from complete games, retaining valid losses. A new model
is initialized if `--from-checkpoint` is omitted. Only the training partition
contributes gradients. `--mode teacher` requires explicit fresh-finalist targets;
the R2 collector does not manufacture search labels from random selections.

```sh
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/r2.py --device cuda train --shards /data/not_backed_up/yxu209/ai-training/artifacts/r2/games-2100.jsonl.gz --from-checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/fresh.pt --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/value.pt --steps 100
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/r2.py --device cuda train --resume --shards /data/not_backed_up/yxu209/ai-training/artifacts/r2/games-2100.jsonl.gz --from-checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/value.pt --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/value-resumed.pt --steps 100
```

Value-only fitting does not teach the choice scorer a better policy. For an
initial policy-learning experiment, `iterate` freezes the current policy for all
four seats, collects fresh complete games, then makes **one** accumulated
terminal REINFORCE update with a permitted-view baseline. It saves each iteration
under a new run directory. This is an experiment before R3 search, not the
recommended final population trainer or a strength claim.

```sh
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/r2.py --device cuda --seed 3100 iterate --from-checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/fresh.pt --iterations 2 --games 4 --workers 4
```

Every seat gets its own terminal win-share reward and baseline. Decisions are
summed within a seat trajectory, then averaged across seat trajectories. No
two-player sign flip or combined bot reward is used. The update rejects random,
stale, temperature-altered or mismatched behavior probabilities. This entry point
starts a new experiment from weights; it does not promise exact restart of a
partially completed self-play iteration. Exact optimizer/RNG resume is supported
and checked for supervised training with the same data and configuration.

Export for local CPU inference, verifying multiple batch/candidate shapes against
PyTorch before publishing the ONNX file and matching JSON manifest:

```sh
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/r2.py export --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/r2/value.pt --shards /data/not_backed_up/yxu209/ai-training/artifacts/r2/games-2100.jsonl.gz --output /data/not_backed_up/yxu209/ai-training/artifacts/r2/model.onnx
```

`evaluate --opponents` takes exactly four checkpoint paths or `random`, rotates
the four assignments, and requires a multiple of four games. It reports per-seat
exposure, win shares, placements and VP; it does not promote checkpoints or
estimate calibrated skill. Use held-out setup seeds and opponent families for
later strength evaluation. [R3/H5](R3_TACTICAL_BANK.md) now provides immutable
legacy versions and evidence-gated, disjoint training/evaluation pools. Use its
mixed-model/bank collector to generate compatible outcome shards; legacy actions
are external behavior, not optimal teacher labels or on-policy neural samples.
The training reader rejects shards marked with evaluation-only purpose, including
those whose setup hashes would otherwise place them in the training partition.

## Interfaces shared with R3/H0–H5

- `Encoder.encode(view, choices, style, intent)` accepts only R1 permitted inputs.
  It returns tensors plus candidate IDs and a stable-to-relative seat map.
  R2 requests are acting-seat decision positions. For a non-acting root's
  afterstate, R3 must add a versioned value-only path or evaluate an appropriate
  acting-seat leaf. Never pair the root's view with another seat's private legal
  choices. General afterstate training is a later extension of this scaffold.
- `Evaluator.predict(list[Request])` batches variable candidate lists with padded
  legal masks. Responses carry request/model/revision/observer identity. The
  evaluator freezes a copy of its model for the match. Requests snapshot their
  permitted inputs, and responses bind the view, choices, style and intent with
  an input digest; this never uses the engine's private canonical-state hash.
- A `Request` carries an **absolute monotonic deadline**. R3 must reuse the same
  whole-turn deadline across planning, action execution and continuation. R2's
  diagnostic collector uses per-decision timeouts; it is not the live-game turn
  coordinator. Expired/cancelled results are discarded before submission, but a
  native inference kernel already in progress cannot be preempted synchronously.
- External opponents implement the permitted-input `select_batch` interface;
  they are never handed the host or checkpoints. The runner records a frozen
  policy version for every seat. Stock H0's unmasked logs are not learner inputs.
- Search teachers provide candidate probabilities, an examined-choice coverage
  mask, confidence, fresh sample count and horizon. Unexamined choices are
  excluded from the policy-loss denominator. Raw scheduler visit counts are not
  accepted as targets. Complete-turn grouping belongs in R3's teacher adapter.

Current R2 batching is synchronous with bounded batches. The GUI bridge,
asynchronous queue, action-plan scheduler, Play Now, failure fallback and search
are R3 work. R2 completion is not required to begin the heuristic bank.

## Reproducibility and compatibility

`features-1.0.0.json` declares the vocabularies, scales, compression and seat
encoding. Replay/checkpoint/export signatures bind this schema, normalized encoder
and architecture source hashes, R1 wire/rules, host producer profile, reference
pins and four-seat targets. Old scalar checkpoints and two-/three-player data
fail early. Checkpoint loads use `weights_only=True` and strict tensor shapes.

Each gzip JSONL shard starts with compatibility/provenance metadata and contains
complete games. Records retain legal choices, chosen index, policy version,
behavior probability, teacher coverage, timing, style/intent and allowed views.
Terminal results are attached separately. Collection failure or the decision cap
aborts the shard; it never fabricates terminal targets. Setup groups receive a
deterministic 80/10/10 train/validation/test assignment, and repeated setups stay
together. Opponent-family holdouts are still the responsibility of later
population evaluation. Generated shards/checkpoints/exports stay ignored by Git,
except curated bootstrap and release weights under `training/models/`. The
[Step 5 guide](STEP5_LAB_TRAINING.md) documents those explicit Git exceptions, the
private rules transfer, and the self-play-first population/meta-RL workflow that
now builds on this scaffold.

The initial loader validates and holds a shard selection in memory. Full-history
schema validation, host IPC and replay encoding can dominate small-network
inference. Profile these before large runs; R2 is not yet a streaming distributed
trainer. Optimizing throughput must preserve R1's information and legality gates.

Provenance and the Daiyu license are recorded in the
[component notice](../src/brass_ai/learning/NOTICE.md). R0's pins and contracts are
unchanged. See [R2 exit checks](R2_EXIT_CHECKS.md) for the measured scope.
