# Teacher outcome auxiliary value revision

Implemented as `teacher-outcome-value/1.0.0`. At the user's request, the auxiliary update is now **enabled by default for fresh self-play configurations**. Diagnostics remain disabled by default. This is an experimental learning change, not evidence of better play.

## Training behavior

Existing collection remains five teacher games, eight search workers per game, shared turn clocks of **9 / 12 / 12 / 12 seconds**, the existing phase reservations and 160-game cadence, up to eight distillation steps, and `search_wall_fraction: null`. PPO, exploration, rewards (including insolvency 0.005/0.075), historical/v4 opponents, architecture and checkpoint tensor shapes remain unchanged. The width-304 utility model still has **2,015,937 parameters**.

Teacher win-share, rank and VP targets reuse the official stable-seat terminal targets. Utility reuses the existing combined-return implementation: terminal win/placement/VP utility minus **future** capped insolvency costs. Each input also carries its four-seat remaining penalty budgets. These are outcomes of the **recorded mixed search/direct policy**, not root-search value estimates or ordinary PPO returns.

Enabled auxiliary learning uses the existing heads and inputs. At each already-eligible distillation optimizer step it adds:

```
0.1 * (0.75 * mean ordinary value loss + 0.25 * mean teacher value loss)
```

Each value loss is the existing win/rank/VP loss plus four-seat utility MSE. `loss_weight` and `teacher_fraction` are separately configurable. Ordinary PPO keeps its current value weight and uses only ordinary learner trajectories; teacher trajectories never enter its actor batches or behavior-probability verification.

Selection is deterministic across games and phases, before auxiliary encoding, and consumes no training RNG. Defaults select at most **24 positions per game and 128 per collection** (normally at most 120 from five games). The auxiliary minibatches share the existing distillation step count, use at most two presentations per selected teacher position, and do not manufacture extra policy labels. Forced, direct and unqualified-search decisions can be valid value positions. Their validity and counts are independent of policy-label eligibility. If the existing diverse-policy-label gate fails, no optimizer steps are added: valid value positions are reported as unused.

After an update, utility readiness is invalidated. Existing head-only calibration retains its ordinary batches, three passes and optimizer step count; each existing step includes the bounded teacher set with the same **75% ordinary / 25% teacher** mean-loss weighting. Both ordinary MSE and weighted mixture MSE must not increase before another collection is admitted. Calibration can present teacher positions more than twice; that is separate from the two-use auxiliary/distillation cap. Reports separate auxiliary and calibration presentations and count their union as distinct positions trained. With `N` ordinary calibration rows, effective batch `B` and `T` selected teacher rows, teacher calibration adds `3 * ceil(N/B) * T` sample presentations, without adding optimizer steps. Its forward/backward cost is explicitly reported.

## Admission, continuation and reporting

Actual actor recipes, shard hashes, requested model/source/reward/config identities and complete-game bindings admit teacher outcomes, including games with no qualified policy labels. Current policy-label quality/freshness checks remain intact. Stale model/source/reward data, changed raw contents, duplicate games/positions and incompatible ordinary learner targets are rejected.

The revision preserves existing model weights, Adam state, RNG, iteration/game counters, population, curriculum and checkpoint lineage. Reviewed continuation admits the immediately preceding loader implementation `014ac0ca...`, the preceding encoder implementation `e794a86e...`, and the already-reviewed `63c9da11...` clock transition. Only the versioned `teacher_value` block may otherwise change. Subsequent opt-in/out or telemetry changes at this source also require explicit reviewed continuation. Pending old-source/config collections still block the transition. Completed compatible v9 IL remains accepted; unfinished IL remains source-bound. The existing dataset does not need regeneration. Derived IL caches may rebuild when their source fingerprint changes; the bounded loader fix remains intact.

Collection reports and compact history version 5 retain separate policy/value eligibility and usage, target/reward identities, completed-game counts, four-phase totals, auxiliary/calibration presentations and timings. They store no per-position prediction vectors or replay contents. Old compact history versions retain their original version and missing fields.

When enabled, diagnostics select at most 128 distinct positions (configurable downward), reuse exactly the same positions/targets at each stage, and report:

- Win-share Brier error, rank/VP MAE, utility MAE/MSE before, post-update and post-calibration.
- Per-head mean/max absolute prediction changes; the existing policy-KL diagnostic measures combined teaching drift.
- Actual probe position/game/phase coverage, extra forward passes/positions, reused fit/calibration positions and diagnostic elapsed time.

Matching existing policy-fit outputs are reused for probe rows with full remaining budgets. Other rows use bounded batched forwards with their actual budgets. The final head-only calibration error pass supplies post-calibration utility predictions; unchanged official heads reuse their post-update predictions. Unavailable stages are marked explicitly. Positions within a game are correlated, and these are training-data diagnostics.

## Start fresh self-play from completed IL

IL is already complete; no retraining or dataset regeneration is needed. After syncing the code, generate a new config from the completed compatible v9 IL run and start self-play in a new run directory. These example names use a new config filename to avoid overwriting an earlier generated config:

```bash
cd /data/not_backed_up/yxu209/ai-training/
python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --output configs/step5/joint7-features4-compact-v9-batch7-teacher-value.json

python scripts/step5.py \
  --config configs/step5/joint7-features4-compact-v9-batch7-teacher-value.json train --fresh \
  --run artifacts/step5/joint7-features4-compact-v9 \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --iterations 1
```

Use your actual completed IL directory and an unused self-play directory. The generated config explicitly records `teacher_value.enabled: true` and `diagnostics: false`. Auxiliary updates follow the existing teacher readiness, collection cadence and policy-label gates. A fresh run does not use `--resume-reviewed-source`. Completed IL from the preceding auxiliary implementation `061e72ea...` is explicitly admitted with unchanged learning compatibility and all original lineage/hash checks. Previously reviewed completed v9 IL sources remain admitted.

An explicit `enabled: false` stays disabled, including in a custom `--base` config. Omitted blocks in old signed configurations keep their original policy-only behavior; the fresh-config generator inserts the enabled block. Existing files are not edited automatically.

## Enable for an already-started self-play run

After syncing the code, create a separate config from the **actual generated config for your run**. Keep its IL binding and all other settings. Adjust these example paths if your run uses different names:

```bash
cd /data/not_backed_up/yxu209/ai-training/
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'src')
from brass_ai.recovery.config import TEACHER_VALUE, validate_config
source = Path('configs/step5/joint7-features4-compact-v9-batch7.json')
target = source.with_name(source.stem + '-teacher-value.json')
assert not target.exists(), 'Choose a new config path'
config = json.loads(source.read_text())
config['teacher_value'] = dict(TEACHER_VALUE, enabled=True, diagnostics=False)
validate_config(config)
target.write_text(json.dumps(config, indent=2) + '\n')
print(target)
PY

python scripts/step5.py \
  --config configs/step5/joint7-features4-compact-v9-batch7-teacher-value.json train \
  --run artifacts/step5/joint7-features4-compact-v9 \
  --imitation-run artifacts/step5/features4-compact-v9-batch7-il \
  --resume-reviewed-source --iterations 1
```

Use the actual committed generation if adding optional `--expected-generation`. No restart script or `--fresh` is needed for continuation. Later calls may request 20 additional iterations using the same config/run. To collect the bounded errors and drift diagnostics, set `diagnostics: true`; `probe_positions` can be reduced below 128. Changes to this block use the same reviewed continuation mechanism at a committed boundary. Keeping `enabled: false` retains policy-only teaching.

## Validation

Final focused-test and saved-data benchmark results are recorded under `artifacts/recovery/teacher-aux-value-20261015/`. The benchmark uses the actual width-304 model, Adam and calibration on five authored native-board game fixtures with synthetic outcomes/search labels, identical saved input/checkpoint files, one warm-up per mode and six alternating repeats. It compares policy-only, auxiliary without diagnostics, and auxiliary with diagnostics. It does not play games, run searches, or modify existing training runs.

**42 focused tests passed in 80.519 seconds.** They cover stable-seat mapping, causal future-cost/budget accounting, independent value eligibility, bounded pre-encoding selection, stale/duplicate rejection, policy gate preservation, exact telemetry on/off weights/Adam/RNG/sample parity, mixed calibration and its ordinary-readiness guard, actual shard/collection integration, checkpoint/restart preservation, pending-work rejection, compact history, unchanged phase clocks/cumulative-limit behavior, and the existing bounded IL loader/update parity smoke. An old readiness fixture was updated to use the compact encoder's native board constants; its preservation assertions remain covered.

Original benchmark implementation: `061e72ea27f3fd4ca8f113971cf8e8711211cd8adaae116d8a8dbcea42421a7f`. The fresh-default revision is `d36679503fe60596a50741b3059bcf343f74785a9443473a6b46438506d899ea`. Learning compatibility remains `747216b5ab873126a6e048ade198883593113f70a815b9012ce40b8f66d443cb`. The benchmark below measures the same enabled algorithm; changing the default does not change its computation.

The default change passed **27 focused tests in 22.805 seconds**, including actual collection/distillation/calibration integration on bounded fixtures, generated-config defaults, explicit opt-out, archived omission, and completed preceding-source IL handoff without changing its checkpoint/state files. Resume and previous source-review protections also pass. Evidence: `artifacts/recovery/teacher-aux-value-default-20261015/focused-tests.txt`. No production IL or self-play run was launched.

The final Windows 11 / PyTorch 2.8 CPU benchmark used two threads:

| Component mode | Median elapsed |
| --- | ---: |
| Policy-only distillation + ordinary calibration | 0.5703 s |
| Auxiliary value + matching mixed calibration | 1.2398 s |
| Same auxiliary update with diagnostics | 1.2192 s |

All modes performed **8 distillation optimizer steps and 3 existing calibration steps**. The auxiliary component cost about **0.6695 seconds more** than policy-only in this tiny workload. The diagnostic on/off median difference was **−0.0206 s**; paired differences ranged from −0.2603 to +0.0669 s, so this is timing noise, not evidence that diagnostics make training faster. Median directly instrumented diagnostic work was **0.002783 s** using a high-resolution timer. This instrumentation excludes ordinary fit/calibration computation reused by the probe.

The 80-position probe covered five authored game records and all four phases (20 positions per phase). It reused 160 existing before/after fit predictions and 80 final calibration predictions, adding **zero forward passes** in this fixture. The focused tests separately exercise unqualified positions and spent budgets requiring extra bounded forwards. Telemetry on/off produced **identical weights, Adam state and RNG**, with identical sample usage and calibration results. Saved data/checkpoint hashes remained unchanged.

There were 80 qualified policy labels and 80 valid value positions; all 80 distinct positions were used. Policy presentations were 128. Value presentations were **128 auxiliary + 240 calibration = 368**, with zero unused value positions; ordinary auxiliary presentations were 128 over 16 distinct ordinary positions. Each phase had 20 valid/used value positions and 92 total presentations. These are correlated authored fixtures, not independent played games.

| Auxiliary fixed-probe error | Before | Post-update | Post-calibration |
| --- | ---: | ---: | ---: |
| Win-share Brier | 0.255563 | 0.250583 | 0.250583 |
| Rank MAE | 1.250000 | 1.250000 | 1.250000 |
| VP MAE | 80.566162 | 25.418875 | 25.418875 |
| Utility MAE | 0.300833 | 0.300833 | 0.300833 |
| Utility MSE | 0.098401 | 0.091372 | 0.091137 |

Mean absolute prediction changes before → post-update were 0.02358 win share, 0.14263 rank, 62.22841 VP and 0.06385 utility. Post-calibration utility change was 0.09071; official heads retained their post-update predictions. Existing exact policy KL was 0.001274 for policy-only and 0.000636 for the auxiliary arm. These errors and KL values verify reporting on fixed synthetic targets; they do not estimate real-world calibration or playing strength.

Evidence: [tests](../artifacts/recovery/teacher-aux-value-20261015/verified-tests.txt), [final benchmark](../artifacts/recovery/teacher-aux-value-20261015/benchmark-verified/report.json). Earlier development benchmark directories are superseded by `benchmark-verified`.

Reproduce the bounded comparison after the usual runtime setup:

```bash
export PYTHONPATH=src:tests/R2:tests/Recovery:tests/FeatureEngineering:tests/Personality
python tests/Recovery/benchmark_teacher_value.py \
  --output artifacts/recovery/teacher-value-local-comparison --repeats 6
```

The **≤1% added median full-iteration wall-time target is unverified**. This component benchmark omits real collection/search, full PPO and representative ordinary calibration volume. It cannot certify that target; diagnostics remain disabled by default. Neither lower training errors nor this smoke comparison demonstrates improved stability or playing strength.

## Changed files

- `src/brass_ai/recovery/teacher_value.py`: outcome admission/selection, auxiliary gradients and bounded probes.
- `updates.py`, `utility.py`, `teacher_collection.py`, `league.py`: existing-step integration, output reuse, matching calibration and reporting.
- `config.py`, `configs/step5/joint7-latest.json`: versioned enabled fresh-run defaults, preserved legacy omission/explicit opt-out behavior and strict bounds.
- `resume.py`, `configs/learning/selfplay-resume-sources.json`, `imitation-handoff-sources.json`: pinned continuation and compatible completed-IL admission.
- `history.py`: bounded version-5 summaries.
- Focused teacher-value/resume tests, the comparison script, current-board readiness fixtures and two history-version assertions.
- This report, lab instructions and the plan's implementation status.
