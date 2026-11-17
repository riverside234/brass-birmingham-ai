# Script workflows

Run commands from the repository root. The workflow indexes link to each
script's runnable path, including entry points kept at the top level.

## Layout

```text
scripts/
|-- training/
|   |-- imitation/         IL, rehearsal preparation, and model tools
|   |-- opponent-banks/    Frozen opponent maintenance and qualification
|   |-- checks/            Training regression and smoke commands
|   |-- rules-host/        Reference rules hosts and contracts
|   |-- checkpoints/       Checkpoint export and read-only calibration
|   |-- migrations/        Historical recipe changes and restarts
|   `-- rust-lab/          Rust lab setup, fixtures, and validation
|       |-- reference-capture/
|       |-- fixtures/
|       `-- validation/
|-- local-game/            Build, install, and verification by patch feature
|   |-- managed-patch/     Shared managed build, staging, and installation
|   |-- turn-history/      Turn history hotfix build, install, and qualification
|   `-- ...                Other patch features and P6 laptop deployment
|-- experiments/           One folder per standalone experiment
|   |-- policy-c-scale/    Shared reader and short- and long-search studies
|   |-- gumbel-is-mcts/    Independent session and coordinator references
|-- reports/               Report exports, plots, and reviews
|-- benchmarks/            Performance and matchup measurements
`-- legacy/                Standalone archival scripts
    `-- rust-lab/          Retired bundle and qualification workflows
```

| Workflow | What to look for |
| --- | --- |
| [Training](training/README.md) | Step 5, continuation, transfer, hosts, and opponent banks. Includes [migrations](training/migrations/README.md) and [Rust lab tools](training/rust-lab/README.md). |
| [Local game](local-game/README.md) | Patch builds, installers, release archives, and verification grouped by feature. |
| [Experiments](experiments/README.md) | C-scale, opening temperature, PFSP, search/value probes, and personality studies. |
| [Reports](reports/README.md) | Training exports, rating charts, audits, and results publication. |
| [Benchmarks](benchmarks/README.md) | Step 5, collection, native inference, search, and matchups. |

## Common commands

On the lab, activate the training environment and set `GEN` to the last
committed generation. This continuation uses the run's existing recipe with a
share target of `0.15` and teacher search-head weight `0`.

```bash
RUN=artifacts/step5/joint7-v11-2-g1190-restart-3
python scripts/continue-rust-g1190-restart-2.py \
  --run "$RUN" --expected-generation "$GEN" --iterations 10 \
  --share-target 0.15 --skip-utility-qualification --teacher-search-head-weight 0

python scripts/qualify-rust-cpu-inference.py --run "$RUN" --expected-generation "$GEN"
python scripts/reports/export-training-reports.py --run "$RUN" --continue --no-push
```

For the managed Windows patch, build from the compatible game base, follow the
release's qualification steps, and close the game before installing. See the
[local patch guide](../docs/LOCAL_PATCH.md) and [feature index](local-game/README.md).

```powershell
pwsh -NoProfile -File scripts/local-game/managed-patch/build-patch.ps1
# Run the release's candidate checks before installing.
pwsh -NoProfile -File scripts/local-game/managed-patch/install-patch.ps1
pwsh -NoProfile -File scripts/local-game/managed-patch/test-unity-patch.ps1
```

Use a fresh output directory for the Windows profile. The C-scale entry points
list the frozen run and search options in their help.

```text
python scripts/benchmarks/profile-rust-windows.py --out artifacts/rust-windows-profile --quick
python scripts/experiments/policy-c-scale/short-search.py --help
python scripts/experiments/policy-c-scale/long-search.py --help
```

## Paths retained for compatibility

Active training and the
[Step 5 release workflow](../.github/workflows/step5-transfer-release.yml)
use the top-level entry points. The [local game index](local-game/README.md)
lists the Rust release tools and installer helper retained at the top level
for source pins and shared callers. The other workflow indexes list the
top-level tools for their own workflows.

Sealed bundles and recorded receipts retain the paths from the checkout
that produced them. Use that checkout to reproduce a sealed workflow.
The 56 historical recipe and restart commands live in
[training migrations](training/migrations/README.md). The 40 retired Rust
bundle and qualification tools are grouped in
[legacy Rust lab](legacy/rust-lab/README.md). See [legacy](legacy/README.md)
for the other archival tools.

The [migration-plan index](../ai-research/migration-plans/README.md) groups the
completed plans and the [further migration plan](../ai-research/migration-plans/FURTHER_SCRIPT_MIGRATION_PLAN.md).
The further migration moved 76 of the reviewed 120 top-level scripts.
The 44 retained paths supply current continuation, transfer/CI, runtime
preparation, release installation, or shared and locked reference functions.
The plan records the complete retained list and verification results.
