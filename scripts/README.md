# Script workflows

Run commands from the repository root. The workflow indexes list every retained
top-level script under its category and link to its runnable path.

## Layout

```text
scripts/
|-- training/
|   |-- migrations/        Historical recipe changes and restarts
|   |-- rust-lab/          Rust lab setup, fixtures, and validation
|-- local-game/            Build, install, and verification by patch feature
|-- experiments/           One group per experiment
|-- reports/               Report exports, plots, and reviews
|-- benchmarks/            Performance and matchup measurements
|-- legacy/                Standalone archival scripts
`-- policy-c-scale/        Existing short- and long-search experiment
```

| Workflow | Scripts at the top level | What to look for |
| --- | ---: | --- |
| [Training](training/README.md) | 188 | Step 5, continuation, transfer, hosts, and opponent banks. Includes [migrations](training/migrations/README.md) and [Rust lab tools](training/rust-lab/README.md). |
| [Local game](local-game/README.md) | 154 | Patch builds, installers, release archives, and verification grouped by feature. |
| [Experiments](experiments/README.md) | 26 | Opening temperature, PFSP, search/value probes, and personality studies. The three C-scale study files are in [policy-c-scale/](policy-c-scale/). |
| [Reports](reports/README.md) | 18 | Training exports, rating charts, evidence review, and results publication. |
| [Benchmarks](benchmarks/README.md) | 20 | Step 5, collection, native inference, search, and matchups. |

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
python scripts/export-training-reports.py --run "$RUN" --continue --no-push
```

For the managed Windows patch, build from the compatible game base, follow the
release's qualification steps, and close the game before installing. See the
[local patch guide](../docs/LOCAL_PATCH.md) and [feature index](local-game/README.md).

```powershell
pwsh -NoProfile -File scripts/build-patch.ps1
# Run the release's candidate checks before installing.
pwsh -NoProfile -File scripts/install-patch.ps1
pwsh -NoProfile -File scripts/test-unity-patch.ps1
```

Use a fresh output directory for the Windows profile. The C-scale entry points
list the frozen run and search options in their help.

```text
python scripts/profile-rust-windows.py --out artifacts/rust-windows-profile --quick
python scripts/policy-c-scale/short-search.py --help
python scripts/policy-c-scale/long-search.py --help
```

## Paths retained for compatibility

Four standalone scripts moved to [legacy/](legacy/README.md). The other
409 scripts keep their original paths. That includes the 406 indexed at the
top level and the three files in `policy-c-scale/`.

- Several Python and PowerShell scripts compute the repository root from
  their location. The lab shell launchers do the same. The C-scale study
  derives its root from its existing depth.
- Continuation, transfer, and install scripts load siblings by filename.
  The [Step 5 release workflow](../.github/workflows/step5-transfer-release.yml)
  calls `scripts/step5-transfer.py` directly. Lab commands, tests, and pinned
  source inventories also use fixed paths.
- The archived scripts' older paths can still appear in historical docs or
  sealed bundle manifests. Those references belong to the original
  checkouts. The [archive index](legacy/README.md) lists their current paths.
