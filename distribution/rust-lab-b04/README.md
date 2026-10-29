# B04: native search_v1 teachers

Implemented; **Linux/CUDA qualification Pending**. Transfer this directory with the updated project checkout and the B01/B02/B02.5/B03 bundles. The ZIP contains source, tests and qualification tools, not Windows binaries or model weights. Keep the existing private training runtime and committed joint7 checkpoints on the server.

The collector uses **4 seconds / K16, one Rust search session with one thread per table**, Gumbel noise, one pending path, c1=50/c2=0.1, and a shared whole-turn deadline with a 0.4-second reserve. search_v1 is explicit; search_v2 and V205 are rejected. Two qualification tables run in parallel through the existing frozen GPU inference service. Only these diagnostic tables use shadow mode; the production run/config and installed game are untouched.

## Extract before running prerequisites

Pause training at a committed boundary and keep it paused throughout qualification. Use the normal `ai-training` environment and the B01 toolchain/runtime prerequisites (Rust 1.99.1, Mono, CUDA/PyTorch and the retained C# bootstrap dependencies).

```bash
conda activate ai-training
unzip -o distribution/rust-lab-b01/b01-search-v1-update.zip
unzip -o distribution/rust-lab-b02/b02-search-v1-update.zip
unzip -o distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip
unzip -o distribution/rust-lab-b03/b03-search-v1-admission.zip
unzip -o distribution/rust-lab-b04/b04-search-v1-teacher.zip

RUN=artifacts/step5/joint7-v11-2-grouped-lr-critic
SUPPLEMENT=distribution/rust-lab-b04/transfer-manifest.json
G=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["iteration"])' "$RUN/state.json")
SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["history"][-1]["sha256"])' "$RUN/state.json")
```

Do not edit a saved signature, rewrite checkpoint origins or create a fresh learner. Source-review transitions retain the current saved recipe, optimizers, RNG, PPG and lineage. Mode/share-only receipt preservation is unchanged. B05/B07 still gate live training/adoption.

## Qualification commands

B01–B03 remain Pending until run on the server. Run them **after the final extraction**, so their source identities match. B03 reuses B02's bounded games and performs no extra games or optimizer updates. The supplement option verifies earlier manifests against their immutable ZIPs, then verifies B04's replacements.

```bash
bash scripts/run-rust-lab-b01-b02.sh "$RUN" cuda:0

# Use the B02 path printed by that command.
B02=artifacts/rust-backend/YOUR_B01_B02_OUTPUT/b02/lab/qualification.json

python scripts/qualify-rust-lab-b025.py \
  --transfer-manifest "$SUPPLEMENT" --run "$RUN" \
  --expected-generation "$G" --device cuda:0

python scripts/qualify-rust-lab-b03.py \
  --transfer-manifest "$SUPPLEMENT" --run "$RUN" --b02 "$B02" \
  --expected-generation "$G" --checkpoint-sha256 "$SHA" --device cuda:0

# Use the B03 qualification path printed above.
B03=artifacts/rust-backend/YOUR_B03_OUTPUT/qualification.json
bash scripts/run-rust-lab-b04.sh "$RUN" "$B02" "$B03" "$G" "$SHA" cuda:0
```

B04 builds a detached copy of the exact B01 core plus read-only export hooks. It runs the native regression suite, calibrates GPU trajectory cost on two existing permitted roots (eight trajectories each), then plays **exactly two complete teacher tables**, the first planned training table and first planned held-out table. Each preserves all 16 reserved turns and the production 4s/K16 profile. The remaining 30 jobs are recorded as unexecuted diagnostic scope. No retries or additional games are launched to manufacture label coverage. The table watchdog is 180 seconds; the entire calibration/collection/admission child is bounded at 240 seconds. Compilation has a separate bound.

The latest non-null current-reward readiness receipt must say ready. An explicit false receipt, reward change, or an unverifiable gap in retained readiness history **defers the smoke with zero tables** and leaves B04 Pending. There is no g890 exception, synthetic receipt or automatic teacher enablement. A subsequent real PPG fit can provide readiness; rerun in a new output directory when the prerequisite is met.

Evidence is written under the printed `artifacts/rust-backend/...-b04/` directory: source/build identities, frozen jobs, readiness, GPU calibration, raw teacher tables, per-search N/K/completed work/rounds/deadlines/overruns, label kinds/phases, Python target/logit checks and GPU queue/forward/IPC timings. No strength or throughput gain is claimed by this qualification.

Optional short CPU checks, without a lab claim:

```bash
python scripts/qualify-rust-lab-b04.py --fixtures-only
```

The local development report additionally records the actual native eight-trajectory/model probe and compiled Rust tests. Full lab qualification remains Pending until the Linux/CUDA runner passes. B05 owns the next PPO/PPG update and committed resume; B07 owns production enablement.
