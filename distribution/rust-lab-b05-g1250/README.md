# B05 g1250 qualification

Linux/CUDA qualification: **Pending**. This is a new immutable B05 bundle;
B04.5 r3 has passed. No earlier qualification needs repeating. No new games,
native rebuild, production update or teacher enablement occurs.

From the lab **code repository**:

```bash
conda activate ai-training
git pull --ff-only
unzip -o distribution/rust-lab-b05-g1250/rust-lab-b05-g1250-r2.zip
python scripts/qualify-rust-lab-b05.py --device cuda:0
```

The run must be stopped at the original committed **g1250** checkpoint:
`artifacts/step5/joint7-v11-2-g1290-trained-restart`, SHA256
`de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`.
Use its exact saved config and full dependencies. If production has advanced,
use an already preserved compatible g1250 run via `--run PATH`; do not rewind
production or rewrite its saved identities. The normal run lock is held during
the lab check. R2 defaults to the existing logical-GPU compatibility path for
this isolated diagnostic: the learner stays on logical `cuda:0`, existing RNG
streams are restored exactly, unavailable inference devices are skipped and
new devices keep their initialized streams. The restoration report records
saved/visible/restored counts. Use `--strict-cuda-topology` to require an equal
device count. The underlying checkpoint loader's strict default is unchanged.
The original `--allow-cuda-topology-change` flag remains accepted.

R1's attempt ending `d6e77efd-221a8f6f` passed all 70 CPU fixtures, then stopped
at the GPU-count check before any diagnostic update. R2 fixes that launcher
default and reports the original child exception directly in qualification.json.
It does not alter training/search code, source identity or checkpoint bytes.

Keep the original sealed B03 replay files in the lab artifacts folder. Their
absolute paths and checksums are recorded in `reference.json`; qualification
does not recollect or relabel them. The B04.5 tables are included in this bundle.
The launcher verifies the checkout and every bundled input before execution.

The short fixture suite checks rejection, interruption, off/shadow equivalence,
optimizer ownership and resume. The CUDA check fully restores g1250 into an
isolated diagnostic fork, exercises bounded PPO/PPG and task-teacher kernels,
reloads their state, and updates/restores only the search head on frozen B04.5
value rows. Historical replay preparation is checked under its original recipe.
The new component checkpoints do not impersonate a production iteration or
advance production gates. See [scope and limitations](../../ai-research/RUST_BACKEND_B05_IMPLEMENTATION.md).

Every attempt prints a fresh folder under `results/rust-lab-qualification/`.
Push that folder **from the results repository**, using the exact commands
printed by the launcher. The source run stays unchanged. CPU-only preflight is
available with `--fixtures-only`; it leaves Linux/CUDA qualification Pending.
