# B03 CUDA topology launcher repair (g570-r9)

Reviewed 2026-11-03, America/Chicago. Source report:
`results/rust-lab-qualification/20261104T014313172885Z-20261104T014145936407Z-g570-all-8a6142e0-696c1276`.

B03 passed its CPU fixtures, exact prerequisite reuse, preflight, diagnostic
copy, PPO/PPG/companion restoration comparisons and other pre-RNG checks. It
failed at the final `resume_optimizer` call with `CUDA device topology differs
on resume`. The failure proves saved and visible GPU counts differ; the old
report does not record the two counts, so no specific counts are assumed.

The ordinary Step 5 launcher limits default training to one visible GPU before
importing Torch. Qualification does not apply that visibility mask. Step 5
training already calls `resume_optimizer(..., allow_cuda_topology_change=True)`.
B03 supports the same explicit option, but the unified launcher neither parsed
nor forwarded it. The previous resume command therefore used strict equality.

r9 exposes and forwards that option for a real B03 run on `cuda:0`, records the
requested policy in the runner and B03 receipts, and copies saved/visible/restored
counts from the restoration report into the lab result and console. It rejects
using this flag for CPU fixtures, a different learner index or a stage selection
that does not include B03. The default remains strict, and B04 gets no such flag.

The existing compatibility path restores the CPU RNG and every available saved
logical GPU stream, including learner GPU 0. Additional visible devices retain
their initialized streams; unavailable saved streams are preserved in the
unchanged checkpoint. B03 still checks restored RNG values exactly and executes
inside `torch.random.fork_rng` so diagnostic state is scoped. No learner model,
Adam, PPG/auxiliary state, update logic or RNG implementation changes.

The r8 trainer identity remains
`8b9614d736c727d682e85753d1c851c3663461c733501eefac568624fca0dfe5`.
All `src/`, `configs/`, native and production-training inputs are unchanged from
r8. The original source-continuation registry and exact prerequisite-reuse
review are byte-identical. B01/B02/B02.5 retain their original successful scope;
no recollection or prerequisite qualification is needed. Resume at B03 using the
[current bundle command](../distribution/rust-lab-g570/README.md), retaining the
original B02 receipt and paused g570 run. B03/B04 lab qualification remains Pending.

Bounded regression checks cover CLI parsing, only-B03 forwarding, resumed B03/B04
selection, unchanged prerequisite/state files, mismatched-count strict rejection,
and expanding/shrinking/equal GPU sets through the real Adam/RNG loader with
mocked CUDA hardware. They verify exact optimizer and CPU/saved GPU RNG values,
untouched new streams and unchanged checkpoint payloads. Local PyTorch is CPU-only;
these tests are development evidence, not actual lab CUDA qualification. Packaging
also verifies a fresh extraction and all transfer/reuse identities. No full games,
production update or previously successful lab qualification is rerun.
