# B03 CUDA restoration check and prerequisite reuse (g570-r8)

Reviewed 2026-11-03, America/Chicago. Source report:
`results/rust-lab-qualification/20261104T012220551953Z-20261104T011422740838Z-g570-all-8bee9d91-e1605e8c`.

The report actually **passed B01, B02 and B02.5**. B02's entire collection command
took 204.513 seconds, within the 240-second watchdog; all three families and the
self-play branch passed, with four useful lanes under the 60-worker ceiling.
The `Timeout (0:01:00)!` stack headings are periodic diagnostics, not failed
qualification. The original receipts are authoritative.

B03 failed during exact PPO Adam restoration verification. `load_checkpoint`
loads CPU tensors; `load_state_dict` moves Adam moments to the model's CUDA
device. Calling `torch.equal` directly across these devices raises before it
can compare values. The same comparator also serves PPG and auxiliary payloads.

r8 uses a diagnostic-local recursive comparator. It requires identical
container types, tensor dtypes/shapes and exact values, comparing detached CPU
copies of tensor leaves. It neither casts nor changes saved/live state and uses
no numerical tolerance. The shared production checkpoint/retention code,
optimizer loading/updating, model, RNG, recipe, Rust core and collection remain
unchanged. CPU/CUDA comparison tests cover changed values, dtypes, shapes,
containers and original-device preservation; the real CUDA test is skipped on
this CPU-only development environment and the full lab restore remains Pending.

Do **not** repeat the successful prerequisites. The bundle's sealed
`prerequisite-reuse.json` binds their exact original receipts, r7 source identity
and the enumerated old/new hashes for the diagnostic comparator, B03 admission
guard, new reuse verifier and source-continuation registry. All other B01
executable/runtime inventory entries must match exactly. B02 must be the exact
original passing receipt; its build, B01, family, runtime, device, state and
checkpoint pins are still validated. The original receipts and families are
never rewritten or relabelled as newly collected data. B03 records the reuse
review in its new receipt. No blanket stale-source bypass exists.

Implementation:
`8b9614d736c727d682e85753d1c851c3663461c733501eefac568624fca0dfe5`.
Exact continuation reviews retain the entire g570 model/Adam/RNG/PPG state and
previously reviewed teacher-adoption routes, without adopting another recipe.
B01/B02/B02.5 remain passed for their original scope and are reusable for this
reviewed diagnostic repair. B03/B04 and production adoption remain Pending.

The [current command](../distribution/rust-lab-g570/README.md) starts at B03,
uses the original B02 receipt under
`artifacts/rust-backend/20261104T011422740838Z-g570-all-8bee9d91/b02/qualification.json`,
then continues to B04. It collects no new B02 games and repeats no B01/B02/B02.5
qualification. Keep that original lab evidence directory and the unchanged g570
run available. Each retry writes a new Git-visible result folder.
