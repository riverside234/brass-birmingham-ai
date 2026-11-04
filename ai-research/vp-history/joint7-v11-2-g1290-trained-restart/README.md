# Ordinary self-play VP backfill

`vp-history.json` is a reporting-only cumulative cache for the neutral trained
g1290 restart, through committed generation 611. It contains exact ordinary
self-play learner VP sums and seat counts from the matching g200, g253, g370,
g570 and g611 export bundles. Their initial checkpoint identities agree and
every overlapping observation is checked for equality. Historical/v4 tables,
branch continuations and evaluation games are excluded.

Coverage: **g71–370 and g441–611**, 471 iterations. The available exports lack
**g1–70 and g371–440**; these gaps remain missing. Generation 611 remains in the
cache for the next ten-iteration group. The g610 chart has 47 complete groups,
covering 470 iterations and 301,760 learner-seat games. The g601–610 pooled mean
is 133.67394578313252 VP. `backfill-receipt.json` records source file hashes and
the prior derived cache/chart/export-manifest hashes.

Sync this folder and the helper to the lab. With training stopped at a committed
boundary, run from the repository root:

```bash
python scripts/backfill-vp-history.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --cache ai-research/vp-history/joint7-v11-2-g1290-trained-restart/vp-history.json
```

This merges into `ratings/vp-history.json` inside the actual run, preserving
newer observations and rejecting conflicting totals or a different initial
checkpoint. Repeating the import is idempotent. It does not edit state,
checkpoints or training configuration. The existing rating plot generator then
reads this cache and continues appending future committed data automatically.

The supplied g611 export's g610 `progress_vp.svg` and cumulative cache have also
been regenerated, with the two derived-file checksums and derivation recorded
in its `export-manifest.json`. Other report evidence remains unchanged.

Validation used the real merged observations and an isolated newer-run fixture:
merge, preservation of newer data, idempotence, conflict rejection, lineage
rejection and updated export checksums passed. The revised SVG was visually
checked. No training or evaluation games were run.
