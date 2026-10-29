# B02 search_v1 lab transfer

**B01 qualification: Pending. B02 qualification: Pending.**

Current package: `b02-search-v1-update.zip` (136,800 bytes).

SHA-256: `62a9d022bb3daa41f8ad874dfbbf40cb24e3b526c478d559db9db115e4725664`

See the [bundle guide and lab commands](../RUST_LAB_SEARCH_V1.md). Extract the new B01 transfer first and B02 second, then run:

```bash
conda activate ai-training
bash scripts/run-rust-lab-b01-b02.sh artifacts/step5/YOUR_RUN cuda:0
```

All current input and transfer ZIPs are under `distribution/` and Git-allowed. The input ZIP is also embedded in the B01 transfer. Keep training paused at the same clean committed boundary throughout both gates. The combined runner stops if B01 fails. No lab training or teacher enablement occurs.

The older `b02-lab-update.zip` remains an immutable historical cache-only bundle; do not extract it over the new package. Old B01 receipts cannot pass the refreshed B02 entry gate.
