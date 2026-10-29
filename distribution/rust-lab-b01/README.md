# B01 search_v1 lab transfer

**B01 qualification: Pending. B02 qualification: Pending.**

Current package: `b01-search-v1-update.zip` (808,469 bytes).

SHA-256: `ae634e746dd1864bee281b36f2415112510aeb66a8c5de7faf8d7e4c75dac685`

See the [bundle guide and lab commands](../RUST_LAB_SEARCH_V1.md). Extract the new B01 transfer first and B02 second, then run:

```bash
conda activate ai-training
bash scripts/run-rust-lab-b01-b02.sh artifacts/step5/YOUR_RUN cuda:0
```

All current input and transfer ZIPs are under `distribution/` and Git-allowed. The input ZIP is also embedded in the B01 transfer. Keep training paused at the same clean committed boundary throughout both gates. The combined runner stops if B01 fails. No lab training or teacher enablement occurs.

The older `b01-lab-update.zip` remains an immutable historical cache-only bundle; do not extract it over the new package. Old B01 receipts cannot pass the refreshed B02 entry gate.
