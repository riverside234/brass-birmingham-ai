# R2 source provenance

This learning component uses the architecture and workflow in
[Yuiffy/brass-birmingham-daiyu-ai](https://github.com/Yuiffy/brass-birmingham-daiyu-ai/tree/387840dd7a784f2c05fe561cb381e4f7d7bc590b)
at commit `387840dd7a784f2c05fe561cb381e4f7d7bc590b` as its starting reference.

- `model.py` adapts the separate state encoder, candidate encoder and shared
  scorer pattern from upstream `training/model.py`. Dense semantic candidate
  features replace `EmbeddingBag`. The scalar heads are replaced with four-seat
  win-share, placement and VP outputs, with explicit stable-seat mapping.
- `batching.py` reimplements the request/evaluate/submit boundary described by
  upstream `training/neural_search.py`. It uses this project's permitted R1
  views, request identities, cancellation and absolute deadlines. No Rust search
  session is copied or invoked.
- `contracts.py`, `checkpoint.py` and `replay.py` use the compatibility approach
  of upstream `training/schema.py`, with different schemas, hashes and targets.
- Collection/training follows the staged-data idea in upstream
  `docs/training-methodology.md`, without its strategy-based rejection filters.

The upstream declaration is GNU AGPL v3. The full, pinned license text is included
in [LICENSE](LICENSE). The R2 learning component in this directory and its R2
Python entry points/tests are provided under AGPL-3.0-only, with these notices
and corresponding source. Project adaptations were made on 27 September 2026.
No upstream pretrained weights, artwork, Rust engine or web application are
included. Preserve these notices and the applicable source/license requirements
when distributing this derived component.

OpenSpiel remains a separately pinned conceptual/semantic reference. This
component does **not** copy its two-player AlphaZero trainer or import OpenSpiel.
Its four-seat accounting agrees with R0's independent multiplayer fixtures.

This notice does not license the proprietary Steam game, its assemblies or
assets. Those remain external dependencies and are excluded from the repository.
R0's immutable provenance/contract files describe the earlier reference-only
stage; [r2-components.json](../../../configs/references/r2-components.json)
records this later adaptation without rewriting R0's historical manifest.
