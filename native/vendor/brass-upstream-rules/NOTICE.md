# Imported upstream rules

Source: https://github.com/riverside234/brass-birmingham-daiyu-ai.git
Commit: `f8f0246718d6d3fb3f301742d4154811c90d5813`.
License: the committed GNU AGPL version 3 text is preserved in `LICENSE`.

The forty modules listed in `UPSTREAM.json` were copied byte-for-byte from
the L01 archive, including the recorded local changes. The original local
patch is retained as `upstream-local.patch`; parts concern modules excluded
from this extraction. The working checkout's deleted LICENSE was not copied
as a deletion.

Local changes on 2026-10-24: new Cargo manifest, library module selection and
game module selection. Web, Python bindings, database, UI assets, upstream
PUCT, trained AI and self-play code are excluded. The three required upstream
library dependencies retain their frozen versions. Rustfmt/Clippy of the new
workspace do not rewrite these historical source bytes. Upstream unit tests
are run separately and their results are recorded alongside wrapper tests.

This import establishes a reusable native dependency. It does not assert
equivalence to the C# rules, feature encoder or Masked ISMCTS. Those remain
the subsequent migration gates. No proprietary Steam assembly is included.
