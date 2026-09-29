# R0 reference provenance and notices

The [reference lock](../configs/references/upstream.lock.json) pins the following
repositories and sixteen inspected files, including both root license texts:

| Reference | Revision | Upstream declaration |
| --- | --- | --- |
| [Yuiffy/brass-birmingham-daiyu-ai](https://github.com/Yuiffy/brass-birmingham-daiyu-ai/tree/387840dd7a784f2c05fe561cb381e4f7d7bc590b) | `387840dd7a784f2c05fe561cb381e4f7d7bc590b` | [GNU AGPL-3.0](https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/LICENSE), as identified by the pinned README; this record does not infer an only/or-later variant |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel/tree/48401890ee9857e611678302371378175a8e4c6b) | `48401890ee9857e611678302371378175a8e4c6b` | [Apache License 2.0](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/LICENSE) |

GitHub commit/tree metadata was resolved at those exact revisions, and the
downloaded source bytes were matched to the selected Git blob IDs. SHA-256 and
byte length are also locked. Neither inspected root tree contains a root
`NOTICE` file; this does not assert that every nested dependency is notice-free.

All ten component records currently have `reuse_status: reference_only`.
Upstream source and full license texts are cached only in ignored
`artifacts/r0-references`. No upstream implementation, artwork or model weights
are vendored into project source by R0. The local contract validator and tiny
four-player arithmetic fixtures are original project implementations of the
documented interface and mathematical behavior.

When a later stage actually copies or adapts a component, update its provenance
record with destination files, source revision, modifications and applicable
license/notice handling. Preserve source attribution and applicable license
texts/requirements in any derived distribution. Daiyu's README also distinguishes
its source license from inherited artwork with unverified rights; artwork is
outside the selected adaptation. This record does not assign any license to the
proprietary Steam engine or grant rights to redistribute it.
