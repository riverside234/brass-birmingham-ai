# Local save integrity repair

This repair covers a confirmed P1 follow-up beyond the original 16 findings. The stock assembly can truncate a previously valid history when serialization or writing fails, append the same revisions again after metadata failure, and recursively delete a game's history when metadata is empty. Its production filesystem cache also publishes failed appends before persistence succeeds. The retained stock proof uses only disposable files under the test directory.

## Commit and recovery

- Existing `events.storage` and `metadata.storage` JSON schemas and locations remain unchanged.
- Both next payloads are serialized completely in memory before a live file changes.
- A `brasspatch-save-v1.pending` binary redo journal records optional event/metadata payloads, lengths, version magic, and SHA-256 checksums. It contains no configurable output paths.
- Each file is staged in the same directory, flushed, then replaced atomically. Previous live files are retained as `.brasspatch-backup` files. Already matching live files are skipped during recovery, retaining their previous backup.
- A read or retry completes an intact pending journal before returning data. A corrupted journal fails without rewriting the live files. The journal is removed only after every replacement succeeds.
- Per-directory locks cover reads, commits, and the complete asynchronous player-metadata read/update/write. A filesystem lock also rejects simultaneous access from another process rather than interleaving writes.
- Append retries compare revision and serialized content. Exact overlaps are skipped, while conflicts, gaps, or stale batches are rejected before publishing a journal. Existing corrupt histories are preserved for recovery rather than silently renumbered.
- Offline repository creation uses the direct filesystem backends instead of the shipped cache that writes in the wrong order. This retains the same serializers, paths and snapshot support; small local histories are read from disk.

This is process-interruption recovery using atomic replacement and flushed staged files. It is not a guarantee against hardware failure, arbitrary filesystem corruption, or power loss on a filesystem that does not honor those operations. Snapshot writing itself is unchanged because no normal snapshot producer was established in the reviewed game path.

## Compatibility and rollback

Completed saves retain stock-readable JSON, subject to the separate patch's versioned game identifiers. A pending journal requires the patched reader to finish recovery before removing the patch. Do not delete a pending journal to suppress an error. Empty/corrupt legacy metadata is reported without deleting the containing directory; automatic restoration from a backup is deliberately avoided because backup revisions may differ.

## Verification

Run `tests/BrassPatch/Persistence/run.ps1 -Stock` in its own Windows PowerShell process to reproduce the installed defects against the fingerprinted original DLL. Run the same script without `-Stock` against the candidate to exercise serialization failures, blocked first/second replacements, fresh-instance recovery, repeated retries, conflicting revisions, corrupted journals, actual repository roundtrips, an asynchronous metadata race, and the production offline factory. Tests create no real user saves.
