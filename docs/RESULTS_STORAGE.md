# Separate training results repository

Code lives in `riverside234/brass-birmingham-ai`. Analysis reports and their
history live in [riverside234/ai-training-results](https://github.com/riverside234/ai-training-results).
The local `results/` directory is an independent Git repository, **not a
submodule**. The code repository ignores it and records no results-version
pointer. Ordinary code clones do not download the results repository.

## Export and upload

The existing command now exports, commits and pushes the report bundle:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --continue
```

On first use, it downloads the independent results repository, preserving
existing local files and restoring missing earlier bundles before incremental
export. Configure Git authentication and your commit name/email on that machine.
For SSH authentication, append
`--results-remote git@github.com:riverside234/ai-training-results.git`.

Only files listed by the export manifest and its incremental dependencies are
staged. Arbitrary files/checkpoints placed beside a bundle are not included.
The code repository's index, commits and remote are untouched. Report bytes are
preserved with `* -text` in the results repository's attributes.

For local-only export, without Git or network access:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --continue --no-push
```

An upload failure leaves the exported files and any completed local commit in
place. Fix authentication/connectivity and rerun the normal command; incremental
export reuses existing reports and pushes retained commits. Publication never
force-pushes or automatically resolves divergent history. Existing staged result
changes are rejected. If a process was forcibly killed, first verify it is no
longer running before removing its `results/.git/report-export.lock`.

## Existing checkouts after the one-time history rewrite

Results were removed from **all code commits**, and their history was extracted
into the results repository. Old code commit IDs consequently change. Do not
merge or push the old code history back into the rewritten repository.

For an existing Linux training checkout, stop training and back up any local
code changes first. The following restores tracked code to the new remote
version while retaining the local `artifacts/` and `results/` directories:

```bash
git bundle create ../code-before-results-migration.bundle --all
git fetch origin
git reset --mixed origin/main
git restore --worktree -- .
```

The final command restores tracked code files, so save local code edits before
running it. `reset --mixed` does not delete working files. Then use the export
command above; it attaches the existing results directory to its new repository.
Model weights, optimizer state, replay caches and training progress under
`artifacts/` are not relocated or reset by this storage migration.

Some archived regression fixtures live only in the results archive. Pure export
and Git-publication tests use small authored data and temporary local remotes;
they do not require downloading production results or contacting GitHub.
