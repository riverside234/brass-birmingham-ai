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

### HTTPS token in an ignored dotenv file

For token-based uploads, configure the credentials **on the training server**:

```bash
python scripts/results_git_auth.py configure
```

Enter the GitHub username and paste the token at the hidden prompt. The helper
writes a private `.env.results-git` in the code checkout root (mode 600 on Linux).
It verifies the file is ignored and untracked before saving. Its format is:

```dotenv
RESULTS_GIT_URL=https://YOUR_USERNAME:YOUR_TOKEN@github.com/riverside234/ai-training-results.git
```

This contains a plaintext secret, so do not share the file or paste its contents
into logs. Use a token with Contents read/write permission for the results
repository. Replace/revoke a token that has been exposed. The ignored file is
not downloaded by `git pull`, so configure it once per machine. Re-run configure
to replace it when rotating credentials.

The exporter reads this file for HTTPS clone/fetch/push. A temporary credential
helper returns credentials only for this exact GitHub results repository. The
token is not put in command arguments, process environment, repository URLs,
Git config, exported reports or commits. No global credential settings change.
Existing SSH/credential-helper authentication still works when the file is absent.

If an export already committed locally and only its push failed, retry without
generating another export:

```bash
python scripts/export-training-reports.py --push-only
```

Subsequent exports use the original command automatically; no repeated login or
special token argument is needed.

Only files listed by the export manifest and its incremental dependencies are
staged. Arbitrary files/checkpoints placed beside a bundle are not included.
The code repository's index, commits and remote are untouched. Report bytes are
preserved with `* -text` in the results repository's attributes.

Publication checks every newly copied report and every **actually reused file**
against the new export's recorded hash. It does not recursively validate or
re-upload unrelated files from historical bundles. Existing historical local
edits are left untouched. For an unpublished local cache, its existing manifest
is included as an index alongside the required physical files; missing unrelated
cache files do not become dependencies of the new export.

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

If an older uploader reports `Report changed before Git publication` for an
**unreferenced historical** file, update the code checkout with `git pull --ff-only`
and rerun the normal export command. Keep the previous export directories;
there is no need to delete them or repeat the history migration. A mismatch in
a newly exported or actually reused file still stops publication.

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
