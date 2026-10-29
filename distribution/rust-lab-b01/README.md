# B01 lab qualification bundle

This directory is explicitly allowed by `.gitignore` so the B01 transfer ZIP can
be committed and pushed with the project. The generated qualification results
and training runs remain under the ignored `artifacts/` directory.

`b01-lab-update.zip` is 1,195,952 bytes (1.14 MiB). Its SHA-256 is:

```text
a95b881a7110cd89f71b91a864035403c9da3b7d8823c83dbb9cf7c07c5b7d22
```

The archive is unchanged from the prepared B01 transfer package. It includes
the qualification tooling and the pinned input ZIP, but no model weights or
proprietary game assemblies. Moving the outer archive does not change its
internal paths or the runner's input-package hash.

After the bundle and B01 implementation have been committed and pushed, run
these commands from the updated lab checkout. Replace `YOUR_RUN` with the actual
training directory containing `state.json`. Pause training at a committed
iteration before qualification. Rust 1.99.1 and the existing CUDA-enabled
`ai-training` environment are required.

```bash
git pull
conda activate ai-training
unzip -o distribution/rust-lab-b01/b01-lab-update.zip
bash scripts/run-rust-lab-b01.sh artifacts/step5/YOUR_RUN cuda:0
```

The extraction restores the immutable input bundle to
`artifacts/rust-backend/20261028-b01/lab/b01-inputs.zip`, where the existing runner
expects it. New qualification evidence is written to a fresh directory under
`artifacts/rust-backend/`. This package does not itself certify the Linux/CUDA
qualification gate or adopt the Rust backend for training.
