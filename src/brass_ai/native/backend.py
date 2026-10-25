"""Explicit selection, with no environment-driven or active-run switching."""
from pathlib import Path

from .protocol import NativeWorker


def select_backend(*, backend="reference", root=None, **kwargs):
    if backend == "reference":
        from brass_ai.searchlab.spiel import BrassSpielGame
        kwargs.setdefault("fast", True)
        return BrassSpielGame(root=root, **kwargs)
    if backend == "rust-diagnostic":
        if root is not None:
            kwargs.setdefault("executable", Path(root) / "artifacts/rust-native-target" /
                              "x86_64-pc-windows-gnu/release/brass-ai-worker.exe")
        return NativeWorker(**kwargs)
    if backend == "rust-simulator":
        from .simulation import NativeSpielGame
        return NativeSpielGame(root=root, **kwargs)
    raise ValueError(f"Backend {backend!r} is unavailable; use reference, rust-simulator or rust-diagnostic")
