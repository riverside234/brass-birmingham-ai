"""Opt-in native diagnostics; the existing simulator remains the default."""
from .backend import select_backend
from .protocol import NativeError, NativeWorker, WorkerFailure

__all__ = ["NativeError", "NativeWorker", "WorkerFailure", "select_backend"]
