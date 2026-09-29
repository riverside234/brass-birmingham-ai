"""Bounded spawned processes: one live game loop/search context per process."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing


def run_jobs(function, jobs, workers, initializer, initargs):
    """Yield compact results, keeping at most `workers` tasks in flight.

    Always spawn, including on Linux/CUDA: never fork an initialized CUDA runtime.
    On failure, stop scheduling work; active jobs finish and pending jobs cancel.
    Large complete trajectories stay in worker-written shards, outside IPC.
    """
    if workers == 1:
        initializer(*initargs)
        for job in jobs:
            yield function(job)
        return
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=initializer, initargs=initargs)
    pending, remaining = set(), iter(jobs)
    try:
        for _ in range(workers):
            job = next(remaining, None)
            if job is None:
                break
            pending.add(executor.submit(function, job))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Resolve all finished futures before queuing more after a failure.
            results = [future.result() for future in finished]
            for result in results:
                yield result
                job = next(remaining, None)
                if job is not None:
                    pending.add(executor.submit(function, job))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
