"""g6547: a stopped or finished trainer never lingers holding its GPU.

Ctrl-C reaches the trainer as KeyboardInterrupt, and its context managers stop the lanes, publishers, CPU workers and
pools. But a worker thread or child blocked in a join kept the interpreter's exit waiting, so the process and its CUDA
context (the GPU memory) stayed until a SIGKILL. The launcher made it worse: `subprocess.call` in
continue-rust-g4970.py SIGKILLs its child a quarter second after Ctrl-C, so the wrapper that stops the trainer
(continue-rust-shadow.run_training) died first and the trainer was orphaned.

In the trainer (`install`, `finished`, from scripts/step5.py):
- SIGTERM and SIGHUP stop it like Ctrl-C (a signal the parent set to ignored, e.g. nohup's SIGHUP, stays ignored);
- from the first stop signal, and again when the command returns or raises, a watchdog allows GRACE_SECONDS; past
  that it writes every thread's stack to stderr (the training log shows what blocked), kills the process's own
  children and ends the process with os._exit, which frees the GPU;
- a second stop signal does the same at once.

In a launcher (`call`): the child is waited for, not killed, on the first Ctrl-C (it got the same Ctrl-C and stops its
trainer within run_training's 15 s); a SIGTERM to the launcher alone reaches the child as Ctrl-C; a second stop kills
the child's whole process tree.
"""
import faulthandler
import os
import signal
import subprocess
import sys
import threading
import time

GRACE_SECONDS = 60.
_armed = threading.Event()
_state = dict(stopping=False)


def _kill_children():
    try:
        import psutil
        for child in reversed(psutil.Process().children(recursive=True)):
            try: child.kill()
            except psutil.Error: pass
    except Exception:
        import multiprocessing
        for child in multiprocessing.active_children():
            try: child.kill()
            except Exception: pass


def _force(code, reason):
    try:
        sys.stderr.write(f'\n{reason}; thread stacks follow, then a forced exit (code {code})\n')
        sys.stderr.write('Threads: ' + ', '.join(f'{t.name} (0x{t.ident or 0:016x}{", daemon" if t.daemon else ""})'
                                                 for t in threading.enumerate()) + '\n'); sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush(); sys.stdout.flush()
    except Exception:
        pass
    finally:
        try: _kill_children()
        finally: os._exit(code)


def arm(code, grace=None):
    """Start the watchdog once: if the process still runs `grace` seconds from now, end it with `code`."""
    if _armed.is_set(): return
    _armed.set(); grace = GRACE_SECONDS if grace is None else grace
    def watch():
        time.sleep(grace)
        _force(code, f'The process is still running {grace:.0f} s after it stopped or finished')
    threading.Thread(target=watch, name='exit-watchdog', daemon=True).start()


def _stop(signum, frame):
    code = 128 + signum
    if _state['stopping']: _force(code, f'Second stop signal ({signal.Signals(signum).name})')
    _state['stopping'] = True
    arm(code)
    raise KeyboardInterrupt


def install():
    """Stop signals unwind like Ctrl-C and arm the watchdog (main thread only; ignored signals stay ignored)."""
    if threading.current_thread() is not threading.main_thread(): return
    for name, default in (('SIGINT', signal.default_int_handler), ('SIGTERM', signal.SIG_DFL), ('SIGHUP', signal.SIG_DFL)):
        number = getattr(signal, name, None)
        if number is not None and signal.getsignal(number) is default: signal.signal(number, _stop)


def finished(code):
    """The command returned or raised: the interpreter's own exit gets GRACE_SECONDS too."""
    arm(code)


def guarded(run):
    """Run `run()` with the guard: install, then arm the watchdog with the exit code it ends with."""
    install()
    try:
        result = run()
    except SystemExit as stop:
        finished(stop.code if isinstance(stop.code, int) else 0 if stop.code is None else 1); raise
    except BaseException:
        finished(1); raise
    finished(result if isinstance(result, int) else 0)
    return result


def _kill_tree(process):
    try:
        import psutil
        try: children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess: children = []
        for child in reversed(children):
            try: child.kill()
            except psutil.Error: pass
    except ImportError:
        pass
    if process.poll() is None: process.kill()


def call(argv, **kwargs):
    """subprocess.call for a launcher whose child stops its own trainer: see the module docstring."""
    process = subprocess.Popen(argv, **kwargs)
    terminated = []
    def term(signum, frame):
        terminated.append(signum); raise KeyboardInterrupt
    main = threading.current_thread() is threading.main_thread() and hasattr(signal, 'SIGTERM')
    previous = signal.signal(signal.SIGTERM, term) if main else None
    stops = 0
    try:
        while True:
            try:
                return process.wait()
            except KeyboardInterrupt:
                stops += 1
                if stops == 1:
                    # Ctrl-C reached the child from the terminal; a SIGTERM reached this launcher alone.
                    if terminated and os.name == 'posix' and process.poll() is None: process.send_signal(signal.SIGINT)
                    print('Stopping: waiting for the training to stop its trainer and workers '
                          '(Ctrl-C again to kill them now)', flush=True)
                    continue
                _kill_tree(process)
                process.wait()
                raise
    finally:
        if main: signal.signal(signal.SIGTERM, previous)
