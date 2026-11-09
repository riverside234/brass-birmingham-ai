"""Selected CPU binaries, ONNX Runtime pin and worker placement.

Nothing here changes a frozen job, checkpoint, reward or optimizer. Absent or
unselected, every lane keeps today's GPU path and its original binaries.
"""
import copy
import os
from pathlib import Path

from brass_ai.learning.contracts import require
from .lab_contract import pin

CONTRACT = 'brass-lab-cpu-inference/1.0.0'
BUILD = 'brass-lab-cpu-build/1.0.0'
SELECTOR = 'artifacts/rust-backend/cpu-inference/build.json'


def selector(root):
    return Path(root)/SELECTOR


def resolve(root, *, teacher_core=None):
    """Load the prepared build and re-pin every file it depends on."""
    from .lab_backend import checked_pin
    from .lab_cpu_overlay import fingerprint
    path = selector(root)
    require(path.is_file(), 'Prepare CPU inference first: python scripts/prepare-rust-cpu-inference.py')
    receipt = pin(path)
    value = checked_pin(receipt)
    require(value['contract'] == BUILD and value['inference'] == CONTRACT and value['overlay_source'] == fingerprint(root),
            'CPU inference build/source differs; rerun scripts/prepare-rust-cpu-inference.py')
    for key in ('runtime', 'collector', 'teacher', 'tests', 'build_log'):
        require(pin(value[key]['path']) == value[key], 'CPU inference runtime changed: '+key)
    require(teacher_core is None or value['teacher_core'] == teacher_core,
            'CPU teacher was built for a different search core; rerun scripts/prepare-rust-cpu-inference.py')
    return dict(value, receipt=receipt)


def ordinary_build(root, gpu):
    """Same producer shape as the GPU build, with its own runtime identity."""
    value = resolve(root)
    previous = gpu.get('_previous_build_sha256s', ())
    allowed = tuple(dict.fromkeys((gpu['_receipt_sha256'], *previous)))
    result = dict(copy.deepcopy({k: v for k, v in gpu.items() if not k.startswith('_')}),
                  collector=value['collector'], _receipt_sha256=value['receipt']['sha256'])
    result['_previous_build_sha256s'] = allowed
    return result, value


def accept_cached_cpu_work(root, build):
    """A GPU rollback may reuse families published by the CPU collector.

    Each cached family is still fully re-validated by a publisher before use.
    Without a prepared CPU build the GPU path is returned untouched.
    """
    path = selector(root)
    if not path.is_file():
        return build
    previous = build.get('_previous_build_sha256s',
                         (build['_previous_build_sha256'],) if '_previous_build_sha256' in build else ())
    return dict(build, _previous_build_sha256s=tuple(dict.fromkeys((*previous, pin(path)['sha256']))))


def core_allocation(count):
    """Spread workers evenly across sockets; empty where affinity is unavailable."""
    if not hasattr(os, 'sched_getaffinity'):
        return []
    cpus = sorted(os.sched_getaffinity(0))
    sockets = {}
    for cpu in cpus:
        try:
            value = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text().strip()
        except OSError:
            value = '0'
        sockets.setdefault(value, []).append(cpu)
    ordered = []
    pools = [sockets[k] for k in sorted(sockets)]
    while any(pools):
        for pool in pools:
            if pool:
                ordered.append(pool.pop(0))
    return [ordered[i % len(ordered)] for i in range(count)]


def raise_open_file_limit(target=65536):
    """64 lanes need more pipes than a default soft limit comfortably allows."""
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if soft != resource.RLIM_INFINITY and soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


def pin_process(cpu):
    """Children inherit this, so a lane and its Rust worker share one core."""
    if cpu is not None and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {cpu})


def pin_child(pid, cpu):
    if cpu is not None and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(pid, {cpu})


def worker_settings(root, block, service, identities, value):
    """Runtime pin plus bundles for the models one collection will use."""
    from .lab_cpu_models import ensure
    bundles, exported = ensure(root, service.models, identities)
    return dict(contract=CONTRACT, runtime=dict(path=value['runtime']['path'], sha256=value['runtime']['sha256']),
                bundles=bundles, exported_bundles=exported, model_cache=block['model_cache'], pinning=block['pinning'])


def ordinary_plan(root, gpu_build, service, jobs, block):
    identities = [p['model_id'] for job in jobs for p in job['policies'] if p['kind'] == 'neural']
    build, value = ordinary_build(root, gpu_build)
    return build, worker_settings(root, block, service, identities, value)


def teacher_plan(root, build, service, jobs, block):
    value = resolve(root, teacher_core=build['core_source'])
    settings = worker_settings(root, block, service, [jobs[0]['expected']['model_id']], value)
    bundle = settings['bundles'][jobs[0]['expected']['model_id']]
    inference = dict(contract=CONTRACT, runtime=settings['runtime'], bundle=bundle, budget=copy.deepcopy(block['teacher_budget']))
    return dict(binary=value['teacher']['path'], inference=inference, receipt=value['receipt'], exported_bundles=settings['exported_bundles'],
                pinning=block['pinning'])


def rating_settings(root, block, service, identities):
    from .lab_cpu_models import ensure
    bundles, exported = ensure(root, service.models, identities)
    return dict(contract=CONTRACT, bundles=bundles, exported_bundles=exported)
