"""B06 diagnostic measurements. No training source, recipe or admission changes."""
from contextlib import contextmanager
import copy
import importlib
import math
import statistics
import threading
import time

from brass_ai.learning.contracts import digest, require

PSUTIL_VERSION = '7.2.2'
REQUIREMENTS = 'distribution/rust-lab-b06-g1250/requirements.txt'


def measurement_dependency():
    """B06's only extra dependency; do not alter the training environment implicitly."""
    command = 'python -m pip install -r ' + REQUIREMENTS
    try:
        module = importlib.import_module('psutil')
    except (ImportError, OSError) as error:
        raise RuntimeError('B06 process measurements require psutil==' + PSUTIL_VERSION +
                           ' in the same Python environment. Run: ' + command) from error
    require(module.__version__ == PSUTIL_VERSION,
            f'B06 requires psutil=={PSUTIL_VERSION}; found {module.__version__}. Run: ' + command)
    return module


def measurement_preflight(device='cpu'):
    """Exercise the process sampler and, in each benchmark child, its CUDA allocator."""
    import torch
    module = measurement_dependency()
    with Meter(device) as meter:
        if device.startswith('cuda'):
            # No RNG/model work: prove allocation, execution and peak telemetry on
            # the selected logical device before any collection or update starts.
            with meter.phase('cuda_preflight_allocation'):
                probe = torch.zeros(256, dtype=torch.uint8, device=device)
            require(probe.device == torch.device(device), 'B06 CUDA probe used a different device')
        report = meter.report()
        if device.startswith('cuda'):
            require(report['device_memory']['cuda_peak_allocated_bytes'] >= probe.numel()*probe.element_size(),
                    'B06 CUDA peak measurement unavailable')
            del probe
    require(report['peak_process_tree_rss_bytes'] > 0, 'B06 process memory measurement unavailable')
    return dict(psutil=module.__version__, requirements=REQUIREMENTS, device=device,
                torch=str(torch.__version__),cuda_build=torch.version.cuda,sampler=report)


def distribution(values):
    values = sorted(float(v) for v in values)
    require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid benchmark measurement')
    return dict(samples=len(values), median=statistics.median(values) if values else None,
                p90=values[max(0, math.ceil(.9*len(values))-1)] if values else None,
                maximum=max(values) if values else None)


def choose_jobs(jobs):
    """First job in each actual (table kind, V4 presence) stratum, at most three.

    The g1250 planner can embed V4 seats in historical tables. Never demand an
    obsolete standalone legacy stratum or redraw jobs to manufacture a branch.
    """
    from brass_ai.rust_backend.lab_collection import route
    selected = []; seen = set()
    for job in jobs:
        key = (job['provenance']['kind'], any(p['kind']!='neural' for p in job['policies']))
        if key not in seen:
            selected.append(copy.deepcopy(job)); seen.add(key)
    require(0 < len(selected) <= 3, 'B06 frozen plan has more than three strata; revise bounded scope explicitly')
    require(any(route(j)[0] == 'rust' for j in selected), 'Frozen batch has no native-supported work')
    return selected


def work(games):
    members = [m for g in games for m in [g] + g.get('branches', [])]
    return dict(families=len(games), branches=len(members)-len(games),
                decisions=sum(len(g['records']) for g in members),
                forced_rows=sum(len(r['choices']) == 1 for g in members for r in g['records']))


def compare_games(left, right):
    """Exact trajectory and targets; bounded float32 inference batch variation.

    GPU queue schedules change batch shapes, and float32 reductions may change
    their last bits. This is separate from replay admission, which reconstructs
    each behavior denominator from its OWN recorded logits/probabilities and RNG.
    """
    import numpy as np
    from brass_ai.recovery.local_scoring_targets import FIELD
    require(work(left) == work(right), 'Matched accepted work differs')
    errors=[]
    for a, b in zip(left, right, strict=True):
        am, bm = [a]+a.get('branches', []), [b]+b.get('branches', [])
        for x, y in zip(am, bm, strict=True):
            ignored = {'records', 'branches', 'branch', FIELD}
            require({k:v for k,v in x.items() if k not in ignored} ==
                    {k:v for k,v in y.items() if k not in ignored}, 'Matched game identity/result differs')
            if 'branch' in x:
                # These are durations only, never selection inputs/likelihoods.
                floats = {'raw_probabilities','probability','marginal_probability','component_probability'}
                omit = {'elapsed_seconds', 'fraction_remaining'} | floats
                np.testing.assert_equal({k:v for k,v in x['branch'].items() if k not in omit},
                                        {k:v for k,v in y['branch'].items() if k not in omit})
                for key in floats:
                    if key in x['branch'] or key in y['branch']:
                        np.testing.assert_allclose(x['branch'][key],y['branch'][key],rtol=1e-5,atol=1e-7,
                            err_msg='Matched GPU-derived branch probability: '+key)
            for ra, rb in zip(x['records'], y['records'], strict=True):
                require({k:v for k,v in ra.items() if k not in ('seconds','behavior')} ==
                        {k:v for k,v in rb.items() if k not in ('seconds','behavior')},
                        f'Matched observation/legal order/action differs: game={x["game_id"]}, step={ra["step"]}, '
                        f'choices={ra["chosen"]}/{rb["chosen"]}, views={digest(ra["view"])}/{digest(rb["view"])}')
                require({k:v for k,v in ra['behavior'].items() if k != 'probability'} ==
                        {k:v for k,v in rb['behavior'].items() if k != 'probability'}, 'Matched behavior identity differs')
                pa,pb=ra['behavior']['probability'],rb['behavior']['probability']
                if len(ra['choices'])==1 or ra['behavior']['kind']=='external':
                    require(pa==pb==1.,'Matched deterministic probability differs')
                else:
                    # Same scale as the existing float32 head/batching checks.
                    # Never applies to actions, raw features, IDs or replay RNG.
                    np.testing.assert_allclose(pa,pb,rtol=1e-5,atol=1e-7,
                        err_msg=f'game={x["game_id"]}, step={ra["step"]}')
                    errors.append((abs(pa-pb),abs(pa-pb)/max(abs(pa),abs(pb),1e-30)))
    return dict(equal=True, work=work(left),probability_comparison=dict(rtol=1e-5,atol=1e-7,
        rows=len(errors),max_absolute=max((a for a,b in errors),default=0.),
        max_relative=max((b for a,b in errors),default=0.),exact_discrete_trajectory=True))


def compare_rows(left, right):
    import numpy as np
    from brass_ai.recovery.local_scoring_targets import map_for
    require(len(left) == len(right), 'Matched prepared rows differ')
    for (a,ta,ia,ea),(b,tb,ib,eb) in zip(left,right,strict=True):
        require(ta == tb and ia == ib, 'Matched targets/trajectory attribution differ')
        for name in ('state','actions','stable_to_relative'):
            np.testing.assert_array_equal(getattr(ea,name),getattr(eb,name))
        np.testing.assert_equal({k:v for k,v in a.items() if k.startswith('_') and k != '_local_scoring'},
                                {k:v for k,v in b.items() if k.startswith('_') and k != '_local_scoring'})
        if '_local_scoring' in a or '_local_scoring' in b:
            require(map_for(a) == map_for(b), 'Matched local-scoring targets differ')
    return len(left)


def performance(reference, candidate, equal):
    require(equal and reference['work'] == candidate['work'], 'Cannot compare unequal accepted work')
    a,b = reference['collection_seconds'],candidate['collection_seconds']
    accepted_a,accepted_b=reference['accepted_work_seconds'],candidate['accepted_work_seconds']
    require(a > 0 and b > 0 and accepted_a>=a and accepted_b>=b, 'Missing collection/admission/preparation timing')
    return dict(collection_ratio_reference_over_rust=a/b, collection_seconds_saved=a-b,
                useful_collection_gain=b < a,accepted_work_ratio_reference_over_rust=accepted_a/accepted_b,
                accepted_work_seconds_saved=accepted_a-accepted_b,useful_accepted_work_gain=accepted_b<accepted_a,
                statistical_claim=False, whole_iteration_speedup=None,
                scope='One bounded frozen batch per arm; startup, admission and preparation reported separately')


class Meter:
    """Synchronized phase wall time, sampled process-tree RSS/CPU, and CUDA peaks.

    Phase durations may nest. Process-tree CPU includes children observed during
    sampling; very short-lived children may be missed, so it is a lower bound.
    """
    def __init__(self, device='cpu'):
        self.device=device; self.phases={}; self.stop=threading.Event(); self.peak_rss=0; self.cpu={}
        self.lock=threading.RLock(); self.error=None; self.samples=0; self.exited_processes=0
        self.cuda_setup=None

    @contextmanager
    def phase(self, name):
        import torch
        if self.device.startswith('cuda'): torch.cuda.synchronize(self.device)
        began=time.perf_counter()
        try: yield
        finally:
            if self.device.startswith('cuda'): torch.cuda.synchronize(self.device)
            with self.lock:self.phases.setdefault(name,[]).append(time.perf_counter()-began)

    def __enter__(self):
        psutil=measurement_dependency()
        import torch
        if self.device.startswith('cuda'):
            initialized=torch.cuda.is_initialized();began=time.perf_counter()
            try:
                # reset_peak_memory_stats does NOT lazy-initialize the allocator.
                # synchronize does, and also drains earlier work before the reset.
                # Each spawned benchmark process needs its own initialization.
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception as error:
                raise RuntimeError(f'B06 CUDA measurement setup failed on {self.device}: {error}') from error
            self.cuda_setup=dict(device=self.device,initialized_before=initialized,
                                 seconds=time.perf_counter()-began)
        process=psutil.Process()
        def sample():
            resident=0
            values={};exited=0
            for p in [process]+process.children(recursive=True):
                try:
                    key=(p.pid,p.create_time());t=p.cpu_times();value=t.user+t.system
                    resident+=p.memory_info().rss;values[key]=value
                except psutil.NoSuchProcess:
                    # A child may exit between enumeration and reading its counters.
                    if p.pid==process.pid:raise
                    exited+=1
            with self.lock:
                for key,value in values.items():
                    # Baseline all already-running children as well as this process.
                    if key not in self.cpu:self.cpu[key]=[value if self.samples==0 else 0.,value]
                    self.cpu[key][1]=value
                self.peak_rss=max(resident,self.peak_rss);self.samples+=1;self.exited_processes+=exited
        sample()
        def loop():
            try:
                while not self.stop.wait(.05):sample()
            except Exception as error:
                with self.lock:self.error=error
                self.stop.set()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
        return self

    def __exit__(self,kind,error,tb):
        self.stop.set();self.thread.join(timeout=2)
        try:
            require(not self.thread.is_alive(),'B06 process sampler did not stop')
            self.check()
        except Exception as sampling_error:
            if error is None:raise
            error.add_note('B06 sampler also failed: '+str(sampling_error))

    def check(self):
        if self.error is not None:
            raise RuntimeError('B06 process sampler failed; measurements are incomplete: '+str(self.error)) from self.error

    def report(self):
        from brass_ai.recovery.telemetry import memory
        with self.lock:
            self.check()
            result=dict(phases={k:dict(total_seconds=sum(v),**distribution(v)) for k,v in self.phases.items()},
                        peak_process_tree_rss_bytes=self.peak_rss,
                        sampled_process_tree_cpu_seconds=sum(max(0,b-a) for a,b in self.cpu.values()),
                        process_samples=self.samples,exited_process_reads=self.exited_processes,
                        cuda_setup=self.cuda_setup,
                        cpu_scope='Sampled lower bound, 50ms; initial processes baselined; nested phases overlap')
        result['device_memory']=memory(self.device)
        return result
