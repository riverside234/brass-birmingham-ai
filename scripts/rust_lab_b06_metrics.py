"""B06 diagnostic measurements. No training source, recipe or admission changes."""
from contextlib import contextmanager
import copy
import math
import statistics
import threading
import time

from brass_ai.learning.contracts import digest, require


def distribution(values):
    values = sorted(float(v) for v in values)
    require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid benchmark measurement')
    return dict(samples=len(values), median=statistics.median(values) if values else None,
                p90=values[max(0, math.ceil(.9*len(values))-1)] if values else None,
                maximum=max(values) if values else None)


def choose_jobs(jobs):
    """First job in each actual (table kind, route) stratum, at most three.

    The g1250 planner can embed V4 seats in historical tables. Never demand an
    obsolete standalone legacy stratum or redraw jobs to manufacture a branch.
    """
    from brass_ai.rust_backend.lab_collection import route
    selected = []; seen = set()
    for job in jobs:
        key = (job['provenance']['kind'], route(job)[0])
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
    """Exact discrete trajectory and targets; only floating probabilities tolerate ULP drift."""
    import numpy as np
    from brass_ai.recovery.local_scoring_targets import FIELD
    require(work(left) == work(right), 'Matched accepted work differs')
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
                        np.testing.assert_allclose(x['branch'][key],y['branch'][key],rtol=1e-14,atol=0)
            for ra, rb in zip(x['records'], y['records'], strict=True):
                require({k:v for k,v in ra.items() if k not in ('seconds','behavior')} ==
                        {k:v for k,v in rb.items() if k not in ('seconds','behavior')},
                        f'Matched observation/legal order/action differs: game={x["game_id"]}, step={ra["step"]}, '
                        f'choices={ra["chosen"]}/{rb["chosen"]}, views={digest(ra["view"])}/{digest(rb["view"])}')
                require({k:v for k,v in ra['behavior'].items() if k != 'probability'} ==
                        {k:v for k,v in rb['behavior'].items() if k != 'probability'}, 'Matched behavior identity differs')
                np.testing.assert_allclose(ra['behavior']['probability'], rb['behavior']['probability'], rtol=3e-6, atol=1e-8)
    return dict(equal=True, work=work(left))


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

    @contextmanager
    def phase(self, name):
        import torch
        if self.device.startswith('cuda'): torch.cuda.synchronize(self.device)
        began=time.perf_counter()
        try: yield
        finally:
            if self.device.startswith('cuda'): torch.cuda.synchronize(self.device)
            self.phases.setdefault(name,[]).append(time.perf_counter()-began)

    def __enter__(self):
        import psutil
        import torch
        process=psutil.Process()
        def sample():
            resident=0
            for p in [process]+process.children(recursive=True):
                try:
                    key=(p.pid,p.create_time());t=p.cpu_times();value=t.user+t.system
                    if key not in self.cpu:self.cpu[key]=[value if p.pid==process.pid else 0.,value]
                    self.cpu[key][1]=value;resident+=p.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
            self.peak_rss=max(resident,self.peak_rss)
        sample()
        def loop():
            while not self.stop.wait(.05):sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
        if self.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self,*_):
        self.stop.set();self.thread.join(timeout=2)

    def report(self):
        from brass_ai.recovery.telemetry import memory
        return dict(phases={k:dict(total_seconds=sum(v),**distribution(v)) for k,v in self.phases.items()},
                    peak_process_tree_rss_bytes=self.peak_rss,
                    sampled_process_tree_cpu_seconds=sum(max(0,b-a) for a,b in self.cpu.values()),
                    cpu_scope='Sampled lower bound, 50ms; nested phases overlap',device_memory=memory(self.device))
