"""Bounded phase telemetry and optional short CPU/CUDA profiler windows."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import os
import platform
import time
import math

import torch


class Timings:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.samples = defaultdict(list)
        self.maximum = defaultdict(float)

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        with torch.profiler.record_function(name):
            try: yield
            finally: self.observe(name,time.perf_counter()-start)

    def observe(self,name,elapsed):
        self.totals[name] += elapsed
        self.counts[name] += 1
        self.maximum[name]=max(self.maximum[name],elapsed)
        # A bounded deterministic sample; totals/counts always include all calls.
        if len(self.samples[name]) < 1024:
            self.samples[name].append(elapsed)

    def report(self):
        def percentile(values, fraction):
            return sorted(values)[min(len(values)-1, max(0,math.ceil(len(values)*fraction)-1))]
        return {name: dict(seconds=seconds, calls=self.counts[name],
                mean_seconds=seconds/self.counts[name],
                p50_seconds=percentile(self.samples[name], .5),
                p95_seconds=percentile(self.samples[name], .95),
                max_seconds=self.maximum[name],first_seconds=self.samples[name][0],
                steady_mean_seconds=(seconds-self.samples[name][0])/max(1,self.counts[name]-1),
                percentile_scope='first bounded 1024 calls; totals and maximum include all calls',
                percentile_sample_count=len(self.samples[name]))
                for name, seconds in sorted(self.totals.items())}


class HostMetrics:
    """Timing-only wrapper; the admitted host's RPC implementation is unchanged."""
    def __init__(self,timings=None): self.timings=timings or Timings()
    def report(self): return self.timings.report()


class _MeasuredStream:
    def __init__(self,stream,metrics,label): self.stream,self.metrics,self.label=stream,metrics,label
    def __getattr__(self,name): return getattr(self.stream,name)
    def write(self,value):
        with self.metrics.timings.phase(self.label+'.write'): return self.stream.write(value)
    def flush(self):
        with self.metrics.timings.phase(self.label+'.flush'): return self.stream.flush()
    def readline(self):
        with self.metrics.timings.phase(self.label+'.read_wait'): return self.stream.readline()


def instrument_host(host,timings=None):
    if hasattr(host,'recovery_metrics'):
        if timings is not None: host.recovery_metrics.timings=timings
        return host
    metrics=HostMetrics(timings);original=host.call
    def call(op,**arguments):
        with metrics.timings.phase('host.rpc.'+op): result=original(op,**arguments)
        native=getattr(host,'last_native_timing',None)
        if native:
            for key in ('elapsed_seconds','cpu_seconds','materialize_serialize_seconds'):
                metrics.timings.observe('host.native.'+key,float(native[key]))
        return result
    host.call=call;host.recovery_metrics=metrics
    host.process.stdin=_MeasuredStream(host.process.stdin,metrics,'host.ipc.stdin')
    host.process.stdout=_MeasuredStream(host.process.stdout,metrics,'host.ipc.stdout')
    return host


def environment(device):
    return dict(platform=platform.platform(), pid=os.getpid(), cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        tensor_threads=torch.get_num_threads(),
        omp_threads=os.environ.get('OMP_NUM_THREADS'), mkl_threads=os.environ.get('MKL_NUM_THREADS'))


def memory(device):
    result={}
    try:
        import psutil
        result['host_resident_bytes']=psutil.Process().memory_info().rss
    except ImportError: result['host_resident_bytes']=None
    if str(device).startswith('cuda'):
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def iteration_utilization(before,after,records,seconds,games,config):
    """Per-collection deltas; elapsed RPC/wait time is not CPU or GPU busy time."""
    batches=after.get('batches',0)-before.get('batches',0)
    positions=after.get('positions',0)-before.get('positions',0)
    phases=defaultdict(float)
    executions=[part for r in records for part in ([r,r['branch_execution']] if r.get('branch_execution') else [r])]
    for record in executions:
        for key,value in record.get('metrics',{}).items():phases[key]+=value['seconds']
    inference={}
    for key,value in after.get('timings',{}).items():
        previous=before.get('timings',{}).get(key,{})
        inference[key]=dict(seconds=value['seconds']-previous.get('seconds',0),calls=value['calls']-previous.get('calls',0))
    cells=after.get('padded_candidate_cells',0)-before.get('padded_candidate_cells',0)
    valid=after.get('valid_candidate_cells',0)-before.get('valid_candidate_cells',0)
    cpu=[r['actor_python_cpu_seconds'] for r in executions if r.get('actor_python_cpu_seconds') is not None]
    complete=bool(executions) and len(cpu)==len(executions)
    return dict(games_per_second=games/max(seconds,1e-9),mean_inference_batch=positions/max(1,batches),
        inference_batches=batches,inference_positions=positions,padding_fraction=1-valid/cells if cells else None,
        actor_phase_seconds=dict(phases),inference_phase_deltas=inference,
        inference_batching=dict(
            batches=after.get('batching',{}).get('batches',0)-before.get('batching',{}).get('batches',0),
            padding_splits=after.get('batching',{}).get('padding_splits',0)-before.get('batching',{}).get('padding_splits',0),
            peak_pending_positions_since_service_start=after.get('batching',{}).get('peak_pending_positions'),
            scope='Per-collection count deltas; pending-position peak is cumulative'),
        actor_python_cpu_seconds=sum(cpu) if complete else None,
        actor_python_average_cores=sum(cpu)/max(seconds,1e-9) if complete else None,
        actor_cpu_recorded_games=len(cpu),actor_cpu_expected_work_items=len(executions),
        actor_cpu_scope='Python actor process CPU during games; includes its threads, excludes Mono children and GPU; unavailable for cached games/multiple host lanes',
        ordinary_actors=config['processes']*config['hosts_per_process'],cpu_work_slots=config['cpu_budget']-config['cpu_reserve'],
        scope='ordinary collection; actor/inference phase times overlap; actor_python_* measures process CPU, not total machine or GPU utilization')


def profile_window(path, device, steps=4):
    """Call profiler.step after batches; only a short scheduled window is traced."""
    if path is None:
        return nullcontext(None)
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith('cuda'):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(path)),
        record_shapes=True, profile_memory=True)
