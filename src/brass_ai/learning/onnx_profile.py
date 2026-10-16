"""Bounded, opt-in ONNX CPU profiling; normal inference writes no traces."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

from .contracts import require
from .onnx_tools import cpu_options, operator_counts
from brass_ai.bank import sha


def summarize_trace(events, warmups=0):
    """Aggregate kernel events only, excluding warmup runs and fence events."""
    runs = sorted((e for e in events if e.get('name') == 'model_run'), key=lambda e: e['ts'])
    require(len(runs) > warmups, 'Trace has no measured model runs')
    intervals = [(e['ts'], e['ts'] + e['dur']) for e in runs[warmups:]]
    nodes = defaultdict(list)
    for event in events:
        if event.get('cat') != 'Node' or not event.get('name', '').endswith('_kernel_time'):
            continue
        if not any(start <= event['ts'] < end for start, end in intervals):
            continue
        args = event.get('args', {})
        key = (event['name'].removesuffix('_kernel_time'), args.get('op_name', 'unknown'), args.get('provider', 'unknown'))
        nodes[key].append(float(event['dur']))
    require(bool(nodes), 'Trace contains no measured kernel events')
    total = sum(sum(values) for values in nodes.values())
    node_rows = [dict(name=name, operation=op, provider=provider, calls=len(values),
        total_us=sum(values), mean_us=statistics.mean(values), percent_node_time=100 * sum(values) / total)
        for (name, op, provider), values in nodes.items()]
    operations = defaultdict(list)
    for (_, operation, provider), values in nodes.items():
        operations[(operation, provider)].extend(values)
    op_rows = [dict(operation=op, provider=provider, calls=len(values), total_us=sum(values),
        mean_us=statistics.mean(values), percent_node_time=100 * sum(values) / total)
        for (op, provider), values in operations.items()]
    return dict(measured_runs=len(intervals), excluded_warmup_runs=warmups, total_kernel_us=total,
        nodes=sorted(node_rows, key=lambda r: -r['total_us']),
        operations=sorted(op_rows, key=lambda r: -r['total_us']))


def profile_model(path, feeds, output, *, threads=1, warmups=3, repeats=21):
    """Save an execution graph, raw Chrome traces, per-node/op totals and latency.

    `feeds` is a sequence of (case name, named NumPy inputs). Latency measurements
    use a separate session with profiling disabled; traces have extra overhead.
    """
    import onnx
    import onnxruntime as ort
    require(type(warmups) is int and warmups >= 0 and type(repeats) is int and repeats > 0,
            'Invalid profile repeat counts')
    path, output = Path(path).resolve(), Path(output).resolve()
    require(path.is_file() and not output.exists(), 'Use an existing model and fresh profile directory')
    require(bool(feeds), 'Profiling requires named input cases')
    output.mkdir(parents=True)
    execution = output / 'ort-execution.onnx'
    session = ort.InferenceSession(str(path), cpu_options(threads, optimized_path=execution),
                                   providers=['CPUExecutionProvider'])
    inputs = {item.name for item in session.get_inputs()}
    rows = []
    for ordinal, (name, feed) in enumerate(feeds):
        require(set(feed) == inputs, 'Profile inputs differ from the model contract')
        for _ in range(warmups):
            session.run(None, feed)
        latency = []
        for _ in range(repeats):
            start = time.perf_counter()
            session.run(None, feed)
            latency.append(1000 * (time.perf_counter() - start))
        profiled = ort.InferenceSession(str(path), cpu_options(threads,
            profile_prefix=output / f'trace-{ordinal}'), providers=['CPUExecutionProvider'])
        try:
            for _ in range(warmups + repeats):
                profiled.run(None, feed)
        finally:
            trace = Path(profiled.end_profiling())
            del profiled
        summary = summarize_trace(json.loads(trace.read_text(encoding='utf-8')), warmups)
        require(summary['measured_runs'] == repeats, 'Profile run count differs')
        rows.append(dict(case=name, shapes={key: list(value.shape) for key, value in feed.items()},
            trace=trace.name, latency_unprofiled_ms=dict(median=statistics.median(latency),
                p95=sorted(latency)[min(len(latency)-1, int(.95 * len(latency)))], samples=latency),
            **summary))
    del session
    report = dict(format='onnx-operation-profile/1.0.0', model=str(path), model_sha256=sha(path),
        runtime=ort.__version__, provider='CPUExecutionProvider', intra_op_threads=threads,
        inter_op_threads=1, graph_optimization='ORT_ENABLE_ALL', warmups=warmups, repeats=repeats,
        optimized_operators=operator_counts(onnx.load(str(execution))), cases=rows,
        scope='Tensor-ready inference. Kernel percentages use summed node time, not wall time; profiling adds overhead. Execution graph is host-specific diagnostic evidence, not a portable export.')
    (output / 'profile.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# ONNX Runtime operation profile', '', report['scope'], '',
        f'ORT {ort.__version__}; CPU; {threads} intra-op threads; ORT_ENABLE_ALL.', '']
    for row in rows:
        lines += [f"## {row['case']}", '',
            f"Unprofiled median: {row['latency_unprofiled_ms']['median']:.3f} ms; {repeats} samples after {warmups} warmups.", '',
            '| Operation | Kernel time share | Calls | Mean kernel µs |', '| --- | ---: | ---: | ---: |']
        lines.extend(f"| {r['operation']} | {r['percent_node_time']:.1f}% | {r['calls']} | {r['mean_us']:.1f} |"
                     for r in row['operations'][:15])
        lines.append('')
    (output / 'PROFILE.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report
