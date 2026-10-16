"""Paired legacy / simplified / seat-batched exports and operation profiles."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, require, architecture_fingerprint
from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.learning.reward import specification
from brass_ai.learning.onnx_tools import cpu_options, simplify_export, operator_counts
from brass_ai.learning.onnx_profile import profile_model
from brass_ai.personality.model import import_parent
from brass_ai.personality.laptop.export import export, compare, BASE_INPUTS
from brass_ai.personality.contracts import CONTEXT_DIM
from test_v11_selectors import trained_branches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT / 'artifacts/feature-engineering') and not out.exists(),
            'Use a fresh isolated benchmark directory')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(716)
    utility = UtilityNet(utility_config(ModelConfig.preset(), specification())).eval()
    trained_branches(utility)
    # Nontrivial independent affine parameters make accidental weight sharing visible.
    with torch.no_grad():
        for role in ('own_player', 'other_player'):
            norm = utility.entity_encoder.encoders[role][2]
            norm.weight.uniform_(.5, 1.5)
            norm.bias.uniform_(-.2, .2)
    actor, _ = import_parent(utility)
    torch.nn.init.normal_(actor.context_projection.weight, std=.02)
    actor.eval()
    source_paths = [SOURCE_ROOT / p for p in (
        'src/brass_ai/learning/model.py', 'src/brass_ai/learning/local_entities.py',
        'src/brass_ai/learning/onnx_tools.py', 'src/brass_ai/learning/onnx_profile.py',
        'src/brass_ai/learning/export.py', 'src/brass_ai/personality/laptop/export.py')]
    source_before = {p.relative_to(SOURCE_ROOT).as_posix(): sha(p) for p in source_paths}
    fixtures = json.loads((SOURCE_ROOT / 'artifacts/feature-engineering/deploy-v9-w352-12s-20261016/inputs.json').read_text())
    positions = {r['name']: Encoder().encode(r['view'], r['choices']) for r in fixtures
                 if r['name'] in ('settled-next-round', 'rail-action')}
    require(len(positions) == 2, 'Missing saved real-position fixtures')
    report = dict(format='onnx-export-optimization-benchmark/1.0.0', threads=1,
        runtime=ort.__version__, source_sha256=source_before, architecture=architecture_fingerprint(),
        diagnostic_weights=True, training_performed=False, models={}, timings=[], profiles={})
    for name, model in (('utility', utility), ('actor', actor)):
        before = copy.deepcopy(model.state_dict())
        paths = {v: out / f'{name}-{v}.onnx' for v in ('baseline', 'simplified', 'optimized')}
        baseline = export(model, paths['baseline'], optimize=False)
        shutil.copyfile(paths['baseline'], paths['simplified'])
        simplified = simplify_export(paths['simplified'])
        optimized = export(model, paths['optimized'])
        report['models'][name] = dict(parameters=model.parameter_count,
            baseline=baseline, simplified=simplified, optimized=optimized,
            operators={v: operator_counts(onnx.load(str(p))) for v, p in paths.items()})
        sessions = {variant: ort.InferenceSession(str(path), cpu_options(), providers=['CPUExecutionProvider'])
                    for variant, path in paths.items()}
        outputs = baseline['outputs']
        feeds = []
        for batch in (1, 2):
            for case, encoded in positions.items():
                data = dict(zip(BASE_INPUTS[:4], [tensor.numpy() for tensor in collate([encoded] * batch)]))
                data['remaining'] = np.full((batch, 4), .075, dtype=np.float32)
                if name == 'actor':
                    data['personality'] = np.full((batch, CONTEXT_DIM), .2, dtype=np.float32)
                expected = dict(zip(outputs, sessions['baseline'].run(outputs, data)))
                parity = {}
                for variant, session in sessions.items():
                    parity[variant] = compare(expected, dict(zip(outputs, session.run(outputs, data))), data['legal_mask'])
                    for _ in range(3):
                        session.run(outputs, data)
                measured = {v: [] for v in paths}
                for repeat in range(21):
                    variants = list(paths)
                    shift = repeat % len(variants)
                    variants = variants[shift:] + variants[:shift]
                    if repeat % 2:
                        variants.reverse()
                    for variant in variants:
                        start = time.perf_counter()
                        sessions[variant].run(outputs, data)
                        measured[variant].append(1000 * (time.perf_counter() - start))
                row = dict(model=name, case=case, batch=batch, candidates=len(encoded.choice_ids), parity=parity,
                    timings={v: dict(median_ms=statistics.median(values), samples_ms=values) for v, values in measured.items()})
                report['timings'].append(row)
                print(name, batch, case, {v: round(r['median_ms'], 4) for v, r in row['timings'].items()}, flush=True)
                if (batch, case) in ((1, 'rail-action'), (2, 'settled-next-round')):
                    feeds.append((f'{case}-batch-{batch}', data))
        del sessions
        report['profiles'][name] = {}
        for variant in ('baseline', 'optimized'):
            result = profile_model(paths[variant], feeds, out / f'profile-{name}-{variant}',
                                   threads=1, warmups=2, repeats=5)
            report['profiles'][name][variant] = dict(path=f'profile-{name}-{variant}/profile.json',
                optimized_operators=result['optimized_operators'],
                cases=[dict(case=r['case'], top_operations=r['operations'][:5]) for r in result['cases']])
        require(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()),
                'Export changed the original model')
    require(source_before == {p.relative_to(SOURCE_ROOT).as_posix(): sha(p) for p in source_paths},
            'Source changed during measurement')
    report['source_unchanged'] = True
    report['geomean_latency_ratios'] = {
        name: {v: math.exp(statistics.mean(math.log(r['timings'][v]['median_ms'] / r['timings']['baseline']['median_ms'])
            for r in report['timings'] if r['model'] == name)) for v in ('simplified', 'optimized')}
        for name in ('utility', 'actor')}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# V11 ONNX export optimization', '',
        'One CPU inference thread; identical nonzero diagnostic weights; real saved Canal/Rail inputs. '
        'Three warmups and 21 alternating samples per condition. No training or search strength evaluation.', '',
        '| Model | Batch | Candidates | Original ms | Simplified ms | Simplified + batched seats ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['timings']:
        lines.append(f"| {row['model']} | {row['batch']} | {row['candidates']} | " +
            ' | '.join(f"{row['timings'][v]['median_ms']:.3f}" for v in ('baseline', 'simplified', 'optimized')) + ' |')
    lines += ['', 'Geometric-mean latency ratios to original: ' + json.dumps(report['geomean_latency_ratios']), '',
        'The portable opset-17 files retain standard erf-based GELU; ORT_ENABLE_ALL fuses it in the execution graph. '
        'Execution-graph checks confirm exact GELU and LayerNorm fusion. Dynamic batch/candidate axes and original '
        'weights/seat mapping pass parity. Training architecture/checkpoint tensors are unchanged.', '',
        'Per-node and operation totals, raw traces and separate unprofiled latency samples:', '']
    for name in ('utility', 'actor'):
        for variant in ('baseline', 'optimized'):
            lines.append(f'- [{name} / {variant}](profile-{name}-{variant}/PROFILE.md)')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Completed:', out / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
