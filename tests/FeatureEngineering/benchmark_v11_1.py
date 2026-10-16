"""Bounded v11.1 ORT comparison: indexed training form vs static W + local MatMul.

Both graphs use identical weights, pre-GELU semantics, simplification and
batched own/other seats. This isolates the candidate execution rewrite.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, require, compatibility
from brass_ai.learning.model import ModelConfig, FourSeatNet, dense_multiply_adds
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.learning.reward import specification
from brass_ai.learning.onnx_tools import prepare_export, cpu_options, simplify_export, inspect_runtime_fusions
from brass_ai.learning.onnx_profile import profile_model
from brass_ai.personality.model import import_parent
from brass_ai.personality.laptop.export import export, Named, BASE_INPUTS, BASE_OUTPUTS, compare
from test_v11_1 import varied_inputs
from test_v11_selectors import trained_branches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    require(not out.exists() and out.is_relative_to(SOURCE_ROOT / 'artifacts/feature-engineering'),
            'Use a new isolated benchmark directory')
    out.mkdir(parents=True); began = time.perf_counter(); torch.set_num_threads(2)
    torch.manual_seed(1116)
    model = UtilityNet(utility_config(ModelConfig.preset('medium'), specification())).eval()
    trained_branches(model)
    model_before = {k: v.clone() for k, v in model.state_dict().items()}
    report = dict(architecture=model.config.architecture, compatibility=compatibility(),
        parameters=dict(base=FourSeatNet().parameter_count, utility=model.parameter_count,
                        personality=import_parent(model)[0].parameter_count),
        training_macs=dense_multiply_adds(model), export_per_candidate_macs=83232+143*160,
        dynamic_values_per_position=143*160, runtime=ort.__version__, torch=torch.__version__,
        provider='CPUExecutionProvider', threads=1, warmups=3, repeats=21,
        scope='Synthetic valid categorical inputs, nonzero diagnostic weights; tensor-ready CPU inference only. '
              'Both variants are v11.1, not a v11 strength comparison. No training/search or production checkpoint changes.',
        variants={}, timings=[])
    path = out / 'selector-matmul.onnx'
    meta = export(model, path)
    report['variants']['selector_matmul'] = meta
    x, a, mask, mapping = varied_inputs(1, 1)
    example = (x, a, mask, mapping, torch.full((1, 4), .075))
    axes = {name: {0: 'positions'} for name in BASE_INPUTS + BASE_OUTPUTS}
    for name in ('actions', 'legal_mask', 'policy_logits'):axes[name][1] = 'candidates'
    indexed_path = out / 'indexed.onnx'
    with torch.inference_mode():
        torch.onnx.export(Named(prepare_export(model, local_selectors=False)), example,
            str(indexed_path), input_names=BASE_INPUTS, output_names=BASE_OUTPUTS,
            dynamic_axes=axes, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(indexed_path)))
    report['variants']['indexed'] = dict(simplification=simplify_export(indexed_path),
        execution=inspect_runtime_fusions(indexed_path), sha256=sha(indexed_path))
    sessions = {name: ort.InferenceSession(str(p), cpu_options(), providers=['CPUExecutionProvider'])
                for name, p in (('indexed', indexed_path), ('selector_matmul', path))}
    feeds = []
    for batch in (1, 2):
        for count in (1, 119, 387, 772):
            x, a, mask, mapping = varied_inputs(batch, count)
            remaining = torch.full((batch, 4), .075)
            data = dict(zip(BASE_INPUTS, [t.numpy() for t in (x, a, mask, mapping, remaining)]))
            with torch.inference_mode():expected = dict(zip(BASE_OUTPUTS, [t.numpy() for t in model(x, a, mask, mapping, remaining)]))
            errors = {name: compare(expected, dict(zip(BASE_OUTPUTS, session.run(BASE_OUTPUTS, data))), data['legal_mask'])
                      for name, session in sessions.items()}
            for session in sessions.values():
                for _ in range(3):session.run(BASE_OUTPUTS, data)
            times = {name: [] for name in sessions}
            for repeat in range(21):
                for name in (list(sessions) if repeat % 2 else list(reversed(sessions))):
                    start = time.perf_counter(); sessions[name].run(BASE_OUTPUTS, data)
                    times[name].append(1000 * (time.perf_counter() - start))
            row = dict(batch=batch, candidates=count, parity=errors,
                timing={name: dict(median_ms=statistics.median(values), samples_ms=values) for name, values in times.items()})
            report['timings'].append(row)
            print(batch, count, {name: round(value['median_ms'], 4) for name, value in row['timing'].items()}, flush=True)
            if count == 772:feeds.append((f'batch-{batch}-candidates-{count}', data))
    del sessions
    profile = profile_model(path, feeds, out / 'profile', threads=1, warmups=2, repeats=5)
    report['profile'] = dict(path='profile/profile.json', operators=profile['optimized_operators'],
        cases=[dict(case=c['case'], batched_seat_nodes=[n for n in c['nodes'] if 'entity_encoder/MatMul' in n['name']])
               for c in profile['cases']])
    require(all(torch.equal(v, model.state_dict()[k]) for k, v in model_before.items()), 'Original weights changed')
    report['weights_unchanged'] = True
    report['elapsed_seconds'] = time.perf_counter() - began
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Report:', out / 'report.json', flush=True)


if __name__ == '__main__':main()
