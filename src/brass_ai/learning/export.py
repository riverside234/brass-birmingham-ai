"""Portable ONNX export with variable batch/candidate axes and parity checks."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .batching import collate, model_id
from .contracts import assert_compatible, compatibility, read_json, require
from .model import ModelConfig
from .onnx_tools import cpu_options, prepare_export, simplify_export, inspect_runtime_fusions, candidate_export_metadata

INPUTS = ['states', 'actions', 'legal_mask', 'stable_to_relative','descriptors']
OUTPUTS = ['policy_logits', 'win_share', 'placement', 'vp']


class WithDescriptors(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,states,actions,legal_mask,stable_to_relative,descriptors):
        return self.model(states,actions,legal_mask,stable_to_relative,descriptors=descriptors)


def export_onnx(model, path, parity_batches):
    model.validate_parameter_count()
    import onnx
    import onnxruntime as ort
    path = Path(path)
    require(len(parity_batches) >= 2, 'Export requires multiple shape checks')
    batches = [collate(batch) for batch in parity_batches]
    from .seat_descriptors import unknown_tensor
    batches=[(*batch,unknown_tensor(batch[0])) for batch in batches]
    require(len({(b[0].shape[0], b[1].shape[1]) for b in batches}) >= 2, 'Parity checks must exercise dynamic shapes')
    reference_model = WithDescriptors(copy.deepcopy(model).cpu().eval())
    frozen = WithDescriptors(prepare_export(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    axes = {name: {0: 'positions'} for name in INPUTS + OUTPUTS}
    axes['descriptors']={0:'descriptor_positions'}
    for name in ('actions', 'legal_mask', 'policy_logits'):
        axes[name][1] = 'candidates'
    try:
        # Explicitly pinned 2.8 exporter. This graph uses only standard operators;
        # the legacy exporter is chosen for stable dynamic axes in this scaffold.
        torch.onnx.export(frozen, batches[0], str(temporary), input_names=INPUTS,
            output_names=OUTPUTS, dynamic_axes=axes, opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(temporary)))
        optimization = simplify_export(temporary)
        optimization['seat_encoding'] = 'one grouped MatMul; original own/other weights and affine parameters'
        optimization['candidate_encoding'] = candidate_export_metadata()
        optimization['execution'] = inspect_runtime_fusions(temporary)
        from .onnx_tools import descriptor_default
        descriptor_default(temporary)
        options = cpu_options()
        session = ort.InferenceSession(str(temporary), options, providers=['CPUExecutionProvider'])
        maximum_error = 0.0
        stored = {}
        for i, batch in enumerate(batches):
            with torch.inference_mode():
                expected = reference_model(*batch)
            actual = session.run(OUTPUTS, {k: t.numpy() for k, t in zip(INPUTS, batch)})
            for name, tensor in zip(INPUTS, batch):
                stored[f'{i}:input:{name}'] = tensor.numpy()
            for name, tensor in zip(OUTPUTS, expected):
                stored[f'{i}:output:{name}'] = tensor.numpy()
            for name, a, b in zip(OUTPUTS, actual, expected):
                reference = b.numpy()
                np.testing.assert_allclose(a, reference, rtol=2e-4, atol=2e-4, err_msg=name)
                finite=np.isfinite(reference)
                if finite.any():maximum_error = max(maximum_error, float(np.max(np.abs(a[finite] - reference[finite]))))
        temporary.replace(path)
        np.savez_compressed(path.with_suffix('.parity.npz'), **stored)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {'format': 'four-seat-onnx/1.0.0', 'compatibility': compatibility(),
        'model_config': model.config.to_dict(), 'model_id': model_id(model),
        'onnx_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'inputs': INPUTS, 'outputs': OUTPUTS, 'parameter_count': model.parameter_count,
        'parity_batches': len(batches), 'parity_max_absolute_error': maximum_error,
        'opset': 17, 'provider_tested': 'CPUExecutionProvider', 'optimization': optimization}
    from brass_ai.recovery.domino_config import ENCODING,mapping
    manifest['supported_contexts']=dict(contract=ENCODING,contexts=mapping(),
        scope='Existing input semantics only; deployment selection and strength qualification remain separate. Training-only DOMiNO heads are not exported.')
    path.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def load_export(path):
    """Return a checked portable runtime; no PyTorch checkpoint deserialization."""
    import onnxruntime as ort
    path = Path(path)
    manifest = read_json(path.with_suffix('.json'))
    require(manifest['format'] == 'four-seat-onnx/1.0.0', 'Unknown export format')
    config = ModelConfig.from_dict(manifest['model_config'])
    assert_compatible(manifest['compatibility'])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest['onnx_sha256'], 'ONNX fingerprint mismatch')
    require(manifest['inputs'] == INPUTS and manifest['outputs'] == OUTPUTS, 'Export tensor contract mismatch')
    options = cpu_options()
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    require({x.name for x in session.get_inputs()+session.get_overridable_initializers()} == set(INPUTS) and
            [x.name for x in session.get_outputs()] == OUTPUTS, 'ONNX graph names differ from manifest')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    require(inputs[0].shape[-1] == config.state_dim and inputs[1].shape[-1] == config.action_dim and
            inputs[2].type == 'tensor(bool)' and inputs[3].type == 'tensor(int64)' and
            inputs[3].shape[-1] == 4 and all(x.shape[-1] == 4 for x in outputs[1:]),
            'ONNX graph dimensions or four-seat heads differ')
    return session, manifest
