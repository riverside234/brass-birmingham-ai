"""Immutable per-model ONNX bundles for CPU inference inside lab workers.

Uses the local game's export pipeline (exact GELU, opset 17, parity at the same
six shapes). A model with a search head gains one output, ``search_delta_4``,
so workers can blend it exactly as the GPU path does. Weights are never changed.

League anchors such as the imitation-learning model have no utility head. Their
bundle carries an all-zero utility output so every bundle has one shape, and its
manifest records ``utility=false`` so evaluators omit that field as the GPU does.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import uuid
import warnings

from brass_ai.bank import sha
from brass_ai.learning.contracts import require

CONTRACT = 'brass-lab-cpu-bundle/1.0.0'
RUNTIME_VERSION = '1.22.1'
DIRECTORY = 'artifacts/rust-backend/cpu-bundles'
RETAINED = 128
SEARCH_DELTA = 'search_delta_4'
NO_REWARD = ''


def bundle_root(root):
    return Path(root)/DIRECTORY


def _graph(model, head, utility, torch):
    """Named outputs of the same forward the GPU teacher path evaluates."""
    from torch import nn
    from brass_ai.learning.onnx_tools import SelectorMatmulExport
    from brass_ai.learning.search_value import correction

    class Graph(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, states, actions, legal_mask, stable_to_relative, remaining, descriptors):
            inner = self.inner
            if not head:
                out = inner(states, actions, legal_mask, stable_to_relative, remaining=remaining, descriptors=descriptors)
                # Multiplying by zero also keeps `remaining` a real graph input.
                return out if utility else (*out, remaining*0)
            encoded = inner.encode_state(states)
            logits = inner.score_prepared(actions, inner.prepare_candidates(encoded)).masked_fill(~legal_mask, -torch.inf)
            heads = inner.heads_from_encoded(encoded, stable_to_relative, remaining, descriptors)
            source = inner.source if isinstance(inner, SelectorMatmulExport) else inner
            return (logits, *heads, correction(source, encoded, stable_to_relative, remaining))
    return Graph(model)


def _manifest_receipt(directory):
    path = Path(directory)/'manifest.json'
    return dict(manifest=str(path.resolve()), sha256=sha(path))


def export_bundle(model, destination):
    """Publish one bundle atomically; a concurrent identical export may win."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from brass_ai.learning.batching import model_id
    from brass_ai.learning.contracts import compatibility, digest
    from brass_ai.learning.onnx_tools import (cpu_options, descriptor_default, inspect_runtime_fusions,
                                              prepare_export, simplify_export)
    from brass_ai.native.features import ENTITY, GROUP, SCHEMA
    from brass_ai.personality.laptop.export import BASE_INPUTS, BASE_OUTPUTS, compare, samples

    require(ort.__version__ == RUNTIME_VERSION, 'CPU bundles require onnxruntime '+RUNTIME_VERSION)
    ort.set_default_logger_severity(3)  # Every export repeats two harmless ORT warnings.
    model.validate_parameter_count()
    destination = Path(destination)
    partial = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    partial.mkdir(parents=True)
    try:
        head = hasattr(model, 'search_value_head')
        outputs = BASE_OUTPUTS+([SEARCH_DELTA] if head else [])
        config = model.config.to_dict()
        utility = config.get('utility') is not None
        require(utility or not head, 'A search head requires a utility model')
        # The shared sampler draws budgets from the reward cap. A model with no
        # utility head ignores them, so any positive range serves its parity check.
        batches = samples(config if utility else dict(config, utility=dict(insolvency=dict(max_cost=1.))), False)
        path = partial/'model.onnx'
        axes = {name: {0: 'positions'} for name in BASE_INPUTS+outputs}
        axes['descriptors'] = {0: 'descriptor_positions'}
        for name in ('actions', 'legal_mask', 'policy_logits'):
            axes[name][1] = 'candidates'
        # Exporting neither trains nor samples; keep any generator use out of
        # the trainer's stream regardless.
        with warnings.catch_warnings(), torch.random.fork_rng(devices=[]), torch.inference_mode():
            warnings.simplefilter('ignore', torch.jit.TracerWarning)
            reference = _graph(copy.deepcopy(model).cpu().eval(), head, utility, torch)
            frozen = _graph(prepare_export(model), head, utility, torch)
            torch.onnx.export(frozen, tuple(torch.from_numpy(batches[0][k]) for k in BASE_INPUTS), str(path),
                input_names=BASE_INPUTS, output_names=outputs, dynamic_axes=axes, opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        simplified = simplify_export(path)
        fusions = inspect_runtime_fusions(path)
        descriptor_default(path)
        session = ort.InferenceSession(str(path), cpu_options(1), providers=['CPUExecutionProvider'])
        require([x.name for x in session.get_outputs()] == outputs, 'Bundle output order differs')
        errors = {name: 0. for name in outputs}
        for batch in batches:
            with torch.inference_mode():
                expected = reference(*(torch.from_numpy(batch[k]) for k in BASE_INPUTS))
            expected = {k: v.numpy() for k, v in zip(outputs, expected, strict=True)}
            actual = dict(zip(outputs, session.run(outputs, batch), strict=True))
            for name, error in compare(expected, actual, batch['legal_mask']).items():
                errors[name] = max(errors[name], error)
        del session
        model_json = dict(contract=CONTRACT, config=config, compatibility=compatibility(), simplifier=simplified,
                          execution=fusions)
        (partial/'model.json').write_text(json.dumps(model_json, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        manifest = dict(contract=CONTRACT, model_id=model_id(model), export_file='model.onnx',
            export_sha256=sha(path), metadata_sha256=sha(partial/'model.json'),
            reward_id=digest(config['utility']) if utility else NO_REWARD,
            max_cost=config['utility']['insolvency']['max_cost'] if utility else 0., utility=utility,
            architecture=config['architecture'], feature_schema=SCHEMA, group_layout=GROUP, entity_layout=ENTITY,
            precision='float32', opset=17, inputs=BASE_INPUTS, outputs=outputs, search_delta=head,
            provider='CPUExecutionProvider', intra_threads=1, inter_threads=1, sequential=True,
            optimization='all-exact-gelu', exporter=dict(torch=torch.__version__, onnx=onnx.__version__,
            onnxruntime=ort.__version__, numpy=np.__version__),
            parity=dict(shapes=[[len(b['states']), b['actions'].shape[1]] for b in batches], errors=errors,
                        tolerance=2e-4, passed=True))
        (partial/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        try:
            os.replace(partial, destination)
        except OSError:
            require((destination/'manifest.json').is_file(), 'Could not publish the CPU bundle')
    finally:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
    return manifest


def bundle_ready(root, identity):
    return (bundle_root(root)/identity/'manifest.json').is_file()


def ensure(root, models, identities, *, materialize=None):
    """Return ``{model_id: {manifest, sha256}}``, exporting anything missing.

    ``materialize`` loads a checkpoint whose load was deferred, for a model
    whose bundle turns out to be gone.
    """
    base = bundle_root(root)
    base.mkdir(parents=True, exist_ok=True)
    wanted = sorted(set(identities))
    require(wanted, 'CPU inference needs at least one frozen model')
    result = {}
    exported = 0
    for identity in wanted:
        directory = base/identity
        if not (directory/'manifest.json').is_file():
            if materialize is not None:
                materialize(identity)
            require(identity in models, 'Unregistered model cannot be exported for CPU inference')
            manifest = export_bundle(models[identity], directory)
            require(manifest['model_id'] == identity, 'CPU bundle model identity differs')
            exported += 1
        receipt = _manifest_receipt(directory)
        value = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        require(value['contract'] == CONTRACT and value['model_id'] == identity, 'Foreign CPU bundle')
        os.utime(directory/'manifest.json')
        result[identity] = receipt
    prune(base, set(wanted))
    return result, exported


def prune(base, keep, limit=RETAINED):
    """Bundles are reproducible caches; drop the least recently used ones."""
    entries = [p for p in Path(base).iterdir() if p.is_dir() and (p/'manifest.json').is_file() and p.name not in keep]
    entries.sort(key=lambda p: (p/'manifest.json').stat().st_mtime)
    for path in entries[:max(0, len(entries)-max(0, limit-len(keep)))]:
        shutil.rmtree(path, ignore_errors=True)
