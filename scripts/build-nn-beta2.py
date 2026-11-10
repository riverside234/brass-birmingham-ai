"""Pinned g1190 beta-2 local package, based on installed search_v1 sources."""
from pathlib import Path
import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/nn-beta2/20261110'
OUT = STUDY / 'release-1.4.55'
BASE = ROOT / 'artifacts/card-tracker-three/20261105/release-1.4.53'
BUNDLE = ROOT / 'test_nn/joint7-v11-2-g1290-restart-g1290-ae347ae9-g1190-f228f4dd7f1a-1af78fc4bcc8'
MODEL = 'd872ea925d6643ee3a3eeec99a52d16c6a78c2de1f757617f1aaaa47d175f864'
CHECKPOINT = 'f228f4dd7f1a913ea11e908cb2ccb53b76cd14ca302da9e2e5479c505d1294f1'
BACKEND = 'brass-gumbel-is-mcts/beta2-g1190-v1'
PROFILE = 'beta2-g1190-neutral/1.0.0'
DIRECTORY = 'beta2-g1190'
VERSION = '1.4.55-rust-local'
SCOPE = ['src/BrassLocalAI/Native/' + n + '.cs' for n in ('NativeModel', 'NativePackage', 'NativeInput')]
NATIVE_SCOPE = ['native/brass-ai-worker/Cargo.toml', 'native/brass-ai-worker/src/local_model.rs',
                'native/brass-ai-worker/src/local.rs']
POLICIES = ('policy.json', 'policy-k8.json', 'policy-k32.json', 'policy-t15.json', 'policy-k8-t15.json', 'policy-k32-t15.json')
HELPERS = runpy.run_path(str(ROOT / 'scripts/fix-sell-replay-speed.py'))
sha, read, write, pins, compile_cs = (HELPERS[k] for k in ('sha', 'read', 'write', 'pins', 'compile_cs'))


def export():
    sys.path[:0] = [str(ROOT / 'src')] + [str(ROOT / p) for p in (
        'artifacts/r2-pythondeps', 'artifacts/r4-pythondeps', 'artifacts/r0-pythonlibs',
        'artifacts/r0-validation-deps', 'artifacts/onnx-export-deps')]
    import torch
    import onnxruntime as ort
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.personality.laptop.export import export as onnx_export
    from brass_ai.native.features import SCHEMA, GROUP, ENTITY
    directory = STUDY / 'model'
    assert not directory.exists(), 'Preserve export evidence'
    exporter = runpy.run_path(str(ROOT / 'scripts/export-test-nn.py'))
    source = exporter['verify_bundle'](BUNDLE)
    assert source['checkpoint']['sha256'] == sha(BUNDLE / 'checkpoint.pt') == CHECKPOINT
    assert source['recorded_checkpoint']['generation'] == 1190
    assert ort.__version__ == '1.22.1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model, raw = load_checkpoint(BUNDLE / 'checkpoint.pt')
    assert raw['model_id'] == MODEL and model.utility_calibration is None
    reward = model.config.to_dict()['utility']
    assert reward['insolvency'] == dict(event_cost=.0025, max_cost=.025)
    directory.mkdir(parents=True)
    sources = [p for folder in ('src/brass_ai/learning', 'src/brass_ai/personality/laptop', 'src/brass_ai/native')
               for p in sorted((ROOT / folder).rglob('*.py'))] + [Path(__file__).resolve()]
    source_pins = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sources]
    began = time.perf_counter()
    meta = onnx_export(model.eval(), directory / 'model.onnx')
    assert meta['model_id'] == MODEL and meta['parity']['passed']
    for p in source_pins:
        assert sha(ROOT / p['path']) == p['sha256'], 'Export source changed'
    with zipfile.ZipFile(directory / 'export-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in source_pins:
            z.write(ROOT / p['path'], p['path'])
    runtime = Path(ort.__file__).parent / 'capi/onnxruntime.dll'
    shutil.copyfile(runtime, directory / 'onnxruntime.dll')
    manifest = dict(format='brass-native-model/1.0.0', model_id=MODEL, checkpoint_sha256=CHECKPOINT,
        export_sha256=sha(directory / 'model.onnx'), export_file='model.onnx', metadata_sha256=sha(directory / 'model.json'),
        runtime_file='onnxruntime.dll', runtime_sha256=sha(runtime), runtime_version=ort.__version__, binding_version='2.0.0-rc.10',
        architecture=meta['config']['architecture'], feature_schema=SCHEMA, group_layout=GROUP, entity_layout=ENTITY,
        reward_id=meta['reward_id'], max_cost=reward['insolvency']['max_cost'], precision='float32', opset=17,
        inputs=meta['inputs'], outputs=meta['outputs'], mechanical_only=True, utility_calibration=None,
        intra_threads=1, inter_threads=1, sequential=True, provider='CPUExecutionProvider', optimization='all-exact-gelu',
        source_archive_sha256=sha(directory / 'export-source.zip'))
    write(directory / 'manifest.json', manifest)
    assert sha(BUNDLE / 'checkpoint.pt') == CHECKPOINT
    write(directory / 'export-receipt.json', dict(passed=True, generation=1190, model_id=MODEL,
        checkpoint_sha256=CHECKPOINT, export_sha256=manifest['export_sha256'], reward_id=meta['reward_id'], reward=reward,
        manifest_sha256=sha(directory / 'manifest.json'), parity=meta['parity'], parameters=model.parameter_count,
        checkpoint_unchanged=True, training_steps=0, utility_calibration=None,
        auxiliary_heads=source['recorded_checkpoint'].get('auxiliary_heads'),
        scope='Original policy and learned utility outputs. Companion heads retained in source checkpoint; search_v1 uses its existing utility output.',
        seconds=time.perf_counter()-began, source_pins=source_pins))
    print(json.dumps(dict(passed=True, model_id=MODEL, export_sha256=manifest['export_sha256'], reward_id=meta['reward_id'], parity=meta['parity'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('export',))
    args = parser.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    globals()[args.stage]()
