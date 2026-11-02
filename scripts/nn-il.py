"""Build the pinned IL game option without fitting or changing its checkpoint."""
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
STUDY = ROOT / 'artifacts/nn-il/20261102'
BASE = ROOT / 'artifacts/nn-beta/20261101/release-1.4.48'
OUT = STUDY / 'release-1.4.49'
BUNDLE = ROOT / 'test_nn/features4-v11-2-batch7-il-7f167b8c-gunknown-657fe1901ab3-21d50ddf1a35'
PARENT = '94bebe397ac8ab1f8fe3cfb46d7d293cdb8bc10116dcd5efaf2d2afdd813270b'
CHECKPOINT = '657fe1901ab35d144e5d86e16b42e795d281946829c87bcb8e0245927b70a263'
BACKEND = 'brass-gumbel-is-mcts/il-v11-2-v1'
PROFILE = 'il-v11-2-neutral/1.0.0'
H = runpy.run_path(str(ROOT / 'scripts/build-nn-search-choice.py'))
sha, read, write, pins = (H[k] for k in ('sha', 'read', 'write', 'pins'))
SCOPE = ['src/BrassLocalAI/' + p for p in (
    'Policy/AiSelectionCatalog.cs', 'Unity/LocalMenu.cs', 'Unity/AiSelectionPanel.cs',
    'Native/NativeModel.cs', 'Native/NativePackage.cs', 'Native/NativeGameState.cs')]


def export():
    sys.path[:0] = [str(ROOT / 'src')] + [str(ROOT / p) for p in (
        'artifacts/r2-pythondeps', 'artifacts/r4-pythondeps', 'artifacts/r0-pythonlibs',
        'artifacts/r0-validation-deps', 'artifacts/onnx-export-deps')]
    import torch
    import onnxruntime as ort
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.learning.utility_model import UtilityNet, utility_config
    from brass_ai.personality.laptop.export import export as onnx_export, samples
    from brass_ai.native.features import SCHEMA, GROUP, ENTITY
    directory = STUDY / 'model'
    assert not directory.exists(), 'Preserve the immutable IL export'
    assert sha(BUNDLE / 'checkpoint.pt') == CHECKPOINT and ort.__version__ == '1.22.1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original, raw = load_checkpoint(BUNDLE / 'checkpoint.pt')
    assert raw['model_id'] == PARENT and getattr(original.config, 'utility', None) is None
    # Reuse the installed search objective and the existing UtilityNet's exact
    # zero-residual initialization. Only an inference copy receives this head.
    # No optimizer, training data, backward pass or readiness receipt is created.
    reward = read(BASE / 'BrassNative/beta-g1290/model/model.json')['config']['utility']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17041)
        model = UtilityNet(utility_config(original.config, reward)).eval()
    missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
    assert not unexpected and set(missing) == {k for k in model.state_dict() if k.startswith('utility_head.')}
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items())
    assert torch.count_nonzero(model.utility_head[-1].weight) == 0
    assert torch.count_nonzero(model.utility_head[-1].bias) == 0
    original.eval()
    with torch.inference_mode():
        for data in samples(model.config.to_dict(), False):
            args = [torch.from_numpy(data[k]) for k in ('states', 'actions', 'legal_mask', 'stable_to_relative')]
            a = original(*args, descriptors=torch.from_numpy(data['descriptors']))
            b = model(*args, remaining=torch.from_numpy(data['remaining']), descriptors=torch.from_numpy(data['descriptors']))
            assert all(torch.equal(x, y) for x, y in zip(a, b[:4]))
            expected = reward['win'] * a[1] + reward['placement'] * (4 - a[2]) / 3 + reward['vp'] * a[3].clamp_max(reward['vp_cap'])
            torch.testing.assert_close(b[-1], expected, rtol=0, atol=0)
    directory.mkdir(parents=True)
    sources = [p for folder in ('src/brass_ai/learning', 'src/brass_ai/personality/laptop', 'src/brass_ai/native')
               for p in sorted((ROOT / folder).rglob('*.py'))] + [Path(__file__).resolve()]
    source_pins = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sources]
    start = time.perf_counter()
    meta = onnx_export(model, directory / 'model.onnx')
    assert meta['parity']['passed'] and model.utility_calibration is None
    for p in source_pins:
        assert sha(ROOT / p['path']) == p['sha256'], 'Export source changed'
    with zipfile.ZipFile(directory / 'export-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in source_pins:
            z.write(ROOT / p['path'], p['path'])
    runtime = Path(ort.__file__).parent / 'capi/onnxruntime.dll'
    shutil.copyfile(runtime, directory / 'onnxruntime.dll')
    manifest = dict(format='brass-native-model/1.0.0', model_id=meta['model_id'], checkpoint_sha256=CHECKPOINT,
        export_sha256=sha(directory / 'model.onnx'), export_file='model.onnx', metadata_sha256=sha(directory / 'model.json'),
        runtime_file='onnxruntime.dll', runtime_sha256=sha(runtime), runtime_version=ort.__version__, binding_version='2.0.0-rc.10',
        architecture=meta['config']['architecture'], feature_schema=SCHEMA, group_layout=GROUP, entity_layout=ENTITY,
        reward_id=meta['reward_id'], max_cost=reward['insolvency']['max_cost'], precision='float32', opset=17,
        inputs=meta['inputs'], outputs=meta['outputs'], mechanical_only=True, utility_calibration=None,
        intra_threads=1, inter_threads=1, sequential=True, provider='CPUExecutionProvider', optimization='all-exact-gelu',
        source_archive_sha256=sha(directory / 'export-source.zip'))
    write(directory / 'manifest.json', manifest)
    assert sha(BUNDLE / 'checkpoint.pt') == CHECKPOINT
    write(directory / 'export-receipt.json', dict(passed=True, parent_model_id=PARENT, checkpoint_sha256=CHECKPOINT,
        model_id=meta['model_id'], export_sha256=manifest['export_sha256'], manifest_sha256=sha(directory / 'manifest.json'),
        parity=meta['parity'], original_tensors_unchanged=True, official_heads_exact=True, zero_residual=True,
        utility='win_share + 0.1*(4-placement)/3 + 0.001*min(vp,180); no learned correction',
        utility_calibration=None, reward=reward, training_steps=0, checkpoint_unchanged=True,
        seconds=time.perf_counter()-start, source_pins=source_pins))
    print(json.dumps(dict(passed=True, model_id=meta['model_id'], export_sha256=manifest['export_sha256'], parity=meta['parity'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('export',))
    stage = parser.parse_args().stage
    STUDY.mkdir(parents=True, exist_ok=True)
    globals()[stage]()
