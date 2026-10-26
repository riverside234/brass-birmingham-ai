"""Immutable L09 CPU export from the frozen reference and temporary g890 weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(reference, out):
    import torch
    import onnxruntime as ort
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.personality.laptop.export import export
    from brass_ai.native.features import SCHEMA, GROUP, ENTITY
    assert not out.exists(), 'Do not overwrite an export'
    assert ort.__version__ == '1.22.1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = next((ROOT/'test_nn').glob('*/checkpoint.pt'))
    digest = sha(checkpoint)
    assert digest == 'ecad3b932a8dd9d3b5d0cc49b5e9bb94a9e98b4536ffdc86a44179d003995cb8'
    model, raw = load_checkpoint(checkpoint)
    assert model.utility_calibration is None
    out.mkdir(parents=True)
    start = time.perf_counter()
    meta = export(model.eval(), out/'model.onnx')
    assert meta['model_id'] == raw['model_id']
    runtime = Path(ort.__file__).parent/'capi/onnxruntime.dll'
    shutil.copyfile(runtime, out/'onnxruntime.dll')
    for notice in ('LICENSE','ThirdPartyNotices.txt','Privacy.md'):
        shutil.copyfile(Path(ort.__file__).parent/notice,out/('onnxruntime-'+notice))
    # The pinned graph always receives explicit descriptors. Keep the reference
    # export's overridable unknown initializer and test both representations.
    manifest = dict(format='brass-native-model/1.0.0', model_id=raw['model_id'],
        checkpoint_sha256=digest, export_sha256=sha(out/'model.onnx'), export_file='model.onnx',
        metadata_sha256=sha(out/'model.json'), runtime_file='onnxruntime.dll',
        runtime_sha256=sha(runtime), runtime_version=ort.__version__, binding_version='2.0.0-rc.10',
        architecture=meta['config']['architecture'], feature_schema=SCHEMA,
        group_layout=GROUP, entity_layout=ENTITY, reward_id=meta['reward_id'],
        max_cost=meta['config']['utility']['insolvency']['max_cost'], precision='float32', opset=17,
        inputs=meta['inputs'], outputs=meta['outputs'], mechanical_only=True,
        utility_calibration=None, intra_threads=1, inter_threads=1, sequential=True,
        provider='CPUExecutionProvider', optimization='all-exact-gelu',
        source_archive_sha256=json.loads((ROOT/'artifacts/rust-backend/20261024-g890-l01-l02/baseline.json').read_text())['reference_source']['archive']['sha256'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (out/'export-receipt.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        manifest_sha256=sha(out/'manifest.json'), reference=str(reference),
        checkpoint_unchanged=sha(checkpoint)==digest, parity=meta['parity'],
        torch=torch.__version__, runtime_source=str(runtime)), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(manifest=str(out/'manifest.json'), sha256=sha(out/'manifest.json'), parity=meta['parity'])), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    sys.path[:0]=[str(a.reference.resolve()/'src')]+[str(ROOT/d) for d in
        ('artifacts/r2-pythondeps','artifacts/r4-pythondeps','artifacts/r0-pythonlibs','artifacts/r0-validation-deps','artifacts/onnx-export-deps')]
    prepare(a.reference.resolve(), a.out.resolve())
