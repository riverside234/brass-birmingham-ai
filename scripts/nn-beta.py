"""Export and qualify the exact g1290 test NN as the local Beta option."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/nn-beta/20261101'
BASE=ROOT/'artifacts/nn-neutral-ui/20261031/release-1.4.47'
OUT=STUDY/'release-1.4.48'
BUNDLE=ROOT/'test_nn/joint7-v11-2-grouped-lr-critic-1b96e228-g1290-ed8edb039a12-d50f4bc95d51'
MODEL='73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff'
CHECKPOINT='ed8edb039a1257d591cc0d2e2b87eb7a2d7aa411ce0f7710a2d7f8ad487dc3b0'
CALIBRATION='f56bd03a91e661c5be0f941765ec9420c4729ef6caca0f8d294d61b3d3550670'
H=runpy.run_path(str(ROOT/'scripts/build-nn-search-choice.py'))
sha,read,write,pins=(H[k] for k in ('sha','read','write','pins'))


def export():
    sys.path[:0]=[str(ROOT/'src')]+[str(ROOT/p) for p in ('artifacts/r2-pythondeps','artifacts/r4-pythondeps',
        'artifacts/r0-pythonlibs','artifacts/r0-validation-deps','artifacts/onnx-export-deps')]
    import torch
    import onnxruntime as ort
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.personality.laptop.export import export as onnx_export
    from brass_ai.native.features import SCHEMA,GROUP,ENTITY
    model_dir=STUDY/'model';assert not model_dir.exists(),'Preserve model export'
    assert sha(BUNDLE/'checkpoint.pt')==CHECKPOINT
    assert read(BUNDLE/'manifest.json')['derived_checkpoint']['generation']==1290
    assert ort.__version__=='1.22.1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,raw=load_checkpoint(BUNDLE/'checkpoint.pt')
    calibration=read(BUNDLE/'calibration.json')
    assert raw['model_id']==MODEL and model.utility_calibration==calibration['calibration_receipt']
    assert model.utility_calibration['request_id']==CALIBRATION and model.utility_calibration['weights_unchanged']
    assert model.utility_calibration['ready'] and model.utility_calibration['steps']==0
    model_dir.mkdir(parents=True)
    sources=[p for folder in ('src/brass_ai/learning','src/brass_ai/personality/laptop','src/brass_ai/native')
             for p in sorted((ROOT/folder).rglob('*.py'))]+[Path(__file__).resolve()]
    source_pins=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sources]
    start=time.perf_counter();meta=onnx_export(model.eval(),model_dir/'model.onnx')
    assert meta['model_id']==MODEL and meta['parity']['passed']
    for p in source_pins:assert sha(ROOT/p['path'])==p['sha256'],'Export source changed during execution'
    with zipfile.ZipFile(model_dir/'export-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in source_pins:z.write(ROOT/p['path'],p['path'])
    runtime=Path(ort.__file__).parent/'capi/onnxruntime.dll'
    shutil.copyfile(runtime,model_dir/'onnxruntime.dll')
    shutil.copyfile(BUNDLE/'calibration.json',model_dir/'calibration.json')
    manifest=dict(format='brass-native-model/1.0.0',model_id=MODEL,checkpoint_sha256=CHECKPOINT,
        export_sha256=sha(model_dir/'model.onnx'),export_file='model.onnx',metadata_sha256=sha(model_dir/'model.json'),
        runtime_file='onnxruntime.dll',runtime_sha256=sha(runtime),runtime_version=ort.__version__,binding_version='2.0.0-rc.10',
        architecture=meta['config']['architecture'],feature_schema=SCHEMA,group_layout=GROUP,entity_layout=ENTITY,
        reward_id=meta['reward_id'],max_cost=meta['config']['utility']['insolvency']['max_cost'],precision='float32',opset=17,
        inputs=meta['inputs'],outputs=meta['outputs'],mechanical_only=False,utility_calibration=model.utility_calibration,
        intra_threads=1,inter_threads=1,sequential=True,provider='CPUExecutionProvider',optimization='all-exact-gelu',
        source_archive_sha256=sha(model_dir/'export-source.zip'))
    write(model_dir/'manifest.json',manifest)
    assert sha(BUNDLE/'checkpoint.pt')==CHECKPOINT
    write(model_dir/'export-receipt.json',dict(passed=True,generation=1290,model_id=MODEL,checkpoint_sha256=CHECKPOINT,
        export_sha256=manifest['export_sha256'],manifest_sha256=sha(model_dir/'manifest.json'),parity=meta['parity'],
        checkpoint_unchanged=True,seconds=time.perf_counter()-start,parameters=model.parameter_count,source_pins=source_pins))
    print(json.dumps(dict(passed=True,model_id=MODEL,export_sha256=manifest['export_sha256'],parity=meta['parity'])),flush=True)


def native():
    out=STUDY/'native-build';assert not out.exists(),'Preserve native build'
    out.mkdir(parents=True)
    support=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'));env,cargo=support['tool_environment']()
    scope=['native/brass-ai-worker/Cargo.toml','native/brass-ai-worker/src/lib.rs',
           'native/brass-ai-worker/src/local.rs','native/brass-ai-worker/src/local_model.rs',
           'native/brass-ai-inference/src/manifest.rs']
    formatter=ROOT/'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    checkout=out/'n'
    with zipfile.ZipFile(ROOT/'BrassNative/alpha-g890/native-source.zip') as z:
        assert all((checkout/n).resolve().is_relative_to(checkout.resolve()) for n in z.namelist());z.extractall(checkout)
    for name in scope:shutil.copyfile(ROOT/name,checkout/name)
    for name in scope:
        if name.endswith('.rs'):subprocess.run([str(formatter),'--edition','2021','--config','skip_children=true',str(checkout/name)],env=env,check=True)
    env.update(CARGO_TARGET_DIR=str(ROOT/'artifacts/nn-temperature/target'),CARGO_INCREMENTAL='0')
    stages=[]
    for label,args in [('build',['build','--release','--bin','brass-ai-local','--features','local-beta']),
                       ('beta-worker-tests',['test','--release','-p','brass-ai-worker','--lib','--features','local-beta']),
                       ('inference-contract-tests',['test','--release','-p','brass-ai-inference','--test','contracts'])]:
        print(label,flush=True)
        log=out/(label+'.log')
        with log.open('wb') as f:
            subprocess.run([str(cargo),*args,'--offline','--locked'],cwd=checkout/'native',env=env,
                stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        print(log.read_text()[-1800:],flush=True);stages.append(dict(stage=label,passed=True,log_sha256=sha(log)))
    support['native_fingerprint'].__globals__['NATIVE']=checkout/'native';identity=support['native_fingerprint']()
    shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'x86_64-pc-windows-gnu/release/brass-ai-local.exe',out/'brass-ai-local.exe')
    with zipfile.ZipFile(out/'native-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(checkout.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(checkout).as_posix())
    write(out/'native-build.json',dict(passed=True,source_sha256=identity,scope=scope,checks=stages,
        binary_sha256=sha(out/'brass-ai-local.exe'),source_archive_sha256=sha(out/'native-source.zip'),feature='local-beta'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('export','native'))
    stage=parser.parse_args().stage;STUDY.mkdir(parents=True,exist_ok=True);globals()[stage]()
