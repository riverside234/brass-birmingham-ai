"""Separate OpenVINO graph/precision diagnostics, never ordinary timing data."""
import argparse
import json
from pathlib import Path
import sys
import time
import os
import shutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/rust-backend/20261026-l14-5/providers'


def main(folder, disable_optimization=False, legacy=False):
    global BASE
    BASE=folder.parent
    sys.path[:0]=[str(BASE/('onnxruntime-openvino' if legacy else 'onnxruntime')),str(BASE/'onnxruntime-ep-openvino'),str(ROOT/'artifacts/r2-pythondeps')]
    handles=[]
    if legacy:
        # The legacy provider uses DLL-directory loading. Co-locate its pinned
        # OpenVINO DLLs instead of depending on a developer's global PATH.
        for source in (BASE/'openvino/openvino/libs').iterdir():
            if source.suffix not in ('.dll','.xml'):continue
            destination=BASE/'onnxruntime-openvino/onnxruntime/capi'/source.name
            if destination.exists():assert source.read_bytes()==destination.read_bytes()
            else:shutil.copyfile(source,destination)
        handles.append(os.add_dll_directory(str(BASE/'openvino/openvino/libs')))
        handles.append(os.add_dll_directory(str(BASE/'onnxruntime-openvino/onnxruntime/capi')))
    import numpy as np
    import onnxruntime as ort
    import onnxruntime_ep_openvino as ep
    ort.disable_telemetry_events()
    devices=[]
    if not legacy:
        ort.register_execution_provider_library('brass_openvino',ep.get_library_path())
        devices=[d for d in ort.get_ep_devices() if d.ep_name=='OpenVINOExecutionProvider' and d.ep_metadata.get('ov_device')=='CPU']
        assert len(devices)==1
    m=json.loads((BASE/'models/B/manifest.json').read_text())
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if disable_optimization: options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for key,value in [('optimization.enable_gelu_approximation','0'),('session.intra_op.allow_spinning','0'),
                      ('session.inter_op.allow_spinning','0'),('session.disable_cpu_ep_fallback','1')]:
        options.add_session_config_entry(key,value)
    options.enable_profiling=True;options.profile_file_prefix=str(folder/'profile')
    if not legacy:
        options.add_provider_for_devices(devices,dict(m['provider_runtime']['options'],dump_subgraphs=str(folder/'subgraphs')))
    begin=time.perf_counter()
    session=ort.InferenceSession(str(BASE/'models/B/model.onnx'),sess_options=options,
        **(dict(providers=[('OpenVINOExecutionProvider',dict(m['provider_runtime']['options'],device_type='CPU',precision='FP32'))]) if legacy else {}))
    session.disable_fallback()
    startup=time.perf_counter()-begin
    cpu_control=None;cpu_errors=[0.0]*5
    if legacy:
        cpu_options=ort.SessionOptions();cpu_options.intra_op_num_threads=1;cpu_options.inter_op_num_threads=1
        cpu_options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        cpu_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_options.add_session_config_entry('optimization.enable_gelu_approximation','0')
        cpu_control=ort.InferenceSession(str(BASE/'models/B/model.onnx'),sess_options=cpu_options,providers=['CPUExecutionProvider'])
    golden=json.loads((ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/golden.json').read_text())
    errors=[0.0]*5;rows=0;violations=[]
    for batch in golden:
        b=len(batch);n=max(len(r['actions'])//327 for r in batch)
        actions=np.zeros((b,n,327),np.float32);mask=np.zeros((b,n),bool)
        for i,r in enumerate(batch):
            count=len(r['actions'])//327;actions[i,:count]=np.array(r['actions'],np.float32).reshape(count,327);mask[i,:count]=True
        feeds=dict(states=np.array([r['state'] for r in batch],np.float32),actions=actions,legal_mask=mask,
            stable_to_relative=np.array([r['mapping'] for r in batch],np.int64),
            remaining=np.array([r['remaining'] for r in batch],np.float32),descriptors=np.array([r['descriptors'] for r in batch],np.float32).reshape(b,4,34))
        outputs=session.run(None,feeds)
        cpu_outputs=cpu_control.run(None,feeds) if cpu_control else None
        assert np.all(np.isneginf(outputs[0][~mask]))
        for i,r in enumerate(batch):
            count=len(r['actions'])//327
            for h in range(5):
                actual=outputs[h][i,:count] if h==0 else outputs[h][i]
                for source in ('torch','ort'):
                    expected=np.array(r[source][h]);assert np.all(np.isfinite(actual))
                    errors[h]=max(errors[h],float(np.max(np.abs(actual-expected))))
                    if cpu_outputs is not None:
                        cpu_actual=cpu_outputs[h][i,:count] if h==0 else cpu_outputs[h][i]
                        cpu_errors[h]=max(cpu_errors[h],float(np.max(np.abs(cpu_actual-expected))))
                        np.testing.assert_allclose(cpu_actual,expected,atol=2e-4,rtol=2e-4)
                    if legacy:
                        if not np.allclose(actual,expected,atol=2e-4,rtol=2e-4):
                            violations.append(dict(row=rows,head=h,source=source,actual=actual.tolist(),expected=expected.tolist()))
                    else: np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
            rows+=1
    profile=Path(session.end_profiling())
    if legacy:
        (folder/'parity.json').write_text(json.dumps(dict(passed=not violations,golden_rows=rows,max_absolute_error=errors,
            same_runtime_cpu_passed=True,same_runtime_cpu_max_absolute_error=cpu_errors,
            violations=violations,profile=str(profile),providers=session.get_providers()),indent=2))
        print('Legacy compatibility parity',not violations,rows,errors)
        return
    events=json.loads(profile.read_text())
    kernels=[e for e in events if e.get('args',{}).get('ep')=='OpenVINOExecutionProvider']
    node_providers=sorted({e['args']['provider'] for e in events if 'provider' in e.get('args',{})})
    assert node_providers==['OpenVINOExecutionProvider'],node_providers
    assert kernels,'Missing actual OpenVINO kernel evidence'
    exec_types=sorted({e['args'].get('ov_exec_type','') for e in kernels if e['args'].get('ov_status')=='EXECUTED'})
    assert not any('bf16' in t.lower() or 'fp16' in t.lower() or 'f16' in t.lower() or 'i8' in t.lower() for t in exec_types),exec_types
    import xml.etree.ElementTree as ET
    gels=[]
    for path in (folder/'subgraphs').glob('*.xml'):
        for layer in ET.parse(path).iter('layer'):
            if 'gelu' in layer.get('type','').lower():
                gels.append(layer.find('data').attrib)
    assert gels and all(d.get('approximation_mode')=='ERF' for d in gels),gels
    result=dict(passed=True,golden_rows=rows,max_absolute_error=errors,device=devices[0].ep_metadata,
        options=m['provider_runtime']['options'],startup_seconds=startup,node_providers=node_providers,
        openvino_kernel_events=len(kernels),executed_kernel_types=exec_types,gelu_operations=gels,
        profile=str(profile.relative_to(ROOT)),precision='f32; actual kernel types checked, IR GELU is ERF',
        cpu_partition_fallback=False)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','golden_rows','max_absolute_error','node_providers','executed_kernel_types')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--disable-optimization',action='store_true');p.add_argument('--legacy',action='store_true')
    a=p.parse_args();main(a.folder.resolve(),a.disable_optimization,a.legacy)
