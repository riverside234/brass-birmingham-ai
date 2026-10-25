"""Read-only reference/model admission and additive local recovery receipts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json, require
from .fixtures import file_sha, write_json


def git(root, *args, binary=False):
    result = subprocess.run(['git','-C',str(root),*args], check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode('utf-8').strip()


def receipt(path, root=None):
    path=Path(path)
    return dict(path=path.relative_to(root).as_posix() if root else str(path),
                bytes=path.stat().st_size, sha256=file_sha(path))


def snapshot_files(root, paths, archive):
    """Archive exact bytes with a per-entry checksum, refusing linked inputs."""
    entries=[]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as target:
        for relative in sorted(set(paths)):
            path=root/relative
            require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Linked/outside source input')
            if not path.is_file(): continue
            before=file_sha(path); target.write(path, relative)
            require(file_sha(path)==before, 'Source changed during snapshot: '+relative)
            entries.append(receipt(path,root))
    return entries


def model_receipt(checkpoint):
    import torch
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.recovery.utility import calibration_ready
    from brass_ai.learning.seat_descriptors import CONTRACT as DESCRIPTORS
    checkpoint=Path(checkpoint).resolve(); bundle=checkpoint.parent
    spec=importlib.util.spec_from_file_location('migration_test_nn_export', SOURCE_ROOT/'scripts/export-test-nn.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    manifest=exporter.verify_bundle(bundle)
    require(file_sha(checkpoint)==manifest['checkpoint']['sha256'] and checkpoint.stat().st_size==manifest['checkpoint']['bytes'],
            'Restored checkpoint differs from export receipt')
    model,raw=load_checkpoint(checkpoint,device='cpu'); model.eval()
    require(raw['model_id']==manifest['recorded_checkpoint']['model_id'], 'Checkpoint model identity differs')
    state=read_json(bundle/'state.json'); ready=calibration_ready(model)
    value=dict(**receipt(checkpoint,SOURCE_ROOT), role='temporary migration test model; not final NN',
               generation=manifest['recorded_checkpoint']['generation'], captured_run_generation=manifest['committed_generation'],
               model_id=raw['model_id'], parameter_count=model.parameter_count, model_config=raw['model_config'],
               compatibility=raw['compatibility'], reward=model.config.utility,
               descriptor_contract=DESCRIPTORS,descriptor_shape=['B',4,34],descriptor_default='unknown',
               value_readiness=dict(ready=ready, receipt=raw['training'].get('utility_calibration'),
                   admission='mechanical tests and timing only' if not ready else 'calibrated test checkpoint',
                   reason=None if ready else 'Existing calibration_ready(model) returned false; no receipt is fabricated'),
               training_metadata_sha256=digest(raw['training']), origin=state['origin'], imitation=state['imitation'],
               imitation_source=state['signature']['imitation_source'],
               source_signature_sha256=digest(state['signature']),
               bundle_files=[receipt(bundle/n,SOURCE_ROOT) for n in ('manifest.json','config.json','state.json','source.json')],
               export=dict(status='pending L08/L09', reason='Bundle is a PyTorch checkpoint, not an ONNX export'),
               frozen_opponents=dict(selection='same supplied frozen checkpoint', model_ids=[raw['model_id']],
                   scope='neutral mechanical benchmark; no unavailable lab population is substituted',
                   exploration=.05, temperature=1.0, focus='neutral', exploration_version=2),
               checks=dict(bundle_hashes=True, strict_checkpoint_loader=True, architecture=True,
                           exact_source_compatibility=True, finite_weights=True, model_digest=True,
                           weights_unchanged=True), pytorch_version=torch.__version__)
    return value


def freeze(root, out, checkpoint, upstream, *, save_root=None):
    root,out,upstream=Path(root).resolve(),Path(out).resolve(),Path(upstream).resolve()
    require(out.is_relative_to(root/'artifacts/rust-backend'), 'Evidence must be under artifacts/rust-backend')
    require(not (out/'baseline.json').exists(), 'Baseline is immutable; choose a new run directory')
    out.mkdir(parents=True,exist_ok=True)
    model=model_receipt(checkpoint)
    # Pin historical rules/search/features/configs, not generated evidence or
    # the new harness itself; later Rust additions must not alter this oracle.
    paths=git(root,'ls-files','-z',binary=True).decode().split('\0')
    references=[p for p in paths if p and not p.startswith(('src/brass_ai/rust_backend/','configs/contracts/rust-backend/'))
        and p!='configs/references/rust-engine.lock.json' and (p.startswith(('src/brass_ai/','src/BrassHost/',
        'src/BrassSearchHost/','src/BrassTrainingHost/','src/BrassLocalAI/','configs/','training/runtime/'))
        or p.startswith('requirements-') or p in ('scripts/build-training-host.py','scripts/build-search-host.py',
        'scripts/build-local-scoring-host.py','scripts/export-test-nn.py','tests/FeatureEngineering/fe_fixtures.py'))]
    files=snapshot_files(root,references,out/'reference-source.zip')
    source=dict(commit=git(root,'rev-parse','HEAD'), status=git(root,'status','--short'), files=files, digest=digest(files),
                archive=receipt(out/'reference-source.zip',root))
    upstream_paths=git(upstream,'ls-files','-z',binary=True).decode().split('\0')
    selected=[p for p in upstream_paths if p and (p.startswith('src/') or p in ('Cargo.toml','Cargo.lock','README.md','LICENSE'))]
    untracked=git(upstream,'ls-files','--others','--exclude-standard','-z',binary=True).decode().split('\0')
    selected += [p for p in untracked if p.startswith(('src/','scripts/cpu_capacity_benchmark/')) and Path(p).suffix in ('.py','.rs','.toml','.md')]
    rust_files=snapshot_files(upstream,selected,out/'upstream-source.zip')
    patch=git(upstream,'diff','--binary','HEAD',binary=True);(out/'upstream-local.patch').write_bytes(patch)
    license_bytes=git(upstream,'show','HEAD:LICENSE',binary=True);(out/'upstream-LICENSE-at-HEAD').write_bytes(license_bytes)
    lock=dict(format='rust-engine-reference-lock/1.0.0', repository=git(upstream,'remote','get-url','origin'),
              checkout=str(upstream), commit=git(upstream,'rev-parse','HEAD'), tree=git(upstream,'rev-parse','HEAD^{tree}'),
              status=git(upstream,'status','--short'), files=rust_files, files_digest=digest(rust_files),
              local_patch=receipt(out/'upstream-local.patch',root), archive=receipt(out/'upstream-source.zip',root),
              license_at_commit=dict(path='LICENSE',sha256=hashlib.sha256(license_bytes).hexdigest(),
                                     evidence=receipt(out/'upstream-LICENSE-at-HEAD',root),
                                     working_copy_present=(upstream/'LICENSE').exists()),
              reuse_status='reference frozen; no Rust source integrated by L01/L02',
              notice='The checkout deletes LICENSE locally. Preserve the committed notice when importing source; this deletion is not a licensing grant.')
    write_json(out/'rust-engine.lock.json',lock)
    lock_path=root/'configs/references/rust-engine.lock.json'
    if lock_path.exists():
        previous=read_json(lock_path)
        require(all(previous[k]==lock[k] for k in ('repository','commit','tree','files_digest'))
                and previous['local_patch']['sha256']==lock['local_patch']['sha256']
                and previous['license_at_commit']['sha256']==lock['license_at_commit']['sha256'],
                'Existing Rust source lock differs; review a new lock rather than overwriting history')
    else: write_json(lock_path,lock)
    installed=read_json(root/'artifacts/patch-installation.json'); fallback=[]
    managed=root/'BrassBirminghamUnity_Data/Managed'; recovery=out/'recovery'; recovery.mkdir()
    for item in installed['files']:
        path=managed/item['name']; require(file_sha(path)==item['sha256'].lower(),'Installed fallback changed')
        shutil.copyfile(path,recovery/item['name']); require(file_sha(path)==file_sha(recovery/item['name']),'Fallback backup differs')
        fallback.append(receipt(recovery/item['name'],root))
    shutil.copyfile(root/'artifacts/patch-installation.json',recovery/'patch-installation.json')
    stock=Path(installed['stockBackup']);require(file_sha(stock)==installed['stockSha256'].lower(),'Stock rollback differs')
    shutil.copyfile(stock,recovery/'stock-Assembly-CSharp.dll')
    saves=[]
    if save_root is not None:
        save_root=Path(save_root).resolve()
        # Exclude logs/telemetry; retain only original game saves and game history.
        paths=[p.relative_to(save_root).as_posix() for directory in ('games','GameHistoryStatistics')
               for p in (save_root/directory).rglob('*') if p.is_file()]
        saves=snapshot_files(save_root,paths,recovery/'original-saves.zip')
    require(save_root is not None,'Original save recovery baseline must be supplied (empty is valid)')
    write_json(recovery/'save-inventory.json',dict(root=str(save_root),files=saves,never_modified=True))
    from brass_ai.recovery.host_runtime import training_host
    runtime=training_host(root,'search')
    from brass_ai.host import RulesHost
    with RulesHost(runtime) as host:
        host.reset(731001); require(host.enable_fastpath(),'Missing fused reference host')
    runtime_manifest=read_json(runtime/'runtime.json')
    write_json(out/'reference-runtime.json',dict(path=str(runtime),manifest=runtime_manifest,
                files=[receipt(runtime/f['name'],root) for f in runtime_manifest['files']]))
    targets=dict(fixed_work_speedup_min=2.0,real_nn_trajectories_speedup_min=1.25,critical_group_regression_max=.05,
                 response_ready_p95_seconds=4.05,response_ready_max_seconds=4.15,
                 minimum_paired_repetitions=5,benchmark_seconds=4,
                 memory=dict(ai_process_peak_mib=2048,post_retirement_growth_mib=64,maximum_pending_per_session=4),
                 zero_completion='retain absolute counts and fallback rates; undefined ratios are not dropped or replaced with epsilon',
                 timing='one absolute ordinary-turn deadline shared by both sessions, all staged choices and both actions; response retirement separately measured')
    baseline=dict(format='rust-local-baseline/1.0.0',created_utc=datetime.now(timezone.utc).isoformat(),
        reference_source=source,upstream_lock=receipt(out/'rust-engine.lock.json',root),model=model,
        rules=dict(profile='steam-brass-patch-1.1.0',players=4,mode='full Canal and Rail',
                   unsupported=['2 players','3 players','intro/tutorial','online'],
                   deployment='research rules oracle is distinct from installed 1.4.34; live parity is L13-L15'),
        installed=dict(receipt=installed,backup=fallback,stock_backup=receipt(recovery/'stock-Assembly-CSharp.dll',root),
                       saves=receipt(recovery/'original-saves.zip',root),save_inventory=receipt(recovery/'save-inventory.json',root),
                       release_archive='artifacts/patch-releases/'+installed['version']),
        execution=dict(settings_ms=dict(easy=4000,mid=8000,hard=10000,max=12000),default='hard',
                       benchmark_ms=4000,sessions=2,threads_per_session=1,torch_intra_op=1,torch_inter_op=1,
                       ort_intra_op=1,ort_inter_op=1,ort_execution='sequential',cross_session_batching=False,
                       legacy=dict(schema='local-ai-execution/1.2.0',readable=['1.0.0','1.1.0','1.2.0'],
                           named=dict(easy=4000,mid=8000,hard=10000),unnamed_ms=10000,signed_bytes='preserved')),
        environment=dict(platform=platform.platform(),python=sys.version,executable=sys.executable,
                         logical_cpus=os.cpu_count(),packages={name:importlib.metadata.version(name)
                          for name in ('torch','numpy','onnxruntime','open_spiel','jsonschema')}),
        performance_targets=targets,reference_runtime=receipt(out/'reference-runtime.json',root),
        status='frozen for user-authorized mechanical testing; live readiness pending',
        unresolved=['Test checkpoint has no utility calibration receipt; no strength/live/teacher admission',
                    'ONNX export and native feature parity belong to L08/L09',
                    'Full original IL artifacts remain on lab; transferred state records lineage receipts only'])
    write_json(out/'baseline.json',baseline)
    write_json(out/'local/progress.json',dict(format='rust-migration-progress/1.0.0',
        steps={f'L{i:02}':dict(status='pending') for i in range(1,16)},
        baseline=receipt(out/'baseline.json',root),commands=[],unresolved=baseline['unresolved']))
    return baseline


def verify_reference(baseline, root=SOURCE_ROOT):
    provenance_drift=[]
    for item in baseline['reference_source']['files']:
        actual=file_sha(Path(root)/item['path'])
        if actual!=item['sha256'] and item['path']=='scripts/export-test-nn.py':
            # This helper was executed ONLY during L01 bundle verification. It
            # is not imported by capture/replay/inference/search. Keep its exact
            # frozen bytes in the verified archive when unrelated export work
            # continues in the shared checkout; never relax an oracle source.
            archived=baseline['reference_source']['archive'];archive=Path(root)/archived['path']
            require(file_sha(archive)==archived['sha256'],'Frozen source archive changed')
            with zipfile.ZipFile(archive) as stream:
                require(hashlib.sha256(stream.read(item['path'])).hexdigest()==item['sha256'],
                        'Frozen bundle-verifier bytes changed')
            provenance_drift.append(dict(path=item['path'],frozen_sha256=item['sha256'],current_sha256=actual,
                disposition='Archived freeze-time bundle verifier preserved; not executed by this operation'))
            continue
        require(actual==item['sha256'],'Frozen reference source changed: '+item['path'])
    require(file_sha(Path(root)/baseline['model']['path'])==baseline['model']['sha256'],'Test weights changed')
    for item in baseline['model']['bundle_files']:
        require(file_sha(Path(root)/item['path'])==item['sha256'],'Test bundle metadata changed: '+item['path'])
    return provenance_drift
