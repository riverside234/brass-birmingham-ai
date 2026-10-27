"""Build an isolated local game package and self-contained Alpha Test runtime.

No installation, lab code or historical qualification is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import runpy
import zipfile

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/model'
BASE=ROOT/'artifacts/patch-releases/1.4.34'
SCOPE=[
 'src/BrassLocalAI/Policy/AiSelectionCatalog.cs','src/BrassLocalAI/Policy/PlayerSelectionDraft.cs',
 'src/BrassLocalAI/Policy/LocalGameState.cs','src/BrassLocalAI/Unity/AiSelectionPanel.cs',
 'src/BrassLocalAI/Unity/AiSetupRow.cs','src/BrassLocalAI/Unity/LocalMenu.cs','src/BrassLocalAI/Unity/LocalTurnBridge.cs',
 'src/BrassPatch/Liquidation/LiquidationFixes.cs',
 'src/BrassPersonalityGame/PublicHistory.cs','src/BrassPersonalityGame/ActionHistory.cs','src/BrassPersonalityGame/LiveChoices.cs']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def package_sources(package):
    notices=package/'notices';notices.mkdir(exist_ok=True)
    support=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
    env,cargo=support['tool_environment']()
    metadata=json.loads(subprocess.check_output([str(cargo),'metadata','--offline','--format-version','1'],cwd=ROOT/'native',env=env))
    dependencies=[]
    for crate in metadata['packages']:
        base=Path(crate['manifest_path']).parent;files=[]
        for source in sorted(base.iterdir()):
            if source.is_file() and source.name.upper().startswith(('LICENSE','COPYING','NOTICE','AUTHORS')):
                target=notices/'crates'/(crate['name']+'-'+crate['version'])/source.name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                files.append(dict(path=target.relative_to(package).as_posix(),sha256=sha(target)))
        dependencies.append(dict(name=crate['name'],version=crate['version'],license=crate['license'],files=files))
    write(notices/'crates.json',dependencies)
    for name in ('LICENSE','NOTICE.md','UPSTREAM.json'):
        shutil.copyfile(ROOT/'native/vendor/brass-upstream-rules'/name,notices/('upstream-'+name))
    for name in ('LICENSE','ThirdPartyNotices.txt'):
        shutil.copyfile(ROOT/'artifacts/r2-pythondeps/onnxruntime'/name,notices/('onnxruntime-'+name))
    # Preserve corresponding local Rust sources and build inputs with the
    # binary; Cargo.lock names the external crate versions exactly.
    with zipfile.ZipFile(package/'native-source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted((ROOT/'native').rglob('*')):
            if source.is_file() and not any(p in ('target','.git') for p in source.parts):
                archive.write(source,source.relative_to(ROOT).as_posix())
    (package/'README.txt').write_text(
        'Alpha Test g890 local Windows x64 runtime, L13. Not an L14/L15 release qualification.\n'
        'Two independent CPU sessions; one compute thread each. No Python inference service.\n'
        'Temporary g890 model; utility calibration remains null. policy.json records local acceptance.\n'
        'Build sources: native-source.zip; locked Rust toolchain and crates are in native/.\n'
        'From native/: cargo build --locked --release --bin brass-ai-local --target x86_64-pc-windows-gnu\n'
        'Upstream AGPL-3.0 text/provenance and ONNX Runtime notices are included in notices/.\n'
        'No proprietary Steam game assembly is included in this native runtime package.\n'
        'Create ../disabled to stop NN tables and remove Alpha Test from setup.\n'
        'Disable/rollback preserves NN saves; they cannot be resumed by V4.\n',encoding='utf-8')
def build(out,binary,background_validation=True):
    out=out.resolve();assert out.is_relative_to(ROOT/'artifacts') and not (out/'qualification.json').exists()
    out.mkdir(parents=True,exist_ok=True);runtime=out/'runtime';managed=out/'Managed';managed.mkdir(exist_ok=True)
    baseline=read(BASE/'build-manifest.json');assert read(BASE/'validation.json')['passed']
    for pin in read(BASE/'archive-manifest.json')['files']:
        p=(BASE/pin['path']).resolve();assert p.is_relative_to(BASE) and sha(p)==pin['sha256'].lower()
    for pin in baseline['sourcePins']:
        src=BASE/pin['path'];dst=runtime/pin['path'];assert sha(src)==pin['sha256'].lower()
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    scope=SCOPE+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src/BrassLocalAI/Native').glob('*.cs')]
    for name in scope:
        dst=runtime/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    for src in (ROOT/'BrassBirminghamUnity_Data/Managed').glob('*.dll'):
        if not any(s in src.name for s in ('Tests','Regression')):shutil.copyfile(src,managed/src.name)
    for name in ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'):shutil.copyfile(BASE/'Managed'/name,managed/name)
    csc=ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    def compile(name,sources,resources=()):
        rsp=out/(name+'.rsp')
        args=['/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{managed/name}"']
        if name=='BrassLegacyAI.dll' and not background_validation:args.append('/define:BRASS_L14_BASELINE_VALIDATION')
        args += [f'/reference:"{p}"' for p in managed.glob('*.dll') if p.name!=name and 'Tests' not in p.name and 'Regression' not in p.name and (name!='BrassPatch.dll' or p.name!='BrassLegacyAI.dll')]
        args += [f'"{p}"' for p in sources]+list(resources)
        rsp.write_text('\n'.join(args),encoding='utf-8-sig');subprocess.run([str(csc),'@'+str(rsp)],check=True,cwd=ROOT)
    compile('BrassPatch.dll',sorted((runtime/'src/BrassPatch').rglob('*.cs')))
    resources=[]
    for key in ('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'):
        for suffix in ('v2','v3','v3-masked','v3-masked-strength','v3-masked-era','v3-masked-industry','v3-masked-economy'):
            resources.append(f'/resource:"{runtime}/configs/legacy-ai/{key}-{suffix}.json",BrassLegacyAI.Profiles.{key}-{suffix}.json')
        resources.append(f'/resource:"{runtime}/configs/legacy-ai/archive/1.4.0/{key}-v3.json",BrassLegacyAI.Profiles.{key}-v3-1.4.0.json')
    sources=sorted((runtime/'src/BrassLocalAI').rglob('*.cs'))+sorted((runtime/'src/BrassMaskedSearch').glob('*.cs'))+sorted((runtime/'src/BrassPersonalityGame').glob('*.cs'))
    compile('BrassLegacyAI.dll',sources,resources)
    package=out/'BrassNative/alpha-g890';(package/'model').mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','model.json','model.onnx','onnxruntime.dll'):shutil.copyfile(MODEL/name,package/'model'/name)
    shutil.copyfile(binary,package/'brass-ai-local.exe')
    case=read(ROOT/'artifacts/rust-backend/20261025-l10-capture/brass-000-request.json')['cases'][0]
    warm=case['input'];warm.pop('shuffle_tape',None);write(package/'warmup.json',warm)
    config=case['config'];config.update(simulations=64,trace=False)
    model=read(MODEL/'manifest.json')
    acceptance=dict(version='brass-local-utility-acceptance/1.0.0',scope='part1-local-alpha-test',decision='user-2026-10-25',
        model_id=model['model_id'],export_sha256=model['export_sha256'],checkpoint_sha256=model['checkpoint_sha256'],utility_calibration=None,final_model=False)
    policy=dict(protocol='brass-local-native/1.0.0',backend='brass-gumbel-is-mcts/alpha-g890-v1',model_manifest_sha256=sha(MODEL/'manifest.json'),
        acceptance=acceptance,config=config,startup_ns=7262519,startup_source='L12 singleton calibration maximum; 5ec4f608931e09c0a3aa4c64bcb6f5355fff8415fa76ee815f6a344c0c59ad7e',warmup_sha256=sha(package/'warmup.json'))
    write(package/'policy.json',policy)
    identity=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))['native_fingerprint']()
    names=['brass-ai-local.exe','policy.json','warmup.json','model/manifest.json','model/model.json','model/model.onnx','model/onnxruntime.dll']
    write(package/'build.json',dict(format='brass-local-package/1.0.0',backend=policy['backend'],scope='local-alpha-test',target='x86_64-pc-windows-gnu',
        source_sha256=identity,files=[dict(path=n,sha256=sha(package/n)) for n in names]))
    package_sources(package)
    pins=[dict(path=p.relative_to(runtime).as_posix(),sha256=sha(p)) for p in sorted(runtime.rglob('*')) if p.is_file()]
    write(out/'build-manifest.json',dict(version='1.4.35-alpha-test',baseVersion='1.4.34',stockSha256=baseline['stockSha256'],sourceScope=scope,
        sourcePins=pins,files=[dict(name=n,sha256=sha(managed/n)) for n in ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')],
        native_build_sha256=sha(package/'build.json'),native_source_sha256=identity,turnOptionsMs=[4000,8000,10000,12000],sessions=2,threads_per_session=1,
        model_acceptance=acceptance,measured_readiness=None,local_release_qualified=False,
        background_validation_candidate=background_validation,save_policy='human-end-checkpoint/1.0.0'))
    print('Built isolated local candidate',out,'background_validation='+str(background_validation),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--baseline-validation',action='store_false',dest='background_validation',help='Comparison only: retain the L13 validation call chain')
    a=p.parse_args();build(a.out,a.binary,a.background_validation)
