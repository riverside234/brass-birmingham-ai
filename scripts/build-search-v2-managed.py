"""Isolated 1.4.38 overlay: projected-state guard and experimental native package."""
from pathlib import Path
import argparse
import runpy
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
T=runpy.run_path(str(ROOT/'scripts/search-v2-tools.py'))
read,write,sha=(T[k] for k in ('read','write','sha'))
BASE=ROOT/'artifacts/setup-preferences/candidate'
OUT=T['OUT']/'local-candidate'
METHOD='search_v2_focus_only'


def main():
    assert not OUT.exists()
    original=read(BASE/'build-manifest.json');assert original['version']=='1.4.38-rust-local'
    runtime=OUT/'runtime';managed=OUT/'Managed';managed.mkdir(parents=True)
    for pin in original['sourcePins']:
        src=BASE/'runtime'/pin['path'];dst=runtime/pin['path'];assert sha(src)==pin['sha256'].lower()
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    scope=['src/BrassLocalAI/Native/NativeInput.cs','src/BrassLocalAI/Unity/LocalTurnBridge.cs']
    for name in scope:shutil.copyfile(ROOT/name,runtime/name)
    for src in (ROOT/'BrassBirminghamUnity_Data/Managed').glob('*.dll'):
        if not any(w in src.name for w in ('Tests','Regression')):shutil.copyfile(src,managed/src.name)
    args=['/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{managed}/BrassLegacyAI.dll"']
    args += [f'/reference:"{p}"' for p in managed.glob('*.dll') if p.name!='BrassLegacyAI.dll']
    for folder in ('BrassLocalAI','BrassMaskedSearch','BrassPersonalityGame'):
        args += [f'"{p}"' for p in sorted((runtime/'src'/folder).rglob('*.cs'))]
    for key in ('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'):
        for suffix in ('v2','v3','v3-masked','v3-masked-strength','v3-masked-era','v3-masked-industry','v3-masked-economy'):
            args.append(f'/resource:"{runtime}/configs/legacy-ai/{key}-{suffix}.json",BrassLegacyAI.Profiles.{key}-{suffix}.json')
        args.append(f'/resource:"{runtime}/configs/legacy-ai/archive/1.4.0/{key}-v3.json",BrassLegacyAI.Profiles.{key}-v3-1.4.0.json')
    rsp=OUT/'legacy.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'),'@'+str(rsp)],check=True)
    package=OUT/'BrassNative/alpha-g890';shutil.copytree(T['OUT']/'control/BrassNative/alpha-g890',package)
    shutil.copyfile(ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-local.exe',package/'brass-ai-local.exe')
    policy=read(package/'policy.json');policy['method']=METHOD;write(package/'policy.json',policy)
    build=read(package/'build.json');build['source_sha256']=T['BASE']['native_fingerprint']()
    with zipfile.ZipFile(package/'native-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((ROOT/'native').rglob('*')):
            if p.is_file() and p.suffix in ('.rs','.toml','.lock','.json'):archive.write(p,p.relative_to(ROOT))
    (package/'README.txt').write_text(f'Isolated {METHOD} study candidate. Not installed or release qualified.\n'
        'Frozen g890/ONNX Runtime CPU; two independent sessions, one compute thread each.\n'
        + ('Conditional descendant reuse enabled. ' if METHOD=='search_v2' else 'No statistical subtree reuse. ')
        + 'Corresponding Rust source: native-source.zip.\n',encoding='utf-8')
    for pin in build['files']:pin['sha256']=sha(package/pin['path'])
    write(package/'build.json',build)
    manifest=dict(original,sourceScope=scope,local_release_qualified=False,search_method=METHOD,
        native_source_sha256=build['source_sha256'],native_build_sha256=sha(package/'build.json'),
        sourcePins=[dict(path=p.relative_to(runtime).as_posix(),sha256=sha(p)) for p in sorted(runtime.rglob('*')) if p.is_file()],
        files=[dict(name=n,sha256=sha(managed/n)) for n in ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')])
    write(OUT/'build-manifest.json',manifest)
    print(OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--method',choices=['search_v2_focus_only','search_v2'],default=METHOD);a=p.parse_args();METHOD=a.method
    OUT=a.output.resolve();assert OUT.is_relative_to(ROOT/'artifacts/search-v2')
    main()
