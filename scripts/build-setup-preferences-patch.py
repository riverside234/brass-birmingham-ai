"""Build the menu-only 1.4.38 overlay, preserving installed native search config."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/city-cancel-fix/candidate'
OUT = ROOT / 'artifacts/setup-preferences/candidate'
SCOPE = ['src/BrassLocalAI/Unity/LocalMenu.cs', 'src/BrassLocalAI/Unity/AiSelectionPanel.cs',
         'src/BrassLocalAI/Policy/LocalSetupPreferences.cs']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, value): Path(p).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def main():
    assert not (OUT/'qualification.json').exists(), 'Qualified candidates are immutable'
    baseline = read(BASE/'build-manifest.json'); installed = read(ROOT/'artifacts/patch-installation.json')
    assert installed['version'] == baseline['version'] == '1.4.37-rust-local'
    assert read(BASE/'qualification.json')['passed']
    runtime, managed = OUT/'runtime', OUT/'Managed'; managed.mkdir(parents=True, exist_ok=True)
    for pin in baseline['sourcePins']:
        src, dst = BASE/'runtime'/pin['path'], runtime/pin['path']
        assert sha(src) == pin['sha256'].lower()
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    for name in SCOPE:
        dst = runtime/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dst)
    for src in (ROOT/'BrassBirminghamUnity_Data/Managed').glob('*.dll'):
        if not any(word in src.name for word in ('Tests', 'Regression')): shutil.copyfile(src, managed/src.name)
    for pin in installed['files']:
        assert sha(managed/pin['name']) == pin['sha256'].lower()
        assert next(p['sha256'] for p in baseline['files'] if p['name']==pin['name']) == pin['sha256']
    args = ['/nologo', '/target:library', '/langversion:7.3', '/nostdlib+', '/deterministic+', '/optimize+', f'/out:"{managed/"BrassLegacyAI.dll"}"']
    args += [f'/reference:"{p}"' for p in managed.glob('*.dll') if p.name != 'BrassLegacyAI.dll']
    for folder in ('BrassLocalAI', 'BrassMaskedSearch', 'BrassPersonalityGame'):
        args += [f'"{p}"' for p in sorted((runtime/'src'/folder).rglob('*.cs'))]
    for key in ('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'):
        for suffix in ('v2','v3','v3-masked','v3-masked-strength','v3-masked-era','v3-masked-industry','v3-masked-economy'):
            args.append(f'/resource:"{runtime}/configs/legacy-ai/{key}-{suffix}.json",BrassLegacyAI.Profiles.{key}-{suffix}.json')
        args.append(f'/resource:"{runtime}/configs/legacy-ai/archive/1.4.0/{key}-v3.json",BrassLegacyAI.Profiles.{key}-v3-1.4.0.json')
    rsp = OUT/'legacy.rsp'; rsp.write_text('\n'.join(args), encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'), '@'+str(rsp)], check=True)
    shutil.copytree(ROOT/'BrassNative', OUT/'BrassNative', dirs_exist_ok=True)
    manifest = dict(baseline)
    manifest.update(version='1.4.38-rust-local', baseVersion=installed['version'], baseFiles=installed['files'],
        base_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'), sourceScope=SCOPE, local_release_qualified=False,
        sourcePins=[dict(path=p.relative_to(runtime).as_posix(), sha256=sha(p)) for p in sorted(runtime.rglob('*')) if p.is_file()],
        files=[dict(name=n,sha256=sha(managed/n)) for n in ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')],
        native_build_sha256=sha(ROOT/'BrassNative/alpha-g890/build.json'),
        native_source_sha256=read(ROOT/'BrassNative/alpha-g890/build.json')['source_sha256'],
        searchKByAllowanceMs=installed['searchKByAllowanceMs'])
    write(OUT/'build-manifest.json', manifest)
    print('Built 1.4.38 menu preferences and NN tab candidate', flush=True)

if __name__ == '__main__': main()
