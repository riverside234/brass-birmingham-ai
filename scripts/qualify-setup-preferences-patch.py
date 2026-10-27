"""Seal the menu patch after bounded restart/UI and runtime-preservation checks."""
from pathlib import Path
import runpy
import shutil

b = runpy.run_path(str(Path(__file__).with_name('build-setup-preferences-patch.py')))
ROOT, BASE, OUT = b['ROOT'], b['BASE'], b['OUT']
sha, read, write = b['sha'], b['read'], b['write']

def pins(directory):
    return [dict(path=p.relative_to(directory).as_posix(), sha256=sha(p)) for p in sorted(directory.rglob('*')) if p.is_file()]

def main():
    assert not (OUT/'qualification.json').exists()
    manifest = read(OUT/'build-manifest.json'); baseline = read(BASE/'build-manifest.json')
    before = {p['path']:p['sha256'] for p in baseline['sourcePins']}
    changed = []
    for pin in manifest['sourcePins']:
        assert sha(OUT/'runtime'/pin['path']) == pin['sha256']
        if before.get(pin['path']) != pin['sha256']:
            changed.append(pin['path']); assert sha(ROOT/pin['path']) == pin['sha256']
    assert set(changed) == set(b['SCOPE'])
    assert pins(OUT/'BrassNative') == pins(ROOT/'BrassNative')
    preservation = read(OUT.parent/'preserved-runtime.json'); assert preservation['passed']
    assert next(p for p in preservation['assemblies'] if p['file']=='BrassLegacyAI.dll')['embeddedResources']==48
    evidence=OUT/'evidence';evidence.mkdir()
    for phase,minimum in [('write-final',808),('read-final',774)]:
        folder=OUT.parent/phase; report=read(folder/'unity.json'); build=read(folder/'test-build.json')
        assert report['passed'] and report['checks']>=minimum and not report['nativeErrors']
        assert build['candidateManifestSha256']==sha(OUT/'build-manifest.json')
        assert build['testDllSha256']==sha(folder/'LocalAiUnityTests.dll')
        for pin in build['sources']: assert sha(folder/'test-source'/pin['path'])==pin['sha256']
        shutil.copytree(folder,evidence/phase)
    shutil.copyfile(OUT.parent/'preserved-runtime.json',evidence/'preserved-runtime.json')
    shutil.copyfile(ROOT/'artifacts/card-tracker-player/artifacts/patch-build/setup-preferences.json',evidence/'restart-preferences.json')
    images=evidence/'images';images.mkdir()
    for p in (ROOT/'artifacts/card-tracker-player/artifacts/patch-build').glob('setup-*-largest.png'): shutil.copyfile(p,images/p.name)
    for name in ('scripts/build-setup-preferences-patch.py','scripts/qualify-setup-preferences-patch.py','scripts/install-setup-preferences-patch.ps1','tests/BrassLocalAI/SetupPreferencesPreservation.cs'):
        target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(OUT/'build-manifest.json',evidence/'tested-build-manifest.json')
    manifest['local_release_qualified']=True;write(OUT/'build-manifest.json',manifest)
    write(OUT/'qualification.json',dict(format='brass-setup-preferences/1.0.0',passed=True,version=manifest['version'],
        local_release_qualified=True,scope='local-game-g890/menu-only',build_manifest_sha256=sha(OUT/'build-manifest.json'),
        files=[p for p in pins(OUT) if p['path'].startswith(('Managed/','runtime/','BrassNative/','evidence/')) or p['path']=='build-manifest.json'],
        native_files=pins(OUT/'BrassNative'),changed_sources=changed,ui_write_checks=808,ui_restart_checks=774,
        preference_storage='PlayerPrefs; isolated tests transfer exact preference bytes across fresh Unity processes',
        final_model=False,lab_deployed=False))
    print('Qualified 1.4.38 menu-only patch; runtime preservation and short UI/restart checks passed')

if __name__=='__main__':main()
