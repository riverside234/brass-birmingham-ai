"""Build isolated host optimizations over the exact installed 1.4.43 source.

The native worker, model, policies and search clocks are copied unchanged.
"""
from pathlib import Path
import argparse
import runpy
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
H = runpy.run_path(str(ROOT / 'scripts/build-nn-search-choice.py'))
sha, read, write, pins = (H[k] for k in ('sha', 'read', 'write', 'pins'))
BASE = ROOT / 'artifacts/nn-k32/candidate-v2'
STUDY = ROOT / 'artifacts/host-delivery/20261029'
SCOPE = [
    'src/BrassLocalAI/Native/NativeInput.cs',
    'src/BrassLocalAI/Native/NativeValidation.cs',
    'src/BrassLocalAI/Host/Model.cs',
    'src/BrassLocalAI/Host/Observation.cs',
    'src/BrassLocalAI/Host/Choices.cs',
    'src/BrassLocalAI/Host/Host.cs',
    'src/BrassLocalAI/Policy/HostExtensions.cs',
    'src/BrassLocalAI/Unity/LocalTurnBridge.cs',
    'src/BrassLocalAI/Unity/AiExecutionResources.cs',
    'src/BrassPersonalityGame/PublicHistory.cs',
]


def build(arm):
    out = STUDY / arm
    assert not out.exists(), 'Preserve prior evidence; choose a new study directory'
    old = read(BASE / 'build-manifest.json')
    installed = read(ROOT / 'artifacts/patch-installation.json')
    assert old['version'] == installed['version'] == '1.4.43-rust-local'
    assert installed['searchMethod'] == 'search_v1' and not installed['v205Enabled']
    assert sha(BASE / 'qualification.json') == installed['nnK32QualificationSha256'].lower()
    for pin in old['sourcePins']:
        assert sha(BASE / 'runtime' / pin['path']) == pin['sha256']
    for pin in old['files']:
        assert sha(BASE / 'Managed' / pin['name']) == pin['sha256']
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / pin['name']) == pin['sha256']
    assert pins(BASE / 'BrassNative') == old['nativeFiles'] == pins(ROOT / 'BrassNative')
    for name in ('runtime', 'Managed', 'BrassNative'):
        shutil.copytree(BASE / name, out / name)
    if arm == 'candidate':
        for name in SCOPE:
            shutil.copyfile(ROOT / name, out / 'runtime' / name)
    rsp = out / 'legacy.rsp'
    rsp.write_text((BASE / 'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE), str(out)), encoding='utf-8-sig')
    if arm == 'candidate':
        subprocess.run([str(ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'), '@' + str(rsp)], check=True)
    new = dict(old)
    new.update(version='1.4.43-host-delivery-' + arm, baseVersion=old['version'],
        sourceScope=SCOPE if arm == 'candidate' else [], sourcePins=pins(out / 'runtime'),
        files=[dict(name=p['name'], sha256=sha(out / 'Managed' / p['name'])) for p in old['files']],
        host_delivery_optimization=arm == 'candidate', installed=False,
        installed_baseline_receipt_sha256=sha(ROOT / 'artifacts/patch-installation.json'))
    write(out / 'build-manifest.json', new)
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=('baseline', 'candidate'))
    build(parser.parse_args().arm)
