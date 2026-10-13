"""Isolated native audit; copies dependencies without installing any helpers."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def run(build, output, source='NativeP56Audit'):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new isolated native audit directory')
    output.mkdir(parents=True)
    for dependency in (ROOT / 'BrassBirminghamUnity_Data/Managed').glob('*.dll'):
        shutil.copyfile(dependency, output / dependency.name)
    for dependency in (Path(build) / 'Managed').glob('*.dll'):
        shutil.copyfile(dependency, output / dependency.name)
    args = ['/nologo', '/target:exe', '/langversion:7.3', '/nostdlib+',
            f'/out:"{output}/{source}.exe"']
    args += [f'/reference:"{p}"' for p in output.glob('*.dll')
             if 'Tests' not in p.name and 'Regression' not in p.name]
    args += [f'"{ROOT}/tests/Personality/{source}.cs"']
    response = output / 'test.rsp'
    response.write_text('\n'.join(args), encoding='utf-8-sig')
    compiler = ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    subprocess.run([str(compiler), '@' + str(response)], check=True)
    subprocess.run([str(output / (source + '.exe')), str(output / 'report.json'),
                    '--brass-local-ai-test'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', choices=['NativeP56Audit', 'NativeP56SaveAudit'], default='NativeP56Audit')
    args = parser.parse_args()
    run(args.build, args.output, args.source)
