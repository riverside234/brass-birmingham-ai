"""Short fixed-reply allocation comparison on the exact Unity Mono runtime.

Embedding permits allocation instrumentation before startup; this is not a
claim to measure Unity presentation/engine allocation or uninstrumented latency.
"""
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
S=runpy.run_path(str(ROOT/'scripts/qualify-rust-release.py'))
read,write,sha,pin,run=(S[k] for k in ('read','write','sha','pin','run'))
OUT=ROOT/'artifacts/rust-backend/20261026-local-release'/('allocation' if len(sys.argv)<2 else sys.argv[1])
if __name__=='__main__':
    assert OUT.resolve().parent==ROOT/'artifacts/rust-backend/20261026-local-release'
    OUT.mkdir(exist_ok=False)
    build=ROOT/'artifacts/rust-backend/20261026-local-release/candidate'
    managed=OUT/'Managed';shutil.copytree(build/'Managed',managed)
    probe=managed/'AllocationBench.exe'
    sources=[ROOT/'tests/RustBackendLocal'/n for n in ('AllocationBench.cs','ValidationBench.cs')]
    rsp=OUT/'compile.rsp'
    rsp.write_text('\n'.join(['/nologo','/target:exe','/main:AllocationBench','/nostdlib+','/langversion:7.3','/optimize+','/deterministic+',f'/out:"{probe}"']+
        [f'/reference:"{p}"' for p in managed.glob('*.dll')]+[f'"{p}"' for p in sources]),encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'),'@'+str(rsp)],check=True)
    host=managed/'allocation-host.exe'
    subprocess.run([str(ROOT/'artifacts/native-toolchain/llvm-mingw-20261020-msvcrt-x86_64/bin/gcc.exe'),'-O2',
        str(ROOT/'tests/RustBackendLocal/AllocationHost.c'),'-o',str(host)],check=True)
    rows=read(ROOT/'artifacts/rust-backend/20261026-l14-study-v2/validation-fixtures.json')
    fixtures=[next(r for r in rows if r['input']['view']['context']['era']==era) for era in ('CanalEra','RailEra')]
    write(OUT/'fixtures.json',fixtures)
    mono=ROOT/'MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll'
    run(OUT,'probe',[host,mono,managed,ROOT/'MonoBleedingEdge/etc',probe,OUT/'fixtures.json',OUT/'result.json'],cwd=managed,timeout=60)
    result=read(OUT/'result.json');assert result['passed'] and result['counter_self_test_main_and_worker']
    totals={a:sum(sum(r['bytes'].values()) for r in result['rows'] if r['arm']==a) for a in ('baseline','candidate')}
    write(OUT/'receipt.json',dict(passed=True,source=[pin(p) for p in sources]+[pin(ROOT/'tests/RustBackendLocal/AllocationHost.c'),pin(Path(__file__))],
        host=pin(host),probe=pin(probe),mono=pin(mono),candidate=pin(build/'build-manifest.json'),result=pin(OUT/'result.json'),
        total_measured_object_bytes=totals,reduction_fraction=1-totals['candidate']/totals['baseline'],
        scope=result['scope'],full_unity_process_allocation_available=False))
    print(totals,flush=True)
