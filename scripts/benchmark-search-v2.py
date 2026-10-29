"""Frozen, short search-v2/native control batches; no installed mutations.

Preserves the initial no-switch feasibility pilot and the subsequent cache-only
matrix as separate protocols. No statistical-reuse or strength claim is implied.
"""
import argparse
import copy
import ctypes
import gzip
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import time
from ctypes import wintypes

ROOT = Path(__file__).resolve().parents[1]
TOOLS = runpy.run_path(str(ROOT / 'scripts/search-v2-tools.py'))
sha, read, write, receipt = (TOOLS[k] for k in ('sha', 'read', 'write', 'receipt'))
OUT = TOOLS['OUT']
L10 = ROOT / 'artifacts/rust-backend/20261025-l10-capture'
CONFIG = ROOT / 'configs/search/search-v2-study.json'


def process(binary, source, target, log_path, watchdog):
    """Process CPU/peak memory include startup; native per-owner spans are separate."""
    memory_type = runpy.run_path(str(ROOT/'scripts/benchmark-rust-coordinator.py'))['Memory']
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*[ctypes.POINTER(wintypes.FILETIME)]*4]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(memory_type),wintypes.DWORD]
    began=time.perf_counter();peak=0
    with log_path.open('wb') as log:
        child=subprocess.Popen([str(binary),str(source),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while child.poll() is None:
            if time.perf_counter()-began>watchdog:
                child.kill();child.wait();raise TimeoutError('60-second batch watchdog')
            memory=memory_type();memory.cb=ctypes.sizeof(memory)
            if psapi.GetProcessMemoryInfo(child._handle,ctypes.byref(memory),memory.cb):peak=max(peak,memory.peak)
            time.sleep(.01)
        times=[wintypes.FILETIME() for _ in range(4)]
        ok=kernel.GetProcessTimes(child._handle,*[ctypes.byref(t) for t in times])
        cpu=sum(((t.dwHighDateTime<<32)+t.dwLowDateTime)/10_000_000 for t in times[2:]) if ok else None
    return dict(exit_code=child.returncode,wall_seconds=time.perf_counter()-began,cpu_seconds=cpu,
                peak_working_set_bytes=peak,binary=receipt(binary),input=receipt(source),
                output=receipt(target) if target.exists() else None)


def freeze():
    target = OUT / 'feasibility'
    target.mkdir(parents=True, exist_ok=False)
    binary = ROOT / 'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/gumbel-pair.exe'
    shutil.copyfile(binary, target / binary.name)
    settings = [('4s_k8', 4000, 8), ('8s_k8', 8000, 8), ('8s_k16', 8000, 16)]
    roots = []
    for index, category in [(3, 'Canal round 3; 413 legal choices; development/build/loan alternatives'),
                            (26, 'Rail round 1; 246 legal choices; network/resource alternatives')]:
        p = L10 / f'brass-{index:03d}-request.json'
        roots.append(dict(id=f'brass-{index:03d}', index=index, fixture=receipt(p), category=category,
                          partition='development', search_seed=7, world_seed=890))
    rows = []
    for ri, root in enumerate(roots):
        for ci, (setting, ms, k) in enumerate(settings):
            order = ['search_v1', 'search_v2_no_switch']
            if (ri + ci) % 2: order.reverse()
            rows.append(dict(batch=f'{root["id"]}-{setting}', root=root['id'], setting=setting,
                             allowance_ms=ms, k=k, order=order))
    protocol = dict(version='search-v2-study/2.0.0', stage='necessary-gate-feasibility',
        method_scope='Instrumented no-switch scheduler; NOT focused search or statistical reuse',
        roots=roots, batches=rows, sessions=2, threads_per_session=1,
        preparation_ns=20_000_000, minimum_support=2, batch_watchdog_seconds=60,
        decision='If neither diagnostic root has a stable, affordable complete first-action recommendation at two advancing owner checkpoints, stop before projection/reuse and report revise. Otherwise implement cache-only focus before the full matrix.',
        maximum_trials=12, search_allowance_seconds=80,
        warmup='One identical 8-trajectory count warmup per arm; reset turn/tree/cache after warmup',
        startup=read(ROOT / 'BrassNative/alpha-g890/policy.json')['startup_ns'],
        model=receipt(OUT / 'control/BrassNative/alpha-g890/model/manifest.json'),
        binary=receipt(target / binary.name), control=receipt(OUT / 'control/manifest.json'),
        native_source_sha256=TOOLS['BASE']['native_fingerprint'](),
        candidate_sources=[receipt(ROOT / p) for p in [
            'native/brass-ai-core/src/search/session/mod.rs',
            'native/brass-ai-core/src/search/session/cooperative.rs',
            'native/brass-ai-core/src/search/session/clock.rs',
            'native/brass-ai-core/src/search/session/focus.rs',
            'native/brass-ai-worker/src/focus.rs',
            'native/brass-ai-worker/src/pair.rs',
            'native/brass-ai-worker/src/pair_harness.rs']],
        heldout_opened=False, full_matrix_pending=True, production_switch=False)
    assert not CONFIG.exists()
    write(CONFIG, protocol); write(target / 'protocol.json', protocol)
    shutil.copyfile(__file__, target / 'benchmark-search-v2.py')
    print('Frozen six short A/B batches; held-out roots remain unopened.')


def run(batch):
    protocol = read(CONFIG)
    folder = OUT / protocol.get('folder', 'feasibility')
    assert read(folder / 'protocol.json') == protocol
    binary = ROOT / protocol['binary']['path']
    assert sha(binary) == protocol['binary']['sha256']
    row = next(r for r in protocol['batches'] if r['batch'] == batch)
    root = next(r for r in protocol['roots'] if r['id'] == row['root'])
    fixture = ROOT / root['fixture']['path']; assert sha(fixture) == root['fixture']['sha256']
    base = read(fixture)['cases'][0]
    model = ROOT / protocol['model']['path']; assert sha(model) == protocol['model']['sha256']
    manifest = read(model)
    cases = []
    for method in row['order']:
        inp = copy.deepcopy(base['input']); inp['shuffle_tape'] = None; inp['sample_seed'] = root['world_seed']
        config = dict(base['config'], seed=root['search_seed'], k=row['k'], simulations=64, trace=False)
        common = dict(input=inp, config=config, method=method,
                      policy=dict(cap=1, pending_limit=1, max_candidate_cells=8192, flush_us=0))
        cases.append(dict(common, name=f'{batch}-{method}-warmup', mode=dict(kind='count'),
                          config=dict(config, simulations=8), allowance_ms=4000))
        ctx = inp['context']
        cases.append(dict(common, name=f'{batch}-{method}', allowance_ms=row['allowance_ms'],
            mode=dict(kind='timed', startup=dict(model_id=ctx['model_id'], export_sha256=manifest['export_sha256'],
                reward_id=ctx['reward_id'], phase=inp['view']['context']['phase'],
                estimate=dict(nanoseconds=protocol['startup'], source='Frozen installed g890 startup estimate')))))
    source = folder / (batch + '-input.json'); target = folder / (batch + '-output.json')
    assert not source.exists() and not target.exists(), 'Never overwrite a trial'
    write(source, dict(manifest=str(model), manifest_sha256=sha(model), cases=cases))
    started = time.perf_counter(); outputs=[]; receipts=[]
    for arm,method in enumerate(row['order']):
        arm_binary=binary;selected=copy.deepcopy(cases[2*arm:2*arm+2])
        if method=='search_v1' and 'control_binary' in protocol:
            arm_binary=ROOT/protocol['control_binary']['path'];assert sha(arm_binary)==protocol['control_binary']['sha256']
            for case in selected:case.pop('method')
        arm_source=folder/(batch+f'-arm{arm}-input.json');arm_target=folder/(batch+f'-arm{arm}-output.json')
        write(arm_source,dict(manifest=str(model),manifest_sha256=sha(model),cases=selected))
        result=process(arm_binary,arm_source,arm_target,folder/(batch+f'-arm{arm}.log'),60-(time.perf_counter()-started))
        receipts.append(dict(method=method,**result));write(folder/(batch+'-receipt.json'),receipts)
        assert result['exit_code']==0, folder/(batch+f'-arm{arm}.log')
        output=read(arm_target);outputs.extend(output['results'])
        assert all(r['result']['status']=='complete' for r in output['results']), output
    write(target,dict(results=outputs))
    for r in read(target)['results']:
        if r['name'].endswith('-warmup'): continue
        assert r['result']['status'] == 'complete', r['result']
        d = r['result']['decision']; f = d.get('focus')
        print(r['name'], 'completed', [l['outcome']['completed'] for l in d['lanes']],
              'gate', f['switch_reason'] if f and 'switch_reason' in f else [e['reason'] for e in f['events']] if f else 'control', flush=True)


def study_freeze():
    target = OUT / 'study'; target.mkdir(parents=True, exist_ok=False)
    (target / 'fixtures').mkdir()
    binary = ROOT / 'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/gumbel-pair.exe'
    shutil.copyfile(binary, target / binary.name)
    roots = []
    def add(name, inp, category, partition, source, authored=None):
        inp = copy.deepcopy(inp); inp['shuffle_tape'] = None
        c = inp['view']['context']; assert c['phase'] == 'action' and c['actions_remaining'] == 2
        path = target / 'fixtures' / (name + '.json')
        write(path, dict(cases=[dict(input=inp, config=read(OUT/'control/BrassNative/alpha-g890/policy.json')['config'])]))
        roots.append(dict(id=name, fixture=receipt(path), category=category, partition=partition,
            source=receipt(source), authored=authored, search_seed=7, world_seed=890,
            expectation='Both actions must be complete and independently legal; resource payments and actor/turn/ledger must match replay. No invented optimal-action claim.'))
    for index, category in [(3, 'Early Canal; build/develop/loan combinations'), (26, 'Rail construction/resource alternatives')]:
        source = L10 / f'brass-{index:03d}-request.json'
        add(f'brass-{index:03d}', read(source)['cases'][0]['input'], category, 'development', source)
    for segment, line, name, category, partition in [
        (3,2,'loan-build','Canal cash=9; loan and follow-up construction','development'),
        (4,1,'dense-canal','Canal round 7; 15 tiles; development and payment choices','development'),
        (8,2,'dense-rail','Rail round 7; 31 tiles/30 links; expensive legality','heldout'),
        (8,4,'near-end','Rail round 8; first-choice commitment risk','heldout')]:
        source=ROOT/f'artifacts/rust-backend/20261026-local-release/nn-{segment:02d}-a-evidence/unity.permitted.jsonl'
        row=json.loads(source.read_text().splitlines()[line])
        add(name,json.loads(row['request']['input_json']),category,partition,source)
    base=ROOT/'artifacts/rust-backend/20261024-g890-l01-l02/local/fixtures/roots'
    template=read(L10/'brass-003-request.json')['cases'][0]['input']
    for name, old, category in [('rail-double','edge-rail-action','Rail double-link; own beer and remote coal'),
                                ('multi-sale','edge-shared-merchant-beer','Two cotton mills; shared merchant beer; multi-sale ordering')]:
        source=base/(old+'.json.gz'); f=json.load(gzip.open(source,'rt')); c=f['projection']['context'];seat=c['actor']
        inp=copy.deepcopy(template);inp.update(view=f['projection']['views'][seat],recall=f['recall'][seat],ledger=f['ledger'])
        inp['context']['remaining']=[inp['context']['max_cost']-v for v in inp['ledger']['spent']]
        authored=None
        if name=='multi-sale':
            inp['view']['context'].update(round=2,actions_remaining=2)
            authored='Change only round 1->2 and actions_remaining 1->2 on the existing authored shared-beer fixture; preserve board/cards/resources.'
        add(name,inp,category,'heldout',source,authored)
    settings=[('4s_k8',4000,8),('8s_k8',8000,8),('8s_k16',8000,16)]
    batches=[]
    for ri,root in enumerate(roots):
        for ci,(setting,ms,k) in enumerate(settings):
            order=['search_v1','search_v2_focus_only']
            if (ri+ci)%2:order.reverse()
            batches.append(dict(batch=f'{root["id"]}-{setting}',root=root['id'],setting=setting,allowance_ms=ms,k=k,order=order))
    for root in roots[:2]:
        batches.append(dict(batch=f'{root["id"]}-no-switch',root=root['id'],setting='4s_k8',allowance_ms=4000,k=8,order=['search_v2_no_switch']))
    protocol=dict(version='search-v2-study/2.0.0',folder='study',stage='cache-only-candidate', roots=roots,batches=batches,
        settings=settings, sessions=2,threads_per_session=1,preparation_ns=20_000_000,minimum_support=2,
        batch_watchdog_seconds=60,maximum_trials=50,search_allowance_seconds=328,
        startup=read(OUT/'control/BrassNative/alpha-g890/policy.json')['startup_ns'],
        model=receipt(OUT/'control/BrassNative/alpha-g890/model/manifest.json'),binary=receipt(target/binary.name),
        control_binary=receipt(OUT/'control/rebuilt-gumbel-pair.exe'),
        native_source_sha256=TOOLS['BASE']['native_fingerprint'](),
        warmup='One identical 8-trajectory count warmup per arm; reset turn/tree/cache after warmup',
        method_scope='Cache-only refinement; zero inherited visits; full statistical subtree reuse not claimed',
        projection_policy='Ordinary same-actor paths excluding Scout/liquidation; exact projected proof and equal hidden-card allocation law',
        retirement='Quiesce at drained trajectory wave; eligibility stays at the last complete round. Discard later partial work only after acceptance; no-switch resumes it unchanged.',
        scope='Single final native request; report incomplete whole-turn plans explicitly. Actual live exceptional replans are a separate Unity/lifecycle check.',
        early_stop='Run first two development roots before remaining matrix; correctness failure blocks further batches.',
        preliminary_evidence=receipt(OUT/'feasibility/protocol.json'),
        production_switch=False, subtree_reuse=False)
    write(CONFIG,protocol);write(target/'protocol.json',protocol)
    shutil.copyfile(__file__,target/'benchmark-search-v2.py')
    with __import__('zipfile').ZipFile(target/'source.zip','w',__import__('zipfile').ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'native').rglob('*')):
            if p.is_file() and p.suffix in ('.rs','.toml','.lock','.json'): z.write(p,p.relative_to(ROOT))
    print('Frozen 8 roots, 24 A/B pairs and 2 no-switch trials; no production changes.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'study-freeze', 'run']); p.add_argument('--batch');p.add_argument('--protocol',type=Path)
    a = p.parse_args()
    if a.protocol: CONFIG=a.protocol.resolve()
    if a.action == 'freeze': freeze()
    elif a.action == 'study-freeze': study_freeze()
    else: run(a.batch)
