"""Seal L11 session/plan correctness; never installs or runs lab training."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import runpy
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
L10=runpy.run_path(str(ROOT/'scripts/qualify-rust-search.py'))
BASE=L10['BASE'];L09=L10['L09']
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))
PRIOR=ROOT/'artifacts/rust-backend/20261025-l10-qualified'
MODEL=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/model'
FIXTURES=ROOT/'artifacts/rust-backend/20261025-l10-capture'

def sources():
    paths=L10['sources']()+[ROOT/p for p in ('scripts/reproduce-gumbel-is-mcts.py','scripts/reproduce-gumbel-sessions.py',
        'tests/RustBackend/test_sessions.py','configs/contracts/rust-backend/gumbel-session-v1.md')]
    paths+=list((ROOT/'tests/GumbelISMCTS').rglob('*.py'))+list((ROOT/'tests/GumbelISMCTS/fixtures').glob('*.json'))
    return sorted(set(paths),key=str)

def qualify(out):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists()
    out.mkdir(parents=True);write(out/'progress.json',dict(status='running',milestone='L11'))
    reference=L10['PRIOR']/'reference-source'
    before=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-before.json',before)
    prior=[receipt(PRIOR/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    assert read(PRIOR/'qualification.json')['status']==read(PRIOR/'final-verification.json')['status']=='passed'
    model=[receipt(p) for p in MODEL.iterdir() if p.is_file()]
    fixture_before=[receipt(p) for p in sorted(FIXTURES.glob('*.json'))]
    write(out/'fixtures.json',fixture_before)
    source_before=[receipt(p) for p in sources()];write(out/'sources-before.json',source_before)
    identity=BASE['native_fingerprint']()
    run=BASE['Qualification'](out);run.env.pop('PROPTEST_RNG_SEED',None);run.env['CARGO_INCREMENTAL']='0'
    run.env['PYTHONPATH']=os.pathsep.join([str(reference/'src'),*[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p)!=ROOT/'src']])
    run.env['BRASS_ONNX_MANIFEST']=str(MODEL/'manifest.json');run.env['BRASS_ONNX_GOLDEN']=str(MODEL.parent/'golden.json')
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    run.run('python-oracles',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/GumbelISMCTS','-v'])
    builds={};reports={}
    for profile,flags in [('debug',[]),('release',['--release']),('profiling',['--profile','profiling'])]:
        if profile!='profiling':
            run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
            run.cargo_run(profile+'-ort-goldens','test','-p','brass-ai-inference','--test','runtime',*flags,'--locked','--offline','--','--ignored','--nocapture')
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        directory=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile
        worker=directory/'brass-ai-worker.exe';trace=directory/'gumbel-trace.exe';sessions=directory/'gumbel-session.exe'
        builds[profile]=dict(worker=receipt(worker),trace=receipt(trace),sessions=receipt(sessions))
        env=dict(run.env,BRASS_NATIVE_WORKER=str(worker),BRASS_SEARCH_BINARY=str(trace),BRASS_SEARCH_FIXTURES=str(FIXTURES),BRASS_SESSION_BINARY=str(sessions))
        for label,folder,pattern in [('sessions','RustBackend','test_sessions.py'),('search-boundary','RustBackend','test_search_trace.py'),
                                     ('transport','RustBackendNative','test_worker.py')]:
            run.run(profile+'-'+label,[sys.executable,'-m','unittest','discover','-s',ROOT/'tests'/folder,'-p',pattern,'-v'],env=env)
        if profile!='profiling':
            run.run(profile+'-l10-parity',[sys.executable,ROOT/'scripts/verify-rust-search.py','--fixtures',FIXTURES,'--binary',trace,'--report',out/(profile+'-l10-parity.json')],timeout=1800)
            run.run(profile+'-plans',[sys.executable,ROOT/'scripts/verify-rust-sessions.py','--binary',sessions,'--out',out/(profile+'-plans')],timeout=1800)
            plans=read(out/(profile+'-plans')/'verification.json')
            assert plans['status']=='passed' and len(plans['cases'])==44 and len(set(map(tuple,plans['root_matrix'])))==40
            reports[profile]=dict(plans=receipt(out/(profile+'-plans')/'verification.json'),l10=receipt(out/(profile+'-l10-parity.json')))
    run.cargo_run('dependencies','tree','--workspace','--edges','features','--locked','--offline')
    after=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-after.json',after)
    assert L09['L05']['preserved_identity'](before)==L09['L05']['preserved_identity'](after)
    assert source_before==[receipt(p) for p in sources()] and identity==BASE['native_fingerprint']()
    assert prior==[receipt(PRIOR/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    assert model==[receipt(p) for p in MODEL.iterdir() if p.is_file()]
    assert fixture_before==[receipt(p) for p in sorted(FIXTURES.glob('*.json'))]
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sources():z.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for profile,b in builds.items():
            for kind,item in b.items():z.write(ROOT/item['path'],profile+'/'+Path(item['path']).name)
    write(out/'qualification.json',dict(status='passed',milestone='L11',completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=identity,
        identity='brass-gumbel-is-mcts/session-v1',sources=source_before,commands=run.commands,builds=builds,reports=reports,
        prior_evidence=prior,model=model,fixtures=fixture_before,archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],
        scalar_round_deadline_fixtures=True,complete_plans_qualified=True,replica_merge_qualified=False,
        deep_polling_and_runtime_retirement_qualified=False,production_latency_qualified=False,matched_four_second_benchmark=False,
        installed_backend_changed=False,training_changed=False,local_g890_utility_accepted=True,calibration_receipt=None,next_milestone='L12'))
    write(out/'progress.json',dict(status='passed',milestone='L11'));print('L11 qualification passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:qualify(a.out.resolve())
    except BaseException as e:
        if a.out.resolve().is_relative_to(ROOT/'artifacts/rust-backend') and a.out.exists() and not (a.out/'qualification.json').exists():
            write(a.out/'progress.json',dict(status='failed',milestone='L11',error=repr(e)))
        raise
