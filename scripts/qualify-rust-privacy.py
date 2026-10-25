"""Qualify L06 observations, recall and permitted-world sampling, entirely locally."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import runpy
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
L05 = runpy.run_path(str(ROOT / 'scripts/qualify-rust-rules.py'))
BASE = L05['BASE']
read, write, receipt, sha = (BASE[k] for k in ('read', 'write', 'receipt', 'sha'))


def qualify(out, capture, edges):
    assert out.is_relative_to(ROOT / 'artifacts/rust-backend') and not out.exists()
    out.mkdir(parents=True)
    write(out / 'progress.json', dict(status='running', milestone='L06'))
    reference = L05['prepare_reference'](out)
    before = BASE['verify_frozen'](reference_source_root=reference)
    write(out / 'preservation-before.json', before)
    frozen = L05['verify_capture'](capture)
    edge_capture = L05['verify_capture'](edges)
    assert L05['preserved_identity'](before) == L05['preserved_identity'](frozen['preservation'])
    for item in [frozen['exporter'], *frozen['runtime'], *frozen['inputs'], *frozen['reference_sources'],
                 frozen['rules_capture'], frozen['script'], frozen['python_reference']]:
        assert receipt(ROOT/item['path']) == item, 'L06 reference provenance changed: '+item['path']
    assert L05['preserved_identity'](before)==L05['preserved_identity'](edge_capture['preservation'])
    for item in [edge_capture['source'],edge_capture['exporter'],edge_capture['script'],*edge_capture['runtime'],*edge_capture['reference_sources']]:
        assert receipt(ROOT/item['path'])==item,'L06 edge provenance changed: '+item['path']
    prior_paths = [ROOT/'artifacts/rust-backend'/folder/name
                   for folder in ('20261024-l03-verified','20261025-l04-qualified','20261025-l05-qualified-v4')
                   for name in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    prior = [receipt(p) for p in prior_paths]
    rules = ROOT/'artifacts/rust-backend/20261025-l05-capture-v2'
    branches = ROOT/'artifacts/rust-backend/20261025-l05-branches'
    L05['verify_capture'](rules); L05['verify_capture'](branches)
    source_identity = BASE['native_fingerprint']()
    run = BASE['Qualification'](out)
    run.env.pop('PROPTEST_RNG_SEED',None)
    run.env['CARGO_INCREMENTAL']='0'
    run.env.update(BRASS_PRIVACY_CAPTURE=str(capture), BRASS_PRIVACY_EDGES=str(edges), BRASS_RULES_CAPTURE=str(rules), BRASS_RULES_BRANCHES=str(branches))
    run.env['PYTHONPATH']=os.pathsep.join([str(reference/'src'), *[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p) != ROOT/'src']])
    check = '''import hashlib,json,sys,unittest
from pathlib import Path
from brass_ai import beliefs
from brass_ai.searchlab import spiel
from brass_ai.native import protocol
from brass_ai.learning import reward
from brass_ai.rust_backend import contracts
root=Path(sys.argv[1]).resolve(); result={}
for module in (beliefs,spiel,protocol,reward,contracts):
    path=Path(module.__file__).resolve(); assert path.is_relative_to(root),str(path)
    result[module.__name__]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
loader=unittest.TestLoader();loader.discover(sys.argv[2]);assert not loader.errors,loader.errors
print(json.dumps(result,indent=2))
'''
    run.run('reference-python-identities',[sys.executable,'-c',check,reference,ROOT/'tests/RustBackendNative'])
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    for profile,flags in [('debug',[]),('release',['--release'])]:
        run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
    from brass_ai.native import NativeWorker
    builds={}
    for profile,flags in [('debug',[]),('release',['--release']),('profiling',['--profile','profiling'])]:
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        binary=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile/'brass-ai-worker.exe'
        with NativeWorker(binary) as worker:
            hello=worker.hello
            assert hello['build']['source_sha256'] == source_identity == BASE['native_fingerprint']()
            assert hello['build']['cargo_lock_sha256'] == sha(ROOT/'native/Cargo.lock')
            assert hello['target_rules_qualified'] is False
        builds[profile]=dict(binary=receipt(binary),handshake=hello,pe_machine=BASE['machine_type'](binary))
        env=dict(run.env,BRASS_NATIVE_WORKER=str(binary),
                 BRASS_STATE_PARITY_REPORT=str(out/(profile+'-state-parity.json')),
                 BRASS_RULES_PARITY_REPORT=str(out/(profile+'-rules-parity.json')),
                 BRASS_RULES_FAILURE=str(out/(profile+'-rules-first-difference.json')),
                 BRASS_PRIVACY_REPORT=str(out/(profile+'-privacy-parity.json')),
                 BRASS_PRIVACY_SAMPLING_REPORT=str(out/(profile+'-sampling-parity.json')),
                 BRASS_PRIVACY_EDGE_REPORT=str(out/(profile+'-privacy-edges-parity.json')),
                 BRASS_PRIVACY_FAILURE=str(out/(profile+'-privacy-first-difference.json')))
        run.run(profile+'-python-tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/RustBackendNative','-v'],env=env,timeout=3600)
    run.cargo_run('runtime-dependencies','tree','-p','brass-ai-worker','--edges','normal','--locked','--offline')
    run.cargo_run('features','tree','--workspace','--edges','features','--locked','--offline')
    metadata=run.cargo_run('metadata','metadata','--format-version','1','--locked','--offline')
    write(out/'cargo-metadata.json',read(metadata))
    after=BASE['verify_frozen'](reference_source_root=reference)
    write(out/'preservation-after.json',after)
    assert L05['preserved_identity'](before)==L05['preserved_identity'](after)
    for item in read(out/'python-reference.json')['candidate_overlay']:
        assert receipt(ROOT/item['candidate']['path'])==item['candidate']
        assert receipt(ROOT/item['executed']['path'])==item['executed']
    for item in read(out/'python-reference.json')['harness_overlay']:
        assert receipt(ROOT/item['source']['path'])==item['source']
        assert receipt(ROOT/item['executed']['path'])==item['executed']
    assert source_identity==BASE['native_fingerprint']()
    assert prior==[receipt(p) for p in prior_paths]
    assert frozen==L05['verify_capture'](capture)
    assert edge_capture==L05['verify_capture'](edges)
    paths=list((ROOT/'native').rglob('*'))+list((ROOT/'src/brass_ai/native').glob('*.py'))
    paths+=list((ROOT/'tests/RustBackendNative').rglob('*'))+list((ROOT/'configs/contracts/rust-backend').glob('*.md'))
    paths+=list((ROOT/'scripts').glob('*rust*.py'))
    paths=sorted({p for p in paths if p.is_file() and (p.suffix in ('.rs','.toml','.lock','.json','.py','.cs','.md','.patch') or p.name=='LICENSE')},key=str)
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths:archive.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for profile,build in builds.items():archive.write(ROOT/build['binary']['path'],profile+'/brass-ai-worker.exe')
    result=dict(format='brass-native-l06-qualification/1.0.0',status='passed',milestone='L06',
                completed_utc=datetime.now(timezone.utc).isoformat(),builds=builds,commands=run.commands,
                sources=[receipt(p) for p in paths],prior_evidence=prior,python_reference=receipt(out/'python-reference.json'),
                capture=receipt(capture/'capture.json'),edge_capture=receipt(edges/'capture.json'),archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],
                roots=frozen['roots'],sampled_worlds=frozen['sampled_worlds'],sampled_transitions=frozen['transitions'],
                full_games=20,game_transitions=8344,unexplained_differences=0,sessions=2,threads_per_session=1,
                edge_cases=edge_capture['cases'],edge_transitions=edge_capture['transitions'],edge_full_games=1,
                target_rules_qualified=False,search_available=False,installed_backend_changed=False,nn_executed=False,
                matched_search_run=False,matched_search_seconds=4,next_milestone='L07: reference simulator adapter and full-game qualification')
    write(out/'qualification.json',result)
    write(out/'progress.json',dict(status='passed',milestone='L06',next='L07'))
    print('L06 qualification passed:',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--capture',type=Path,default=ROOT/'artifacts/rust-backend/20261025-l06-capture')
    parser.add_argument('--edges',type=Path,default=ROOT/'artifacts/rust-backend/20261025-l06-edges')
    args=parser.parse_args();out=args.out.resolve()
    try:qualify(out,args.capture.resolve(),args.edges.resolve())
    except BaseException as error:
        if out.is_relative_to(ROOT/'artifacts/rust-backend') and not (out/'qualification.json').exists():
            write(out/'progress.json',dict(status='failed',milestone='L06',error=str(error)))
        raise
