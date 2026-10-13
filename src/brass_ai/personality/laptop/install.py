"""Explicit release certification, dry-run installation and hash-guarded rollback."""
from pathlib import Path
import os
import shutil
import subprocess
import uuid

from brass_ai.bank import sha
from brass_ai.learning.contracts import require,SOURCE_ROOT
from .contracts import read,write,seal,unseal,child,BENCHMARK,BACKENDS,identity
from .integration import BUILD
from .package import validate

RELEASE='personality-laptop-release/1.0.0'
INSTALL='personality-install-journal/1.0.0'
OWNED=('BrassLegacyAI.dll','BrassPatch.dll')


def preflight(package, python, backend, *, fixture=False):
    """Use exactly the interpreter/module path the game will launch, before writes."""
    require(backend in BACKENDS, 'Unknown installed backend')
    env=dict(os.environ, PYTHONPATH=str(Path(package)/'runtime/src'), PYTHONDONTWRITEBYTECODE='1')
    command=[str(python),'-m','brass_ai.personality.laptop.cli','check-runtime',
             '--package',str(package),'--backend',backend]+(['--fixture'] if fixture else [])
    result=subprocess.run(command,cwd=package,env=env,capture_output=True,text=True,timeout=60,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode==0, 'Selected Python/backend cannot run this package. Install its dependencies before applying: '+result.stderr[-2000:])
    import json
    receipt=json.loads(result.stdout)
    require(receipt['ready'] is True and receipt['backend']==backend, 'Invalid runtime preflight')
    return receipt


def build_check(directory):
    d=Path(directory).resolve();m=unseal(read(d/'build.json'),BUILD)
    require(m['contract']==identity(),'Build source changed; restage the helper')
    for f in m['files']:require(sha(child(d/'Managed',f['name']))==f['sha256'],'Staged helper changed')
    for f in m['source_pins']:require(sha(child(d/'sources',f['path']))==f['sha256'],'Staged source changed')
    return m


def certify(package,build,benchmark,acceptance,output,*,backend='ort-cpu'):
    p=validate(package,allow_fixture=False);b=build_check(build)
    report=unseal(read(benchmark),BENCHMARK);review=read(acceptance)
    require(backend in BACKENDS and report['package_id']==p['id'] and not report['fixture'] and report['unity_running'],
            'Benchmark the real package with the game running on the target laptop')
    cpu=report.get('environment',{}).get('processor','').casefold()
    require('intel' in cpu and '135u' in cpu, 'Measured target laptop CPU is missing or different')
    runs=[r for r in report['results'] if r['backend']==backend and r['available']];require(len(runs)==1,'Backend has no valid whole-turn measurement')
    rows=runs[0]['records'];ordinary=[r for r in rows if not r['interrupt']]
    completed_turns=[r for r in rows if not r['interrupt'] or r['interrupt']=='play-now']
    require(all(r['status']=='ok' and r['metrics']['deadline_resets']==0 and
        r['metrics']['turn_overrun_seconds']<=.05 for r in completed_turns),'Deadline/failure gate failed')
    require(all(len({r['seed'] for r in ordinary if r['tier']==t and r['era']==era})>=2
        and sum(r['metrics']['completed'] for r in ordinary if r['tier']==t and r['era']==era)>0
        for t in ('easy','medium','hard') for era in ('CanalEra','RailEra')), 'Both eras, two seeds and useful search required for every tier')
    require({'build','develop','sell','network'}<=set(k for r in ordinary for k in r['available_kinds']), 'Representative investment/sale/network positions missing')
    require(any(r['interrupt']=='cancel' and r['status']=='cancelled' and r['metrics'].get('cancelled') is True for r in rows) and
            any(r['interrupt']=='play-now' and r['status']=='ok' and r['metrics'].get('play_now') is True
                and r['metrics'].get('cancelled') is False for r in rows),'Measured cancellation/Play Now missing')
    checks=('four_player_game','sixteen_styles','save_load_pending_action','history_rewind','play_now','menu_cancel','resource_restoration','old_save_unchanged','same_profile_strength_review')
    require(review.get('format')=='personality-laptop-acceptance/1.0.0' and review.get('package_id')==p['id'] and review.get('build_id')==b['id']
        and review.get('backend')==backend and review.get('target')=='Intel Core Ultra 5 135U' and review.get('reviewer')
        and all(review.get('checks',{}).get(k) is True for k in checks), 'Real laptop/Unity acceptance and backend/profile review are pending')
    value=seal(dict(format=RELEASE,qualified=True,fixture=False,package_id=p['id'],build_id=b['id'],backend=backend,
        manifest_sha256=sha(Path(package)/'manifest.json'),integration_sha256=sha(Path(build)/'Managed/BrassLegacyAI.dll'),
        integration_files={n:sha(Path(build)/'Managed'/n) for n in OWNED},
        benchmark_sha256=sha(benchmark),acceptance_sha256=sha(acceptance),p4_id=p['p4_id'],p5_id=p['p5_id'],
        scope='This exact package, backend, two-worker profiles and target-laptop integration only'))
    require(not Path(output).exists(),'Use a new release certificate');write(output,value);return value


def _closed(game):
    import psutil
    for process in psutil.process_iter(['name','exe']):
        if 'brassbirmingham' in (process.info['name'] or '').lower():raise ValueError('Close Brass Birmingham before installation/rollback')


def _sandbox(game):
    game=Path(game).resolve();root=(SOURCE_ROOT/'artifacts').resolve()
    require(game.is_relative_to(root) and (game/'.p6-fixture-sandbox').is_file(), 'Fixture installation requires an explicit isolated workspace-artifacts sandbox')


def _bundle_path(game,manifest):
    # Full hashes remain in the manifest/pointer. Short directory names avoid
    # MAX_PATH failures in the game's older Mono runtime; collisions fail closed.
    bundle=child(Path(game),'BrassPersonality/p/'+manifest['id'][:16])
    if os.name=='nt':
        longest=max(len(str(bundle/f['path'])) for f in manifest['files'])
        require(longest+14<260,'Package paths exceed the Windows game runtime limit; use a shorter isolated game directory')
    return bundle


def install(package,build,game,*,certificate=None,python=None,apply=False,sandbox_fixture=False,intent_hints=False,review_copy=False,review_backend='ort-cpu'):
    package,build,game=map(lambda x:Path(x).resolve(),(package,build,game))
    review_copy=review_copy or sandbox_fixture
    p=validate(package,allow_fixture=review_copy);b=build_check(build)
    require(p['profiles']['hint_mode']!='intent-hints' or intent_hints,'This qualified profile reveals broad AI intentions; opt in explicitly with --intent-hints')
    if review_copy:
        _sandbox(game);require(review_backend in BACKENDS,'Unknown review backend')
    else:
        require(certificate is not None,'A real P6 release certificate is required')
        c=unseal(read(certificate),RELEASE)
        require(c['qualified'] and not c['fixture'] and c['package_id']==p['id'] and c['build_id']==b['id']
            and c['manifest_sha256']==sha(package/'manifest.json')
            and c['integration_files']=={n:sha(build/'Managed'/n) for n in OWNED}, 'Release certificate does not cover these bytes')
    require(python is not None and Path(python).is_file(),'Supply the tested absolute Python executable')
    managed=game/'BrassBirminghamUnity_Data/Managed';target=managed/'BrassLegacyAI.dll'
    baseline={f['name']:f['sha256'].lower() for f in b['base_files']}
    pointer=game/'BrassPersonality/active.json';old=read(pointer) if pointer.exists() else None
    expected=dict(baseline)
    if old:
        expected.update(old.get('installed_files',{'BrassLegacyAI.dll':old['installed_helper_sha256']}))
    for name in (*OWNED,'Assembly-CSharp.dll'):require(sha(managed/name)==expected[name],'Installed helper/rules/UI differs; refuse to overwrite unknown work')
    bundle=_bundle_path(game,p)
    plan=dict(package_id=p['id'],target=str(managed),replace_only=list(OWNED),fixture=p['fixture'],apply=apply,
              preserves=['Assembly-CSharp.dll','saves','training'],installed=False)
    plan['runtime']=preflight(package,Path(python).resolve(),review_backend if review_copy else c['backend'],fixture=review_copy)
    require(plan['runtime']['package_id']==p['id'], 'Interpreter loaded a different package')
    if not apply:return plan
    _closed(game)
    folder=game/'BrassPersonality';backup=folder/'backups'/uuid.uuid4().hex[:16];backup.mkdir(parents=True)
    for name in OWNED:shutil.copyfile(managed/name,backup/name)
    if pointer.exists():shutil.copyfile(pointer,backup/'active.json')
    receipt_path=folder/'certificates'/((c['id'][:32] if not review_copy else 'review')+'.json')
    staging=bundle.with_name(bundle.name+'.new-'+uuid.uuid4().hex[:8])
    journal=seal(dict(format=INSTALL,game=str(game),package_id=p['id'],previous_sha256=sha(target),
        installed_sha256=sha(build/'Managed/BrassLegacyAI.dll'),previous_pointer_sha256=sha(pointer) if pointer.exists() else None,
        previous_files={n:sha(managed/n) for n in OWNED},installed_files={n:sha(build/'Managed'/n) for n in OWNED},
        backup=backup.relative_to(game).as_posix(),status='prepared',fixture=p['fixture']))
    write(backup/'journal.json',journal)
    try:
        if bundle.exists():require(validate(bundle)['id']==p['id'],'Existing pinned package differs')
        else:
            shutil.copytree(package,staging,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            require(validate(staging)['id']==p['id'],'Staged package copy differs');os.replace(staging,bundle)
        # Dependencies are provisioned/tested by the operator before acceptance;
        # installation never runs pip or downloads a model behind their back.
        if not review_copy:
            receipt_path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(certificate,receipt_path)
        for name in OWNED:
            dest=managed/name;temp=dest.with_suffix('.p6-new');shutil.copyfile(build/'Managed'/name,temp);os.replace(temp,dest)
        new=dict(format='personality-install-pointer/1.0.0',package=bundle.relative_to(game).as_posix(),manifest_sha256=sha(bundle/'manifest.json'),
            certificate=receipt_path.relative_to(game).as_posix(),certificate_sha256=sha(receipt_path) if not review_copy else None,
            backend=c['backend'] if not review_copy else review_backend,python=str(Path(python).resolve()),review_only=review_copy,
            installed_helper_sha256=sha(target),fixture=p['fixture'])
        new['installed_files']={n:sha(managed/n) for n in OWNED}
        new['intent_hints_opt_in']=bool(intent_hints)
        write(pointer,new)
        journal=seal(dict({k:v for k,v in journal.items() if k!='id'},status='installed',pointer_sha256=sha(pointer)))
        write(backup/'journal.json',journal)
    except BaseException:
        for name in OWNED:
            dest=managed/name;temp=dest.with_suffix('.p6-restore');shutil.copyfile(backup/name,temp);os.replace(temp,dest)
        if old is None:pointer.unlink(missing_ok=True)
        else:shutil.copyfile(backup/'active.json',pointer)
        raise
    finally:
        require(staging.resolve().is_relative_to((game/'BrassPersonality/p').resolve()),'Unsafe temporary package path')
        if staging.exists():shutil.rmtree(staging)
    return dict(plan,installed=True,journal=str(backup/'journal.json'))


def rollback(journal_path,*,apply=False,sandbox_fixture=False):
    path=Path(journal_path).resolve();j=unseal(read(path),INSTALL);game=Path(j['game']).resolve()
    if j['fixture']:require(sandbox_fixture,'Fixture sandbox rollback flag required');_sandbox(game)
    backup=child(game,j['backup']);require(path==backup/'journal.json','Unexpected rollback journal path')
    target=game/'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll';pointer=game/'BrassPersonality/active.json'
    require(j['status']=='installed' and sha(target)==j['installed_sha256'] and sha(pointer)==j['pointer_sha256'],'Installation changed since this backup')
    installed=j.get('installed_files',{'BrassLegacyAI.dll':j['installed_sha256']})
    previous=j.get('previous_files',{'BrassLegacyAI.dll':j['previous_sha256']})
    require(set(installed)==set(previous) and set(installed)<=set(OWNED),'Unknown rollback file set')
    for name in installed:
        require(sha(target.parent/name)==installed[name] and sha(backup/name)==previous[name],'Rollback/current helper changed')
    if j['previous_pointer_sha256']:require(sha(backup/'active.json')==j['previous_pointer_sha256'],'Prior pointer backup changed')
    if not apply:return dict(rollback_ready=True,apply=False,game=str(game))
    _closed(game)
    # Rollback must itself be recoverable on ordinary I/O failures. Keep the
    # current small helper/pointer bytes until all replacements have committed.
    current={target.parent/name:(target.parent/name).read_bytes() for name in installed}
    current[pointer]=pointer.read_bytes()
    try:
        for name in installed:
            dest=target.parent/name;temp=dest.with_suffix('.p6-rollback');shutil.copyfile(backup/name,temp);os.replace(temp,dest)
        if j['previous_pointer_sha256']:
            temp=pointer.with_suffix('.p6-rollback');shutil.copyfile(backup/'active.json',temp);os.replace(temp,pointer)
        else:pointer.unlink()
        write(backup/'rollback.json',dict(restored_sha256=sha(target),package_retained=True,saves_untouched=True))
    except BaseException:
        for dest,data in current.items():
            temp=dest.with_suffix('.p6-restore');temp.write_bytes(data);os.replace(temp,dest)
        raise
    finally:
        for dest in current:dest.with_suffix('.p6-rollback').unlink(missing_ok=True)
    return dict(rolled_back=True,restored_sha256=sha(target),package_retained=True)
