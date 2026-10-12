"""Immutable private deployment bundles; fixtures can never be released."""
from pathlib import Path
import shutil

from brass_ai.bank import sha
from brass_ai.learning.contracts import require,digest
from brass_ai.personality.deployment import roster
from brass_ai.personality.deployment.contracts import validate_roster
from brass_ai.personality.difficulty import panels,evaluation,ui
from brass_ai.personality.difficulty.contracts import default_profiles,validate_profiles,resolve_search,TIERS,validate_report_summary
from brass_ai.personality.search.readiness import load as load_ready
from brass_ai.search.common import SearchConfig
from .contracts import PACKAGE,identity,read,write,seal,unseal,child,inventory
from .export import export,validate_metadata


def build(root,roster_directory,output,*,qualification=None,fixture=False):
    root=Path(root).resolve();out=Path(output).resolve()
    require(not out.exists(),'Use a new package directory')
    value=roster.load(Path(roster_directory));validate_roster(value)
    require(value['full_size'] and len(value['archetypes'])==16 and len(value['members'])==128,'A full 16 x 8 roster is required')
    panels.source_check(root,value,fixture)
    report=evaluation.load_report(root,qualification) if qualification else None
    require(fixture or (report and report['p5_qualified'] and not report['fixture'] and report['roster_id']==value['id']),
            'Real qualified P4 and P5 artifacts are required for a release candidate')
    profiles=report['profiles'] if report else default_profiles();validate_profiles(profiles)
    require(profiles['workers']==2 and profiles['cpu_role']=='laptop','Qualify the two-worker laptop profiles first')
    model,parents,ready,handoff,pool=load_ready(root,value['shared_model']['ready'])
    require(pool==value['pool_id'] and ready['model']['model_id']==value['shared_model']['model_id'],'Roster/model/pool differs')
    out.mkdir(parents=True)
    graphs=[]
    for i,m in enumerate([model]+parents):
        path=f'models/{"actor" if i==0 else "reference-"+str(i)}.onnx'
        meta=export(m,out/path);graphs.append(dict(path=path,model_id=meta['model_id'],actor=i==0))
    # The auxiliary frozen rollout pool preserves P2's opponent model. One actor
    # serves all 128 members; no descriptor, optimizer or learner is shipped.
    write(out/'roster.json',value);write(out/'qualification.json',report)
    if not fixture:
        # Retain the exact P4 qualification receipt as portable provenance; its
        # original byte hash is pinned by the roster. Raw games remain on the lab.
        shutil.copyfile(root/value['qualification']['report'],out/'p4-qualification.json')
    public=ui.menu(value,report,fixture=fixture);write(out/'public-menu.json',public)
    write(out/'profiles.json',profiles)
    compact=dict(id=value['id'],archetypes=[{k:g[k] for k in ('id','name','description','members')} for g in value['archetypes']],
                 members=[{k:m[k] for k in ('id','archetype_id','profile')} for m in value['members']])
    # Carry the existing, fingerprint-checked native search host. These licensed
    # game dependencies are a private local bundle, not a redistributable asset.
    native=root/'artifacts/r4-host';runtime=read(native/'runtime.json')
    deps=read(root/'configs/rules/runtime-dependencies-1.1.0.json')
    require({x['name']:x['sha256'] for x in runtime['files'] if x['name']!='BrassHost.exe'}==
            {x['name']:x['sha256'] for x in deps['files']},'Native rules dependencies differ')
    for entry in runtime['files']:
        source=child(native,entry['name']);require(sha(source)==entry['sha256'],'Native dependency changed')
        target=out/'runtime/artifacts/r4-host'/entry['name'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    write(out/'runtime/artifacts/r4-host/runtime.json',runtime)
    for name in ('configs/rules/runtime-dependencies-1.1.0.json','configs/contracts/v0.2.0/card-composition.json',
                 'configs/contracts/v0.2.0/wire.schema.json','configs/contracts/v0.2.0/manifest.json',
                 'configs/learning/features-1.0.0.json','configs/learning/host-1.0.0.json','configs/references/upstream.lock.json'):
        dest=out/'runtime'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
    # The search dependencies stay byte-identical. No training entry point or
    # training artifacts are included; Python modules are library dependencies.
    for source in sorted((root/'src/brass_ai').rglob('*.py')):
        dest=out/'runtime/src'/source.relative_to(root/'src');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    for module in ('BrassPersonalityGame','BrassPersonalityHost'):
        for source in sorted((root/'src'/module).glob('*.cs')):
            dest=out/'runtime/src'/module/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    manifest=seal(dict(format=PACKAGE,contract=identity(),fixture=bool(fixture),release_qualified=False,
        status='FIXTURE — not qualified for play' if fixture else 'Candidate — laptop acceptance pending',
        models=graphs,roster=compact,profiles=profiles,searches={t:resolve_search(value,profiles,t).to_dict() for t in TIERS},
        calibration=ready['goal_calibration'],reward=ready['reward'],pool_id=pool,
        p4_id=value['id'],p5_id=report['id'] if report else None,
        qualified_sources=bool(not fixture and value['p4_qualified'] and report and report['p5_qualified']),
        private_game_dependencies=True,files=inventory(out)))
    write(out/'manifest.json',manifest);validate(out);return manifest


def validate(directory,*,allow_fixture=True):
    directory=Path(directory).resolve();m=unseal(read(directory/'manifest.json'),PACKAGE)
    require(m['contract']==identity(),'P6 runtime/source differs; use the immutable bundled runtime or rebuild this candidate')
    require(allow_fixture or not m['fixture'],'Fixture package cannot be installed or released')
    require(not m['release_qualified'],'Release certificates are separate from the immutable candidate')
    require(m['files']==inventory(directory),'Package bytes/files changed')
    # inventory() already checks paths and hashes every byte; do not reread the
    # whole package a second time on every laptop service start.
    value=read(directory/'roster.json');validate_roster(value)
    require(value['id']==m['p4_id']==m['roster']['id'] and value['full_size'],'Roster identity differs')
    expected=dict(id=value['id'],archetypes=[{k:g[k] for k in ('id','name','description','members')} for g in value['archetypes']],
                 members=[{k:r[k] for k in ('id','archetype_id','profile')} for r in value['members']])
    require(m['roster']==expected,'Compact roster differs')
    report=read(directory/'qualification.json');p=validate_profiles(read(directory/'profiles.json'))
    if report is not None:validate_report_summary(report,value)
    require(m['profiles']==p and p['workers']==2 and p['cpu_role']=='laptop','Laptop resource profile differs')
    require(m['searches']=={t:resolve_search(value,p,t).to_dict() for t in TIERS},'Search behavior differs')
    require(read(directory/'public-menu.json')==ui.menu(value,report,fixture=m['fixture']),'Public menu differs')
    require(len(m['models']) in range(2,6) and sum(x['actor'] for x in m['models'])==1 and m['models'][0]['actor'], 'One actor and a frozen rollout pool required')
    require(m['models'][0]['model_id']==value['shared_model']['model_id'], 'Actor differs from qualified roster')
    for entry in m['models']:
        graph=child(directory,entry['path']);meta=validate_metadata(read(graph.with_suffix('.json')))
        require(meta['model_id']==entry['model_id'] and meta['actor']==entry['actor'] and meta['parity']['passed']
                and meta['config']['utility']==m['reward']
                and meta['sha256']==next(f['sha256'] for f in m['files'] if f['path']==entry['path']), 'Export model/reward/parity differs')
    if not m['fixture']:
        require(m['qualified_sources'] and value['p4_qualified'] and report and report['p5_qualified']
                and not report['fixture'] and all(report['checks'].values()) and report['id']==m['p5_id']
                and report['roster_id']==value['id'],'Missing qualified P4/P5 evidence')
        evidence=directory/'p4-qualification.json'
        require(sha(evidence)==value['qualification']['sha256'], 'Pinned P4 evidence changed')
        panels.validate_p4_evidence(value,read(evidence))
    else:require(not m['qualified_sources'],'Fixtures cannot claim qualified sources')
    return m
