"""P6 export, native staging, measured backend selection and explicit installation."""
import argparse
import json
from pathlib import Path
from brass_ai.learning.contracts import SOURCE_ROOT
from .contracts import read


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('export');b.add_argument('--roster',type=Path,required=True);b.add_argument('--qualification',type=Path)
    b.add_argument('--output',type=Path,required=True);b.add_argument('--fixture',action='store_true')
    v=s.add_parser('validate');v.add_argument('--package',type=Path,required=True)
    v=s.add_parser('check-runtime');v.add_argument('--package',type=Path,required=True)
    v.add_argument('--backend',default='ort-cpu');v.add_argument('--fixture',action='store_true');v.add_argument('--mono')
    g=s.add_parser('stage-game');g.add_argument('--output',type=Path,required=True);g.add_argument('--base-version',default='1.4.31')
    e=s.add_parser('benchmark');e.add_argument('--package',type=Path,required=True);e.add_argument('--cases',type=Path,required=True)
    e.add_argument('--output',type=Path,required=True);e.add_argument('--backends',nargs='+',default=['ort-cpu'])
    e.add_argument('--tiers',nargs='+',default=['easy','medium','hard']);e.add_argument('--fixture',action='store_true')
    e.add_argument('--unity-running',action='store_true');e.add_argument('--mono')
    c=s.add_parser('certify')
    for name in ('package','build','benchmark','acceptance','output'):c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--backend',default='ort-cpu')
    i=s.add_parser('install')
    for name in ('package','build','game','python'):i.add_argument('--'+name,type=Path,required=True)
    i.add_argument('--certificate',type=Path);i.add_argument('--apply',action='store_true')
    i.add_argument('--intent-hints',action='store_true',help='Explicitly accept broad AI intention disclosure, only for a matching P5-qualified hint profile')
    i.add_argument('--review-copy',action='store_true',help='Only an explicitly marked isolated artifacts copy, launched with --brass-personality-review; never a release installation')
    i.add_argument('--review-backend',default='ort-cpu',help='Explicit backend for an isolated review copy; release uses its certificate')
    r=s.add_parser('rollback');r.add_argument('--journal',type=Path,required=True);r.add_argument('--apply',action='store_true')
    r.add_argument('--review-copy',action='store_true',help='Allow rollback of an isolated fixture review copy')
    e=s.add_parser('serve');e.add_argument('--package',type=Path,required=True);e.add_argument('--backend',default='ort-cpu')
    e.add_argument('--fixture',action='store_true');e.add_argument('--mono')
    a=p.parse_args()
    if a.command=='export':
        from .package import build
        result=build(SOURCE_ROOT,a.roster,a.output,qualification=a.qualification,fixture=a.fixture)
        result={k:result[k] for k in ('id','status','qualified_sources','fixture')}
    elif a.command=='validate':
        from .package import validate
        result=validate(a.package);result={k:result[k] for k in ('id','status','qualified_sources','fixture')}
    elif a.command=='stage-game':
        from .integration import stage
        result=stage(SOURCE_ROOT,a.output,base_version=a.base_version);result={k:result[k] for k in ('id','base_version','installed','qualified','files')}
    elif a.command=='check-runtime':
        from .runtime import Runtime
        runtime=Runtime(a.package,a.backend,fixture=a.fixture,mono=a.mono)
        try:
            runtime.game.new_initial_state()  # Native dependency/process probe; no game campaign.
            result=dict(ready=True,package_id=runtime.package['id'],backend=a.backend,fixture=runtime.package['fixture'],parity=runtime.service.parity)
        finally:runtime.close()
    elif a.command=='benchmark':
        from .benchmark import measure,recommendation
        result=recommendation(measure(a.package,read(a.cases),a.output,backends=a.backends,tiers=a.tiers,
            fixture=a.fixture,mono=a.mono,unity_running=a.unity_running))
    elif a.command=='certify':
        from .install import certify
        result=certify(a.package,a.build,a.benchmark,a.acceptance,a.output,backend=a.backend)
    elif a.command=='install':
        from .install import install
        result=install(a.package,a.build,a.game,certificate=a.certificate,python=a.python,apply=a.apply,intent_hints=a.intent_hints,review_copy=a.review_copy,review_backend=a.review_backend)
    elif a.command=='rollback':
        from .install import rollback
        result=rollback(a.journal,apply=a.apply,sandbox_fixture=a.review_copy)
    else:
        from .server import serve
        serve(a.package,a.backend,fixture=a.fixture,mono=a.mono);return
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
