"""Install additive authored hosts using already-admitted private dependencies."""
from pathlib import Path
import shutil
import threading

from brass_ai.bank import sha
from brass_ai.learning.contracts import read_json, require, project_path

_lock=threading.Lock()
_ready={}

def training_host(root,kind='ordinary'):
    root=Path(root).resolve()
    require(kind in ('ordinary','search'),'Unknown training host kind')
    source=project_path(root,'training/runtime/throughput/'+kind)
    if not (source/'runtime.json').exists():
        # Older installations remain usable; the guide makes the selected
        # runtime and missing fast-path package visible in preflight.
        return project_path(root,'artifacts/r4-host' if kind=='search' else 'artifacts/r1-host/guarded')
    metadata=project_path(root,source/'runtime.json')
    identity=sha(metadata);key=(str(root),kind,identity)
    with _lock:
        manifest=read_json(metadata)
        out=project_path(root,'artifacts/training-host/'+kind)
        project_path(root,out/'runtime.json')
        base=project_path(root,'artifacts/r1-host/guarded')
        for entry in manifest['files']:
            name=entry['name'];require(Path(name).name==name and name not in ('.','..'),'Unsafe runtime filename')
            project_path(root,out/name)
            project_path(root,(source if name=='BrassHost.exe' else base)/name)
        # Cached setup may skip repeated DLL hashes, never path containment.
        if key in _ready:return _ready[key]
        original=read_json(project_path(root,base/'runtime.json'))
        require(manifest['optimization']=='training-host/1.0.0' and manifest['kind']==kind,'Unknown training runtime')
        import hashlib
        require(hashlib.sha256((root/'scripts/build-training-host.py').read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()==manifest['builder_sha256'],
                'Training builder changed; rebuild additive runtime')
        for relative,expected in manifest['authored_sources'].items():
            path=project_path(root,relative)
            require(hashlib.sha256(path.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()==expected,
                    'Training host source changed; rebuild additive runtime: '+relative)
        require(manifest['baseline_host_sha256']==sha(base/'BrassHost.exe'),'Training runtime baseline changed')
        require({f['name']:f['sha256'] for f in manifest['files'] if f['name']!='BrassHost.exe'}==
                {f['name']:f['sha256'] for f in original['files'] if f['name']!='BrassHost.exe'},'Training rules dependencies changed')
        out.mkdir(parents=True,exist_ok=True)
        # Multiple spawned actors may initialize concurrently. Same immutable
        # source bytes, private temporary names and atomic replacements.
        import os
        for entry in manifest['files']:
            name=entry['name'];require(Path(name).name==name,'Unsafe runtime filename')
            origin=(source if name=='BrassHost.exe' else base)/name
            require(sha(origin)==entry['sha256'],'Changed training worker/dependency: '+name)
            target=out/name
            if target.exists() and sha(target)==entry['sha256']:continue
            temp=out/(name+'.'+str(os.getpid())+'.tmp');shutil.copyfile(origin,temp);temp.replace(target)
        if not (out/'runtime.json').exists() or sha(out/'runtime.json')!=identity:
            temp=out/('runtime.'+str(os.getpid())+'.tmp');shutil.copyfile(source/'runtime.json',temp);temp.replace(out/'runtime.json')
        _ready[key]=out
        return out


def preflight_workers(root,mono=None):
    """Expose selected workers and exercise fused replies/restoration on the lab."""
    from brass_ai.host import RulesHost
    result={}
    for kind in ('ordinary','search'):
        selected=training_host(root,kind)
        with RulesHost(selected,mono=mono) as host:
            context=host.reset(153);fast=host.enable_fastpath()
            if fast:
                reply=host.call('search_snapshot') if kind=='search' else host.decision()
                require(reply['context']==context,'Training worker context mismatch')
                require(reply['choices']==host.choices(),'Training worker legal choices differ')
                view=reply['views'][context['actor']] if kind=='search' else reply['view']
                require(view==host.observe(context['actor']),'Training worker observation differs')
                checkpoint=host.call('checkpoint');before=host.call('canonical_hash')
                host.call('restore',checkpoint=checkpoint)
                require(host.call('canonical_hash')==before,'Training worker restoration differs')
            result[kind]=dict(optimized=fast,path=str(selected),runtime_sha256=sha(selected/'runtime.json'))
    return result
