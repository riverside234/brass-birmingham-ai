"""Recover archive-pinned repository files in a clone, with immutable backups.

Private artifacts/binaries and untracked files never receive this exception.
New transport tools keep running; their archived versions are verified in
separate snapshots because they are outside the training source fingerprint.
"""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import uuid

TOOLS=frozenset(('scripts/step5-transfer.py','scripts/step5_transfer_runtime.py','scripts/step5_transfer_restore.py','scripts/step5_transfer_frozen.py'))
PREFIXES=('src/','scripts/','configs/','training/opponents/','training/runtime/','lab-native/')
TEXT=frozenset(('.py','.json','.cs','.rs','.toml','.lock','.txt','.md','.sh','.ps1','.yml','.yaml'))
CONTRACT='step5-clone-reconciliation/1.0.0'


def tracked_files(root):
    try:
        top=subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel'],stderr=subprocess.DEVNULL)
        if Path(os.fsdecode(top).strip()).resolve()!=root:return set()
        raw=subprocess.check_output(['git','-C',str(root),'ls-files','-z'],stderr=subprocess.DEVNULL)
        return {os.fsdecode(p).replace('\\','/') for p in raw.split(b'\0') if p}
    except (OSError,subprocess.CalledProcessError):return set()


def git_bytes(root,manifest,record):
    commit=manifest.get('git_commit')
    if not isinstance(commit,str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',commit):return None
    try:
        result=subprocess.run(['git','-C',str(root),'cat-file','blob',commit+':'+record['path']],
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True,timeout=30,
            env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0'))
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):return None
    for candidate in newline_variants(result.stdout,record['bytes']):
        if len(candidate)==record['bytes'] and hashlib.sha256(candidate).hexdigest()==record['sha256']:return candidate
    return None


def newline_variants(data,size):
    """Bounded legacy recovery; every candidate still needs the full archive SHA.

    Old exporters trusted normalized Git status and could omit mixed endings.
    A single LF/CRLF transition can be recovered in linear time: the recorded
    byte count fixes the number of CR bytes, leaving just prefix/suffix choices.
    Do not enumerate arbitrary newline combinations or accept normalized hashes.
    """
    yield data
    lf=data.replace(b'\r\n',b'\n')
    yield lf
    yield lf.replace(b'\n',b'\r\n')
    extra=size-len(lf)
    if 0<extra<lf.count(b'\n'):
        yield b'\r\n'.join(lf.split(b'\n',extra))
        yield b'\r\n'.join(lf.rsplit(b'\n',extra))


def reproducible_git_files(root,commit,names,*,expected=None):
    """Prove exact raw/uniform-CRLF bytes exist in the recorded Git tree.

    Git diff can hide mixed endings and clean-filter transformations. Compare
    unfiltered blob identities instead; bundle files that cannot be proved.
    Read the tree once and one candidate file at a time, without executing filters.
    """
    raw=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','-z',commit,'--'],stderr=subprocess.DEVNULL)
    blobs={}
    for entry in raw.split(b'\0'):
        if not entry:continue
        header,name=entry.split(b'\t',1);mode,kind,oid=header.split()
        if kind==b'blob':blobs[os.fsdecode(name)]=oid.decode('ascii')
    result=set()
    for name in names:
        oid=blobs.get(name)
        if oid is None or not (root/name).is_file():continue
        data=(root/name).read_bytes()
        if expected is not None:
            record=expected[name]
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:continue
        def matches(value):
            algorithm='sha1' if len(oid)==40 else 'sha256'
            h=hashlib.new(algorithm);h.update(b'blob '+str(len(value)).encode('ascii')+b'\0');h.update(value)
            return h.hexdigest()==oid
        if matches(data):
            result.add(name);continue
        lf=data.replace(b'\r\n',b'\n')
        if data==lf.replace(b'\n',b'\r\n') and matches(lf):result.add(name)
    return result


class RestorePlan:
    def __init__(self,root,manifest,*,below,sha,require,read,encode,publish,verify_only=False,strict=False,
                 original_state_present=False,use_checkout_code=False):
        self.root,self.manifest=root,manifest
        self.below,self.sha,self.require,self.read,self.encode,self.publish=below,sha,require,read,encode,publish
        self.targets={};self.replacements={};self.external={};self.actions={}
        self.use_checkout_code=use_checkout_code;self.checkout={}
        self.base=below(root,'artifacts/step5-transfer/restore-'+manifest['id'][:12])
        self.header=self.base/'archive.json'
        self.identity=dict(contract=CONTRACT,archive_id=manifest['id'],original_state_present=original_state_present)
        self.entries={}
        if self.base.exists():
            if self.header.is_file():
                saved=read(self.header)
                require(set(saved)==set(self.identity) and saved['contract']==CONTRACT and saved['archive_id']==manifest['id']
                    and type(saved['original_state_present']) is bool,'Conflicting repository restore journal')
                self.identity=saved
            require(self.base.is_dir() and (self.header.is_file() or
                not self.header.exists() and all(p.is_file() and re.fullmatch(r'\.restore-[0-9a-f]{32}\.partial',p.name)
                    for p in self.base.iterdir())), 'Conflicting repository restore journal')
            for file in self.base.glob('f-*.json'):
                entry=read(file);name=entry['path'];record=next((r for r in manifest['files'] if r['path']==name),None)
                require(record is not None and file==self.entry_path(name) and entry.get('contract')==CONTRACT and entry['archive_id']==manifest['id']
                    and entry['archive_sha256']==record['sha256'] and entry['kind'] in ('replace','transport')
                    and entry['id']==self.digest({k:v for k,v in entry.items() if k!='id'}),'Repository restore journal binding changed')
                if entry['kind']=='replace':
                    backup=below(root,entry['backup'])
                    require(backup.parent==self.base and backup.is_file() and sha(backup)==entry['before_sha256'],
                        'Repository restore backup missing/changed')
                self.entries[name]=entry
        tracked=tracked_files(root);blocked=[]
        for record in manifest['files']:
            name=record['path'];path=below(root,name);entry=self.entries.get(name)
            require(not path.is_relative_to(self.base),'Archive overlaps the private restore journal')
            repo=(name in tracked and (name.startswith(PREFIXES) or name.startswith('requirements') and '/' not in name)
                  and Path(name).suffix in TEXT)
            if use_checkout_code and repo and path.is_file():
                # Explicitly keep the current checkout; bind its bytes for the
                # duration of restore/verify without claiming archive identity.
                self.checkout[name]=sha(path)
                continue
            # Reuse an already published transport snapshot on repeat/verify.
            shadow=(entry is not None and entry['kind']=='transport')
            if shadow:
                require(name in TOOLS,'Unexpected transport snapshot')
                path=self.base/('t-'+record['sha256'][:32]);self.targets[name]=path
            elif path.is_file() and sha(path)==record['sha256']:
                continue
            elif repo and name in TOOLS and path.is_file() and not verify_only and not strict:
                path=self.base/('t-'+record['sha256'][:32]);self.targets[name]=path
                self.actions[name]=dict(kind='transport')
            if path.is_file() and sha(path)==record['sha256']:continue
            if verify_only:
                blocked.append(('Conflicting destination: ' if path.exists() else 'Missing restored file: ')+name);continue
            if path.exists():
                if not path.is_file() or not repo or strict or name in self.targets:
                    blocked.append('Conflicting destination; private/untracked file: '+name);continue
                current=sha(path)
                if entry is not None and (entry['kind']!='replace' or current!=entry['before_sha256']):
                    blocked.append('Conflicting destination changed after its restore backup: '+name);continue
                self.replacements[name]=current;self.actions[name]=dict(kind='replace',before_sha256=current)
            if record.get('external',False):
                data=git_bytes(root,manifest,record) if repo else None
                if data is None:
                    blocked.append('Missing external/archive-pinned Git dependency: '+name+
                        ' (need exact archive bytes from Git commit '+str(manifest.get('git_commit'))+
                        '; Git normalization may have omitted non-reproducible working-tree bytes)')
                else:self.external[name]=data
        require(not blocked,'Restore preflight found '+str(len(blocked))+' blocking file(s); nothing overwritten:\n'+'\n'.join(blocked))

    def digest(self,value):return hashlib.sha256(self.encode_canonical(value)).hexdigest()

    @staticmethod
    def encode_canonical(value):
        import json
        return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

    def entry_path(self,name):return self.base/('f-'+hashlib.sha256(name.encode()).hexdigest()[:24]+'.json')

    def target(self,record):return self.targets.get(record['path'],self.below(self.root,record['path']))

    def matches(self,record):
        path=self.target(record)
        return path.is_file() and self.sha(path)==self.checkout.get(record['path'],record['sha256'])

    def write_once(self,path,data):
        if path.exists():
            self.require(path.is_file() and path.read_bytes()==data,'Repository restore evidence changed: '+str(path));return
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name('.restore-'+uuid.uuid4().hex+'.partial')
        try:
            with temporary.open('xb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            self.publish(temporary,path)
        finally:temporary.unlink(missing_ok=True)

    def prepare(self):
        # Retain the first destination's identity even if a process stops after
        # restoring state.json but before installing its fresh-buffer marker.
        self.write_once(self.header,self.encode(self.identity))
        records={r['path']:r for r in self.manifest['files']}
        # Back up EVERY replacement before altering any original file.
        for name,action in self.actions.items():
            if name in self.entries:continue
            record=records[name];entry=dict(contract=CONTRACT,archive_id=self.manifest['id'],path=name,
                archive_sha256=record['sha256'],kind=action['kind'])
            if action['kind']=='replace':
                path=self.below(self.root,name);data=path.read_bytes()
                self.require(hashlib.sha256(data).hexdigest()==action['before_sha256'],'Destination changed during restore planning: '+name)
                backup=self.base/('b-'+action['before_sha256'][:32]);self.write_once(backup,data)
                entry.update(before_sha256=action['before_sha256'],before_bytes=len(data),before_mode=path.stat().st_mode,
                    backup=backup.relative_to(self.root).as_posix())
            entry['id']=self.digest(entry);self.write_once(self.entry_path(name),self.encode(entry));self.entries[name]=entry

    def install(self,record,temporary):
        path=self.target(record);name=record['path']
        if name in self.targets and path.is_file() and self.sha(path)==record['sha256']:return
        if name in self.replacements:
            self.require(path.is_file() and self.sha(path)==self.replacements[name],
                'Destination changed after restore backup: '+name)
            temporary.replace(path)
        else:
            self.require(not path.exists(),'Destination appeared during restore: '+name)
            self.publish(temporary,path)

    def report(self):
        records={r['path']:r for r in self.manifest['files']}
        return dict(policy=CONTRACT,receipt=self.header.relative_to(self.root).as_posix() if self.header.is_file() else None,
            backups=[dict(path=e['path'],backup=e['backup'],sha256=e['before_sha256'])
                for e in self.entries.values() if e['kind']=='replace'],
            preserved_transport=[dict(path=n,snapshot=self.target({'path':n}).relative_to(self.root).as_posix())
                for n,e in self.entries.items() if e['kind']=='transport' and n not in self.checkout],
            checkout_code=dict(enabled=self.use_checkout_code,files=len(self.checkout),
                differing_files=[dict(path=n,checkout_sha256=h,archive_sha256=records[n]['sha256'])
                    for n,h in self.checkout.items() if h!=records[n]['sha256']]))
