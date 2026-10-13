"""Versioned P6 artifacts, portable paths and private save envelopes."""
from functools import lru_cache
from pathlib import Path
import json
import os

from brass_ai.learning.contracts import digest, require
from brass_ai.bank import sha
from brass_ai.personality.difficulty.contracts import identity as p5_identity

PACKAGE='personality-laptop-package/1.2.0'
EXPORT='personality-named-onnx/1.0.0'
SAVE='personality-runtime-save/1.0.0'
PROTOCOL='personality-game-rpc/1.0.0'
SAMPLING='personality-seat-sha256/1.0.0'
BENCHMARK='personality-laptop-benchmark/1.0.0'
BACKENDS=('ort-cpu','openvino-cpu','openvino-gpu')


@lru_cache(maxsize=1)
def identity():
    base=Path(__file__).parent;root=base.parents[3]
    files=list(base.glob('*.py'))+list((root/'src/BrassPersonalityGame').glob('*.cs'))
    return dict(version=PACKAGE,p5=p5_identity(),source=digest({
        p.relative_to(root).as_posix():p.read_text(encoding='utf-8').replace('\r\n','\n') for p in sorted(files)}))


def child(root,name):
    root=Path(root).resolve();p=Path(name)
    require(isinstance(name,str) and name and not p.is_absolute() and '\\' not in name
            and ':' not in name and '..' not in p.parts,'Unsafe package path')
    path=(root/p).resolve();require(path.is_relative_to(root) and path!=root,'Path escapes package')
    require(not any((root/Path(*p.parts[:i])).is_symlink() for i in range(1,len(p.parts)+1)), 'Package symlinks are unsupported')
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def seal(value):return dict(value,id=digest(value))


def unseal(value,version):
    require(value.get('format')==version and value.get('id')==digest({k:v for k,v in value.items() if k!='id'}),
            'Artifact format or hash mismatch')
    return value


def inventory(directory):
    root=Path(directory).resolve()
    return [dict(path=p.relative_to(root).as_posix(),sha256=sha(child(root,p.relative_to(root).as_posix())),bytes=p.stat().st_size)
            for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'
            and '__pycache__' not in p.parts and p.suffix!='.pyc']
