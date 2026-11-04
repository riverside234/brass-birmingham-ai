"""Fresh r8 extraction and exact overlay against the passing lab inventory."""
import hashlib,importlib.util,json,os,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/rust-backend/g570-b03-cuda'
sys.path.insert(0,str(ROOT/'src'))
from brass_ai.rust_backend.lab_contract import sha
from brass_ai.rust_backend.lab_transfer import verify,DIRECTORY
from brass_ai.rust_backend.lab_prerequisite_reuse import reviewed,verify_delta

def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())

receipt=read(ROOT/DIRECTORY/'bundle.json');manifest=read(ROOT/DIRECTORY/'transfer-manifest.json')
assert receipt['path'].endswith('g570-r8.zip')
stages=[verify(stage) for stage in ('b01','b02','b025','b03','b04')]
destination=ROOT/'artifacts/r8-final';assert not destination.exists()
old=read(OUT/'lab-evidence/evidence/b01/python-runtime-sources.json')
def relevant(name):
    if Path(name).suffix not in ('.py','.json','.cs','.toml','.dll','.exe','.so'):return False
    return name.startswith(('src/','configs/','training/runtime/')) or name in (
        'scripts/build-training-host.py','scripts/build-search-host.py','scripts/build-local-scoring-host.py')
previous={item['path']:item['sha256'] for item in old if relevant(item['path'])};current=dict(previous)
with zipfile.ZipFile(ROOT/receipt['path']) as archive:
    assert archive.testzip() is None
    for name,expected in manifest['files'].items():
        raw=archive.read(name)
        assert raw==(ROOT/name).read_bytes(),name
        assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==expected,name
        if relevant(name):current[name]=hashlib.sha256(raw).hexdigest()
    for name,expected in manifest['archives'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==expected
    assert not any(name.startswith('results/rust-lab-qualification/') and name!='results/rust-lab-qualification/README.md' for name in archive.namelist())
    archive.extractall(destination)
    # The overlay leaves all unrelated baseline/runtime bytes from the passing
    # lab untouched. Check its entire resulting relevant source inventory.
    verify_delta(previous,current,reviewed())
import shutil
shutil.copyfile(ROOT/DIRECTORY/'bundle.json',destination/DIRECTORY/'bundle.json')
shutil.copyfile(ROOT/receipt['path'],destination/receipt['path'])
env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(destination/name) for name in (
    'src','scripts','tests/R2','tests/Recovery','tests/FeatureEngineering','tests/RustBackendLab')),CUDA_VISIBLE_DEVICES='')
commands=[('integrity',[sys.executable,'-X','utf8','-c',
    'from brass_ai.rust_backend.lab_transfer import verify; from brass_ai.rust_backend.lab_prerequisite_reuse import reviewed; print([verify(s)["milestone"] for s in ("b01","b02","b025","b03","b04")]); print(reviewed()["scope"])']),
    ('regression',[sys.executable,'-X','utf8','-m','unittest','-v','test_b03_cuda_restore','test_qualification_runner','test_lab_transfer'])]
for name,command in commands:
    print('Fresh bundle: '+name,flush=True)
    with (OUT/('extracted-'+name+'.log')).open('wb') as log:
        result=subprocess.run(command,cwd=destination,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    assert result.returncode==0,name
report=dict(passed=True,archive=receipt,fresh_extraction=True,transfer_stages=stages,
    exact_lab_inventory_overlay=True,changed_files=list(reviewed()['changes']),
    original_result_folders_untouched=True,prerequisites_repeated=False,
    scope='Bundle/inventory reuse and focused CPU tests only; prior B01/B02/B02.5 passes retained; B03/B04 lab Pending',
    logs={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),*OUT.glob('extracted-*.log'))})
write(OUT/'package-check.json',report);write(ROOT/DIRECTORY/'package-check.json',report)
print(json.dumps(dict(passed=True,reuse_inventory=True,prerequisites_repeated=False)))
