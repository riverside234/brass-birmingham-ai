"""B07 wire repair: exact native runtime pins and prior ordinary baseline."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, read_json, require
from brass_ai.rust_backend.lab_contract import SOURCE, extract_new, pin, native_fingerprint
from brass_ai.rust_backend.lab_teacher_overlay import apply, source_fingerprint
from brass_ai.step5.league import implementation_id, write_json

DIRECTORY=ROOT/'distribution/rust-lab-b07-g1250'
INPUTS=ROOT/'distribution/rust-lab-b06-g1250/evidence'


def module(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/'scripts'/name)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def original_runtime(reference):
    ordinary=read_json(INPUTS/'ordinary-build.json');teacher=read_json(INPUTS/'teacher-build.json')
    require(ordinary['shared_core']==SOURCE and ordinary['search_enabled'] is False and
        ordinary['search_method']=='search_v1' and ordinary['v205_enabled'] is False,'Original collector scope differs')
    require(teacher['producer']==reference['teacher_wire_review']['from_producer'],
            'Unexpected original teacher producer')
    require(source_fingerprint()==reference['teacher_wire_review']['to_producer'],'Teacher repair source review differs')
    for name,expected in reference['ordinary_native_source_review'].items():
        data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        require(hashlib.sha256(data).hexdigest()==expected,'Reviewed ordinary native source changed: '+name)
    pins=[ordinary['collector']]
    for build in (ordinary,teacher):pins += [build['bootstrap']['binary'],*build['bootstrap']['dependencies']]
    for item in pins:require(pin(item['path'])==item,'Original runtime changed: '+item['path'])
    return ordinary,teacher


def verify_teacher(value,original):
    from rust_lab_b07_completion import reuse_build
    value=reuse_build(value,'teacher')
    require(value['contract']=='brass-lab-b07-teacher-build/1.0.0' and
            value['producer']==source_fingerprint() and value['implementation']==implementation_id() and
            value['core_source']==original['core_source'] and value['bootstrap']==original['bootstrap'] and
            value['wire_regressions_passed'] is True,'Teacher repair build/source differs')
    for item in [value['binary'],value['tests'],value['build_log'],value['bootstrap']['binary'],
                 *value['bootstrap']['dependencies']]:
        require(pin(item['path'])==item,'Teacher repair runtime/evidence changed: '+item['path'])
    return value


def verify_ordinary(value,original):
    from rust_lab_b07_completion import reuse_build
    value=reuse_build(value,'ordinary')
    require(value['contract']=='brass-lab-b07-ordinary-build/1.0.0' and value['shared_core']==SOURCE
            and value['producer']==source_fingerprint() and value['implementation']==implementation_id()
            and value['bootstrap']==original['bootstrap'] and value['b01']==original['b01']
            and value['search_enabled'] is False and value['search_method']=='search_v1' and value['v205_enabled'] is False
            and value['collection_wire']=='compact-arrays-forced/1' and value['wire_regressions_passed'] is True,
            'Ordinary repair build/source differs')
    for item in [value['collector'],value['tests'],value['build_log'],value['bootstrap']['binary'],*value['bootstrap']['dependencies']]:
        require(pin(item['path'])==item,'Ordinary repair runtime/evidence changed: '+item['path'])
    return value


def build_teacher(out,original,**kwargs):return build_runtime(out,original,teacher=True,**kwargs)
def build_ordinary(out,original,**kwargs):return build_runtime(out,original,teacher=False,**kwargs)


def build_runtime(out,original,*,teacher,offline=False,environment=None):
    """Rebuild each adapter against its original sealed core; reuse C# pins."""
    base=module('qualify-rust-lab.py');b02=module('qualify-rust-lab-b02.py')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    env=dict(environment or base.python_environment())
    for key in list(env):
        if key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','CARGO_TARGET_DIR') or (key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS')):
            env.pop(key)
    extract_new(ROOT/'distribution/rust-lab-g1250/inputs.zip',out/'inputs')
    extract_new(out/'inputs/source.zip',out/'checkout')
    base.prepare_test_inputs(out/'checkout/native')
    overlay=apply(out/'checkout/native') if teacher else None
    if teacher:require(overlay['core_source']==original['core_source'],'Teacher repair must preserve shared core source')
    else:require(native_fingerprint(out/'checkout/native')==SOURCE,'Ordinary repair must preserve shared core source')
    shutil.copytree(ROOT/'lab-native',out/'checkout/lab-native',ignore=shutil.ignore_patterns('target'))
    b02.check_shared_dependencies(out/'checkout/native/Cargo.lock',out/'checkout/lab-native/Cargo.lock')
    version=subprocess.check_output(['rustc','+1.99.1','-vV'],text=True,env=env,timeout=15)
    target=next(x.split(': ',1)[1] for x in version.splitlines() if x.startswith('host: '))
    common=['--manifest-path',out/'checkout/lab-native/Cargo.toml','--release',
            '--target',target,'--target-dir',out/'target','--locked','-j','2']
    if teacher:common+=['--features','teacher']
    if offline:common+=['--offline']
    # Cargo runs the wire regression before any live model/game qualification.
    for op in ('test','build'):
        base.command(['cargo','+1.99.1',op,*common,*(['--bin','brass-lab-teacher' if teacher else 'brass-lab-collector'] if op=='build' else [])],
                     cwd=ROOT,env=env,log=out/(op+'.log'),timeout=900)
    if not teacher:
        result=dict(original,contract='brass-lab-b07-ordinary-build/1.0.0',producer=source_fingerprint(),
            implementation=implementation_id(),collector=pin(out/'target'/target/'release'/('brass-lab-collector'+('.exe' if os.name=='nt' else ''))),
            tests=pin(out/'test.log'),build_log=pin(out/'build.log'),wire_regressions_passed=True,rustc=version,
            collection_wire='compact-arrays-forced/1')
        write_json(out/'build.json',result)
        return verify_ordinary(result,original)
    result=dict(contract='brass-lab-b07-teacher-build/1.0.0',producer=source_fingerprint(),
        implementation=implementation_id(),core_source=overlay['core_source'],overlay=overlay,
        binary=pin(out/'target'/target/'release'/('brass-lab-teacher'+('.exe' if os.name=='nt' else ''))),
        bootstrap=copy.deepcopy(original['bootstrap']),tests=pin(out/'test.log'),build_log=pin(out/'build.log'),
        wire_regressions_passed=True,rustc=version,ordinary_runtime_replaced=False)
    write_json(out/'build.json',result)
    return verify_teacher(result,original)


def prior_ordinary(reference,snapshot):
    """Retain prior-source timing as history, never a current-source pass."""
    reuse=reference['ordinary_baseline']
    values={}
    for name,entry in reuse['evidence'].items():
        path=DIRECTORY/'evidence'/entry['file']
        require(hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],'Reused ordinary evidence changed')
        values[name]=read_json(path)
    q,checks=values['qualification'],values['ordinary']
    require(q['milestone']=='B07' and q['implementation']==reuse['from_implementation'] and
            q['failed_phase']=='canary-1' and checks['passed'] is True and checks['matched']['equal'] is True,
            'Ordinary evidence is not the reviewed successful phase')
    require(q['generation']==snapshot['generation'] and q['checkpoint_sha256']==snapshot['checkpoint']['sha256'] and
            q['config_sha256']==snapshot['config_sha256'],'Reused ordinary checkpoint/config changed')
    require(reuse['to_implementation']==implementation_id(),'Ordinary reuse source review differs')
    result=copy.deepcopy(checks)
    result['historical_baseline']=dict(current_qualification=False,original_implementation=q['implementation'],
        current_implementation=implementation_id(),original_result=reuse['result'],
        reason='Prior B07 phase; changed ordinary publication/reads require a new matched bounded comparison')
    return result
