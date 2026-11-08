"""Bounded ordinary data-path/check-policy regressions; no training update."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery.replay import check_game
from brass_ai.rust_backend import lab_shared as S,lab_pool as P,lab_perf as V,lab_admission as A
from brass_ai.rust_backend.lab_delta import Decoder
from test_b03 import family_fixture


def shared_fixture_actors(connections,root,build,execution,mono,lease,cpu,stop,directory):
    from concurrent.futures import ThreadPoolExecutor
    from brass_ai.recovery.telemetry import Timings
    def lane(connection):
        connection.send(dict(ready=True))
        while not stop.is_set():
            if not connection.poll(.05):continue
            command=connection.recv()
            if command is None:return
            with cpu.acquire(1,Timings()):part=exported(json.loads(Path(build['fixture']).read_text()))
            def allocate(size):
                connection.send(dict(shared_allocate=dict(size=size)))
                return connection.recv()['shared_lease']
            result=S.write(part,allocate,admit=lambda:cpu.acquire(1,Timings()))
            connection.send(dict(result=result))
    with ThreadPoolExecutor(max_workers=len(connections)) as workers:list(workers.map(lane,connections))


def exported(part):
    """Authored counterpart of Rust pages, preserving byte seals and arrays."""
    part=copy.deepcopy(part);game=check_game(part['game']);records=game['records'];payload=bytearray();lengths=[]
    hashes=[hashlib.sha256(b'['),hashlib.sha256(b'[')]
    for index,(row,evidence) in enumerate(zip(records,part.pop('evidence'),strict=True)):
        pair=[json.dumps(v,separators=(',',':'),allow_nan=False).encode() for v in (row,evidence)]
        lengths.append(list(map(len,pair)))
        for h,raw in zip(hashes,pair,strict=True):
            if index:h.update(b',')
            h.update(raw);payload.extend(raw)
    for h in hashes:h.update(b']')
    token='authored-token';receipt=part['native_receipt']
    receipt.update(token=token,records_sha256=hashes[0].hexdigest(),evidence_sha256=hashes[1].hexdigest())
    arrays=[]
    for row in records:
        e=game.encoded(row,None)
        arrays.append(e.state.astype('<f4').tobytes()+e.actions.astype('<f4').tobytes()+e.stable_to_relative.astype('<i8').tobytes())
    part['game']={k:v for k,v in game.items() if k!='records'};part['actor_validated']=True
    page=dict(wire='ordinary-records/1',token=token,start=0,end=len(records),rows=len(records),lengths=lengths,bytes=len(payload))
    return S.Export(part,[(page,bytes(payload))],arrays)


class Performance(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(temp.cleanup);self.out=Path(temp.name)
        self.cancel=threading.Event();self.registry=S.Registry(self.cancel);self.addCleanup(self.registry.close)

    def test_shared_publication_has_no_temporary_payload_and_equal_training_arrays(self):
        f,a,c=family_fixture(decisions=32);original=A.convert(f,a)
        handle=S.write(exported(f['parent']),self.registry.allocate)
        self.registry.verify(handle)
        with patch.object(P,'part_write',side_effect=AssertionError('temporary file')):
            record=P.Publisher(self.out,f['producer'],c,'ordinary').publish(f['frozen_job'],handle,None)
        self.registry.release(handle)
        actual=A.read(record['path'],record['native_admission'])
        self.assertEqual(digest(actual),digest(original))
        for before,after in zip(original['records'],actual['records'],strict=True):
            for name in ('state','actions','stable_to_relative'):
                np.testing.assert_array_equal(getattr(original.encoded(before,None),name),getattr(actual.encoded(after,None),name))
        self.assertFalse((self.out/'parts').exists());self.assertEqual(self.registry.report()['active_bytes'],0)
        self.assertEqual(handle['transfer_timings']['temporary_payload_write_bytes'],0)

    def test_chunk_boundary_tamper_lease_identity_and_release(self):
        part=S.Export(dict(branch_expected=False),[],[b'abc'*100])
        with patch.object(S,'CHUNK',128):
            handle=S.write(part,self.registry.allocate);restored=S.read(handle)
            self.assertEqual(bytes(restored.features[0]),b'abc'*100)
            altered=copy.deepcopy(handle);altered['chunks'][0]['name']='foreign'
            with self.assertRaisesRegex(ValueError,'identity'):self.registry.verify(altered)
            shm=S.shared_memory.SharedMemory(name=handle['chunks'][0]['name'])
            try:shm.buf[0]^=1
            finally:shm.close()
            with self.assertRaisesRegex(ValueError,'changed'):S.read(handle)
        names=[x['name'] for x in handle['chunks']];self.registry.release(handle);self.registry.release(handle)
        for name in names:
            with self.assertRaises(FileNotFoundError):S.shared_memory.SharedMemory(name=name)
        self.assertEqual(self.registry.report()['active_leases'],0)

    def test_allocation_backpressure_cancellation_and_failed_publisher_cleanup(self):
        registry=S.Registry(self.cancel,budget=256);self.addCleanup(registry.close)
        first=registry.allocate(200);errors=[]
        def blocked():
            try:registry.allocate(200)
            except BaseException as e:errors.append(e)
        thread=threading.Thread(target=blocked);thread.start();self.cancel.set();thread.join(timeout=2)
        self.assertFalse(thread.is_alive());self.assertEqual(len(errors),1);registry.release(first)
        self.assertEqual(registry.used,0)

    def test_cancellation_after_allocation_keeps_parent_ownership_until_cleanup(self):
        part=S.Export(dict(branch_expected=False),[],[b'x'*64])
        def allocate(size):
            value=self.registry.allocate(size);self.cancel.set();return value
        with self.assertRaisesRegex(ValueError,'export canceled'):S.write(part,allocate,cancel=self.cancel)
        self.assertGreater(self.registry.used,0);self.registry.close();self.assertEqual(self.registry.used,0)

    def test_raw_seal_context_and_row_order_are_rejected(self):
        f,_,_=family_fixture();base=exported(f['parent'])
        changed=copy.deepcopy(base);changed.metadata['native_receipt']['records_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'seal'):S.decode(changed)
        changed=copy.deepcopy(base);changed.records[0][0]['start']=1
        with self.assertRaisesRegex(ValueError,'binding'):S.decode(changed)
        changed=copy.deepcopy(base);values=np.frombuffer(changed.features[0],dtype='<f4').copy();values[3127]=1.
        changed.features[0]=values.tobytes()
        with self.assertRaisesRegex(ValueError,'context'):S.decode(changed)

    def test_fast_skips_rng_probability_reconstruction_and_strict_detects_fault(self):
        f,a,c=family_fixture(decisions=32);part=S.decode(exported(f['parent']))
        selection=V.selection(f['frozen_job'],ordinal=1,selected_mode='production_fast')
        self.assertFalse(selection['semantic'])
        with patch('brass_ai.rust_backend.lab_families.exploration_distribution',side_effect=AssertionError('RNG replay')):
            record=P.Publisher(self.out,f['producer'],c,'ordinary').publish(f['frozen_job'],
                P.part_write(self.out/'parts',part),None,validation=selection)
        receipt=record['native_admission'];self.assertFalse(receipt['validation']['coverage']['semantic_audited'])
        self.assertEqual(receipt['validation']['contract'],'brass-native-validation/2.0.0')
        self.assertEqual(digest(A.read(record['path'],receipt)),digest(A.read(record['path'],receipt,reuse_validation=False)))
        bad=copy.deepcopy(f);bad['parent']['game']['records'][0]['behavior']['probability']*=.9
        bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'denominator'):A.convert(bad,a)

    def test_fast_mandatory_identity_and_finite_probability_checks_remain(self):
        f,a,_=family_fixture();part=S.decode(exported(f['parent']));sel=V.selection(f['frozen_job'],ordinal=1,selected_mode='production_fast')
        for bad in ('seat','probability','action','evidence','kind'):
            changed=copy.deepcopy(part);row=changed['game']['records'][0]
            if bad=='seat':row['seat']=9
            elif bad=='probability':row['behavior']['probability']=float('nan')
            elif bad=='action':row['chosen']=999
            elif bad=='kind':row['behavior']['kind']='external'
            else:changed['evidence'][0]['semantic_id']='foreign'
            family=dict(f,parent=changed);family['sha256']='private-owned-seal'
            with self.assertRaises((ValueError,KeyError)):
                A.convert(family,a,_owned=True,_coverage=V.coverage(sel))

    def test_coverage_is_bound_and_cold_strict_can_recheck_fast_receipt(self):
        f,_,c=family_fixture();sel=V.selection(f['frozen_job'],ordinal=1,selected_mode='production_fast')
        record=P.Publisher(self.out,f['producer'],c,'ordinary').publish(f['frozen_job'],
            P.part_write(self.out/'parts',S.decode(exported(f['parent']))),None,validation=sel)
        wrong=copy.deepcopy(record['native_admission']);wrong['validation']['coverage']['semantic_audited']=True
        with self.assertRaisesRegex(ValueError,'receipt changed'):A.read(record['path'],wrong)
        with patch('brass_ai.rust_backend.lab_families.exploration_distribution',side_effect=AssertionError('strict replay')):
            A.read(record['path'],record['native_admission'])
            with self.assertRaisesRegex(AssertionError,'strict replay'):
                A.read(record['path'],record['native_admission'],reuse_validation=False)
            with patch.dict(os.environ,BRASS_COLLECTOR_VALIDATION='strict'):
                with self.assertRaisesRegex(AssertionError,'strict replay'):A.read(record['path'],record['native_admission'])

    def test_unsampled_semantic_fault_is_not_misreported_as_fully_checked(self):
        f,a,_=family_fixture();f['parent']['game']['records'][0]['behavior']['probability']*=.9
        f['parent']=S.decode(exported(f['parent']));f['sha256']=digest({k:v for k,v in f.items() if k!='sha256'})
        selection=V.selection(f['frozen_job'],ordinal=1,selected_mode='production_fast')
        fast=A.convert(f,a,_owned=True,_coverage=V.coverage(selection))
        self.assertTrue(fast['records'])
        with self.assertRaisesRegex(ValueError,'denominator'):A.convert(f,a)

    def test_shared_branch_reconstruction_and_original_targets_are_preserved(self):
        f,a,_=family_fixture(branch=True);before=A.convert(f,a,allow_diagnostic=True)
        for role in ('parent','child'):f[role]=S.decode(exported(f[role]))
        f['sha256']=digest({k:v for k,v in f.items() if k!='sha256'})
        after=A.convert(f,a,allow_diagnostic=True)
        self.assertEqual(digest(before),digest(after));self.assertEqual(len(after['branches']),1)

    def test_runtime_upgrade_preserves_only_the_exact_original_cache_producer(self):
        f,_,c=family_fixture();job=f['frozen_job'];old=P.Publisher(self.out,f['producer'],c,'ordinary')
        record=old.publish(job,P.part_write(self.out/'parts',f['parent']),None)
        changed=dict(f['producer'],build_sha256='f'*64)
        upgraded=P.Publisher(self.out,changed,c,'ordinary',previous_producer=f['producer'])
        restored=upgraded.load(job)
        self.assertEqual(restored['sha256'],record['sha256'])
        self.assertEqual(restored['native_admission']['admission']['producer'],f['producer'])
        with self.assertRaisesRegex(ValueError,'different job/producer'):
            P.Publisher(self.out,changed,c,'ordinary').load(job)

    def test_runtime_resolver_verifies_source_parent_and_pinned_build_evidence(self):
        from brass_ai.rust_backend import lab_perf_runtime as R
        from brass_ai.rust_backend.lab_contract import pin,SOURCE
        from brass_ai.step5.league import write_json
        path=self.out/'artifacts/rust-backend/collector-performance/build.json';path.parent.mkdir(parents=True)
        for name in ('collector','bootstrap','tests','build_log','performance_tests'):(self.out/name).write_bytes(name.encode())
        original=dict(_receipt_sha256='a'*64,b01=dict(sha256='b'*64),
            bootstrap=dict(binary=pin(self.out/'bootstrap'),dependencies=[]))
        built=dict(original,contract=R.CONTRACT,ordinary_source=R.fingerprint(),parent_build_sha256='a'*64,
            shared_core=SOURCE,search_enabled=False,search_method='search_v1',v205_enabled=False,
            performance_wire='ordinary-collector-fast/1',wire_regressions_passed=True,collector=pin(self.out/'collector'),
            **{name:pin(self.out/name) for name in ('tests','build_log','performance_tests')})
        built.pop('_receipt_sha256');write_json(path,built)
        self.assertEqual(R.resolve(self.out,original)['_previous_build_sha256'],'a'*64)
        write_json(path,dict(built,ordinary_source='0'*64))
        with self.assertRaisesRegex(ValueError,'runtime/source/parent'):R.resolve(self.out,original)
        write_json(path,built);(self.out/'performance_tests').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'evidence changed'):R.resolve(self.out,original)

    def launcher(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pi_prepare',SOURCE_ROOT/'scripts/prepare-rust-collector-performance.py')
        result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);result.ROOT=self.out
        return result

    def test_prepare_launcher_pins_generation_reuses_runtime_and_never_changes_checkpoint(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import rust_lab_b07_retry as retry
        from brass_ai.rust_backend import lab_backend as B,lab_perf_runtime as R
        from brass_ai.rust_backend.lab_contract import pin
        from brass_ai.step5.league import write_json
        launch=self.launcher();run=self.out/'artifacts/run';run.mkdir(parents=True)
        checkpoint=run/'model.pt';checkpoint.write_bytes(b'unchanged checkpoint fixture')
        state=dict(iteration=1257,history=[dict(generation=1257,path='artifacts/run/model.pt',sha256=pin(checkpoint)['sha256'])])
        write_json(run/'state.json',state);before=(run/'state.json').read_bytes()
        original=dict(_receipt_sha256='a'*64,b01=dict(path=str(self.out/'b01/qualification.json')),source_reuse=dict(scope='historical unchanged binary'))
        def build(out,*args,**kwargs):
            out.mkdir();binary=out/'collector';binary.write_bytes(b'fixture runtime')
            return dict(collector=pin(binary))
        args=SimpleNamespace(run='artifacts/run',expected_generation=1257,offline=True);evidence=self.out/'evidence';evidence.mkdir()
        with patch.object(B,'validate_state',return_value=dict(mode='rust',build={})),\
                patch.object(B,'check_build',return_value=original),patch.object(R,'resolve',return_value={}),\
                patch.object(retry,'build_ordinary',side_effect=build) as builder,patch.object(launch.subprocess,'run'):
            one=launch.prepare(args,evidence);two=launch.prepare(args,evidence)
            self.assertFalse(one['reused']);self.assertTrue(two['reused']);self.assertEqual(builder.call_count,1)
            self.assertNotIn('source_reuse',builder.call_args.args[1]);self.assertNotIn('_receipt_sha256',builder.call_args.args[1])
            args.expected_generation=1258
            with self.assertRaisesRegex(ValueError,'Stale generation'):launch.prepare(args,evidence)
        self.assertEqual(before,(run/'state.json').read_bytes());self.assertEqual(checkpoint.read_bytes(),b'unchanged checkpoint fixture')

    def test_prepare_launcher_exports_first_error_without_replacing_it(self):
        import sys
        import rust_lab_results
        launch=self.launcher();exported_paths=[]
        with patch.object(sys,'argv',['prepare','--expected-generation','1257']),\
                patch.object(launch,'prepare',side_effect=RuntimeError('original compiler failure')),\
                patch.object(rust_lab_results,'export',side_effect=lambda p:exported_paths.append(p) or self.out/'results'):
            with self.assertRaisesRegex(RuntimeError,'original compiler failure'):launch.main()
        self.assertEqual(len(exported_paths),1)
        report=json.loads((exported_paths[0]/'qualification.json').read_text())
        self.assertFalse(report['passed']);self.assertEqual(report['error'],'original compiler failure')

    def test_component_probe_exports_failed_report_without_running_a_benchmark(self):
        import importlib.util,sys
        import rust_lab_results
        spec=importlib.util.spec_from_file_location('pi_profile',SOURCE_ROOT/'tests/RustBackendLab/profile_collector_performance.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=self.out
        output=self.out/'artifacts/rust-backend/failed-profile';paths=[]
        with patch.object(sys,'argv',['profile','--out',str(output),'--decisions','0','--export-results']),\
                patch.object(rust_lab_results,'export',side_effect=lambda p:paths.append(p) or self.out/'results'):
            with self.assertRaisesRegex(ValueError,'bounded'):module.main()
        self.assertEqual(paths,[output]);self.assertFalse(json.loads((output/'report.json').read_text())['passed'])

    def matched_launcher(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pi_matched',SOURCE_ROOT/'scripts/profile-rust-collector-performance.py')
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);value.ROOT=self.out
        return value

    def test_matched_snapshot_is_checked_on_failure_and_keeps_first_error_without_gpu(self):
        from brass_ai.rust_backend import lab_snapshot
        from brass_ai.step5.league import write_json
        run=self.out/'run';run.mkdir();write_json(run/'state.json',dict(iteration=1257))
        module=self.matched_launcher();snapshot=dict(identity='frozen')
        with patch.object(lab_snapshot,'capture',return_value=(snapshot,None,None)) as capture,\
                patch.object(lab_snapshot,'verify_unchanged',side_effect=ValueError('also changed')) as verify:
            with self.assertRaisesRegex(RuntimeError,'original phase') as found:
                with module.frozen_run(self.out,run,1257,'cpu'):raise RuntimeError('original phase')
            self.assertTrue(any('also changed' in note for note in found.exception.__notes__))
            self.assertEqual(verify.call_count,1)
            with self.assertRaisesRegex(ValueError,'Stale'):
                with module.frozen_run(self.out,run,1258,'cpu'):pass
            self.assertEqual(capture.call_count,1)
        self.assertFalse((run/'.active-run.json').exists())

    def test_matched_generation_failure_exports_without_collection_or_checkpoint_load(self):
        import sys
        import rust_lab_results
        from brass_ai.step5.league import write_json
        module=self.matched_launcher();run=self.out/'artifacts/run';run.mkdir(parents=True)
        write_json(run/'state.json',dict(iteration=1257));paths=[]
        with patch.object(sys,'argv',['matched','--run','artifacts/run','--expected-generation','1258','--device','cpu']),\
                patch.object(rust_lab_results,'export',side_effect=lambda p:paths.append(p) or self.out/'results'):
            with self.assertRaisesRegex(ValueError,'Stale'):module.main()
        self.assertEqual(len(paths),1);report=json.loads((paths[0]/'qualification.json').read_text())
        self.assertFalse(report['passed']);self.assertIn('Stale',report['error'])

    def test_sampling_plan_is_frozen_and_does_not_consume_random(self):
        f,_,_=family_fixture();jobs=[dict(f['frozen_job'],index=i) for i in range(130)]
        before=random.getstate();one=V.plan(jobs,selected_mode='production_fast')
        self.assertEqual(before,random.getstate());self.assertEqual(one,V.plan(jobs,selected_mode='production_fast'))
        self.assertEqual([i for i,p in one.items() if p['semantic']],[0,64,128])
        self.assertTrue(all(v['semantic'] for v in V.plan(jobs,selected_mode='strict').values()))

    def test_frame_delta_reuses_immutable_prefix_rejects_stale_and_resets(self):
        decoder=Decoder();first=dict(decision=dict(view=dict(context=dict(revision=1),players=[dict(cash=4)])),
            history_delta=dict(base=0,previous_revision=None,length=1,tail=[dict(action='pass')]),view_reuse=[])
        decoder.reply(first)
        frame=dict(decision=dict(view=dict(context=dict(revision=2))),
            history_delta=dict(base=1,previous_revision=1,length=2,tail=[dict(action='loan')]),view_reuse=['players'])
        decoder.reply(frame)
        self.assertEqual(frame['decision']['view']['history'],[dict(action='pass'),dict(action='loan')])
        with self.assertRaises(TypeError):frame['decision']['view']['players'][0]['cash']=999
        stale=copy.deepcopy(first)
        stale['history_delta']=dict(base=0,previous_revision=999,length=0,tail=[]);stale['view_reuse']=[]
        with self.assertRaisesRegex(ValueError,'Stale'):decoder.reply(stale)

    def test_reused_candidate_roots_do_not_replace_previous_admission(self):
        from brass_ai.recovery.admitted_inputs import freeze,admitted_digest
        a,c=freeze(dict(context=dict(revision=1)),[dict(id='pass')],'first')
        b,d=freeze(dict(context=dict(revision=2)),c,'second')
        self.assertEqual(admitted_digest(a,c),'first');self.assertEqual(admitted_digest(b,d),'second')
        self.assertIsNot(c,d);self.assertIs(c[0],d[0])

    def test_real_pool_shared_handoff_recovery_cpu_cap_and_cleanup(self):
        from test_b07 import B07
        pool,job,cpu=B07().pool(self.out)
        with patch.object(P,'_actors',shared_fixture_actors):records=pool.collect([job])
        self.assertEqual(pool.shared.report()['active_bytes'],0)
        self.assertEqual(pool.last_collection_report['counts']['rust'],1)
        self.assertGreater(pool.last_collection_report['shared_buffers']['peak_bytes'],0)
        self.assertFalse((self.out/'native-ordinary/parts').exists())
        self.assertEqual(cpu.report()['active'],0)
        with patch.object(P,'_actors',side_effect=AssertionError('cache started actors')):
            again=pool.collect([job])
        self.assertEqual(again,records)
        self.assertEqual(pool.last_collection_report['counts']['cached'],1)


@unittest.skipUnless(os.environ.get('BRASS_B07_COLLECTOR'),'Set BRASS_B07_COLLECTOR for bounded native wire checks')
class NativeWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=os.environ['BRASS_B07_COLLECTOR']
        cls.roots=json.loads(Path(os.environ['BRASS_B07_FIXTURES']).read_text())['roots']

    def test_same_frozen_origins_actions_rng_features_records_and_branch_opportunities(self):
        from rust_lab_b07_transport import walk
        for root in (max(self.roots,key=lambda r:len(r['expected']['choices'])),self.roots[-1]):
            origin=dict(state=root['expected']['state'],rail_keys=list(range(64)))
            before=walk(self.binary,origin,optimize=False,limit=24 if root!=self.roots[-1] else 96,raw_export=True)
            after=walk(self.binary,origin,optimize=True,limit=24 if root!=self.roots[-1] else 96,raw_export=True)
            for key in ('trace','features','branches','rng','complete','forced','training_features'):
                self.assertEqual(before[key],after[key],key)
            self.assertLess(after['metrics']['ipc_received_bytes'],before['metrics']['ipc_received_bytes'])
            if after['complete']:
                self.assertGreater(after['metrics']['ipc_training_reused_bytes'],0)
                self.assertGreater(after['metrics']['ipc_training_reused_rows'],0)

    def test_new_game_does_not_reuse_prior_delta_or_features(self):
        from brass_ai.rust_backend.lab_transport import Collector,DecisionEncoder
        root=self.roots[0];origin=dict(state=root['expected']['state'],rail_keys=list(range(64)))
        with Collector(self.binary) as lane:
            lane.begin('one',origin,100);view=copy.deepcopy(lane.current['decision'])
            repeated=lane.call('decision')
            self.assertEqual(repeated['decision'],view)
            self.assertIs(repeated['decision']['choices'],lane.delta.choices)
            DecisionEncoder(lane,'fixture').encode(view['view'],view['choices'],[0.]*4,[0.]*8)
            self.assertTrue(lane.retained_features);lane.release();lane.begin('two',origin,100)
            self.assertFalse(lane.retained_features);self.assertEqual(lane.current['decision'],view)

    def test_actual_collection_adapter_raw_and_decoded_outputs_match_without_nn(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from brass_ai.rust_backend import lab_session as session_module
        from brass_ai.rust_backend.lab_collection import collect_lane
        from brass_ai.step5.policy import NeuralPolicy
        from brass_ai.recovery.admitted_inputs import request as admitted_request
        from test_inference_deadline_race import response
        def make(spec,*args):
            evaluator=SimpleNamespace(model_id=spec['model_id'],request_factory=admitted_request)
            def predict(requests):
                results=[]
                for r in requests:
                    evaluator.encoder.encode(r.view,r.choices,r.style,r.intent)
                    value=response(r);raw=np.zeros(len(r.choices));raw[next((i for i,c in enumerate(r.choices) if c['kind']=='pass'),0)]=1.
                    value['probabilities']=raw.tolist();results.append(value)
                return results
            evaluator.predict=predict
            return NeuralPolicy(None,evaluator=evaluator,focus=spec['focus'],exploration=spec['exploration'],
                                exploration_version=3,opening=spec.get('opening'))
        f,_,_=family_fixture();job=copy.deepcopy(f['frozen_job']);job.pop('branching',None)
        origin=dict(state=self.roots[-1]['expected']['state'],rail_keys=list(range(64)))
        outputs=[]
        for raw in (False,True):
            selected=V.selection(job,ordinal=1,selected_mode='production_fast') if raw else None
            with patch.object(session_module,'make_policy',side_effect=make):
                value=collect_lane(SOURCE_ROOT,self.binary,Mock(),job,None,dict(max_decisions=256),origin=origin,
                                   raw_export=raw,validation=selected)
            outputs.append(S.decode(value) if raw else value)
        before,after=outputs
        self.assertEqual(before['game']['result'],after['game']['result'])
        self.assertEqual(before['scoring'],after['scoring']);self.assertEqual(before['evidence'],after['evidence'])
        for left,right in zip(before['game']['records'],after['game']['records'],strict=True):
            self.assertEqual(dict(left,seconds=0.),dict(right,seconds=0.))
            for name in ('state','actions','stable_to_relative'):
                np.testing.assert_array_equal(getattr(before['game'].encoded(left,None),name),getattr(after['game'].encoded(right,None),name))


if __name__=='__main__':unittest.main()
