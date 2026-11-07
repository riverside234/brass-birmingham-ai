"""Short B07 publication/process/restart/adoption boundary regressions."""
import copy
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.step5.league import write_json,implementation_id
from brass_ai.rust_backend import lab_pool as P,lab_admission as A,lab_backend as B
from brass_ai.rust_backend.lab_families import FamilyStore
from brass_ai.recovery.resources import CPUAdmission
from test_b03 import family_fixture


def fixture_actors(connections,root,build,execution,mono,lease,cpu,stop,directory):
    """Spawned wire peer; real publisher, files, scheduler and CPU gate are used."""
    def lane(connection):
        connection.send(dict(ready=True))
        while not stop.is_set():
            if not connection.poll(.05):continue
            command=connection.recv()
            if command is None:return
            if build.get('fail'):
                connection.send(dict(error='original native lane failure'));return
            part=json.loads(Path(build['fixture']).read_text())
            connection.send(dict(result=P.part_write(directory,part)))
    with ThreadPoolExecutor(max_workers=len(connections)) as pool:list(pool.map(lane,connections))


class B07(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def temporary(self):
        value=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(value.cleanup);return Path(value.name)

    def test_one_full_validation_preserves_replay_and_targets(self):
        from rust_lab_b06_metrics import compare_games,compare_rows
        from rust_lab_b06_ordinary import prepared_rows
        f,a,c=family_fixture();out=self.temporary();job=f['frozen_job']
        parent=P.part_write(out/'parts',f['parent']);publisher=P.Publisher(out,f['producer'],c,'ordinary')
        original=A.validate_family
        with patch.object(A,'validate_family',wraps=original) as validation:
            record=publisher.publish(job,parent,None)
            self.assertEqual(validation.call_count,1)
        self.assertFalse(Path(parent['path']).exists())
        game=A.read(record['path'],record['native_admission']);baseline=A.convert(f,a)
        compare_games([game],[baseline])
        learner=job['provenance']['learner_behavior']['policy_id']
        compare_rows(prepared_rows([game],c,learner),prepared_rows([baseline],c,learner))
        self.assertEqual(publisher.load(job),record)

    def test_unsealed_partial_family_cannot_be_loaded_and_retry_recovers_publication_gap(self):
        f,a,c=family_fixture();out=self.temporary();job=f['frozen_job'];publisher=P.Publisher(out,f['producer'],c,'ordinary')
        self.assertIsNone(publisher.load(job))
        handle=P.part_write(out/'parts',f['parent'])
        with patch.object(P,'write_json',side_effect=RuntimeError('crash before receipt')):
            with self.assertRaisesRegex(RuntimeError,'crash before receipt'):publisher.publish(job,handle,None)
        self.assertIsNone(publisher.load(job));self.assertTrue(publisher.path(job).exists())
        record=publisher.publish(job,handle,None)
        self.assertEqual(publisher.load(job),record)

    def test_changed_parts_receipts_and_scope_fail_closed(self):
        f,a,c=family_fixture();out=self.temporary();publisher=P.Publisher(out,f['producer'],c,'diagnostic');job=f['frozen_job']
        handle=P.part_write(out/'parts',f['parent'])
        for bad in (dict(handle,sha256='0'*64),dict(handle,path=str(out/'outside'))):
            with self.assertRaises((ValueError,FileNotFoundError)):publisher.publish(job,bad,None)
        record=publisher.publish(job,handle,None)
        with self.assertRaisesRegex(ValueError,'Diagnostic'):A.read(record['path'],record['native_admission'])
        bad=copy.deepcopy(record);bad['native_admission']['admission']['job']['seed']+=1
        write_json(Path(record['path']+'.record.json'),bad)
        with self.assertRaises(ValueError):publisher.load(job)

    def test_invalid_evidence_never_gets_a_writer_receipt(self):
        f,a,c=family_fixture();out=self.temporary();publisher=P.Publisher(out,f['producer'],c,'ordinary')
        f['parent']['evidence'][0]['raw_probabilities']=[float('nan')]
        with self.assertRaises(ValueError):P.part_write(out/'parts',f['parent'])
        self.assertFalse(publisher.path(f['frozen_job']).exists())

    def test_attested_reads_equal_full_validation_including_branches_and_reference_fallback(self):
        from rust_lab_b06_metrics import compare_games,compare_rows
        from rust_lab_b06_ordinary import prepared_rows
        for options in ({},{'kind':'historical'},{'kind':'legacy'},{'branch':True}):
            with self.subTest(options=options):
                f,a,c=family_fixture(**options);out=self.temporary();path=out/'replay.jsonl.gz'
                record=A.publish(path,f,a,validation_receipt=True)
                original=A.read(path,record['native_admission'],allow_diagnostic=True,reuse_validation=False)
                with patch.object(A,'validate_family',side_effect=AssertionError('duplicate full validation')):
                    game=A.read(path,record['native_admission'],allow_diagnostic=True)
                compare_games([game],[original])
                learner=f['frozen_job']['provenance']['learner_behavior']['policy_id']
                compare_rows(prepared_rows([game],c,learner),prepared_rows([original],c,learner))
                # Changed in-memory labels still invalidate the CheckedGame.
                row=game['records'][0];row['behavior']['probability']*=.5
                with self.assertRaisesRegex(ValueError,'record changed'):game.encoded(row,None)

    def test_validation_reuse_rejects_changed_bytes_and_receipts_and_invalidates_old_source(self):
        f,a,c=family_fixture();out=self.temporary();path=out/'replay.jsonl.gz'
        r=A.publish(path,f,a,validation_receipt=True)['native_admission'];original=A.validate_family
        with patch.object(A,'validate_family',wraps=original) as checked,patch.object(A,'_validation_version',return_value='old'):
            A.read(path,r);self.assertEqual(checked.call_count,1)
        bad=copy.deepcopy(r);bad['admission']['scope']='diagnostic'
        with self.assertRaises(ValueError):A.read(path,bad,allow_diagnostic=True)
        bad=copy.deepcopy(r);bad['validation']['receipt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'validation receipt'):A.read(path,bad)
        raw=path.read_bytes();path.write_bytes(raw+b'changed')
        with self.assertRaisesRegex(ValueError,'writer receipt'):A.read(path,r)
        path.write_bytes(raw)
        feature=path.with_name(path.name+'.features-v4.npz');data=feature.read_bytes();feature.write_bytes(data+b'changed')
        with self.assertRaisesRegex(ValueError,'receipt changed'):A.read(path,r)
        feature.write_bytes(data)
        feature.unlink()
        with patch.object(A,'validate_family',wraps=original) as checked:
            A.read(path,r);self.assertEqual(checked.call_count,1)

    def test_disk_receipt_is_fully_revalidated_on_restart(self):
        f,a,c=family_fixture();out=self.temporary();job=f['frozen_job'];publisher=P.Publisher(out,f['producer'],c,'ordinary')
        record=publisher.publish(job,P.part_write(out/'parts',f['parent']),None)
        with patch.object(A,'validate_family',wraps=A.validate_family) as checked:
            publisher.load(job);self.assertEqual(checked.call_count,1)
        # A self-attested disk sidecar cannot promote invalid behavior.
        path=Path(record['path']);header,family=map(json.loads,gzip.decompress(path.read_bytes()).splitlines())
        family['parent']['evidence'][0]['raw_probabilities']=[-1.,1.,1.]
        family['sha256']=digest({k:v for k,v in family.items() if k!='sha256'})
        path.write_bytes(gzip.compress((json.dumps(header)+'\n'+json.dumps(family)+'\n').encode()))
        receipt=record['native_admission'];receipt.pop('validation');receipt.pop('features')
        receipt.update(replay_sha256=sha(path),family_sha256=family['sha256']);A._attest(receipt)
        record['sha256']=sha(path);write_json(Path(record['path']+'.record.json'),record)
        with self.assertRaisesRegex(ValueError,'raw policy evidence'):publisher.load(job)

    def pool(self,out,*,fail=False):
        f,a,c=family_fixture();job=f['frozen_job'];job['path']=str(out/'job.jsonl.gz')
        path=out/'fixture.json';write_json(path,f['parent'])
        mp=multiprocessing.get_context('spawn');cpu=CPUAdmission(mp,4,1)
        service=SimpleNamespace(cpu=cpu,error=None,channels=lambda count:None)
        execution=dict(processes=1,hosts_per_process=2,worker_threads=1,cpu_budget=4,cpu_reserve=1)
        build=dict(_receipt_sha256=f['producer']['build_sha256'],b01={'sha256':f['producer']['b01_sha256']},fixture=str(path),fail=fail)
        return P.NativePool(SOURCE_ROOT,service,execution,build,c,f['producer']['signature'],scope='ordinary'),job,cpu

    def test_spawned_pool_streams_only_receipts_reloads_and_closes(self):
        out=self.temporary();pool,job,cpu=self.pool(out);received=[]
        with patch.object(P,'_actors',fixture_actors):
            records=pool.collect([job],on_records=received.extend)
            self.assertEqual(records,received);self.assertNotIn('game',records[0])
            self.assertFalse(pool.processes);self.assertEqual(cpu.report()['active'],0)
            cached=pool.collect([job])
        self.assertEqual(cached,records);self.assertEqual(pool.last_collection_report['counts']['cached'],1)
        self.assertFalse(pool.processes)

    def test_spawned_failure_keeps_first_error_and_cancels_without_publication(self):
        out=self.temporary();pool,job,cpu=self.pool(out,fail=True)
        with patch.object(P,'_actors',fixture_actors):
            with self.assertRaisesRegex(ValueError,'original native lane failure'):pool.collect([job])
        self.assertFalse(pool.processes);self.assertTrue(cpu.report()['aborted'])
        self.assertFalse(list(out.glob('native-ordinary/replay/*.record.json')))

    def test_writer_failure_and_reader_backpressure_failure_cancel_all_workers(self):
        for where in ('writer','reader'):
            with self.subTest(where=where):
                out=self.temporary();pool,job,cpu=self.pool(out)
                if where=='writer':
                    path=Path(pool.build['fixture']);part=json.loads(path.read_text())
                    part['evidence'][0]['raw_probabilities']=[-1.,1.,1.];write_json(path,part)
                def failed(_):raise RuntimeError('reader failed')
                with patch.object(P,'_actors',fixture_actors):
                    with self.assertRaisesRegex((ValueError,RuntimeError),'raw policy evidence|reader failed'):
                        pool.collect([job],on_records=failed if where=='reader' else None)
                self.assertFalse(pool.processes);self.assertTrue(cpu.report()['aborted'])
                if where=='writer':self.assertFalse(list(out.glob('native-ordinary/replay/*.record.json')))

    def test_duplicate_branch_cancels_before_waiting_for_running_workers(self):
        from brass_ai.rust_backend.lab_families import collect_families
        f,_,_=family_fixture();cancel=threading.Event();store=Mock();store.load.return_value=None;errors=[]
        def malformed(job,child,submit):
            packet={'row':f['parent']['game']['records'][0]}
            submit(packet);submit(packet)
            self.assertTrue(cancel.wait(2),'Dispatcher waited for workers before canceling')
            return {'branch_expected':True}
        def failed(error):errors.append(str(error));cancel.set()
        with self.assertRaisesRegex(ValueError,'Duplicate/stale fork'):
            collect_families([f['frozen_job']],store,malformed,malformed,lanes=1,cancel=cancel,on_error=failed)
        self.assertTrue(errors);store.publish.assert_not_called()

    def qualification(self):
        state=dict(iteration=1250,history=[{'sha256':'a'*64}],signature={'config':{'teacher_phase':{'mode':'off'}}})
        q=dict(milestone='B07',passed=True,implementation=implementation_id(),generation=1250,checkpoint_sha256='a'*64,
            config_sha256=digest(state['signature']['config']),ordinary={'matched':{'equal':True},'performance':{'useful_accepted_work_gain':True}},
            canary={'passes':2,'passed':True},continuation={'passed':True})
        return q,state

    def test_adoption_requires_measured_gain_current_checkpoint_and_complete_gates(self):
        q,state=self.qualification();B.qualify_receipt(q,state)
        for key,value in (('generation',1251),('checkpoint_sha256','b'*64),('implementation','c'*64),('passed',False)):
            with self.subTest(key=key),self.assertRaises(ValueError):B.qualify_receipt(dict(q,**{key:value}),state)
        bad=copy.deepcopy(q);bad['ordinary']['performance']['useful_accepted_work_gain']=False
        with self.assertRaisesRegex(ValueError,'measured useful'):B.qualify_receipt(bad,state)
        bad=copy.deepcopy(q);bad['canary']['passes']=1
        with self.assertRaises(ValueError):B.qualify_receipt(bad,state)

    def test_lost_backend_checkpoint_state_cannot_silently_select_reference(self):
        with self.assertRaisesRegex(ValueError,'lost backend'):
            B.validate_state({},dict(training={'native_backend':{'mode':'rust'}}))
        self.assertIsNone(B.validate_state({},dict(training={})))

    def test_reference_switch_preserves_checkpoint_progress_and_has_explicit_gap_receipt(self):
        out=self.temporary();checkpoint=out/'model.pt';checkpoint.write_bytes(b'full-state-fixture')
        entry=dict(path=str(checkpoint),sha256=sha(checkpoint),generation=1251,model_id='m')
        value=dict(mode='rust');state=dict(iteration=1251,games_done=400320,history=[entry],native_backend=value,
                                         signature={'implementation':'source'})
        write_json(out/'state.json',state)
        with patch.object(B,'validate_state',return_value=value),patch.object(B,'rollback_fork'),patch('brass_ai.learning.checkpoint.load_checkpoint',
             return_value=(None,{'training':{'native_backend':value}})):
            report=B.reference_switch(SOURCE_ROOT,out)
        restored=json.loads((out/'state.json').read_text())
        self.assertEqual(restored['history'],state['history']);self.assertEqual(restored['games_done'],400320)
        self.assertEqual(restored['native_backend']['mode'],'reference')
        self.assertEqual(restored['native_backend_switch']['before'],value)
        self.assertTrue(report['ordinary_progress_preserved']);self.assertEqual(sha(checkpoint),entry['sha256'])

    def adoption_fixture(self):
        from test_b03 import continuation_fixture
        from brass_ai.rust_backend.lab_contract import pin
        out=self.temporary();run,state=continuation_fixture(out);q,_=self.qualification()
        build=out/'build.json';write_json(build,{})
        q.update(generation=state['iteration'],checkpoint_sha256=state['history'][-1]['sha256'],
            config_sha256=digest(state['signature']['config']),ordinary_build=pin(build))
        qualification=out/'qualification.json';write_json(qualification,q)
        return out,run,state,qualification

    def test_real_adoption_restore_reference_switch_and_rollback_dependency_tamper(self):
        from brass_ai.rust_backend import lab_continuation as C
        from brass_ai.recovery.resume import load_resume
        out,run,state,qualification=self.adoption_fixture();before=sha(SOURCE_ROOT/state['history'][-1]['path'])
        with patch.object(B,'check_build',return_value={}):
            result=B.adopt(SOURCE_ROOT,run,qualification,out/'rollback')
            current=json.loads((run/'state.json').read_text())
            _,raw,_=load_resume(SOURCE_ROOT,run,current,C.target_signature(current))
            B.validate_state(current,raw)
            self.assertTrue(result['adopted']);self.assertFalse(result['training_advanced'])
            self.assertEqual((current['iteration'],current['games_done'],current['history']),
                             (state['iteration'],state['games_done'],state['history']))
            fork=B.rollback_fork(SOURCE_ROOT,current['native_backend'])
            restored=C.restore(fork);self.assertEqual(restored.state,state)
            B.reference_switch(SOURCE_ROOT,run)
        current=json.loads((run/'state.json').read_text())
        with patch.object(B,'check_build',side_effect=AssertionError('reference needs no native runtime')):
            B.validate_state(current,raw)
        self.assertEqual(current['native_backend']['mode'],'reference')
        self.assertEqual(sha(SOURCE_ROOT/state['history'][-1]['path']),before)
        copied=fork.resolve(SOURCE_ROOT/state['history'][-1]['path']);copied.write_bytes(b'changed rollback bytes')
        with self.assertRaisesRegex(ValueError,'diagnostic dependency'):B.rollback_fork(SOURCE_ROOT,current['native_backend'])

    def test_retention_follows_backend_rollback_and_keeps_current_feature_cache(self):
        from brass_ai.recovery.retention import references
        from brass_ai.rust_backend.lab_contract import pin
        root=self.temporary();run=root/'artifacts/run';run.mkdir(parents=True)
        checkpoint=run/'old.pt';checkpoint.write_bytes(b'full-checkpoint')
        replay=run/'iteration-0000/native-ordinary/replay/game.jsonl.gz';replay.parent.mkdir(parents=True);replay.write_bytes(b'replay')
        features=[Path(str(replay)+'.features-v4.'+ext) for ext in ('npz','json')]
        features[0].write_bytes(b'encoded');write_json(features[1],{})
        manifest=root/'artifacts/rollback.json';write_json(manifest,{'checkpoint':pin(checkpoint),'replay':pin(replay)})
        state=dict(format='step5-recovery-league/2.2.0',native_backend={'rollback':pin(manifest)},signature={'config':{}})
        write_json(run/'state.json',state);pinned=references(root,run)
        self.assertTrue(all(p.resolve() in pinned for p in [checkpoint,replay,*features]))

    def test_failed_optimizer_restore_cannot_adopt_or_modify_source_state(self):
        out,run,state,qualification=self.adoption_fixture();before=(run/'state.json').read_bytes()
        with patch.object(B,'check_build',return_value={}),patch('brass_ai.rust_backend.lab_continuation.restore',
                side_effect=ValueError('broken Adam payload')):
            with self.assertRaisesRegex(ValueError,'broken Adam'):B.adopt(SOURCE_ROOT,run,qualification,out/'rollback')
        self.assertEqual((run/'state.json').read_bytes(),before)

    def test_routing_keeps_evaluation_reference_and_ordinary_callback_compact(self):
        state={'signature':{'config':{}},'native_backend':{'mode':'rust','build':{}}}
        with patch('brass_ai.recovery.execution.ActorPool') as old,patch.object(B,'validate_state',return_value=state['native_backend']),\
             patch.object(B,'check_build',return_value={}),patch.object(P,'NativePool') as new:
            native=new.return_value.__enter__.return_value;native.collect.return_value=[{'receipt':1}]
            old.return_value.collect.return_value=[{'reference':1}]
            old.return_value.__enter__.return_value=old.return_value
            pool=B.RoutedPool(SOURCE_ROOT,'host',Mock(),{'execution':{}},state)
            callback=Mock();training=[{'provenance':{'purpose':'training'}}]
            self.assertEqual(pool.collect(training,on_records=callback),[{'receipt':1}])
            native.collect.assert_called_once_with(training,on_records=callback)
            self.assertEqual(pool.collect([{'provenance':{'purpose':'evaluation'}}]),[{'reference':1}])
            self.assertEqual(new.call_count,1)


if __name__=='__main__':unittest.main()
