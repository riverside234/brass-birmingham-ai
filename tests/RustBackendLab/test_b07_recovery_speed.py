"""Bounded native publication/recovery scheduling; no games or model updates."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.recovery import ingress as I
from brass_ai.recovery.replay import check_game
from brass_ai.recovery import replay as replay_module
from brass_ai.recovery.utility import reward_spec
from brass_ai.rust_backend import lab_pool as P,lab_admission as A,lab_replay_compat as R
from brass_ai.rust_backend import lab_features as F
from brass_ai.rust_backend.lab_families import collect_families
from brass_ai.step5.league import write_json
import test_b07 as fixtures
from test_b03 import family_fixture


class RecoverySpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def temporary(self):
        value=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(value.cleanup)
        return Path(value.name)

    def test_recovery_is_bounded_parallel_and_returns_frozen_order(self):
        jobs=[dict(index=i) for i in range(12)];lock=threading.Lock();barrier=threading.Barrier(3)
        active=peak=0;stop=threading.Event()
        def load(job):
            nonlocal active,peak
            with lock:active+=1;peak=max(active,peak)
            try:
                if job['index']<3:barrier.wait(timeout=2)
                return dict(job=job['index']) if job['index']!=5 else None
            finally:
                with lock:active-=1
        records=P.recover_cached(jobs,load,workers=3,cancel=stop,on_error=lambda e:stop.set())
        self.assertEqual([r['job'] for r in records],[i for i in range(12) if i!=5])
        self.assertEqual(peak,3);self.assertFalse(stop.is_set())

    def test_recovery_failure_and_cancellation_release_siblings_before_join(self):
        stop=threading.Event();entered=threading.Event()
        def load(job):
            if job['index']==0:
                entered.set();self.assertTrue(stop.wait(2));return dict(job=0)
            self.assertTrue(entered.wait(2));raise ValueError('original recovery failure')
        with self.assertRaisesRegex(ValueError,'original recovery failure'):
            P.recover_cached([dict(index=0),dict(index=1)],load,workers=2,cancel=stop,on_error=lambda e:stop.set())
        called=Mock();stop.set()
        with self.assertRaisesRegex(ValueError,'canceled'):
            P.recover_cached([dict(index=0)],called,workers=1,cancel=stop,on_error=lambda e:stop.set())
        called.assert_not_called()

    def test_cold_validation_reissues_only_an_in_memory_receipt_and_reader_reuses_it(self):
        f,a,c=family_fixture();out=self.temporary();publisher=P.Publisher(out,f['producer'],c,'ordinary')
        record=publisher.publish(f['frozen_job'],P.part_write(out/'parts',f['parent']),None)
        sidecar=Path(record['path']+'.record.json');old=copy.deepcopy(record)
        old['native_admission']['validation']['validator']='old-reviewed-source'
        write_json(sidecar,old);before=sidecar.read_bytes();replay=Path(record['path']).read_bytes()
        with patch.object(A,'validate_family',wraps=A.validate_family) as full:
            restored=publisher.load(f['frozen_job']);self.assertEqual(full.call_count,1)
            game=A.read(record['path'],restored['native_admission']);self.assertEqual(full.call_count,1)
        self.assertEqual(sidecar.read_bytes(),before);self.assertEqual(Path(record['path']).read_bytes(),replay)
        self.assertEqual(restored['features'],record['features'])
        original=A.convert(f,a)
        for x,y in zip(original['records'],game['records'],strict=True):
            self.assertEqual(x,y)
            np.testing.assert_array_equal(original.encoded(x,None).state,game.encoded(y,None).state)
        Path(record['path']).write_bytes(replay+b'changed')
        with self.assertRaisesRegex(ValueError,'writer receipt'):A.read(record['path'],restored['native_admission'])

    def test_cached_record_metadata_cannot_disagree_with_full_validation(self):
        f,_,c=family_fixture();out=self.temporary();publisher=P.Publisher(out,f['producer'],c,'ordinary')
        record=publisher.publish(f['frozen_job'],P.part_write(out/'parts',f['parent']),None)
        for key,value in (('job',999),('game_id','foreign'),('sha256','0'*64),('features',None)):
            changed=dict(record,**{key:value});write_json(Path(record['path']+'.record.json'),changed)
            with self.assertRaisesRegex(ValueError,'record differs'):publisher.load(f['frozen_job'])

    def test_cache_only_restart_starts_no_actors_or_inference_channels(self):
        out=self.temporary();pool,job,cpu=fixtures.B07().pool(out)
        with patch.object(P,'_actors',fixtures.fixture_actors):records=pool.collect([job])
        pool.service.channels=Mock(side_effect=AssertionError('cache-only engine startup'))
        with patch.object(P,'_actors',side_effect=AssertionError('cache-only actor startup')):
            restored=pool.collect([job])
        self.assertEqual(records,restored);pool.service.channels.assert_not_called()
        report=pool.last_collection_report
        self.assertEqual((report['processes'],report['lanes'],report['actor_startup_seconds']),(0,0,0.))
        self.assertEqual(report['counts'],dict(rust=0,reference=0,cached=1))
        self.assertEqual(cpu.report()['active'],0);self.assertFalse(pool.processes)

    def test_partial_restart_starts_only_missing_parents(self):
        out=self.temporary();pool,job,cpu=fixtures.B07().pool(out)
        with patch.object(P,'_actors',fixtures.fixture_actors):pool.collect([job])
        f,_,_=family_fixture(seed=40);second=f['frozen_job'];second['path']=str(out/'second.jsonl.gz')
        path=out/'second.json';write_json(path,f['parent']);pool.build['fixtures']={str(second['index']):str(path)}
        pool.execution['processes']=4;received=[]
        with patch.object(P,'_actors',fixtures.fixture_actors):records=pool.collect([job,second],on_records=received.extend)
        self.assertEqual(pool.last_collection_report['processes'],1)
        self.assertEqual(pool.last_collection_report['counts'],dict(rust=1,reference=0,cached=1))
        self.assertEqual([r['job'] for r in received],[job['index'],second['index']])
        self.assertEqual([r['job'] for r in records],[job['index'],second['index']]);self.assertEqual(cpu.report()['active'],0)

    def test_publication_uses_more_than_two_bounded_workers_and_keeps_order(self):
        f,_,_=family_fixture();jobs=[dict(f['frozen_job'],index=i) for i in range(4)]
        barrier=threading.Barrier(4);store=Mock()
        def publish(job,*args,**kwargs):barrier.wait(timeout=3);return job['index']
        store.publish.side_effect=publish
        def collect(*args):return dict(branch_expected=False)
        records,_=collect_families(jobs,store,collect,collect,lanes=4,publication_workers=4,cached_records=[])
        self.assertEqual(records,list(range(4)));store.load.assert_not_called()
        self.assertEqual(P.publisher_workers([{}]*320,60),24)
        self.assertEqual(P.publisher_workers([{}]*320,3),3)
        self.assertEqual(P.publisher_workers([{}]*8,60),2)
        self.assertEqual(P.publisher_workers([{}]*64,60),8)

    def test_twenty_four_publication_lanes_are_supported_with_the_same_family_bound(self):
        f,_,_=family_fixture();jobs=[dict(f['frozen_job'],index=i) for i in range(24)]
        barrier=threading.Barrier(24);store=Mock()
        def publish(job,*args,**kwargs):barrier.wait(timeout=3);return job['index']
        store.publish.side_effect=publish
        def collect(*args):return dict(branch_expected=False)
        records,report=collect_families(jobs,store,collect,collect,lanes=24,
            publication_workers=24,cached_records=[])
        self.assertEqual(records,list(range(24)));self.assertEqual(report['limit'],48)
        self.assertEqual(report['counts']['rust'],24)
        cached_barrier=threading.Barrier(24);cancel=threading.Event()
        def load(job):cached_barrier.wait(timeout=3);return dict(job=job['index'])
        restored=P.recover_cached(jobs,load,workers=24,cancel=cancel,on_error=lambda error:cancel.set())
        self.assertEqual([r['job'] for r in restored],list(range(24)))
        with self.assertRaisesRegex(ValueError,'publication workers'):
            collect_families(jobs,store,collect,collect,lanes=24,publication_workers=25,cached_records=[])
        with self.assertRaisesRegex(ValueError,'recovery worker'):
            P.recover_cached(jobs,load,workers=25,cancel=cancel,on_error=lambda error:cancel.set())

    def test_ingress_reads_native_bytes_once_and_rejects_outer_receipt_mismatch(self):
        f,a,c=family_fixture();out=self.temporary();record=A.publish(out/'replay.gz',f,a,validation_receipt=True)
        job=f['frozen_job'];policy=job['provenance']['learner_behavior']['policy_id']
        task=([record],policy,{f['parent']['game']['setup_group']:'selfplay'},c.get('insolvency'),reward_spec(c))
        with patch.object(I,'sha',side_effect=AssertionError('duplicate file hash')),\
                patch.object(I,'read_shards',side_effect=AssertionError('duplicate native header parse')):
            headers,rows,_,_=I._prepare_inner(task)
        self.assertEqual(headers[0]['game_id'],record['game_id']);self.assertTrue(rows)
        bad=copy.deepcopy(record);bad['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'receipt differs'):I._prepare_inner(([bad],*task[1:]))

    def test_publisher_prepares_identical_rows_without_rereading_its_files(self):
        from brass_ai.recovery import ingress
        family,_,config=family_fixture()
        part=family['parent']
        job=family['frozen_job']
        kind={part['game']['setup_group']:job['provenance']['kind']}
        arguments=(job['provenance']['learner_behavior']['policy_id'],kind,config.get('insolvency'),reward_spec(config))
        directory=self.temporary()
        handle=P.part_write(directory/'parts',part)
        publisher=P.Publisher(directory,family['producer'],config,'ordinary')
        published=[]
        with patch.object(A,'read',side_effect=AssertionError('publisher reread its own replay')):
            record,prepared=publisher.publish(job,handle,None,preparation=arguments,
                published=lambda:published.append(publisher.path(job).with_name(publisher.path(job).name+'.record.json').is_file()))
        self.assertEqual(published,[True])
        cold=ingress._prepare_inner(([record],*arguments))
        self.assertEqual(prepared[0],cold[0])
        self.assertEqual(prepared[2],cold[2])
        for left,right in zip(prepared[1],cold[1],strict=True):
            np.testing.assert_equal(left[0],right[0])
            np.testing.assert_equal(left[1],right[1])
            self.assertEqual(left[2],right[2])
            for field in ('state','actions','stable_to_relative'):
                np.testing.assert_array_equal(getattr(left[3],field),getattr(right[3],field))
        self.assertEqual(prepared[3]['publisher_prepared_shards'],1)
        with ingress.StreamingCollection(None,1,*arguments) as stream:
            stream.submit([record],prepared=prepared)
            with self.assertRaisesRegex(ValueError,'Repeated streaming shard'):
                stream.submit([record],prepared=prepared)
            metrics={}
            games,data,_=stream.finish([record],metrics)
            self.assertEqual(len(games),1)
            self.assertEqual(len(data),len(cold[1]))
            self.assertEqual(metrics['publisher_prepared_shards'],1)

    def test_exact_predecessor_execution_can_resume_but_mutations_and_unknown_code_fail(self):
        review=json.loads(R.REVIEW.read_text());old=review['predecessor_executions'][-1]
        self.assertTrue(R.accepts_execution(old));self.assertTrue(R.accepts_execution(R.execution()))
        for key in old:
            self.assertFalse(R.accepts_execution(dict(old,**{key:'0'*64})))
        with patch.object(R,'implementation_id',return_value='f'*64):self.assertFalse(R.accepts_execution(old))

    def test_verified_feature_binding_reuses_input_hashes_and_keeps_mutation_guards(self):
        f,_,_=family_fixture();game=check_game(f['parent']['game'])
        raw,meta=F.pack([game],'a'*64);original=F.digest
        calls=[]
        def digesting(value):
            if isinstance(value,dict) and set(value)=={'view','choices'}:calls.append(1)
            return original(value)
        with patch.object(F,'digest',side_effect=digesting),patch.object(replay_module,'digest',side_effect=digesting):
            arrays=F.unpack(raw,[game],meta)
            checked=F.bind(game,arrays)
        self.assertEqual(len(calls),len(game['records']))
        checked.assert_metadata();row=checked['records'][0];checked.encoded(row,None)
        row['behavior']['probability']*=.5
        with self.assertRaisesRegex(ValueError,'record changed'):checked.encoded(row,None)
        row=checked['records'][1];row['style'][0]+=1.
        with self.assertRaisesRegex(ValueError,'context changed'):checked.encoded(row,None)
        row=checked['records'][2];row['view']['context']['revision']+=1
        with self.assertRaisesRegex(ValueError,'record changed'):checked.encoded(row,None)
        changed=dict(game,records=list(reversed(game['records'])))
        with self.assertRaisesRegex(ValueError,'identity/order'):F.bind(changed,arrays)

    def test_warm_receipts_disable_duplicate_deep_checks_but_cold_recovery_keeps_them(self):
        f,a,_=family_fixture();out=self.temporary();record=A.publish(out/'replay.gz',f,a,validation_receipt=True)
        with patch.object(F,'check_action_selectors',side_effect=AssertionError('duplicate deep validation')):
            A.read(record['path'],record['native_admission'])
        with patch.object(F,'check_action_selectors',wraps=F.check_action_selectors) as checked:
            A.read(record['path'],record['native_admission'],reuse_validation=False)
            self.assertEqual(checked.call_count,len(f['parent']['game']['records']))
        receipt=copy.deepcopy(record['native_admission']);receipt['validation']['receipt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'validation receipt'):A.read(record['path'],receipt)

    def test_nonfinite_features_cannot_be_packed_for_an_independent_receipt(self):
        for invalid in (float('nan'),float('inf'),float('-inf')):
            f,_,_=family_fixture();game=check_game(f['parent']['game'])
            encoded=game.encoded(game['records'][0],None)
            encoded.state=encoded.state.copy();encoded.state[0]=invalid
            with self.assertRaisesRegex(ValueError,'Non-finite'):F.pack([game],'a'*64)

    def test_worker_health_scans_are_throttled_but_exit_and_cancel_still_fail(self):
        pool=P.NativePool.__new__(P.NativePool);pool.stop=threading.Event()
        from types import SimpleNamespace
        pool.service=SimpleNamespace(error=None);worker=Mock();pool.processes=[worker]
        worker.is_alive.return_value=True;connection=Mock()
        connection.poll.side_effect=[False,False,True];connection.recv.return_value={'result':1}
        ticks=iter(i*.05 for i in range(100))
        with patch.object(P.time,'monotonic',side_effect=lambda:next(ticks)):
            self.assertEqual(pool.receive(connection),{'result':1})
        self.assertEqual(worker.is_alive.call_count,1)
        worker.is_alive.return_value=False;connection.poll.side_effect=[False]
        with self.assertRaisesRegex(ValueError,'worker exited'):pool.receive(connection)
        pool.stop.set()
        with self.assertRaisesRegex(ValueError,'canceled'):pool.receive(connection)


if __name__=='__main__':unittest.main()
