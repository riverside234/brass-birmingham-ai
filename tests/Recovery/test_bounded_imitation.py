"""Bounded IL ingress, exact learning parity and real abrupt-worker recovery."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import io
import os
from pathlib import Path
import pickle
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.replay import split_for
from brass_ai.recovery import demonstrations as demos,prepared_store as store
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.replay import write_shard,cache_receipt
from brass_ai.recovery.updates import Dataset,supervised_epoch
from brass_ai.step5.league import seeds_for,write_json
from fe_fixtures import Factory
from fixtures import game,rotate


def fail_once_job(job):
    task,directory,key=job;directory=Path(directory);marker=directory/'.injected-worker-death'
    if Path(task[0]).name=='1.jsonl.gz' and not marker.exists():
        marker.write_text('intentional test fault',encoding='utf-8')
        deadline=time.monotonic()+30
        while not list(directory.glob('*.json')) and time.monotonic()<deadline:time.sleep(.05)
        os._exit(91)
    return store._prepare_job(job)


def die_job(job):os._exit(92)


class BoundedImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        factory=Factory()
        try:cls.view,cls.choices=factory.position()
        finally:factory.close()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        teacher=next(iter(read(SOURCE_ROOT/'training/opponents/pool.json')['training']))
        entries=[]
        for i,partition in enumerate(('train','train','validation')):
            seed=seeds_for(75000+i*100,1,partition)[0][0];episode=game()
            episode.update(game_id=f'bounded-fixture-{i}',setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed)),opponents=[teacher]*4)
            episode['split']=split_for(episode['setup_group'])
            for step,row in enumerate(episode['records']):
                row['behavior'].update(kind='external',policy_id=teacher)
                row['view']=rotate(self.view,(row['seat']-self.view['seat'])%4)
                row['view']['context']['revision']=row['step']
                # Vary cardinality (including a forced row), label family and
                # balancing groups without encoding synthetic board facts.
                choices=copy.deepcopy(self.choices)
                row['choices']=choices[:1] if step==0 else choices[:max(2,len(choices)//(step+1))]
                row['chosen']=min(i+step,len(row['choices'])-1)
            path=self.directory/f'{i}.jsonl.gz';write_shard(path,[episode],dict(purpose='training',seed=seed))
            entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),game_id=episode['game_id'],
                setup_group=episode['setup_group'],split=partition,transitions_verified=True,features=cache_receipt(path)))
        self.manifest=self.directory/'manifest.json'
        write_json(self.manifest,dict(format='bc-dataset/1.0.0',compatibility=compatibility(),label_kind='qualified-v3-choice/1.0.0',
            context='neutral-predecision/1.0.0',teachers=[teacher],entries=entries))
        self.tasks=[(str(SOURCE_ROOT/e['path']),[e],e['split'],{teacher},True) for e in entries]
        self.storage=self.directory/'prepared'

    def rows(self,workers=2):
        rows,_=demos.load_dataset(SOURCE_ROOT,self.manifest,'train',workers=workers,compact=True,storage=self.storage)
        self.addCleanup(rows.close);return rows

    def test_disk_parallel_rows_batching_and_optimizer_are_exact(self):
        original,_=demos.load_dataset(SOURCE_ROOT,self.manifest,'train',compact=True)
        disk=self.rows();self.assertEqual(len(disk),len(original))
        self.assertEqual(disk.game_ids,{d[2][0] for d in original})
        for a,b in zip(original,disk):
            self.assertEqual(a[:3],b[:3])
            self.assertEqual(a[3].choice_ids,b[3].choice_ids)
            for field in ('state','actions','stable_to_relative'):
                np.testing.assert_array_equal(getattr(a[3],field),getattr(b[3],field))
                self.assertFalse(getattr(b[3],field).flags.writeable)
        np.testing.assert_array_equal(demos.balance_weights(original),demos.balance_weights(disk))
        left=Dataset(demos.balance(original),'cpu',policy=False)
        right=store.PreparedDataset(disk,'cpu',demos.balance_weights(disk))
        for bucket in (False,True):
            indices=[7,1,3,0,5,4,2,6]
            self.assertEqual(list(left.microbatches(indices,3,10000,bucket)),list(right.microbatches(indices,3,10000,bucket)))
        with self.assertRaisesRegex(ValueError,'Single position'):list(right.microbatches(range(8),2,1))
        with patch.object(torch.Tensor,'pin_memory',lambda self:self),ThreadPoolExecutor(max_workers=2) as pool:
            actual=list(right.supervised_batches(iter([[3,0],[1,5],[7,2,4,6]]),pool))
        for part,ix,inputs in actual:
            expected_ix,expected,_=left.batch(part)
            self.assertTrue(torch.equal(ix,expected_ix))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(inputs,expected)))
        cfg=dict(UPDATE_V21,effective_batch_size=8,microbatch_size=2)
        first=FourSeatNet(ModelConfig.preset('small'));second=copy.deepcopy(first)
        first_opt=torch.optim.AdamW(first.parameters(),lr=1e-4);second_opt=torch.optim.AdamW(second.parameters(),lr=1e-4)
        torch.manual_seed(314);a=supervised_epoch(first,first_opt,left,cfg)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(314);b=supervised_epoch(second,second_opt,right,cfg)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key in a:
            if key not in ('timings','memory'):self.assertEqual(a[key],b[key],key)
        for key,value in first.state_dict().items():self.assertTrue(torch.equal(value,second.state_dict()[key]),key)
        for i,state in first_opt.state_dict()['state'].items():
            for name,value in state.items():self.assertTrue(torch.equal(value,second_opt.state_dict()['state'][i][name]))
        natural=Dataset(original,'cpu',policy=False);lazy=store.PreparedDataset(disk,'cpu')
        a=supervised_epoch(first,None,natural,cfg,training=False,validation_weights=demos.balance_weights(original))
        b=supervised_epoch(first,None,lazy,cfg,training=False,validation_weights=demos.balance_weights(disk))
        for before,after in zip(a,b):
            for key in before:
                if key not in ('timings','memory'):self.assertEqual(before[key],after[key],key)

    def test_receipts_resume_and_bad_cache_rebuild_without_raw_relabeling(self):
        disk=self.rows();self.assertEqual(disk.report['cache_hits'],0);disk.close()
        again=self.rows(workers=1);self.assertEqual(again.report['cache_hits'],2);again.close()
        key=store.task_key(self.tasks[0],store.preparation_identity());files=store.paths(self.storage,key)
        original=sha(files['data']);files['data'].write_bytes(files['data'].read_bytes()+b'corrupt')
        receipt,reused=store._prepare_job((self.tasks[0],str(self.storage),key))
        self.assertFalse(reused);self.assertEqual(original,sha(files['data']))
        self.assertEqual(receipt['data_sha256'],original)
        files['index'].write_bytes(b'broken zip');_,reused=store._prepare_job((self.tasks[0],str(self.storage),key))
        self.assertFalse(reused)
        files['receipt'].unlink()
        # Interrupted/uncommitted output is rebuilt, not trusted on size alone.
        _,reused=store._prepare_job((self.tasks[0],str(self.storage),key));self.assertFalse(reused)
        edited=dict(receipt,game_ids=['forged-game'])
        write_json(files['receipt'],edited)
        repaired,reused=store._prepare_job((self.tasks[0],str(self.storage),key))
        self.assertFalse(reused);self.assertEqual(repaired['game_ids'],receipt['game_ids'])
        raw=Path(self.tasks[0][0]);raw.write_bytes(raw.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'source changed'):
            store._prepare_job((self.tasks[0],str(self.storage),key))
        changed=list(self.tasks[1]);changed[3]={'not-admitted'}
        self.assertNotEqual(store.task_key(changed,store.preparation_identity()),store.task_key(self.tasks[1],store.preparation_identity()))

    def test_metadata_and_microbatch_planning_do_not_read_dense_features(self):
        rows=self.rows();weights=demos.balance_weights(rows)
        with patch.object(rows,'record',side_effect=AssertionError('unexpected feature or metadata read')):
            dataset=store.PreparedDataset(rows,'cpu',weights)
            self.assertEqual(sum(map(len,dataset.microbatches(range(8),3,10000))),8)
            self.assertEqual(len(rows.game_ids),2)
        with patch.object(rows,'record',wraps=rows.record) as fetch:
            self.assertIsInstance(rows[0][0],dict);fetch.assert_called_once_with(0,features=False)
        receipt=rows.receipts[0]
        self.assertLess(len(pickle.dumps(receipt)),4096)
        self.assertLess(rows.index_bytes,200*len(rows))
        rows.max_open_files=1
        for i in (0,4,1,5):rows[i][3]
        self.assertEqual(len(rows.handles),1)
        rows.close();self.assertFalse(rows.handles)
        with self.assertRaisesRegex(ValueError,'closed'):rows[0][3]

    def test_real_worker_death_reduces_concurrency_and_retains_committed_shard(self):
        with patch.object(store,'_prepare_job',fail_once_job),contextlib.redirect_stdout(io.StringIO()):
            rows=self.rows(workers=2)
        self.assertEqual(len(rows),8)
        self.assertEqual(rows.report['worker_retries'],[dict(previous_workers=2,workers=1,completed_shards=1)])
        self.assertEqual(rows.report['final_workers'],1)
        self.assertEqual(len(rows.game_ids),2)

    def test_persistent_worker_death_fails_without_skipping_data(self):
        with patch.object(store,'_prepare_job',die_job):
            with self.assertRaisesRegex(RuntimeError,'one worker; 0/1 shards.*0.jsonl.gz'):
                store.prepare(self.tasks[:1],self.storage,1,'train')
        self.assertFalse(list(self.storage.glob('*.json')))

    def test_unadmitted_manifest_and_raw_neural_replay_are_still_rejected(self):
        manifest=read(self.manifest);manifest['teachers']=['not-admitted'];write_json(self.manifest,manifest)
        with self.assertRaisesRegex(ValueError,'teacher no longer admitted'):self.rows()
        task=list(self.tasks[0]);task[3]={'not-admitted'}
        with self.assertRaisesRegex(ValueError,'audited heuristics'):
            demos._prepared_shard(task)

    def test_train_smoke_reuses_loading_cache_and_completed_state(self):
        run=self.directory/'bc';cfg=dict(UPDATE_V21,effective_batch_size=8,microbatch_size=2)
        real_loader=demos.load_dataset
        def interrupted(root,manifest,partition,**kwargs):
            if partition=='validation':raise RuntimeError('injected pre-epoch interruption')
            return real_loader(root,manifest,partition,**kwargs)
        with patch.object(demos,'load_dataset',interrupted),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'pre-epoch interruption'):
                demos.train_bc(SOURCE_ROOT,run,self.manifest,None,cfg,epochs=1,min_epochs=1,patience=1,workers=2)
        self.assertFalse((run/'state.json').exists());self.assertFalse((run/'initial.pt').exists())
        self.assertFalse((run/'.active-run.json').exists())
        with contextlib.redirect_stdout(io.StringIO()):
            report=demos.train_bc(SOURCE_ROOT,run,self.manifest,None,cfg,epochs=1,min_epochs=1,patience=1,workers=2)
            before={p.name:sha(p) for p in run.glob('*.pt')};state=read(run/'state.json')
            repeated=demos.train_bc(SOURCE_ROOT,run,self.manifest,None,cfg,epochs=1,min_epochs=1,patience=1,workers=1)
        self.assertEqual(report['history'],repeated['history'])
        self.assertEqual(report['preparation']['train']['cache_hits'],2)
        self.assertEqual(state,read(run/'state.json'))
        self.assertEqual(before,{p.name:sha(p) for p in run.glob('*.pt')})
        self.assertEqual(repeated['preparation']['train']['cache_hits'],2)
        self.assertEqual(repeated['preparation']['validation']['cache_hits'],1)
        self.assertEqual(report['history'][0]['train']['optimizer_steps'],1)
        _,origin=demos.completed_imitation(SOURCE_ROOT,run)
        self.assertEqual(origin['sha256'],report['candidate_sha256'])


if __name__=='__main__':unittest.main()
