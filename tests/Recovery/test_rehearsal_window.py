"""Disposable rehearsal windows, train-only sampling, exact loss and probe reuse."""
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import gzip
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import rehearsal_window as windows,demonstrations as demos
from brass_ai.recovery import il_kl_probe as probes,il_kl_limit as limits,updates
from brass_ai.recovery.config import UPDATE_V21,latest_model_config,validate_config
from brass_ai.step5.league import write_json
import test_bounded_imitation as fixture
import test_auto_il_kl as probe_fixture


def killed_rehearsal_reader(job):os._exit(93)


class RehearsalWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.BoundedImitationTests.setUpClass()

    def setUp(self):
        self.fixture=fixture.BoundedImitationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory=self.fixture.directory;self.manifest=self.fixture.manifest
        manifest=read(self.manifest)
        for entry in manifest['entries']:entry['teachers']={manifest['teachers'][0]:4}
        write_json(self.manifest,manifest)
        self.run=self.directory/'selfplay'
        self.config=dict(windows.DEFAULT,rows_per_iteration=8,shard_draws=4,max_cache_bytes=16*1024**2)

    def window(self,index=None,iteration=0,config=None,**execution):
        return windows.window(SOURCE_ROOT,self.run,index or windows.TrainIndex(SOURCE_ROOT,self.manifest),
            config or self.config,seed=49,iteration=iteration,device='cpu',**execution)

    def test_manifest_only_index_selection_rotation_and_rng_continuity(self):
        with patch.object(demos,'_prepared_shard',side_effect=AssertionError('Eager shard read')):
            index=windows.TrainIndex(SOURCE_ROOT,self.manifest)
        self.assertEqual((index.rows,index.games,len(index.tasks)),(8,2,2))
        rng=torch.get_rng_state().clone();first=index.select(self.config,49,0)
        restored=windows.TrainIndex(SOURCE_ROOT,self.manifest)
        self.assertEqual(first,restored.select(self.config,49,0))
        self.assertNotEqual(first,index.select(self.config,49,1))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        # Unequal shard sizes must not turn into equal-shard weighting.
        index.counts=[1,3];index.ends=[1,4];index.rows=4
        counts=Counter()
        for iteration in range(1000):
            for shard,rows in index.select(self.config,49,iteration).items():
                counts.update((shard,row) for row in rows)
        self.assertEqual(len(counts),4)
        self.assertTrue(all(850<n<1150 for n in counts.values()),counts)

    def test_missing_train_cache_streams_selected_sources_and_cleans_up(self):
        index=windows.TrainIndex(SOURCE_ROOT,self.manifest)
        original=demos._prepared_shard;calls=[]
        # Simulate the user's deleted train caches; leave validation untouched.
        for task in index.tasks:
            for path in Path(task[0]).parent.glob(Path(task[0]).name+'.features-v4.*'):path.unlink()
        pinned={p:sha(p) for p in self.directory.iterdir() if p.is_file()}
        def tracked(task,**kwargs):calls.append(task[0]);return original(task,**kwargs)
        rng=torch.get_rng_state().clone()
        with patch.object(demos,'_prepared_shard',side_effect=tracked),contextlib.redirect_stdout(io.StringIO()):
            with self.window(index) as dataset:
                self.assertEqual(len(dataset.data),8)
                self.assertLessEqual(dataset.report['peak_cache_bytes'],self.config['max_cache_bytes'])
                self.assertFalse(dataset.report['cache_deleted'])
                for number,(shard,local) in enumerate(dataset.identities):
                    expected=original(index.tasks[shard])[local];actual=dataset.data[number]
                    self.assertEqual(actual[:3],expected[:3])
                    np.testing.assert_array_equal(actual[3].state,expected[3].state)
                    np.testing.assert_array_equal(actual[3].actions,expected[3].actions)
                eligible=[i for i in range(len(dataset.data)) if not dataset.forced_cpu[i]][:3]
                for _ in dataset.supervised_batches([eligible],None):pass
                self.assertEqual(dataset.report['used_presentations'],len(eligible))
                self.assertEqual(dataset.report['distinct_rows_used'],len({dataset.identities[i] for i in eligible}))
            self.assertTrue(dataset.report['cache_deleted'])
            from brass_ai.recovery.history import compact
            self.assertEqual(compact(dict(actor=dict(rehearsal_sampling=dataset.report)))['actor']['rehearsal_sampling'],dataset.report)
        self.assertEqual(set(calls),{index.tasks[i][0] for i in index.select(self.config,49,0)})
        self.assertFalse((self.run/'.rehearsal-cache').exists())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(pinned,{p:sha(p) for p in pinned})
        self.assertFalse(any(self.directory.glob('0.jsonl.gz.features*')))

    def test_selected_neutral_preparation_is_exact_with_repeats_and_skips_unused_rows(self):
        task=self.fixture.tasks[0];task[1][0]['teachers']={next(iter(task[3])):4}
        full=demos._prepared_shard(task);draws=[3,0,3,2];real=demos.prepared;prepared_counts=[]
        def prepare(*args,**kwargs):
            data=real(*args,**kwargs);prepared_counts.append(len(data));return data
        with patch.object(demos,'prepared',side_effect=prepare):chosen=demos._prepared_shard(task,selection=draws)
        self.assertEqual(prepared_counts,[3])
        for actual,i in zip(chosen,draws):
            self.assertEqual(actual[:3],full[i][:3])
            for field in ('state','actions','stable_to_relative'):
                np.testing.assert_array_equal(getattr(actual[3],field),getattr(full[i][3],field))
        # An illegal label in an UNSELECTED row must still be rejected.
        path=Path(task[0])
        with gzip.open(path,'rt',encoding='utf-8') as handle:header,episode=[json.loads(line) for line in handle]
        episode['records'][1]['chosen']=len(episode['records'][1]['choices'])
        with gzip.open(path,'wt',encoding='utf-8') as handle:
            handle.write(json.dumps(header)+'\n'+json.dumps(episode)+'\n')
        task[1][0].update(sha256=sha(path),features=None)
        with self.assertRaisesRegex(ValueError,'Illegal chosen'):
            demos._prepared_shard(task,selection=draws)

    def test_parallel_reuses_readers_preserves_sample_order_bytes_and_rng(self):
        from brass_ai.recovery.ingress import reader_pool
        index=windows.TrainIndex(SOURCE_ROOT,self.manifest)
        def snapshot(ds):
            return (ds.identities,[(r['data_sha256'],r['index_sha256']) for r in ds.data.receipts],
                    copy.deepcopy(ds.report))
        rng=torch.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()),self.window(index) as ds:serial=snapshot(ds)
        with reader_pool(2) as readers,contextlib.redirect_stdout(io.StringIO()):
            with self.window(index,workers=2,executor=readers) as ds:parallel=snapshot(ds)
        self.assertEqual(serial[:2],parallel[:2])
        self.assertEqual(serial[2]['selection_sha256'],parallel[2]['selection_sha256'])
        self.assertTrue(parallel[2]['reused_reader_pool']);self.assertEqual(parallel[2]['workers'],2)
        self.assertEqual(parallel[2]['dense_ipc_bytes'],0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse((self.run/'.rehearsal-cache').exists())

    def test_parallel_budget_is_shared_and_failures_join_writers_before_cleanup(self):
        # Both shards individually fit; their combined output does not. Serial
        # and parallel must reject rather than give each worker its own cap.
        with contextlib.redirect_stdout(io.StringIO()),self.window() as ds:
            sizes=[sum(p.stat().st_size for p in f.values()) for f in ds.data.files]
        cap=max(sizes)+65536+8192
        self.assertLess(cap,sum(sizes))
        with contextlib.redirect_stdout(io.StringIO()):
            for workers in (1,2):
                with self.assertRaisesRegex(ValueError,'max_cache_bytes'):
                    with self.window(config=dict(self.config,max_cache_bytes=cap),workers=workers):pass
                self.assertFalse((self.run/'.rehearsal-cache').exists())
            with self.assertRaisesRegex(ValueError,'source changed'):
                index=windows.TrainIndex(SOURCE_ROOT,self.manifest)
                path=Path(index.tasks[0][0]);path.write_bytes(path.read_bytes()+b'changed')
                with self.window(index,workers=2):pass
            self.assertFalse((self.run/'.rehearsal-cache').exists())

    def test_dead_parallel_reader_does_not_leave_writers_or_partial_cache(self):
        from concurrent.futures.process import BrokenProcessPool
        pinned={p:sha(p) for p in self.directory.iterdir() if p.is_file()}
        with patch.object(windows,'_prepare_selection',killed_rehearsal_reader),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BrokenProcessPool):
                with self.window(workers=2):pass
        self.assertFalse((self.run/'.rehearsal-cache').exists())
        self.assertEqual(pinned,{p:sha(p) for p in pinned})

    def test_disk_window_preserves_ppo_loss_for_identical_selected_rows(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.step5.policy import NeuralPolicy
        model=FourSeatNet(ModelConfig.preset('small'));policy=NeuralPolicy(model,focus='neutral',exploration=.08)
        with gzip.open(self.fixture.tasks[0][0],'rt',encoding='utf-8') as handle:
            next(handle);game=json.loads(next(handle))
        game['opponents']=[policy.policy_id]*4
        for row in game['records']:
            chosen,probability,_=policy.select_batch([(row['view'],row['choices'],random.Random(row['step']))])[0]
            row.update(chosen=chosen,behavior=dict(kind='model',policy_id=policy.policy_id,probability=probability,temperature=1.))
        cfg=dict(UPDATE_V21,effective_batch_size=4,critic_batch_size=4,microbatch_size=2,epochs=1)
        with contextlib.redirect_stdout(io.StringIO()),self.window() as dataset:
            rows=[dataset.data.record(i,features=True) for i in range(len(dataset.data))]
            left=copy.deepcopy(model);right=copy.deepcopy(model)
            torch.manual_seed(142)
            a=updates.ppo_update(left,torch.optim.AdamW(left.parameters(),lr=1e-6),[game],policy,cfg,
                rehearsal=rows,rehearsal_weight=.1)
            rng=torch.get_rng_state().clone();torch.manual_seed(142)
            b=updates.ppo_update(right,torch.optim.AdamW(right.parameters(),lr=1e-6),[game],policy,cfg,
                rehearsal=dataset,rehearsal_weight=.1)
            self.assertEqual(a['epochs'],b['epochs'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items()))
            self.assertGreater(dataset.report['used_presentations'],0)

    def test_prefetch_tracks_only_consumed_samples_once(self):
        with contextlib.redirect_stdout(io.StringIO()),self.window() as dataset:
            with patch.object(torch.Tensor,'pin_memory',lambda self:self),ThreadPoolExecutor(max_workers=2) as pool:
                list(dataset.supervised_batches(iter([[0,1],[2,3]]),pool))
            self.assertEqual(dataset.report['used_presentations'],4)

    def test_interrupt_cap_and_corruption_cleanup_without_touching_sources(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                with self.window():raise KeyboardInterrupt()
            self.assertFalse((self.run/'.rehearsal-cache').exists())
            with self.assertRaisesRegex(ValueError,'max_cache_bytes'):
                with self.window(config=dict(self.config,max_cache_bytes=1024)):pass
            self.assertFalse((self.run/'.rehearsal-cache').exists())
            index=windows.TrainIndex(SOURCE_ROOT,self.manifest)
            selected=next(iter(index.select(self.config,49,0)));path=Path(index.tasks[selected][0])
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):
                with self.window(index):pass
            self.assertFalse((self.run/'.rehearsal-cache').exists())

    def test_index_counts_and_cache_scope_fail_closed(self):
        meta=read(self.manifest);meta['entries'][0]['teachers'][meta['teachers'][0]]=False
        write_json(self.manifest,meta)
        with self.assertRaisesRegex(ValueError,'decision counts'):windows.TrainIndex(SOURCE_ROOT,self.manifest)
        directory=self.run/'.rehearsal-cache';directory.mkdir(parents=True)
        write_json(directory/'owner.json',dict(format=windows.OWNER))
        other=directory/'unknown';other.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'Unexpected'):windows.cleanup(SOURCE_ROOT,self.run)
        self.assertEqual(other.read_bytes(),b'keep')
        other.unlink();windows.cleanup(SOURCE_ROOT,self.run)

    def test_config_is_strict_and_generator_preserves_unrelated_recipe(self):
        cfg=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');validate_config(cfg)
        legacy=copy.deepcopy(cfg);legacy.pop('rehearsal_sampling');validate_config(legacy)
        rebuilt=latest_model_config(legacy,legacy['imitation_checkpoint'])
        self.assertEqual(rebuilt['rehearsal_sampling'],windows.DEFAULT)
        for key in ('update','learning_rate','rehearsal_floor','il_kl_limit','table_plan','search','teacher_enabled'):
            self.assertEqual(rebuilt[key],cfg[key])
        for key,value in (('rows_per_iteration',True),('shard_draws',0),('max_cache_bytes',float('nan')),
                          ('delete_after_update',False),('contract','unknown')):
            with self.assertRaises(ValueError):windows.validate(dict(windows.DEFAULT,**{key:value}))


class ProbeImportTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.donor=self.root/'artifacts/old';self.run=self.root/'artifacts/new'
        source=self.root/'validation.jsonl.gz';source.write_bytes(b'pinned fixture validation source')
        manifest=self.root/'manifest.json'
        write_json(manifest,dict(entries=[dict(split='validation',path=source.name,sha256=sha(source))]))
        self.origin=dict(manifest=manifest.name,manifest_sha256=sha(manifest),model_id='a'*64,sha256='b'*64)
        self.model=probe_fixture.ToyPolicy();self.config=copy.deepcopy(limits.DEFAULT)
        self.update=dict(microbatch_size=3,max_candidate_cells=100)
        with patch.object(demos,'load_dataset',return_value=(probe_fixture.ProbeTests().data(),{})):
            self.probe,self.receipt=probes.load(self.root,self.donor,self.origin,self.model,self.config,self.update,1)
        checkpoint=self.donor/'initial.pt'
        torch.save(dict(training=dict(il_kl_limit=dict(probe=self.receipt))),checkpoint)
        write_json(self.donor/'state.json',dict(il_kl_limit=dict(probe=self.receipt),history=[dict(
            path=checkpoint.relative_to(self.root).as_posix(),sha256=sha(checkpoint))]))
        prepared=self.donor/'il-kl/.prepared';prepared.mkdir();(prepared/'keep.bin').write_bytes(b'donor large cache')

    def test_import_reuses_probe_without_loading_validation_or_ppo(self):
        before={p:sha(p) for p in self.donor.rglob('*') if p.is_file()};rng=torch.get_rng_state().clone()
        with patch.object(demos,'load_dataset',side_effect=AssertionError('Validation rebuilt')):
            probe,receipt=probes.import_probe(self.root,self.run,self.donor,self.origin,self.model,self.config,self.update)
            resumed,again=probes.load(self.root,self.run,self.origin,self.model,self.config,self.update,1,receipt)
        self.assertEqual(receipt['probe_hash'],self.receipt['probe_hash']);self.assertEqual(receipt,again)
        self.assertEqual(probe.measure(self.model,self.update)['kl_mean'],0.)
        self.assertEqual(resumed.measure(self.model,self.update)['kl_mean'],0.)
        self.assertFalse((self.run/'il-kl/.prepared').exists());self.assertFalse((self.run/'state.json').exists())
        self.assertFalse(list(self.run.glob('*.pt')))
        self.assertEqual(before,{p:sha(p) for p in before});self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_changed_origin_config_bytes_and_committed_receipt_are_rejected(self):
        for change in (dict(model_id='c'*64),dict(sha256='d'*64)):
            with self.assertRaisesRegex(ValueError,'source/config'):
                probes.import_probe(self.root,self.run,self.donor,dict(self.origin,**change),self.model,self.config,self.update)
        with self.assertRaisesRegex(ValueError,'source/config'):
            probes.import_probe(self.root,self.run,self.donor,self.origin,self.model,dict(self.config,probe_seed=9),self.update)
        path=self.donor/'il-kl/probe.npz';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'cache changed'):
            probes.import_probe(self.root,self.run,self.donor,self.origin,self.model,self.config,self.update)


if __name__=='__main__':unittest.main()
