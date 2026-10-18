"""Disk-backed critic reuse, leakage boundaries, durable cleanup and resume."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_gae_value_phase import fixture,examples,ppo,finish,config
from brass_ai.bank import sha,read
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import value_phase,value_buffer,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json


def source(behavior):
    # Authored unit fixtures have no native replays. Production sources are
    # admitted via DiskBuffer.source; missing fixture caches must fail closed.
    return dict(policy_id=behavior.policy_id,model_id=behavior.evaluator.model_id,records=[],kind_by_group={})


def commit(run,phases,generation):
    write_json(run/'state.json',dict(iteration=generation))
    phases.committed(generation)


class Buffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup);self.run=Path(self.temp.name)/'run'

    def test_default_legacy_and_retention_contract(self):
        cfg=config();self.assertEqual(cfg['value_phase']['buffer_iterations'],2);validate_config(cfg)
        old=copy.deepcopy(cfg);old['value_phase'].update(contract='ppg-value-phase/1.0.0',buffer_iterations=1)
        validate_config(old)
        for n in (True,0,3):
            bad=copy.deepcopy(cfg);bad['value_phase']['buffer_iterations']=n
            with self.assertRaises(ValueError):validate_config(bad)
        cfg['retention']['replay_rounds']=1
        with self.assertRaisesRegex(ValueError,'2 replay rounds'):validate_config(cfg)
        cfg['retention']['enabled']=False
        with self.assertRaisesRegex(ValueError,'2 replay rounds'):validate_config(cfg)
        cfg['value_phase']['enabled']=False;cfg['advantage']['gate']['mode']='fixed'
        for key in ('critic_data','seat_descriptors'):cfg.pop(key)
        validate_config(cfg)

    def test_retention_keeps_committed_rebuild_sources_despite_pending_report(self):
        from brass_ai.recovery.retention import plan,RETENTION
        cfg=config();state=dict(format='step5-recovery-league/2.2.0',iteration=3,history=[],signature=dict(config=cfg))
        self.run.mkdir(parents=True);write_json(self.run/'state.json',state)
        for i in range(4):
            directory=self.run/f'iteration-{i:04d}';directory.mkdir()
            write_json(directory/'report.json',dict(format='step5-recovery-report/2.2.0'))
            (directory/'ordinary.jsonl.gz').write_bytes(b'replay')
        with patch('brass_ai.recovery.retention.references',return_value={}),patch('brass_ai.recovery.ratings.archived_history',return_value={'slots':[]}):
            options=dict(RETENTION,replay_rounds=2)
            result=plan(SOURCE_ROOT,self.run,options)
            paths={r['path'] for r in result['entries']}
            self.assertIn('iteration-0000/ordinary.jsonl.gz',paths)
            for i in (1,2,3):self.assertNotIn(f'iteration-{i:04d}/ordinary.jsonl.gz',paths)
            with self.assertRaisesRegex(ValueError,'2 replay rounds'):
                plan(SOURCE_ROOT,self.run,dict(options,replay_rounds=1))

    def make(self):
        model,opt,cfg,(behavior,data)=fixture();cfg['value_phase']['buffer_iterations']=2
        ds=updates.Dataset(data,'cpu');train,held,_=value_phase.split(data,.05)
        buf=value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=0)
        return model,opt,cfg,behavior,data,ds,train,held,buf

    def test_disk_rows_match_targets_values_and_do_not_leak_holdout(self):
        model,_,cfg,behavior,data,ds,train,held,buf=self.make()
        rng=torch.get_rng_state().clone();r=buf.stage(ds,train,source(behavior),model)
        receipt=copy.deepcopy(buf.pending);write_json(self.run/'state.json',dict(iteration=1));buf.committed(1)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(receipt['rows'],len(train));self.assertGreater(r['current_forced_rows'],0)
        self.assertFalse(set(receipt['game_ids']) & {data[i][2][0] for i in held})
        self.assertLess(receipt['bytes'],20000*len(train))
        _,current=examples(model,prefix='next-');current=updates.Dataset(current,'cpu')
        with buf.previous() as disk:
            self.assertIsInstance(disk.array,np.memmap)
            combined=value_buffer.ValueDataset(current,disk)
            ids=list(range(len(current.data),len(combined.data)))
            _,old_inputs,_=ds.value_batch(train,model)
            _,new_inputs,_=combined.value_batch(ids,model)
            for a,b in zip(old_inputs,new_inputs):self.assertTrue(torch.equal(a,b))
            for a,b in zip(ds.targets,combined.targets):self.assertTrue(torch.equal(a[train],b[ids]))
            for name in ('utility','remaining','seats'):
                self.assertTrue(torch.equal(getattr(ds,name)[train],getattr(combined,name)[ids]))
            with torch.no_grad():
                a=model(*old_inputs,remaining=ds.remaining[train])
                b=model(*new_inputs,remaining=combined.remaining[ids])
            for x,y in zip(a[1:],b[1:]):self.assertTrue(torch.equal(x,y))
            for j,i in enumerate(train):self.assertEqual(disk[j][2],data[i][2])

    def test_commit_only_cleanup_and_interrupted_staging(self):
        model,_,cfg,behavior,data,ds,train,held,buf=self.make()
        buf.stage(ds,train,source(behavior),model);receipt=copy.deepcopy(buf.pending)
        with self.assertRaises((ValueError,FileNotFoundError)):buf.committed(1)
        self.assertTrue(buf.path(1,'npy').exists())
        write_json(self.run/'state.json',dict(iteration=1));buf.committed(1)
        b,d=examples(model,prefix='second-');ds2=updates.Dataset(d,'cpu');ids,_,_=value_phase.split(d,.05)
        buf.stage(ds2,ids,source(b),model)
        self.assertTrue(buf.path(1,'npy').exists());self.assertTrue(buf.path(2,'npy').exists())
        buf.path(2,'npy.partial').write_bytes(b'incomplete')
        resumed=value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=1,receipt=receipt)
        self.assertTrue(resumed.path(1,'npy').exists());self.assertFalse(resumed.path(2,'npy').exists())
        self.assertFalse(resumed.path(2,'npy.partial').exists())
        resumed.stage(ds2,ids,source(b),model)
        write_json(self.run/'state.json',dict(iteration=2));cleanup=resumed.committed(2)
        self.assertFalse(resumed.path(1,'npy').exists());self.assertEqual(len(cleanup['deleted_files']),2)
        self.assertTrue(resumed.path(2,'npy').exists())
        c,e=examples(model,prefix='third-');ds3=updates.Dataset(e,'cpu');ids,_,_=value_phase.split(e,.05)
        resumed.stage(ds3,ids,source(c),model);write_json(self.run/'state.json',dict(iteration=3));resumed.committed(3)
        self.assertEqual([p.name for p in resumed.directory.glob('*.npy')],['generation-000003.npy'])

    def test_missing_corrupt_rebuild_exact_and_failed_sources_are_not_skipped(self):
        model,_,cfg,behavior,data,ds,train,held,buf=self.make()
        buf.stage(ds,train,source(behavior),model);receipt=copy.deepcopy(buf.pending)
        write_json(self.run/'state.json',dict(iteration=1));buf.committed(1)
        buf.path(1,'npy').unlink()
        with self.assertRaisesRegex(ValueError,'no sealed replay'):
            value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=1,receipt=receipt)
        rng=torch.get_rng_state().clone()
        with patch.object(value_buffer.DiskBuffer,'_rebuild_rows',lambda _:iter(data[i] for i in train)):
            recovered=value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=1,receipt=receipt)
        self.assertTrue(recovered.rebuilt);self.assertEqual(sha(recovered.path(1,'npy')),receipt['sha256'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        recovered.path(1,'npy').write_bytes(b'corrupt')
        with patch.object(value_buffer.DiskBuffer,'_rebuild_rows',lambda _:iter(data[i] for i in reversed(train))):
            with self.assertRaisesRegex(ValueError,'differs from committed receipt'):
                value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=1,receipt=receipt)
        with patch.object(value_buffer.DiskBuffer,'_rebuild_rows',lambda _:iter(data[i] for i in train)):
            recovered=value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=1,receipt=receipt)
        self.assertFalse(recovered.path(1,'npy.partial').exists())

    def test_unknown_files_overlap_holdout_reward_and_generation_rejected(self):
        model,_,cfg,behavior,data,ds,train,held,buf=self.make()
        with self.assertRaisesRegex(ValueError,'Held-out row'):buf.stage(ds,held,source(behavior),model)
        bad=source(behavior);bad['policy_id']='stale'
        with self.assertRaisesRegex(ValueError,'identity differs'):buf.stage(ds,train,bad,model)
        buf.stage(ds,train,source(behavior),model);receipt=copy.deepcopy(buf.pending)
        write_json(self.run/'state.json',dict(iteration=1));buf.committed(1)
        with self.assertRaisesRegex(ValueError,'overlap'):buf.stage(ds,train,source(behavior),model)
        bad=copy.deepcopy(receipt);bad['reward_id']='stale'
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=1,receipt=bad)
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=2,receipt=receipt)
        unexpected=buf.directory/'do-not-delete.txt';unexpected.write_text('user file')
        with self.assertRaisesRegex(ValueError,'Unknown PPG cache artifact'):buf.cleanup(1)
        self.assertEqual(unexpected.read_text(),'user file');self.assertTrue(buf.path(1,'npy').exists())

    def test_owner_and_receipt_publication_interruption(self):
        model,_,cfg,behavior,data,ds,train,held,buf=self.make()
        owner=buf.directory/'owner.json';owner.rename(owner.with_suffix('.json.tmp'))
        buf=value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=0)
        self.assertTrue(owner.exists());self.assertFalse(owner.with_suffix('.json.tmp').exists())
        buf.stage(ds,train,source(behavior),model);receipt=copy.deepcopy(buf.pending)
        write_json(self.run/'state.json',dict(iteration=1));buf.committed(1)
        checksum=sha(buf.path(1,'npy'));buf.path(1,'json').write_bytes(b'{unfinished')
        resumed=value_buffer.DiskBuffer(SOURCE_ROOT,self.run,cfg,generation=1,receipt=receipt)
        self.assertEqual(read(buf.path(1,'json')),receipt);self.assertEqual(sha(buf.path(1,'npy')),checksum)
        self.assertFalse(resumed.rebuilt)

    def test_empty_train_cache_keeps_explicit_empty_holdout_skip(self):
        model,_,cfg,behavior,data,ds,train,held,buf=self.make()
        buf.stage(ds,[],source(behavior),model);write_json(self.run/'state.json',dict(iteration=1));buf.committed(1)
        with buf.previous() as disk:
            self.assertEqual(len(disk),0)
            combined=value_buffer.ValueDataset(ds,disk);self.assertEqual(len(combined.data),len(ds.data))

    def test_two_iteration_training_clone_scope_and_exact_resume(self):
        model,opt,cfg,(behavior,data)=fixture();cfg['value_phase']['buffer_iterations']=2
        cfg['value_phase'].update(batch_size=128,max_epochs=1)
        phases=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=self.run)
        phases.collection_source=source(behavior)
        first=ppo(model,opt,cfg,behavior,data,phases);value=phases.fit(model);finish(phases,first,1)
        self.assertEqual(value['previous_train_rows'],0)
        path=self.run/'generation-0001.pt';save_checkpoint(path,model,opt,step=1,training={KEY:phases.payload()})
        commit(self.run,phases,1);phases.release();checksum=sha(path)
        other_run=self.run.parent/'resumed';shutil.copytree(self.run,other_run)
        restored,raw=load_checkpoint(other_run/path.name);other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,other)
        resumed=TrainingPhases(restored,cfg,raw=raw,generation=1,root=SOURCE_ROOT,run=other_run)
        behavior,current=examples(model,prefix='second-');rng=torch.get_rng_state().clone();results=[]
        real=value_phase.RawPolicyReference.__init__
        def reference(ref,net,ds,ids,options):
            self.assertEqual({d[2][0] for d in ds.data},{d[2][0] for d in current})
            return real(ref,net,ds,ids,options)
        with patch.object(value_phase.RawPolicyReference,'__init__',reference):
            for net,adam,phase in ((model,opt,phases),(restored,other,resumed)):
                torch.set_rng_state(rng);phase.collection_source=source(behavior)
                r=ppo(net,adam,cfg,behavior,current,phase);state=copy.deepcopy(adam.state_dict());v=phase.fit(net);finish(phase,r,2)
                self.assertTrue(_same_payload(state,adam.state_dict()))
                self.assertEqual(r['learner_decisions'],len(current))
                self.assertEqual(v['previous_train_rows'],len(value_phase.split(data,.05)[0]))
                self.assertEqual(v['train_rows'],v['current_train_rows']+v['previous_train_rows'])
                self.assertGreater(v['optimizer_steps'],0);self.assertGreater(v['distinct_previous_rows_used'],0)
                self.assertLessEqual(v['probe_kl'],.002+1e-7)
                self.assertEqual(v['distinct_value_rows_used'],v['distinct_current_rows_used']+v['distinct_previous_rows_used'])
                results.append((r,v,torch.get_rng_state().clone()))
        self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()))
        self.assertTrue(_same_payload(opt.state_dict(),other.state_dict()))
        self.assertTrue(_same_payload(phases.payload(),resumed.payload()))
        self.assertTrue(torch.equal(results[0][2],results[1][2]))
        self.assertEqual(sha(path),checksum)
        commit(self.run,phases,2);commit(other_run,resumed,2)
        self.assertFalse(phases.buffer.path(1,'npy').exists())
        from brass_ai.recovery.history import compact
        self.assertEqual(compact(dict(value_phase=results[0][1]))['value_phase'],results[0][1])


if __name__=='__main__':unittest.main()
