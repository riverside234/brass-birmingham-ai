"""Dense replay, latest-loss retention, ordered workers and exact adoption."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import dcpg_config as D,ppg_config as P,ppg_auxiliary as A,gae,ppg_epochs,snapshot_r2
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.ppg_selection import Selection,LATEST_DEFAULT
from brass_ai.recovery.ppo_preparation import PreparationPool,Preparation
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX
from brass_ai.step5.league import write_json
from test_ppg_sixteen import config as old_config
from test_ppg_eight import stage_only
from test_local_scoring import annotate
import test_ppg_eight as base


def config():
    from brass_ai.recovery.ppo_kl_control import configure as exact_kl_config
    model,opt,c=old_config();c=exact_kl_config(D.fast_options(c));c['update']['preparation_workers']=0
    return model,opt,c


class DenseEight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_last_accepted_reference_can_move_above_minimum_and_restore_matching_adam(self):
        m=torch.nn.Linear(1,1);opt=torch.optim.Adam(m.parameters())
        s=Selection(m,opt,dict(a=1.,b=1.),config=LATEST_DEFAULT)
        def step(games,n):
            for p in m.parameters():p.grad=torch.ones_like(p)
            opt.step();opt.zero_grad(set_to_none=True)
            return s.consider(m,opt,games,list(range(n)),[],n,n)
        self.assertTrue(step(dict(a=1.03,b=.99),16))
        self.assertTrue(step(dict(a=1.06,b=.98),32))
        saved=copy.deepcopy((m.state_dict(),opt.state_dict()))
        self.assertGreater(s.report()['retained_loss'],s.report()['minimum_loss'])
        self.assertFalse(step(dict(a=1.20,b=1.20),48))
        s.restore(m,opt);self.assertTrue(_same_payload(saved,(m.state_dict(),opt.state_dict())))
        self.assertEqual(s.steps,32)

    def test_dense_process_parity_metadata_gradients_and_cleanup(self):
        m,_,c=config();c['local_scoring_auxiliary']=copy.deepcopy(AUX)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
            original=base.collect
            def labeled(*args,**kwargs):
                b,data=original(*args,**kwargs);annotate(data);annotate(p.snapshot_data);return b,data
            with patch.object(base,'collect',labeled):
                for g in (1,2):
                    cache=stage_only(m,p,g);self.assertEqual(cache['packed_generations'],0)
                    if g==1:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            with p.buffer.datasets('cpu') as (ds,_,receipts),PreparationPool(4) as pool,Preparation(pool,directory=run) as prep:
                n=len(ds.data.parts[0]);parts=[[n+2,0,n+2,1],[3,n+1],[0],[1]]
                expected=[(ds.batch(ids),tuple(t[ids] for t in ds.targets),ds.utility[ids],ds.descriptors[ids]) for ids in parts]
                rng=torch.get_rng_state().clone()
                for j,(ids,ix,inputs) in enumerate(ds.value_batches(parts,m,prep)):
                    x=expected[j]
                    self.assertTrue(_same_payload((ix,inputs),x[0][:2]))
                    self.assertTrue(_same_payload(tuple(t[ix] for t in ds.targets),x[1]))
                    self.assertTrue(torch.equal(ds.utility[ix],x[2]));self.assertTrue(torch.equal(ds.descriptors[ix],x[3]))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertIsNone(ds.data.prepared)
                self.assertEqual(prep.metrics['input_bytes'],0);self.assertEqual(prep.metrics['batches'],4)
                gen=ds.value_batches(parts,m,prep);next(gen);gen.close()
                self.assertIsNone(ds.data.prepared);self.assertFalse(prep.futures)
                self.assertEqual(list(prep.path.glob('batch-*.bin')),[])
                companion=Companion(m,c['local_scoring_auxiliary']);companion.age=7
                ids=parts[0];metadata=companion.metadata(ds,ids)
                for kind,count in [('slots',49),('links',69)]:
                    arrays=companion.target_arrays(metadata,kind,count)
                    for b,(target,_,_,mapping) in enumerate(metadata):
                        for j,row in enumerate(target[kind]):
                            owner=row['owner'];relative=mapping[owner-1] if owner else None
                            self.assertEqual(arrays[0][b,j],relative+1 if owner else 0)
                            self.assertAlmostEqual(float(arrays[1][b,j].sum())*companion.config['vp_scale'],row['vp'],places=5)
                            self.assertEqual(arrays[2][b,j],row['valid'])
                # Compare optimized metadata with the original row-oriented
                # path, including every core/head gradient and reported loss.
                ix,inputs,_=ds.batch(ids);outcomes=[]
                for fast in (False,True):
                    m.zero_grad();companion.heads.zero_grad();companion.begin('ppg')
                    method=ds.scoring_metadata
                    if not fast:
                        def slow(chosen):
                            result=[]
                            from brass_ai.recovery.local_scoring_targets import map_for
                            for i in chosen:
                                row=ds.rows[i];t=map_for(row);mapping=ds.encoded[i].stable_to_relative;context=row['view']['context']
                                result.append((t,(row['_local_scoring']['envelope']['sha256'],t['era'],int(row['seat']),tuple(mapping)),
                                    (ds.data[i][2][0],context['era'],int(context['revision']),int(row['seat'])),mapping))
                            return result
                        ds.scoring_metadata=slow
                    try:
                        out,loss=companion.forward(m,inputs,ds,ix);loss.mean().backward()
                        outcomes.append((loss.detach().clone(),[None if x.grad is None else x.grad.clone() for x in list(m.parameters())+list(companion.heads.parameters())]))
                    finally:ds.scoring_metadata=method
                self.assertTrue(_same_payload(*outcomes))

    def test_adoption_clears_only_owned_cache_after_commit_preserving_state(self):
        m,opt,old=old_config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            p.schedule=P.advance(p.schedule,1,False);p.beta=4.
            p.buffer.pending=copy.deepcopy(p.buffer.receipts);p.pending_schedule=p.schedule;p.pending_epoch_control=p.epoch_control
            raw=dict(training={KEY:p.payload()});c=D.fast_options(old)
            before=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={x.name:sha(x) for x in p.buffer.directory.iterdir()}
            q=TrainingPhases(m,c,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.beta,4.);self.assertEqual(q.cache_origin,1);self.assertEqual(P.next_phase(q.schedule),9)
            self.assertEqual(q.buffer.receipts,[]);self.assertEqual(hashes,{x.name:sha(x) for x in p.buffer.directory.iterdir()})
            after=(m.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())
            self.assertTrue(_same_payload(before,after))
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=c)))
            cleanup=q.startup(1);self.assertTrue(cleanup['deleted_files'])
            self.assertEqual([x.name for x in p.buffer.directory.iterdir()],['owner.json'])
            # Restart immediately after reviewed adoption but before collection.
            again=TrainingPhases(m,c,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(again.schedule,q.schedule);self.assertEqual(again.startup(1)['deleted_files'],[])

    def test_real_phase_retains_last_safe_interval_even_with_high_policy_kl(self):
        m,opt,c=config();c['value_phase']['batch_size']=16
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(m,p,g)
                if g<8:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            def count():return max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            step=p.optimizer.step;saved={};ppo_before=copy.deepcopy(opt.state_dict())
            def capture(*args,**kwargs):
                result=step(*args,**kwargs)
                if count()==48:saved.update(weights=copy.deepcopy(m.state_dict()),adam=copy.deepcopy(p.optimizer.state_dict()))
                return result
            def losses(net,ds,ids,update,**kwargs):
                games=sorted({ds.data[i][2][0] for i in ids});sign={g:(1 if j%2 else -1) for j,g in enumerate(games)}
                n=count();base_loss=1. if not n else .99 if n==16 else .991 if n==32 else .992 if n==48 else 1.2
                spread=.02 if n==32 else .04 if n==48 else 0.
                return {i:base_loss+spread*sign[ds.data[i][2][0]] for i in ids}
            probe=A.Reference.probes
            def high_kl(ref,net):return dict(probe(ref,net),overall=.02,age0=.03)
            with patch.object(p.optimizer,'step',capture),patch.object(A,'losses',losses),patch.object(A.Reference,'probes',high_kl):
                r=A.fit(m,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                    c['value_phase'],c['update'],beta=4.,snapshot_weight=1.,dropout=None,run_phase=True,
                    ppo_work=p.ppo_work,epoch_cap=5,ppo_peak=D.rate_receipt(6e-6,8,32))
            # This fixture has 55 batches/epoch: the mandatory epoch-tail
            # check rejects before the next regular check at step 64.
            self.assertEqual(r['attempted_steps'],55);self.assertEqual(r['optimizer_steps'],48)
            self.assertEqual([x['loss_reference_step'] for x in r['interval_reports']],[0,16,32,48])
            self.assertEqual(r['acceptance_generations'],list(range(8,0,-1)))
            self.assertGreater(r['selection']['retained_loss'],r['selection']['minimum_loss'])
            self.assertEqual(r['early_stop_reason'],'validation_patience');self.assertFalse(r['kl_guard_tripped'])
            self.assertEqual(r['beta_clone_final'],4.);self.assertTrue(_same_payload(saved['weights'],m.state_dict()))
            self.assertTrue(_same_payload(saved['adam'],p.optimizer.state_dict()));self.assertTrue(_same_payload(ppo_before,opt.state_dict()))

if __name__=='__main__':unittest.main()
