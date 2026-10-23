"""Reviewed g620 reward transition and startup-only retirement of stale targets."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.reward import CostLedger,MAX_INSOLVENCY_COST
from brass_ai.learning.batching import Request,utility_budget,predict_encoded
from brass_ai.recovery import dcpg_config as D,ppg_config as P,gae,snapshot_r2,ppg_epochs
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import (insolvency_010_config,insolvency_020_config,
    insolvency_040_cap100_config,vp_cap_config,validate_config)
from brass_ai.recovery.utility import reward_spec,migrate,event_cost_upgrade
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,stage_only

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0620-20261023T133150568578Z'


class Reward(unittest.TestCase):
    def test_exact_migration_and_per_event_cap(self):
        old=read(EXPORT/'recorded-config.json');new=insolvency_040_cap100_config(old)
        validate_config(new)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'insolvency','search'})
        self.assertEqual(new['insolvency'],dict(event_cost=.04,max_cost=.10))
        self.assertEqual(new['search']['reward'],reward_spec(new))
        self.assertEqual(insolvency_040_cap100_config(new),new)
        self.assertTrue(event_cost_upgrade(reward_spec(old),reward_spec(new)))
        from brass_ai.recovery.config import insolvency_040_config
        earlier=insolvency_040_config(old)
        self.assertEqual(earlier['insolvency'],dict(event_cost=.04,max_cost=.16))
        self.assertEqual(insolvency_040_cap100_config(earlier),new)
        self.assertTrue(event_cost_upgrade(reward_spec(earlier),reward_spec(new)))
        self.assertEqual(MAX_INSOLVENCY_COST,.16)
        ledger=CostLedger(reward_spec(new))
        for n in range(1,8):
            ledger.charge(0)
            self.assertAlmostEqual(ledger.spent[0],min(n*.04,.10))
        self.assertEqual(ledger.spent[1:].tolist(),[0.,0.,0.])
        bad=copy.deepcopy(new);bad['insolvency']['max_cost']=.161
        bad['search']['reward']['insolvency']['max_cost']=.161
        with self.assertRaises(ValueError):validate_config(bad)
        self.assertFalse(event_cost_upgrade(reward_spec(new),reward_spec(old)))
        # The historical .020 migration must not start using the new default.
        older=copy.deepcopy(old);older['insolvency']=dict(event_cost=.01,max_cost=.075)
        older['search']['reward']['insolvency']=copy.deepcopy(older['insolvency'])
        self.assertEqual(insolvency_020_config(older),old)

    def test_request_admission_and_frozen_opponent_cap(self):
        model,_,cfg=setup();model.config=replace(model.config,utility=reward_spec(
            insolvency_040_cap100_config(insolvency_020_config(insolvency_010_config(cfg)))))
        request=Request('one','model',{},[],float('inf'),remaining=(.10,.08,.04,0.))
        self.assertEqual(request.remaining,(.10,.08,.04,0.))
        self.assertEqual(utility_budget(model,[request.remaining]).shape,(1,4))
        with self.assertRaises(ValueError):Request('bad','model',{},[],float('inf'),remaining=(.161,0,0,0))
        with self.assertRaises(ValueError):Request('bad','model',{},[],float('inf'),remaining=(float('nan'),0,0,0))
        old_model,_,_=setup()
        with self.assertRaisesRegex(ValueError,'utility budget'):utility_budget(old_model,[request.remaining])


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_eight_to_sixteen_reward_reset_and_restart(self):self.exercise(False)

    def test_existing_sixteen_reward_reset_and_restart(self):self.exercise(True)

    def exercise(self,already_sixteen):
        model,opt,cfg=setup()
        old=vp_cap_config(insolvency_020_config(insolvency_010_config(D.beta_options(D.configure(cfg)))))
        if already_sixteen:old=D.sixteen_options(old)
        # Author the old checkpoint with the exact g620 reward; no live run.
        model.config=replace(model.config,utility=reward_spec(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            for g in range(1,11):
                stage_only(model,p,g);write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            p.state=gae.initial(old['advantage'],10);p.critic_state['generation']=10
            p.critic_state['weight_recovery']=snapshot_r2.initial(10,p.critic_state['snapshot_weight'])
            clock=P.initial(0,old['value_phase'])
            for g in range(1,11):clock=P.advance(clock,g,P.due(clock))
            p.schedule=clock;p.epoch_control=dict(ppg_epochs.initial(clock['last_phase'],old['value_phase']),phases=clock['phases'])
            p.beta=4.;p.trips=[False]
            for optimizer in (opt,p.optimizer):
                next(model.parameters()).grad=torch.ones_like(next(model.parameters()))
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            p.pending_schedule=p.schedule;p.pending_epoch_control=p.epoch_control
            raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(p.payload())})
            p.buffer.pending=None
            weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            new=insolvency_040_cap100_config(D.sixteen_options(old))
            with self.assertRaisesRegex(ValueError,'explicit reward migration'):migrate(model,opt,reward_spec(new))
            model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(new))
            q=TrainingPhases(model,new,raw=raw,generation=10,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.cache_origin,10);self.assertEqual(P.next_phase(q.schedule),26)
            self.assertEqual(q.schedule['phases'],clock['phases']);self.assertEqual(q.epoch_control,p.epoch_control)
            self.assertEqual(q.state,p.state);self.assertEqual(q.critic_state,p.critic_state)
            self.assertEqual(q.beta,4.);self.assertEqual(q.trips,p.trips)
            self.assertTrue(_same_payload(q.optimizer.state_dict(),p.optimizer.state_dict()))
            self.assertTrue(_same_payload(adam,opt.state_dict()));self.assertTrue(_same_payload(weights,model.state_dict()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(q.buffer.receipts)
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            write_json(run/'state.json',dict(iteration=10,signature=dict(config=old)))
            with self.assertRaisesRegex(ValueError,'committed reward'):q.startup(10)
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            write_json(run/'state.json',dict(iteration=10,signature=dict(config=new)))
            unknown=q.buffer.directory/'user-data.txt';unknown.write_text('keep')
            with self.assertRaisesRegex(ValueError,'Unknown'):q.startup(10)
            self.assertTrue(p.buffer.path(10,'meta.npy' if already_sixteen else 'critic.npy').exists())
            unknown.unlink();cleanup=q.startup(10)
            self.assertEqual(cleanup['next_phase'],26);self.assertGreater(cleanup['deleted_bytes'],0)
            self.assertEqual([f.name for f in q.buffer.directory.iterdir()],['owner.json'])
            # Interruption after adoption, before first new checkpoint is safe.
            repeat=TrainingPhases(model,new,raw=raw,generation=10,root=SOURCE_ROOT,run=run)
            self.assertEqual(repeat.startup(10)['deleted_files'],[])
            stage_only(model,q,11)
            q.pending_schedule=P.advance(q.schedule,11,False);q.pending_epoch_control=q.epoch_control
            q.state=gae.initial(new['advantage'],11);q.critic_state['generation']=11
            q.critic_state['weight_recovery']=snapshot_r2.initial(11,q.critic_state['snapshot_weight'])
            new_raw=dict(model_config=model.config.to_dict(),training={KEY:q.payload()})
            write_json(run/'state.json',dict(iteration=11,signature=dict(config=new)));q.buffer.committed(11)
            resumed=TrainingPhases(model,new,raw=new_raw,generation=11,root=SOURCE_ROOT,run=run)
            self.assertIsNone(resumed.startup(11));self.assertEqual(P.next_phase(resumed.schedule),26)
            self.assertEqual([r['generation'] for r in resumed.buffer.receipts],[11])
            self.assertEqual(resumed.buffer.receipts[0]['reward_id'],digest(reward_spec(new)))
            resumed.buffer.pending=copy.deepcopy(resumed.buffer.receipts)
            with resumed.buffer.datasets('cpu') as (critic,policy,receipts):
                ids=list(range(len(critic.rows)))
                self.assertTrue(torch.equal(critic.utility[ids],q.critic_dataset.utility))
                self.assertTrue((critic.data.column(ids,'remaining')==.10).all())
                remaining=utility_budget(model,[None])
                with torch.no_grad():
                    outputs=predict_encoded(model,[policy.encoded[0]],'cpu',131072,torch.from_numpy(remaining))
                self.assertTrue(all(torch.isfinite(torch.as_tensor(v)).all() for v in outputs))


if __name__=='__main__':unittest.main()
