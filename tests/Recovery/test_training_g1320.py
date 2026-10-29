"""Reward-only continuation and durable, restart-safe PPG target refresh."""
import copy
from dataclasses import replace
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.reward import CostLedger,replay_returns,terminal_utility
from brass_ai.recovery import gae,snapshot_r2,ppg_g210 as G
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.training_revision_g1320 import configure,CURRENT,PREVIOUS
from brass_ai.recovery.utility import reward_spec,migrate,reviewed_reward_upgrade
from brass_ai.step5.league import write_json,implementation_id
from test_training_g1120_kl32 import smoke_config as prior_config
from test_ppg_eight import stage_only
from test_insolvency_recovery import row
from fixtures import result

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1320-20261027T075134827445Z'


def exported_state():
    manifest=read(EXPORT/'export-manifest.json')
    entry=next(e for e in manifest['reused_files'] if e['path']=='state.json')
    path=SOURCE_ROOT/entry['previous_export']/entry['previous_path']
    # Git checks this text fixture out with CRLF on Windows; the Linux export
    # binds LF. Production binary checkpoint validation remains byte-exact.
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==entry['sha256']
    return read(path)


def smoke_config():
    model,opt,c=prior_config();c=configure(c)
    model.config=replace(model.config,utility=reward_spec(c))
    return model,opt,c


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_change_fresh_default_and_archived_config(self):
        old=read(EXPORT/'recorded-config.json');original=copy.deepcopy(old)
        new=validate_config(configure(old))
        self.assertEqual(old,original);self.assertEqual(configure(new),new)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'insolvency','search'})
        self.assertEqual(new['insolvency'],CURRENT)
        self.assertEqual(new['search'],dict(old['search'],reward=reward_spec(new)))
        self.assertEqual(validate_config(old),old)
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh['insolvency'],CURRENT);self.assertEqual(fresh['search']['reward'],reward_spec(fresh))
        self.assertEqual(new['value_phase'],old['value_phase'])
        for field,value in [('max_cost',.08),('event_cost',.005)]:
            bad=copy.deepcopy(old);bad['insolvency'][field]=value
            with self.assertRaises(ValueError):configure(bad)
        bad=copy.deepcopy(old);bad['search']['reward']['insolvency']['event_cost']=.01
        with self.assertRaisesRegex(ValueError,'Search/learner'):configure(bad)

    def test_reward_transition_is_exact_and_fresh_returns_keep_the_cap(self):
        old=read(EXPORT/'recorded-config.json');new=configure(old)
        previous,current=reward_spec(old),reward_spec(new)
        self.assertTrue(reviewed_reward_upgrade(previous,current))
        self.assertFalse(reviewed_reward_upgrade(current,previous))
        bad=copy.deepcopy(current);bad['vp']=.002
        self.assertFalse(reviewed_reward_upgrade(previous,bad))
        rows=[]
        for n in range(1,10):
            rows.extend([row(0,round=n),row(0,round=n,phase='debt',kind='liquidate',debts={0:5}),
                         row(0,round=n,phase='debt',kind='liquidate',debts={0:1})])
        targets,summary=replay_returns(rows,result(),current);ledger=CostLedger(current)
        for record,label in zip(rows,targets):
            ledger.observe(record['view']);np.testing.assert_allclose(ledger.remaining,label['remaining'])
            ledger.chosen(record['view'],record['choices'][record['chosen']])
        self.assertAlmostEqual(summary['cost'][0],.075)
        for count in (1,2,7,8,9):
            _,s=replay_returns(rows[:count*3],result(),current)
            self.assertAlmostEqual(s['cost'][0],min(count*.01,.075))
        one,summary=replay_returns(rows[:3],result(),current)
        self.assertAlmostEqual(terminal_utility(result(),current)[0]-one[0]['utility'][0],.01)
        self.assertEqual(summary['reward_id'],digest(current))

    def test_refresh_is_after_commit_retry_safe_and_preserves_training_state(self):
        model,opt,old=prior_config();new=configure(old)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            for g in (1,2):
                stage_only(model,p,g);write_json(run/'state.json',dict(iteration=g,signature=dict(config=old)))
                p.buffer.committed(g);p.state=gae.initial(old['advantage'],g)
                p.critic_state['generation']=g;p.critic_state['weight_recovery']=snapshot_r2.initial(g,p.critic_state['snapshot_weight'])
            for optimizer in (opt,p.optimizer):
                parameter=next(model.parameters());parameter.grad=torch.ones_like(parameter)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            p.beta=2.;p.trips=[True,False];p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(p.payload())})
            old_raw=copy.deepcopy(raw);p.buffer.pending=None
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            before=copy.deepcopy((model.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            with self.assertRaisesRegex(ValueError,'explicit reward migration'):migrate(model,opt,reward_spec(new))
            model._reviewed_cap_upgrade=True;receipt=migrate(model,opt,reward_spec(new))
            self.assertIsNotNone(receipt);self.assertIsNone(model.utility_calibration)
            q=TrainingPhases(model,new,raw=raw,generation=2,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.cache_origin,2);self.assertEqual(q.buffer.receipts,[])
            self.assertEqual(q.g210_migration['invalidated_generations'],[1,2])
            self.assertEqual((q.beta,q.trips),(p.beta,p.trips))
            self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            self.assertTrue(_same_payload(old_raw,raw))
            with self.assertRaisesRegex(ValueError,'committed reward'):q.startup(2)
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            write_json(run/'state.json',dict(iteration=2,signature=dict(config=new)))
            unknown=q.buffer.directory/'user-data.txt';unknown.write_text('keep')
            with self.assertRaisesRegex(ValueError,'Unknown PPG cache artifact'):q.startup(2)
            unknown.unlink()
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            cleanup=q.startup(2);self.assertEqual(cleanup['next_phase'],3);self.assertGreater(cleanup['deleted_bytes'],0)
            self.assertEqual([f.name for f in q.buffer.directory.iterdir()],['owner.json'])
            retry=TrainingPhases(model,new,raw=raw,generation=2,root=SOURCE_ROOT,run=run)
            self.assertEqual(retry.startup(2)['deleted_files'],[])
            stage_only(model,q,3)
            self.assertEqual([r['generation'] for r in q.buffer.pending],[3])
            self.assertEqual(q.buffer.pending[0]['reward_id'],digest(reward_spec(new)))
            q.state=gae.initial(new['advantage'],3);q.critic_state['generation']=3
            q.critic_state['weight_recovery']=snapshot_r2.initial(3,q.critic_state['snapshot_weight'])
            raw_new=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(q.payload())})
            write_json(run/'state.json',dict(iteration=3,signature=dict(config=new)));q.buffer.committed(3)
            hashes={f.name:sha(f) for f in q.buffer.directory.iterdir()}
            resumed=TrainingPhases(model,new,raw=raw_new,generation=3,root=SOURCE_ROOT,run=run)
            self.assertIsNone(resumed.startup(3))
            self.assertEqual(hashes,{f.name:sha(f) for f in q.buffer.directory.iterdir()})
            stage_only(model,resumed,4)
            self.assertEqual([r['generation'] for r in resumed.buffer.pending],[3,4])
            self.assertTrue(all(r['reward_id']==digest(reward_spec(new)) for r in resumed.buffer.pending))

    def test_helper_pins_g1320_read_only_and_rejects_other_changes(self):
        spec=importlib.util.spec_from_file_location('g1320_helper',SOURCE_ROOT/'scripts/configure-training-g1320.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state=exported_state();self.assertEqual(state['iteration'],1320)
        self.assertEqual(state['signature']['config'],read(EXPORT/'recorded-config.json'))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated checkpoint stand-in')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            output='configs/step5/g1320.json';before=(sha(run/'state.json'),sha(checkpoint))
            report=helper.generate(root,run,output,1320)
            self.assertEqual(report,helper.generate(root,run,output,1320));self.assertFalse(report['run_modified'])
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            signature=dict(state['signature'],implementation=implementation_id(),config=configure(state['signature']['config']))
            self.assertTrue(check_signature(state,signature,reviewed=True))
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=False)
            signature['config']['exploration']=0.
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=True)
            with self.assertRaisesRegex(ValueError,'Stale expected generation'):helper.generate(root,run,output,1319)
            # Retrying after durable adoption is read-only and writes the same config.
            state['signature'].update(implementation=implementation_id(),config=configure(state['signature']['config']))
            write_json(run/'state.json',state)
            self.assertEqual(report,helper.generate(root,run,output,1320))
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint missing/changed'):helper.generate(root,run,output,1320)


if __name__=='__main__':unittest.main()
