"""Focused gates for changed learning semantics, cache trust and retention."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from fixtures import game,position
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.features import Encoder
from brass_ai.learning.replay import split_for
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.step5.league import seeds_for,write_json
from brass_ai.step5.evaluation import evaluation_config
from brass_ai.recovery.config import UPDATE_V21,RETENTION,validate_config
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.inputs import InputBoundary
from brass_ai.recovery.replay import read_shards,write_shard,cache_receipt
from brass_ai.recovery.retention import plan,prune,restore,run_lock,update_population
from brass_ai.recovery.updates import ppo_update,prepared,supervised_epoch


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);(SOURCE_ROOT/'artifacts/recovery').mkdir(parents=True,exist_ok=True)

    def model(self):
        torch.manual_seed(7111);return FourSeatNet(ModelConfig.preset('small'))

    def onpolicy(self,forced):
        model=self.model();policy=NeuralPolicy(model,focus='neutral',exploration=.05);g=game()
        seed=seeds_for(17100000,1)[0][0];g['setup_group']=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
        g.update(split=split_for(g['setup_group']),opponents=[policy.policy_id]*4)
        for i,row in enumerate(g['records']):
            if i in forced: row['choices']=row['choices'][:1]
            index,p,_=policy.select_batch([(row['view'],row['choices'],random.Random(i))])[0]
            row.update(chosen=index,behavior=dict(kind='model',policy_id=policy.policy_id,probability=p,temperature=1.))
        return model,policy,g

    def test_cache_roundtrip_seats_and_mutation(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts/recovery') as tmp:
            path=Path(tmp)/'games.jsonl.gz';g=game()
            for row in g['records']: row.update(style=[.3,-.2,.6,.1],intent=[.1]*8)
            g2=copy.deepcopy(g);g2['game_id']='another'
            write_shard(path,[g,g2],dict(purpose='training'))
            receipt=cache_receipt(path);rows=read_shards([path,path],receipts={str(path):receipt})
            self.assertEqual(len(rows),2)
            for r in rows[0]['records']:
                actual=rows[0].encoded(r,PreparedEncoder());expected=Encoder().encode(r['view'],r['choices'],r['style'],r['intent'])
                np.testing.assert_array_equal(actual.state,expected.state)
                np.testing.assert_array_equal(actual.actions,expected.actions)
                np.testing.assert_array_equal(actual.stable_to_relative,expected.stable_to_relative)
                neutral=dict(r,style=[0.]*4,intent=[0.]*8)
                derived=rows[0].encoded(neutral,PreparedEncoder(),allow_context=True)
                control=Encoder().encode(r['view'],r['choices'],neutral['style'],neutral['intent'])
                np.testing.assert_array_equal(derived.state,control.state)
                np.testing.assert_array_equal(derived.actions,control.actions)
                self.assertFalse(derived.state.flags.writeable)
            # Plausible schema-valid alterations also invalidate the checked row.
            changed=copy.deepcopy(rows[0]['records'][0]);changed['view']['players'][0]['cash']+=1
            with self.assertRaisesRegex(ValueError,'record changed'): rows[0].encoded(changed,PreparedEncoder())
            changed=copy.deepcopy(rows[0]['records'][0]);changed['chosen']=(changed['chosen']+1)%len(changed['choices'])
            with self.assertRaisesRegex(ValueError,'record changed'): rows[0].encoded(changed,PreparedEncoder())
            rows[0]['result']['seats'][0]['vp']+=1
            with self.assertRaisesRegex(ValueError,'metadata changed'): rows[0].assert_metadata()
            changed=copy.deepcopy(rows[0]['records'][0]);changed['view']['players'][1]['own_hand']=['location:Birmingham']
            with self.assertRaises(Exception): rows[0].encoded(changed,PreparedEncoder())
            data=Path(str(path)+'.features.npz');data.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'receipt'): read_shards([path],receipts={str(path):receipt})
            # An unsigned foreign cache is ignored, and source records are checked.
            self.assertEqual(len(read_shards([path])),2)

    def test_ingress_cache_never_hides_unknown_content(self):
        boundary=InputBoundary();v,c=position();boundary.check(v,c);boundary.check(v,c)
        self.assertEqual(boundary.report()['frame_cache_hits'],1)
        v['unknown']=1
        with self.assertRaises(Exception): boundary.check(v,c)

    def test_actor_statistics_exclude_forced_critic_can_learn_all_forced(self):
        model,policy,g=self.onpolicy({0,1,2,3});before={k:v.clone() for k,v in model.state_dict().items()}
        config=dict(UPDATE_V21,effective_batch_size=2,microbatch_size=1,critic_batch_size=2,epochs=1)
        report=ppo_update(model,torch.optim.AdamW(model.parameters(),lr=1e-4),[g],policy,config)
        self.assertEqual(report['meaningful_decisions'],0);self.assertEqual(report['forced_decisions'],4)
        self.assertEqual(report['epochs'][0]['actor_positions'],0);self.assertEqual(report['epochs'][0]['critic_positions'],4)
        self.assertEqual(report['optimizer_steps'],2);self.assertFalse(report['rehearsal_active'])
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        model,policy,g=self.onpolicy({0,1});report=ppo_update(model,torch.optim.AdamW(model.parameters(),lr=1e-6),[g],policy,config)
        self.assertEqual(report['meaningful_decisions'],2);self.assertEqual(report['advantage']['scope'],'nonforced actor decisions')
        self.assertEqual(report['kl_probe_rows'],2)

    def test_bc_nonforced_denominator_and_root_metrics(self):
        model,policy,g=self.onpolicy({0,1});data=prepared([g]);cfg=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2)
        report=supervised_epoch(model,None,data,cfg,training=False)
        self.assertEqual(report['meaningful_decisions'],2);self.assertEqual(report['decisions'],4)
        self.assertAlmostEqual(report['loss'],report['policy_loss']+.5*report['value_loss'])
        control=supervised_epoch(model,None,data,dict(cfg,actor_nonforced=False),training=False)
        self.assertAlmostEqual(report['natural_policy_loss'],2*control['natural_policy_loss'])
        self.assertTrue(0<=report['root_action_top1']<=1)
        self.assertEqual(sum(v['decisions'] for v in report['by_teacher_era_phase_action'].values()),2)

    def test_monitor_and_broader_have_exact_small_counts(self):
        config=read(SOURCE_ROOT/'configs/step5/lab.json');validate_config(config)
        monitor=evaluation_config(config);self.assertEqual(monitor['setups']*4*len(monitor['lineups']),8)
        broader=evaluation_config(config,lineups=['mixed-neural','mixed-v3'],include_legacy=True)
        self.assertEqual(broader['setups']*4*len(broader['lineups']),16)
        with self.assertRaises(ValueError): evaluation_config(config,lineups=['heldout-pottery-v3'],include_legacy=True)
        from brass_ai.step5.evaluation import validate_evaluation_config
        invalid=copy.deepcopy(monitor);invalid['retention']['replay_rounds']=-1
        with self.assertRaises(ValueError): validate_evaluation_config(invalid)

    def test_retention_pins_quarantine_restore_and_active_run(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts/recovery') as tmp:
            run=Path(tmp);model=self.model();history=[]
            for i in range(5):
                with torch.no_grad(): next(model.parameters()).add_(.0001)
                path=run/f'generation-{i:04d}.pt';identity=save_checkpoint(path,model,torch.optim.AdamW(model.parameters()),step=i)
                history.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=i))
            state=dict(format='step5-recovery-league/2.1.0',signature=dict(config=dict(retention=RETENTION)),iteration=5,history=history)
            update_population(SOURCE_ROOT,run,state,dict(recent=2,anchors=1));write_json(run/'state.json',state)
            # An indirect BC/dataset dependency must survive cleanup too.
            dependency=run/'iteration-0001/games.jsonl.gz';dependency.parent.mkdir();dependency.write_bytes(b'BC source')
            write_json(dependency.parent/'report.json',{})
            manifest=run/'dataset.json';write_json(manifest,dict(entries=[dict(path=dependency.relative_to(SOURCE_ROOT).as_posix())]))
            state['signature']['config']['rehearsal_manifest']=manifest.relative_to(SOURCE_ROOT).as_posix();write_json(run/'state.json',state)
            protected=run/'iteration-0000/games.jsonl.gz';protected.parent.mkdir();protected.write_bytes(b'dataset')
            write_json(protected.parent/'report.json',{})
            write_json(run/'pins.json',dict(paths=[protected.relative_to(SOURCE_ROOT).as_posix()]))
            decision=plan(SOURCE_ROOT,run)
            self.assertNotIn(protected.relative_to(run).as_posix(),[e['path'] for e in decision['entries']])
            self.assertNotIn(dependency.relative_to(run).as_posix(),[e['path'] for e in decision['entries']])
            self.assertEqual(len(state['population']),3)
            for entry in state['population']:
                _,raw=load_checkpoint(SOURCE_ROOT/entry['path']);self.assertIsNone(raw['optimizer'])
            result=prune(SOURCE_ROOT,run,apply=True)
            self.assertTrue(all((run/e['quarantine']).exists() for e in result['entries']))
            self.assertEqual(restore(SOURCE_ROOT,result['manifest'])['restored_files'],len(result['entries']))
            with run_lock(run):
                with self.assertRaisesRegex(ValueError,'active'): restore(SOURCE_ROOT,result['manifest'])
            with self.assertRaises(ValueError): prune(SOURCE_ROOT,SOURCE_ROOT/'results/1.1',apply=True)

    def test_parallel_matched_matrix_private_hosts_and_deadlines(self):
        from brass_ai.recovery.parallel import ParallelS2
        from brass_ai.step5.search import SearchIncomplete
        import time
        class Worker:
            def __init__(self): self.base=type('Base',(),{'request_deadline':None})();self.rollouts=[];self.calls=[]
            def scenario(self,view,plan,seed,context,deadline):
                self.calls.append((tuple(plan),seed,view['seat'],self.base.request_deadline));return seed+len(plan)
        planner=object.__new__(ParallelS2);planner.telemetry=dict(parallel_batches=0);planner.replicas=[Worker(),Worker()]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            planner.parallel_pool=pool;deadline=time.monotonic()+1
            values=planner.matrix(dict(seat=2),[['a'],['a','b']],[1,2],[[],[]],deadline)
            self.assertEqual(values,[[2,3],[3,4]])
            self.assertTrue(all(len(w.calls)==2 and w.calls[0][1]==w.calls[1][1] for w in planner.replicas))
            self.assertTrue(all(w.base.request_deadline is None for w in planner.replicas))
            with self.assertRaises(SearchIncomplete): planner.matrix(dict(seat=2),[['a']],[1],None,time.monotonic()-1)
