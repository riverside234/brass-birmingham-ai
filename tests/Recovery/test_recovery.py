import copy
from concurrent.futures import ThreadPoolExecutor
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixtures import game,position
from brass_ai.learning.batching import Evaluator,Request
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.contracts import SOURCE_ROOT,read_json,digest
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.step5.league import seeds_for
from brass_ai.learning.replay import split_for
from brass_ai.recovery.config import UPDATE,EXECUTION,CURRICULUM,table_kind,validate_config
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.inference import InferenceService,RemoteEvaluator
from brass_ai.recovery.updates import bc_loss,ppo_update


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(671)
        self.model=FourSeatNet(ModelConfig.preset('small'))

    def onpolicy(self):
        policy=NeuralPolicy(self.model,focus='neutral',exploration=.05)
        g=game();seed=seeds_for(17100000,1)[0][0]
        group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
        g.update(setup_group=group,split=split_for(group),opponents=[policy.policy_id]*4)
        for row in g['records']:
            rng=random.Random(row['seat']);chosen,prob,_=policy.select_batch([(row['view'],row['choices'],rng)])[0]
            row.update(chosen=chosen,behavior=dict(kind='model',policy_id=policy.policy_id,probability=prob,temperature=1.0))
        return policy,g

    def test_fast_encoder_preserves_exact_tensors_all_seats(self):
        old,new=Encoder(),PreparedEncoder()
        for seat in range(4):
            view,choices=position(seat)
            a,b=old.encode(view,choices),new.encode(view,choices)
            for field in ('state','actions','stable_to_relative'):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
            self.assertEqual(a.choice_ids,b.choice_ids)
        for kind in old.kinds:
            c=dict(kind=kind,card=old.cards[0])
            np.testing.assert_array_equal(old.choice_features(c),new.choice_features(c))

    def test_fast_ingress_rejects_opponent_hand_and_unknown_fields(self):
        view,choices=position();view['opponent_hands']=[['location:Birmingham']]
        with self.assertRaises(Exception): PreparedEncoder().encode(view,choices)

    def test_bc_has_gradient_against_every_legal_alternative(self):
        logits=torch.zeros((1,3),requires_grad=True)
        loss=bc_loss(logits,torch.tensor([1])).mean();loss.backward()
        self.assertLess(float(logits.grad[0,1]),0)
        self.assertTrue(bool((logits.grad[0,[0,2]]>0).all()))
        updated=logits.detach()-.1*logits.grad
        self.assertGreater(float(updated.softmax(-1)[0,1]),1/3)

    def test_cached_bc_batches_preserve_real_update(self):
        from brass_ai.recovery.updates import prepared,Dataset,supervised_epoch
        data=prepared([game()],neutral=True)
        a,b=copy.deepcopy(self.model),copy.deepcopy(self.model)
        config=dict(UPDATE,effective_batch_size=2,microbatch_size=1)
        state=torch.get_rng_state()
        left=supervised_epoch(a,torch.optim.SGD(a.parameters(),lr=1e-4),data,config)
        torch.set_rng_state(state)
        cached=Dataset(data,'cpu',policy=False)
        right=supervised_epoch(b,torch.optim.SGD(b.parameters(),lr=1e-4),cached,config)
        self.assertIsNone(cached.q)
        self.assertEqual(left['optimizer_steps'],2)
        self.assertAlmostEqual(left['loss'],right['loss'],places=6)
        for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y)

    def test_host_timing_preserves_rpc_bytes_and_returned_private_view(self):
        import io,json
        from types import SimpleNamespace
        from brass_ai.host import RulesHost
        from brass_ai.recovery.telemetry import instrument_host
        view={'own_hand':['private-card']}
        hosts=[]
        for _ in range(2):
            host=RulesHost.__new__(RulesHost);host.sequence=0
            host.process=SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO(json.dumps({'id':'1','ok':True,'data':view})+'\n'))
            hosts.append(host)
        instrument_host(hosts[1])
        self.assertEqual(hosts[0].call('observe',seat=2),hosts[1].call('observe',seat=2))
        self.assertEqual(hosts[0].process.stdin.getvalue(),hosts[1].process.stdin.getvalue())
        metrics=hosts[1].recovery_metrics.report()
        self.assertEqual(metrics['host.rpc.observe']['calls'],1)
        self.assertEqual(metrics['host.ipc.stdout.read_wait']['calls'],1)
        self.assertNotIn('private-card',json.dumps(metrics))

    def test_effective_batches_create_multiple_optimizer_steps(self):
        policy,g=self.onpolicy()
        config=dict(UPDATE,effective_batch_size=2,microbatch_size=1,target_kl=.1,epochs=2)
        report=ppo_update(self.model,torch.optim.AdamW(self.model.parameters(),lr=1e-5),[g],policy,config)
        self.assertEqual(report['optimizer_steps'],4)
        self.assertEqual(report['learner_trajectories'],4)
        self.assertTrue(all('post_update_kl' in e for e in report['epochs']))

    def test_memory_microbatches_preserve_effective_gradient(self):
        policy,g=self.onpolicy();a=copy.deepcopy(self.model);b=copy.deepcopy(self.model)
        config=dict(UPDATE,effective_batch_size=4,epochs=1,target_kl=.1)
        rng=torch.get_rng_state()
        ppo_update(a,torch.optim.SGD(a.parameters(),lr=1e-4),[g],policy,dict(config,microbatch_size=1))
        torch.set_rng_state(rng)
        ppo_update(b,torch.optim.SGD(b.parameters(),lr=1e-4),[g],policy,dict(config,microbatch_size=4))
        for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-7)

    def test_kl_guard_stops_large_updates(self):
        policy,g=self.onpolicy()
        report=ppo_update(self.model,torch.optim.AdamW(self.model.parameters(),lr=.01),[g],policy,
            dict(UPDATE,effective_batch_size=2,microbatch_size=1,target_kl=1e-10,epochs=4))
        self.assertTrue(report['kl_early_stop']);self.assertEqual(report['optimizer_steps'],1)

    def test_curriculum_quotas_and_resume_offset(self):
        for start,expected in [(0,(128,64,128)),(2400,(176,80,64)),(6000,(208,80,32)),(11000,(224,80,16))]:
            kinds=[table_kind(i,CURRICULUM) for i in range(start,start+320)]
            self.assertEqual(tuple(kinds.count(k) for k in ('selfplay','historical','legacy')),expected)
        whole=[table_kind(i,CURRICULUM) for i in range(320)]
        restarted=[table_kind(i,CURRICULUM) for i in range(160)]+[table_kind(i,CURRICULUM) for i in range(160,320)]
        self.assertEqual(whole,restarted)
        cfg=read_json(SOURCE_ROOT/'configs/step5/lab.json');validate_config(cfg)
        with self.assertRaises(ValueError): validate_config(dict(cfg,legacy_every=2))

    def test_shared_service_routes_private_observers_and_preserves_outputs(self):
        config=dict(EXECUTION,batch_size=4,flush_ms=50.0)
        old=Evaluator(self.model)
        with InferenceService(config,'cpu') as service:
            identity=service.register(self.model);lease=service.channels(2)
            remotes=[RemoteEvaluator(lease.acquire(),identity) for _ in range(2)]
            requests=[]
            import time
            for seat in range(2):
                v,c=position(seat)
                requests.append(Request('observer-'+str(seat),identity,v,c,time.monotonic()+30))
            expected=old.predict(requests)
            with ThreadPoolExecutor(max_workers=2) as executor:
                actual=list(executor.map(lambda pair:pair[0].predict([pair[1]])[0],zip(remotes,requests)))
            for a,e in zip(actual,expected):
                self.assertEqual(a['observer'],e['observer']);self.assertEqual(a['input_digest'],e['input_digest'])
                np.testing.assert_allclose(a['probabilities'],e['probabilities'],rtol=1e-5,atol=1e-7)
            self.assertEqual(service.metrics['positions'],2)
            self.assertEqual(service.metrics['max_batch_positions'],2)

    def test_remote_rejects_wrong_frozen_model(self):
        config=dict(EXECUTION,batch_size=4)
        with InferenceService(config,'cpu') as service:
            identity=service.register(self.model);client=RemoteEvaluator(service.channels(1).acquire(),identity)
            import time
            v,c=position()
            with self.assertRaises(ValueError): client.predict([Request('bad','wrong',v,c,time.monotonic()+30)])

    def test_search_deadline_reaches_inference_and_restores_policy(self):
        from brass_ai.step5.policy import inference_deadline
        from brass_ai.search.common import SearchIncomplete
        import time
        policy=NeuralPolicy(self.model,focus='neutral')
        v,c=position();rng=random.Random(81)
        with patch.object(policy.evaluator,'predict',wraps=policy.evaluator.predict) as predict:
            deadline=time.monotonic()+10
            with inference_deadline([policy],deadline): policy.predict([(v,c,rng)])
            self.assertEqual(predict.call_args.args[0][0].deadline,deadline)
            with inference_deadline([policy],time.monotonic()-1):
                with self.assertRaises(SearchIncomplete): policy.predict([(v,c,rng)])
            self.assertIsNone(policy.request_deadline)

    def test_bc_early_stop_resume_and_checkpoint_integrity(self):
        from brass_ai.recovery import demonstrations as demos
        from brass_ai.recovery.updates import prepared
        from brass_ai.learning.checkpoint import save_checkpoint
        from brass_ai.bank import read
        train=game();train['game_id']='training-game'
        val=game();val['game_id']='validation-game'
        data={'train':prepared([train],neutral=True),'validation':prepared([val],neutral=True)}
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            directory=Path(directory);initial=directory/'initial.pt'
            save_checkpoint(initial,self.model)
            manifest=directory/'manifest.json';manifest.write_text('{}',encoding='utf-8')
            run=directory/'bc'
            with patch.object(demos,'load_dataset',side_effect=lambda root,path,split:(data[split],{})), \
                 patch.object(demos,'supervised_epoch',return_value={'loss':1.0}) as update:
                with self.assertRaisesRegex(ValueError,'random residual weights'):
                    demos.train_bc(SOURCE_ROOT,run,manifest,initial,UPDATE,epochs=10,min_epochs=1,patience=1)
                demos.train_bc(SOURCE_ROOT,run,manifest,None,UPDATE,epochs=10,min_epochs=1,patience=1)
                self.assertEqual(update.call_count,6)  # Natural + balanced validation each epoch.
                report=demos.train_bc(SOURCE_ROOT,run,manifest,None,UPDATE,epochs=10,min_epochs=1,patience=1)
                self.assertEqual(report['epochs'],2);self.assertEqual(update.call_count,6)
                state=read(run/'state.json');checkpoint=run/state['last']['path']
                checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                    demos.train_bc(SOURCE_ROOT,run,manifest,None,UPDATE,epochs=10,min_epochs=1,patience=1)


if __name__=='__main__': unittest.main()
