"""Compact contracts/numerics checks for the one active search backend."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import random
from threading import Event
import time
import unittest

import numpy as np
import torch

from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.batching import Evaluator,Request,accept_response
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
from brass_ai.learning.features import Encoder
from brass_ai.learning.replay import validate_teacher
from brass_ai.search.common import SearchConfig,expansion_indices
from brass_ai.search.ismcts import MaskedISMCTS,Node,Edge,visit_teacher,select_edge,supported_continuation
from brass_ai.recovery.parallel import merge_statistics
from brass_ai.recovery.config import validate_config
from brass_ai.step5.evaluation import validate_evaluation_config
from brass_ai.host import RulesHost

ROOT=Path(__file__).resolve().parents[2]

class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model=FourSeatNet(ModelConfig.preset('medium'))
        with RulesHost(ROOT/'artifacts/r1-host/guarded') as host:
            cls.positions=[]
            for seed in range(401,405):
                c=host.reset(seed);cls.positions.append((host.observe(c['actor']),host.choices()))

    def test_references_and_active_configs(self):
        lock=json.loads((ROOT/'configs/references/masked-ismcts.lock.json').read_text())
        self.assertEqual(lock['format'],'masked-ismcts-references/1.0.0')
        import hashlib
        for pin in lock['open_spiel']['files']:
            local=ROOT/'artifacts/references/masked-ismcts'/Path(pin['path']).name
            if local.exists():self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),pin['sha256'])
        for path in (ROOT/'configs/step5').glob('*.json'):
            if path.name.startswith('previous-'):continue
            config=json.loads(path.read_text())
            if config['format'].startswith('step5-training/'):validate_config(config)
            else:validate_evaluation_config(config)
        with self.assertRaises(ValueError):SearchConfig(backend='s3').validate()

    def test_selection_support_and_branch_separation(self):
        config=SearchConfig().validate()
        node=Node(visits=4,returns=np.array([3.6,.2,.1,.1]),edges={'a':Edge(visits=4,returns=np.array([3.6,.2,.1,.1]))})
        self.assertEqual(select_edge(node,['a','b'],[.95,.05],[0,1],0,config),0) # unseen has no unconditional priority
        from brass_ai.learning.contracts import digest
        weak=[('root','a'),('second','poor')];strong=[('root','a'),('second','good')]
        stats=dict(plans={digest(weak):dict(signature=digest(weak),first='a',trace=weak,visits=1,returns=[1.,0.,0.,0.]),
                          digest(strong):dict(signature=digest(strong),first='a',trace=strong,visits=2,returns=[.6,.6,.6,.2])})
        self.assertEqual(supported_continuation(stats,'a',0),dict(strong))
        with self.assertRaises(ValueError):replace(config,q_transform='regularized_sibling').validate()
        with self.assertRaises(ValueError):replace(config,leaf_policy='settled_reply_prefix').validate()
        with self.assertRaises(TypeError):SearchConfig(future_turns=1)

    def test_virtual_loss_is_selection_only(self):
        node=Node(visits=2,returns=np.array([1.595,0,0,0]),edges={
            'a':Edge(visits=1,pending=1,returns=np.array([.8,0,0,0])),
            'b':Edge(visits=1,returns=np.array([.795,0,0,0]))})
        cfg=replace(SearchConfig(),exploration=1e-9)
        self.assertEqual(select_edge(node,['a','b'],[.5,.5],[0,1],0,replace(cfg,virtual_loss=0)),0)
        self.assertEqual(select_edge(node,['a','b'],[.5,.5],[0,1],0,cfg),1)
        self.assertEqual(node.edges['a'].visits,1)
        np.testing.assert_array_equal(node.edges['a'].returns,[.8,0,0,0])

    def test_cpu_admission_and_semantic_action_cache(self):
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor
        from brass_ai.recovery.resources import CPUAdmission
        from brass_ai.recovery.telemetry import Timings
        from brass_ai.recovery.encoding import PreparedEncoder
        gate=CPUAdmission(multiprocessing.get_context('spawn'),6,2)
        def work(_):
            with gate.acquire(2,Timings()):time.sleep(.02)
        with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(work,range(6)))
        self.assertEqual(gate.report(),dict(capacity=4,active=0,peak=4,queued=0))
        prepared=PreparedEncoder();pinned=Encoder()
        for view,choices in self.positions:
            actual=prepared.encode(view,choices);expected=pinned.encode(view,choices)
            np.testing.assert_array_equal(actual.state,expected.state)
            np.testing.assert_array_equal(actual.actions,expected.actions)
        self.assertTrue(prepared.action_cache)

    def test_dynamic_nn_queue_private_channel(self):
        from brass_ai.recovery.inference import InferenceService,RemoteEvaluator
        from brass_ai.recovery.config import EXECUTION_V21
        from concurrent.futures import ThreadPoolExecutor
        with InferenceService(dict(EXECUTION_V21,processes=1),'cpu') as service:
            identity=service.register(self.model);lease=service.channels(4)
            evaluators=[RemoteEvaluator(lease.acquire(),identity) for _ in range(4)]
            rows=[Request('queued-'+str(i),identity,v,c,time.monotonic()+30) for i,(v,c) in enumerate(self.positions)]
            ticket=evaluators[0].submit([rows[0]])
            with self.assertRaises(ValueError):evaluators[0].submit([rows[0]])
            evaluators[0].poll(ticket,wait=True)
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses=list(pool.map(lambda i:evaluators[i].predict([rows[i]])[0],range(4)))
            for row,response in zip(rows,responses):accept_response(row,response,row.view['context']['revision'],identity)
            self.assertTrue(all(e.endpoint.pending is None for e in evaluators))
            self.assertGreaterEqual(service.report()['positions'],5)

    def test_total_parameters_and_checkpoint(self):
        self.assertEqual(self.model.parameter_count,1840109)
        self.assertLessEqual(self.model.parameter_count,2000000)
        torch.manual_seed(12);other=FourSeatNet(ModelConfig.preset('medium'))
        self.assertFalse(torch.equal(other.state_dict()['state_encoder.0.weight'],self.model.state_dict()['state_encoder.0.weight']))
        path=ROOT/'artifacts/masked-exit/unit-fresh.pt'
        identity=save_checkpoint(path,self.model,training=dict(initialization='random',fresh_run='artifacts/masked-exit'))
        loaded,raw=load_checkpoint(path)
        self.assertEqual(raw['model_id'],identity);self.assertEqual(loaded.parameter_count,1840109)
        from brass_ai.learning.contracts import compatibility,assert_compatible,assert_heuristic_data_compatible
        historical=dict(compatibility(),architecture_sha256='a'*64)
        assert_heuristic_data_compatible(historical)
        with self.assertRaises(ValueError):assert_compatible(historical)
        with self.assertRaises(ValueError):assert_heuristic_data_compatible(dict(historical,encoder_sha256='a'*64))

    def test_cpu_batches_chunks_masks_and_cancel(self):
        evaluator=Evaluator(self.model,batch_size=8,max_candidate_cells=8192)
        rows=[]
        for i,(view,choices) in enumerate(self.positions):
            # Full legal sets; different shape checks also use a permitted subset
            # only for the arithmetic collation test, not as a game search mask.
            rows.append(Request(str(i),evaluator.model_id,view,choices,time.monotonic()+120))
        ticket=evaluator.submit(rows)
        batched=evaluator.poll(ticket,wait=True)
        for row,actual in zip(rows,batched):
            single=evaluator.predict([row])[0]
            accept_response(row,actual,row.view['context']['revision'],evaluator.model_id)
            for field in ('probabilities','win_share','placement','vp'):
                np.testing.assert_allclose(actual[field],single[field],rtol=2e-5,atol=2e-5)
        chunked=Evaluator(self.model,batch_size=8,max_candidate_cells=17)
        for row,actual in zip(rows,chunked.predict(rows)):
            baseline=evaluator.predict([row])[0]
            np.testing.assert_allclose(actual['probabilities'],baseline['probabilities'],rtol=2e-5,atol=2e-5)
            self.assertEqual(actual['choice_ids'],[c['id'] for c in row.choices])
        cancelled=Event();cancelled.set()
        view,choices=self.positions[0]
        row=Request('cancel',evaluator.model_id,view,choices,time.monotonic()+120,cancelled=cancelled)
        self.assertEqual(evaluator.predict([row])[0]['status'],'cancelled')
        self.assertGreaterEqual(evaluator.telemetry['max_batch_positions'],4)
        evaluator.close();chunked.close()

    def test_protected_expansion_and_unpruning(self):
        choices=[dict(id=str(i),kind='build',industry='CottonMill') for i in range(40)]
        choices += [dict(id='pottery',kind='build',industry='Pottery'),dict(id='loan',kind='loan'),
                    dict(id='develop',kind='develop'),dict(id='scout',kind='scout'),
                    dict(id='merchant',kind='beer',merchant='Oxford:0')]
        before=copy.deepcopy(choices);p=[.9/40]*40+[.1/5]*5
        config=replace(SearchConfig(),root_top_k=2,interior_top_k=2)
        initial=expansion_indices(choices,p,0,{},random.Random(11),config=config)
        self.assertTrue(set(range(40,45))<=set(initial))
        later=expansion_indices(choices,p,1000,{},random.Random(11),config=config)
        self.assertEqual(later,list(range(len(choices))));self.assertEqual(choices,before)

    def test_completed_visits_forest_and_teacher(self):
        choices=[dict(id='a'),dict(id='b'),dict(id='pottery')]
        item=dict(completed=3,interrupted=1,opponent_nodes=0,max_pending=3,opponent_choices=5,
                  expansion=[True,True,False],root={'a':dict(visits=2,returns=[1.,.4,.4,.2],available=3),
                                                   'b':dict(visits=1,returns=[.1,.2,.3,.4],available=3)})
        merged=merge_statistics([item,item],choices)
        self.assertEqual(merged['root']['a']['visits'],4)
        np.testing.assert_allclose(merged['root']['a']['returns'],[2.,.8,.8,.4])
        teacher=visit_teacher(choices,merged,[.4,.4,.2],model='a'*64,observer=3,config=SearchConfig())
        validate_teacher(teacher,3);self.assertGreater(teacher['probabilities'][2],0)
        self.assertEqual(teacher['coverage'],[True]*3);self.assertFalse(teacher['expansion'][2])
        teacher['visits'][0]+=1
        with self.assertRaises(ValueError):validate_teacher(teacher,3)
        bot=MaskedISMCTS(None,None,[],SearchConfig(),0,None,random.Random(7),time.monotonic()-1)
        stats=bot.run({},choices)
        self.assertEqual(stats['completed'],0);self.assertEqual(stats['root'],{})

if __name__=='__main__':unittest.main()
