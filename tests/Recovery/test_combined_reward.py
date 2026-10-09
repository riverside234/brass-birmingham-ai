"""Shared undiscounted returns, budget-aware search and checkpoint extension."""
import copy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from fixtures import game,position,result
from brass_ai.bank import read
from brass_ai.learning.batching import collate,Evaluator,Request,accept_response,model_id
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.learning.replay import terminal_targets,validate_teacher
from brass_ai.learning.reward import specification,terminal_components,terminal_utility,replay_returns,CostLedger
from brass_ai.recovery.config import UPDATE_V21,INSOLVENCY,validate_config
from brass_ai.recovery.updates import prepared,Dataset,ppo_update,policy_measure,distill_update
from brass_ai.recovery.utility import migrate,calibrate,calibration_ready,reward_spec
from brass_ai.recovery.telemetry import Timings
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import MaskedISMCTS,Node,Edge,select_edge,visit_teacher
from brass_ai.step5.policy import NeuralPolicy
import test_insolvency_recovery as debt_fixtures


SPEC=specification()


def scored(scores):
    value=result();best=max(scores)
    for s,score in zip(value['seats'],scores):
        s.update(vp=score[0],income=score[1],cash=score[2],rank=1+sum(b>score for b in scores),
                 utility=1/scores.count(best) if score==best else 0.)
    return value


def config(**kw):return SearchConfig(contract='masked-ismcts/1.2.0',reward=SPEC,utility_scale=1.2,**kw)


class RewardTests(unittest.TestCase):
    def test_official_tiebreakers_shared_wins_and_half_credit(self):
        r=scored([(100,4,20),(100,4,20),(100,4,19),(100,3,100)])
        parts=terminal_components(r)
        np.testing.assert_allclose(parts['win'],[.5,.5,0,0])
        np.testing.assert_allclose(parts['placement'],[5/6,5/6,1/3,0])
        np.testing.assert_allclose(terminal_utility(r,SPEC),[.5+.1*5/6+.05,.5+.1*5/6+.05,.1/3+.05,.05])
        ties=terminal_components(scored([(150,5,10)]*4))
        self.assertEqual(ties['win'],[.25]*4);self.assertEqual(ties['placement'],[.5]*4)
        # Lower-place ties also split pairwise credit, not competition rank.
        self.assertEqual(terminal_components(scored([(110,0,0),(100,0,0),(100,0,0),(90,0,0)]))['placement'],[1,.5,.5,0])

    def test_linear_vp_has_constant_marginal_at_high_scores(self):
        for base in (40,200,1000,100000):
            a=[(base+40,0,0),(base+20,0,0),(base+10,0,0),(base,0,0)]
            b=[(v+7,i,c) for v,i,c in a]
            np.testing.assert_allclose(terminal_utility(scored(b),SPEC)-terminal_utility(scored(a),SPEC),[.0035]*4,atol=1e-10)
        self.assertGreater(terminal_utility(scored(b),SPEC)[0],1.)
        custom=specification(dict(win=2.,placement=.3,vp=.001))
        self.assertAlmostEqual(terminal_utility(scored(b),custom)[0],2.3+.001*b[0][0])

    def test_causal_vector_costs_match_incremental_ledger_and_cap(self):
        rows=[]
        for n in range(1,8):
            rows += [debt_fixtures.row(0,round=n),debt_fixtures.row(0,round=n,phase='debt',kind='liquidate',debts={0:5,2:3}),
                     debt_fixtures.row(0,round=n,phase='debt',kind='liquidate',debts={0:1})]
        targets,summary=replay_returns(rows,result(),SPEC);terminal=terminal_utility(result(),SPEC)
        ledger=CostLedger(SPEC)
        for row,label in zip(rows,targets):
            ledger.observe(row['view']);np.testing.assert_allclose(ledger.remaining,label['remaining'])
            expected=terminal-(np.asarray(summary['cost'])-ledger.spent)
            np.testing.assert_allclose(label['utility'],expected)
            ledger.chosen(row['view'],row['choices'][row['chosen']])
        self.assertAlmostEqual(summary['cost'][0],.10)
        self.assertAlmostEqual(targets[0]['utility'][0]-terminal[0],-.10)
        self.assertAlmostEqual(targets[1]['utility'][0]-terminal[0],-.08)
        np.testing.assert_allclose(targets[-1]['utility'],terminal)

    def test_automatic_no_asset_debt_and_final_rail_exemption(self):
        r=debt_fixtures.row
        rows=[r(),r(3,phase='continuation',kind='finish',cash={0:0},income={0:-2}),r(round=2)]
        ledger=CostLedger(SPEC)
        for row in rows:
            ledger.observe(row['view']);ledger.chosen(row['view'],row['choices'][0])
        self.assertAlmostEqual(ledger.spent[0],.02)
        last=r(round=8,era='RailEra',phase='continuation',kind='finish',cash={0:0},income={0:-8})
        ledger=CostLedger(SPEC);ledger.observe(last['view']);ledger.chosen(last['view'],last['choices'][0])
        terminal=copy.deepcopy(last['view']);terminal['context']['terminal']=True;ledger.observe(terminal)
        np.testing.assert_array_equal(ledger.spent,[0]*4)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(340);model=FourSeatNet(ModelConfig.preset('small'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        migrate(model,optimizer,SPEC)
        return model,optimizer

    def test_versioned_migration_preserves_old_weights_adam_rng_and_roundtrip(self):
        torch.manual_seed(81);model=FourSeatNet(ModelConfig.preset('small'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        sum(p.square().mean() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        old=copy.deepcopy(model.state_dict());moments=copy.deepcopy(optimizer.state_dict()['state']);rng=torch.get_rng_state().clone()
        receipt=migrate(model,optimizer,SPEC)
        self.assertEqual(receipt['version'],'four-seat-utility-extension/1.0.0')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in old.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        for k,fields in moments.items():
            for n,v in fields.items():torch.testing.assert_close(v,optimizer.state_dict()['state'][k][n],rtol=0,atol=0)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'new.pt';save_checkpoint(path,model,optimizer,step=70)
            loaded,raw=load_checkpoint(path);restored=torch.optim.AdamW(loaded.parameters(),lr=1e-5);resume_optimizer(raw,restored)
            self.assertEqual(raw['format'],'four-seat-checkpoint/2.0.0');self.assertEqual(raw['step'],70)
            self.assertEqual(model_id(model),model_id(loaded));self.assertEqual(len(restored.param_groups[0]['params']),len(list(model.parameters())))
            self.assertFalse(calibration_ready(loaded));self.assertIsNone(migrate(loaded,restored,SPEC))
        medium=UtilityNet(utility_config(ModelConfig.preset('medium'),SPEC))
        self.assertEqual(medium.parameter_count,1850129);self.assertLessEqual(medium.parameter_count,2000000)

    def test_calibration_changes_only_new_head_and_keeps_data_and_official_targets(self):
        model,optimizer=self.model();g=game();original=copy.deepcopy(g)
        data=prepared([g],insolvency=INSOLVENCY,reward=SPEC);ds=Dataset(data,'cpu');before=copy.deepcopy(model.state_dict())
        for d in data:self.assertEqual(d[1],terminal_targets(g['result']))
        report=calibrate(model,optimizer,ds,dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2))
        self.assertTrue(report['receipt']['ready']);self.assertEqual(g,original)
        for k,v in before.items():
            if not k.startswith('utility_head.'):torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('utility_head.')))
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-5)

    def test_ppo_baseline_uses_same_four_seat_remaining_return(self):
        model,opt=self.model();g=game();policy=NeuralPolicy(model,focus='neutral',exploration=.05)
        self.addCleanup(policy.close);g['split']='train';g['opponents']=[policy.policy_id]*4
        for i,row in enumerate(g['records']):
            index,p,_=policy.select_batch([(row['view'],row['choices'],random.Random(i))])[0]
            row.update(chosen=index,behavior=dict(kind='model',policy_id=policy.policy_id,probability=p,temperature=1.))
        data=prepared([g],insolvency=INSOLVENCY,reward=SPEC);ds=Dataset(data,'cpu');cfg=dict(UPDATE_V21,epochs=1,effective_batch_size=4,microbatch_size=2,critic_batch_size=4)
        adv=torch.zeros(4);policy_measure(model,ds,range(4),policy,cfg,Timings(),True,adv)
        with torch.no_grad():out=model(*collate(ds.encoded),remaining=ds.remaining)
        expected=ds.utility[range(4),ds.seats]-out[4][range(4),ds.seats]
        torch.testing.assert_close(adv,expected)
        report=ppo_update(model,opt,[g],policy,cfg,insolvency=INSOLVENCY,reward=SPEC)
        self.assertEqual(report['reward']['spec'],SPEC);self.assertIn('utility_prediction',report['epochs'][0])
        self.assertTrue(report['utility_calibration']['receipt']['ready'])

    def test_budget_aware_batched_chunked_and_remote_inference(self):
        from brass_ai.recovery.inference import InferenceService,RemoteEvaluator
        from brass_ai.recovery.config import EXECUTION_V21
        model,_=self.model()
        with torch.no_grad():model.utility_head[-1].bias.fill_(2.)
        local=Evaluator(model,max_candidate_cells=100);chunked=Evaluator(model,max_candidate_cells=1)
        self.addCleanup(local.close);self.addCleanup(chunked.close)
        requests=[Request(str(i),local.model_id,*position(i),time.monotonic()+120,remaining=tuple(.02*j for j in range(4))) for i in range(4)]
        ordinary=local.predict(requests);chunks=chunked.predict(requests)
        with InferenceService(dict(EXECUTION_V21,processes=1),'cpu') as service:
            service.register(model);remote=RemoteEvaluator(service.channels(1).acquire(),local.model_id)
            remote_outputs=remote.predict(requests)
        for req,a,b,c in zip(requests,ordinary,chunks,remote_outputs):
            accept_response(req,a,req.view['context']['revision'],local.model_id)
            for key in ('utility','vp','placement','win_share','probabilities'):
                np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-5);np.testing.assert_allclose(a[key],c[key],rtol=1e-5,atol=1e-5)
            self.assertGreater(min(a['utility']),1.);self.assertEqual(a['reward_id'],digest(SPEC))

    def test_budget_inputs_preserve_stable_seat_mapping(self):
        model,_=self.model()
        with torch.no_grad():
            for p in model.utility_head.parameters():p.zero_()
            for i in range(4):
                model.utility_head[0].weight[i,model.config.hidden+i]=1.
                model.utility_head[-1].weight[i,i]=1.
            context=torch.zeros(4,model.config.hidden);budgets=torch.tensor([[0.,.02,.06,.1]]*4)
            mapping=torch.tensor([[(s-observer)%4 for s in range(4)] for observer in range(4)])
            heads=(torch.zeros(4,4),torch.full((4,4),4.),torch.zeros(4,4))
            value=model.predict_utility(context,mapping,heads,budgets)
            torch.testing.assert_close(value,torch.nn.functional.gelu(budgets))

    def test_old_teachers_readable_but_rejected_by_new_distillation(self):
        model,opt=self.model();g=game();g['split']='train';row=g['records'][0]
        stats=dict(root={c['id']:dict(visits=2,returns=[.5]*4) for c in row['choices']},expansion=[True]*3)
        teacher=visit_teacher(row['choices'],stats,[1/3]*3,model='a'*64,observer=0,config=SearchConfig())
        validate_teacher(teacher,3);row['teacher']=teacher
        with self.assertRaisesRegex(ValueError,'Historical'):distill_update(model,opt,[g],UPDATE_V21)
        teacher=visit_teacher(row['choices'],stats,[1/3]*3,model='a'*64,observer=0,config=config())
        validate_teacher(teacher,3);row['teacher']=teacher
        self.assertEqual(teacher['source'],'ismcts-root-visits/1.2.0')
        self.assertFalse(distill_update(model,opt,[g],UPDATE_V21)['updated'])

    def test_fresh_distillation_updates_policy_with_utility_calibrated_and_versioned(self):
        model,opt=self.model();data=prepared([game()],insolvency=INSOLVENCY,reward=SPEC)
        calibrate(model,opt,Dataset(data,'cpu'),dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2))
        identity=model_id(model);g=game();g['split']='train'
        for row,(era,round_) in zip(g['records'],[('CanalEra',1),('CanalEra',5),('RailEra',1),('RailEra',7)]):
            row['view']['context'].update(era=era,round=round_)
            stats=dict(root={c['id']:dict(visits=2) for c in row['choices']},expansion=[True]*3)
            row['teacher']=visit_teacher(row['choices'],stats,[.8,.1,.1],model=identity,observer=row['seat'],config=config())
        official={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('outcomes.','utility_head.'))}
        cfg=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2)
        quality=dict(min_samples=4,min_games=1,min_samples_per_phase=1,batch_size=4,max_reuse=1)
        outcome=distill_update(model,opt,[g],cfg,steps=1,quality=quality)
        self.assertEqual(outcome['optimizer_steps'],1);self.assertFalse(outcome['value_loss_training'])
        for k,v in official.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'fresh current-model'):distill_update(model,opt,[g],cfg,steps=1,quality=quality)

    def test_reviewed_resume_migrates_after_restoring_optimizer_and_preserves_schedule(self):
        import test_reviewed_resume as reviewed
        from brass_ai.recovery.resume import load_resume,commit_transition
        from brass_ai.step5.league import implementation_id
        reviewed.ReviewedResumeTests.setUpClass();fixture=reviewed.ReviewedResumeTests()
        try:
            run,state,_,original,oldopt=fixture.fixture()
            cfg=read(SOURCE_ROOT/'configs/step5/lab.json');cfg.update(model_size='small',rehearsal_manifest=state['signature']['imitation_source']['manifest'])
            signature=dict(state['signature'],config=cfg,implementation=implementation_id())
            model,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
            self.assertTrue(changed);opt=torch.optim.AdamW(model.parameters(),lr=1e-5);resume_optimizer(raw,opt)
            migrate(model,opt,reward_spec(cfg));next_state=commit_transition(run,state,signature)
            self.assertEqual(next_state['iteration'],state['iteration']);self.assertEqual(next_state['games_done'],state['games_done'])
            for k,v in original.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
            for k,fields in oldopt.state_dict()['state'].items():
                for n,v in fields.items():torch.testing.assert_close(v,opt.state_dict()['state'][k][n],rtol=0,atol=0)
            # Restart before the first new commit still loads the exact original.
            again,_,changed=load_resume(SOURCE_ROOT,run,next_state,signature)
            self.assertFalse(changed);self.assertIsNone(getattr(again.config,'utility',None))
        finally:fixture.doCleanups()


class SearchTests(unittest.TestCase):
    def test_linear_search_values_above_one_and_below_zero_are_not_clipped(self):
        cfg=config(exploration=1e-9)
        node=Node(visits=4,returns=np.array([12.,0,0,0]),bootstrap=np.array([3.,0,0,0]),
                  edges={'a':Edge(visits=4,returns=np.array([8.,0,0,0]))})
        self.assertEqual(select_edge(node,['a','b'],[.5,.5],[0,1],0,cfg),1)
        node.returns[0]=-8.;node.edges['a'].returns[0]=-4.
        self.assertEqual(select_edge(node,['a','b'],[.5,.5],[0,1],0,cfg),0)

    def test_search_node_local_backup_matches_terminal_and_leaf_reward(self):
        # Root action triggers one debt event; two forced liquidation choices
        # belong to that same event. A later leaf owes no part of that past cost.
        for terminal in (False,True):
            views=[];choices=[]
            for i in range(4):
                v,c=position(0);v['context'].update(round=1 if i<3 else 2,revision=i+1,
                    phase='action' if i in (0,3) else 'debt',terminal=terminal and i==3)
                if i in (1,2):v['players'][0]['debt']=3
                views.append(v);choices.append(c if i in (0,3) else [dict(id='debt-'+str(i),kind='liquidate')])
            final=scored([(180,0,0),(80,0,0),(70,0,0),(60,0,0)])
            leaf=terminal_utility(final,SPEC)
            class State:
                def __init__(self):self.i=0;self.recall=[[] for _ in range(4)];self.final_result=final;self.sync()
                def sync(self):self.context=views[self.i]['context'];self.views=[views[self.i]]*4;self.legal=choices[self.i]
                def current_player(self):return -2 if self.is_terminal() else 0
                def is_terminal(self):return terminal and self.i==3
                def is_chance_node(self):return False
                def apply_action(self,_):self.i+=1;self.sync()
            host=SimpleNamespace(from_observation=lambda *_:State())
            policy=SimpleNamespace(contexts=weakref.WeakKeyDictionary(),utility_remaining=weakref.WeakKeyDictionary(),
                context_features=lambda *_:None,evaluator=SimpleNamespace(model_id='a'*64),policy_id='test')
            cfg=config();bot=MaskedISMCTS(host,policy,[policy],cfg,0,None,random.Random(2),time.monotonic()+30,
                reward_context=dict(spent=[.09,0,0,0],settled=[]))
            g=bot.trajectory(views[0],choices[0])
            try:
                request=next(g)
                while True:
                    count=len(request[3]);response=dict(utility=leaf.tolist(),reward_id=digest(SPEC))
                    request=g.send(([1/count]*count,response))
            except StopIteration:pass
            self.assertEqual(bot.completed,1)
            root=bot.tree[bot.root_key];self.assertAlmostEqual(root.returns[0],leaf[0]-.01)
            nonroot=[n for k,n in bot.tree.items() if k!=bot.root_key and n.visits]
            self.assertEqual(len(nonroot),2)
            for n in nonroot:self.assertAlmostEqual(n.returns[0],leaf[0])
            self.assertTrue(all(e.pending==0 for n in bot.tree.values() for e in n.edges.values()))

    def test_active_presets_preserve_other_settings_and_ninety_percent_budget(self):
        from brass_ai.recovery.teacher_schedule import search_allowance
        from brass_ai.recovery.config import SEARCH_WALL_FRACTION
        self.assertEqual(SEARCH_WALL_FRACTION,.9);self.assertAlmostEqual(search_allowance(100,0),900.)
        for name in ('lab','lab-16x2','lab-24x4','lab-reference','lab-small','smoke','recovery-smoke'):
            cfg=validate_config(read(SOURCE_ROOT/f'configs/step5/{name}.json'))
            self.assertEqual(reward_spec(cfg),cfg['search']['reward']);self.assertEqual(cfg['search_wall_fraction'],.9)
            if name.startswith('lab'):
                self.assertEqual((cfg['search_every_games'],cfg['search_games'],cfg['distill_steps']),(160,4,8))
                self.assertEqual((cfg['search']['seconds'],cfg['execution']['search_workers_per_turn']),(12.,4))
                self.assertEqual(cfg['learning_rate'],1e-5)


if __name__=='__main__':unittest.main()
