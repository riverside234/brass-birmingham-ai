"""VP-180 utility, unchanged official scores, and explicit cache-safe resume."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,digest,compatibility
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.replay import terminal_targets
from brass_ai.learning.reward import specification,terminal_components,terminal_utility,replay_returns,validate,CAPPED_VERSION
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery import ppg_config as P,ppg_epochs,gae,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY,ppo_evidence
from brass_ai.recovery.config import vp_cap_config,validate_config,latest_model_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import load_resume,commit_transition,check_signature
from brass_ai.recovery.utility import reward_spec,migrate,vp_cap_upgrade,event_cost_upgrade
from brass_ai.recovery.updates import prepared
from brass_ai.step5.league import write_json,implementation_id
import test_combined_reward as combined
import test_insolvency_recovery as debt
from test_ppg_eight import setup,stage_only
from fixtures import game


EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0540-20261022T144917069131Z'
BASE=SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-one-se-g540.json'
module=importlib.util.spec_from_file_location('vp_cap_helper',SOURCE_ROOT/'scripts/joint7-vp-cap.py')
helper=importlib.util.module_from_spec(module);module.loader.exec_module(helper)


class Reward(unittest.TestCase):
    def test_boundary_and_real_winner_above_cap(self):
        c=reward_spec(vp_cap_config(read(BASE)));old=reward_spec(read(BASE))
        for score in (0,179,180,181,250,100000):
            result=combined.scored([(score,0,0)]*4)
            self.assertAlmostEqual(terminal_utility(result,c)[0],.25+.05+.001*min(score,180))
            self.assertEqual(terminal_components(result)['vp'],[score]*4)
            if score<=180:np.testing.assert_array_equal(terminal_utility(result,c),terminal_utility(result,old))
        result=combined.scored([(250,0,0),(240,0,0),(181,0,0),(180,0,0)])
        official=terminal_targets(result)
        self.assertEqual(official['vp'],[250,240,181,180])
        self.assertEqual(official['win_share'],[1.,0.,0.,0.])
        self.assertEqual(official['placement'],[1,2,3,4])
        np.testing.assert_allclose(terminal_utility(result,c),[1.28,.1*2/3+.18,.1/3+.18,.18])

    def test_legacy_v1_is_still_linear_and_cap_is_strictly_validated(self):
        old=reward_spec(read(BASE));c=reward_spec(vp_cap_config(read(BASE)))
        self.assertNotEqual(digest(old),digest(c));self.assertEqual(c['version'],CAPPED_VERSION)
        for score in (40,200,100000):
            a=combined.scored([(score,0,0)]*4);b=combined.scored([(score+7,0,0)]*4)
            np.testing.assert_allclose(terminal_utility(b,old)-terminal_utility(a,old),[.007]*4)
        for cap in (None,True,0,-1,179,181,float('nan'),float('inf'),'180'):
            with self.subTest(cap=cap),self.assertRaises(ValueError):validate(dict(c,vp_cap=cap))
        with self.assertRaises(ValueError):validate(dict(old,vp_cap=180))
        bad=dict(c);del bad['vp_cap']
        with self.assertRaises(ValueError):validate(bad)

    def test_replay_costs_remain_causal_and_reward_reporting_uses_capped_vp(self):
        c=reward_spec(vp_cap_config(read(BASE)))
        records=[r for n in range(1,6) for r in (debt.row(0,round=n),
            debt.row(0,round=n,phase='debt',kind='liquidate',debts={0:5}))]
        result=combined.scored([(220,0,0)]*4)
        labels,summary=replay_returns(records,result,c)
        self.assertEqual(summary['components']['vp'],[180]*4)
        self.assertAlmostEqual(summary['cost'][0],.080)
        self.assertAlmostEqual(labels[0]['utility'][0],.25+.05+.18-.08)
        self.assertAlmostEqual(labels[1]['utility'][0]-labels[0]['utility'][0],.020)
        episode=game();episode['result']=result
        # This authored replay checks return derivation, not map encoding.
        with patch('brass_ai.recovery.updates.PreparedEncoder'):
            data=prepared([episode],insolvency=c['insolvency'],reward=c)
        self.assertTrue(all(d[1]['vp']==[220]*4 for d in data))
        self.assertTrue(all(d[0]['_reward_id']==digest(c) for d in data))
        self.assertTrue(all(d[0]['_reward_trajectory']['vp']==180 for d in data if '_reward_trajectory' in d[0]))


class ModelAndConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_utility_prior_caps_vp_only_and_legacy_forward_is_exact(self):
        old=reward_spec(read(BASE));c=reward_spec(vp_cap_config(read(BASE)))
        a=UtilityNet(utility_config(ModelConfig.preset('small'),old)).eval()
        b=UtilityNet(utility_config(ModelConfig.preset('small'),c)).eval();b.load_state_dict(a.state_dict())
        context=torch.zeros(2,a.config.hidden);mapping=torch.arange(4).repeat(2,1)
        win=torch.tensor([[1.,0.,0.,0.]]*2);rank=torch.tensor([[1.,2.,3.,4.]]*2)
        vp=torch.tensor([[179.,180.,181.,999.],[0.,50.,100.,180.]])
        heads=(win,rank,vp);before=vp.clone()
        raw=(old['win']*win+old['placement']*(4-rank)/3+old['vp']*vp)
        capped=(c['win']*win+c['placement']*(4-rank)/3+c['vp']*vp.clamp_max(180))
        self.assertTrue(torch.equal(a.predict_utility(context,mapping,heads),raw))
        self.assertTrue(torch.equal(b.predict_utility(context,mapping,heads),capped))
        self.assertTrue(torch.equal(vp,before));self.assertNotEqual(model_id(a),model_id(b))
        self.assertEqual(a.parameter_count,b.parameter_count)
        self.assertEqual(digest(compatibility()),'ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54')

    def test_explicit_model_migration_keeps_all_tensors_Adam_and_rng(self):
        old=reward_spec(read(BASE));new=reward_spec(vp_cap_config(read(BASE)))
        model=UtilityNet(utility_config(ModelConfig.preset('small'),old))
        optimizer=torch.optim.AdamW(model.parameters(),lr=6e-6)
        p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,'explicit reward migration'):migrate(model,optimizer,new)
        model._reviewed_cap_upgrade=True;report=migrate(model,optimizer,new)
        self.assertEqual(report['version'],'vp-capped-return/1.0.0')
        self.assertTrue(_same_payload(weights,model.state_dict()));self.assertTrue(_same_payload(adam,optimizer.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertIsNone(model.utility_calibration)
        self.assertEqual(model.config.utility,new);self.assertIsNone(migrate(model,optimizer,new))

    def test_vp_capped_utility_exports_with_parity_at_batches_one_and_two(self):
        import onnxruntime as ort
        class Prior(torch.nn.Module):
            def __init__(self,net):super().__init__();self.net=net
            def forward(self,context,mapping,remaining,win,rank,vp):
                return self.net.predict_utility(context,mapping,(win,rank,vp),remaining)
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(vp_cap_config(read(BASE))))).eval()
        with torch.no_grad():model.utility_head[-1].weight.normal_(std=.01)
        wrapper=Prior(model).eval();names=('context','mapping','remaining','win','rank','vp')
        sample=(torch.randn(2,model.config.hidden),torch.tensor([[0,1,2,3],[3,2,0,1]]),
            torch.full((2,4),.08),torch.full((2,4),.25),torch.full((2,4),2.5),
            torch.tensor([[179.,180.,181.,300.],[250.,180.,50.,0.]]))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'capped-utility.onnx'
            torch.onnx.export(wrapper,sample,str(path),input_names=list(names),output_names=['utility'],
                dynamic_axes={n:{0:'positions'} for n in (*names,'utility')},opset_version=17,dynamo=False)
            options=ort.SessionOptions();options.intra_op_num_threads=1
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            runtime=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            for count in (1,2):
                batch=tuple(t[:count] for t in sample)
                with torch.no_grad():expected=wrapper(*batch).numpy()
                actual=runtime.run(None,{k:v.numpy() for k,v in zip(names,batch)})[0]
                np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)

    def test_config_changes_only_reward_search_and_fresh_default_has_cap(self):
        old=read(BASE);before=copy.deepcopy(old);new=validate_config(vp_cap_config(old))
        self.assertEqual({k for k in old if old[k]!=new[k]},{'reward','search'})
        self.assertEqual(old,before);self.assertEqual(vp_cap_config(new),new)
        self.assertEqual(new['insolvency'],dict(event_cost=.020,max_cost=.080))
        self.assertEqual(new['search']['reward'],reward_spec(new))
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh['reward'],new['reward']);self.assertEqual(fresh['value_phase'],old['value_phase'])
        self.assertEqual(fresh['imitation_checkpoint'],old['imitation_checkpoint'])
        bad=copy.deepcopy(new);bad['search']['reward']['vp_cap']=181
        with self.assertRaises(ValueError):validate_config(bad)
        self.assertTrue(vp_cap_upgrade(reward_spec(old),reward_spec(new)))
        self.assertTrue(vp_cap_upgrade(reward_spec(read(EXPORT/'recorded-config.json')),reward_spec(new)))
        self.assertFalse(event_cost_upgrade(reward_spec(old),reward_spec(new)))
        bad=copy.deepcopy(new);bad['reward']['win']=2
        self.assertFalse(vp_cap_upgrade(reward_spec(old),reward_spec(bad)))

    def test_actual_g540_helper_is_idempotent_preserves_run_and_rejects_unrelated_edit(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';(root/'configs/step5').mkdir(parents=True)
            state=read(EXPORT/'state.json');write_json(run/'state.json',state);checksum=sha(run/'state.json')
            write_json(root/'configs/step5/base.json',read(BASE))
            result=helper.generate(root,run,'configs/step5/new.json',config_path='configs/step5/base.json',expected_generation=540)
            self.assertFalse(result['run_modified']);self.assertEqual(result['first_fresh_ppg_generation'],548)
            self.assertEqual(sha(run/'state.json'),checksum)
            self.assertEqual(read(root/'configs/step5/new.json'),vp_cap_config(read(BASE)))
            helper.generate(root,run,'configs/step5/new.json',config_path='configs/step5/base.json',expected_generation=540)
            bad=read(root/'configs/step5/base.json');bad['learning_rate']*=2
            write_json(root/'configs/step5/bad.json',bad)
            with self.assertRaises(ValueError):helper.generate(root,run,'configs/step5/bad-out.json',config_path='configs/step5/bad.json')


class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_existing_eight_age_cache_is_reset_only_after_adoption_and_once(self):
        self.cache_reset(True)

    def test_legacy_eight_age_recipe_also_excludes_old_reward_targets(self):
        self.cache_reset(False)

    def cache_reset(self,coverage):
        model,opt,old=setup();old=critic_data.fixed_config(old)
        old=P.coverage_options(old) if coverage else P.short_options(old)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(model,p,g);write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            p.state=gae.initial(old['advantage'],8);p.critic_state['generation']=8
            clock=P.initial(0)
            for g in range(1,9):clock=P.advance(clock,g,P.due(clock))
            epoch=dict(ppg_epochs.initial(0,old['value_phase']),cap=4,phases=1,last_phase=8,window_origin=1)
            saved=dict(contract=p.contract,options=p.options,gae=p.state,critic_data=p.critic_state,
                critic_evidence=ppo_evidence(0),beta_clone=8.,kl_trips=[False],optimizer=p.optimizer.state_dict(),
                schedule=clock,epoch_control=epoch,rolling_buffer=p.buffer.receipts,
                **({'cache_origin':0} if coverage else {}))
            raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(saved)})
            new=vp_cap_config(old);weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
            model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(new))
            phase=TrainingPhases(model,new,raw=raw,generation=8,root=SOURCE_ROOT,run=run)
            self.assertEqual(phase.schedule,P.initial(8));self.assertEqual(phase.epoch_control['cap'],4)
            self.assertEqual(phase.state,p.state);self.assertEqual(phase.critic_state,p.critic_state)
            self.assertEqual(phase.beta,8.);self.assertEqual(phase.trips,[])
            self.assertTrue(_same_payload(phase.optimizer.state_dict(),p.optimizer.state_dict()))
            self.assertTrue(_same_payload(model.state_dict(),weights));self.assertTrue(_same_payload(opt.state_dict(),adam))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(phase.buffer.receipts)
            write_json(run/'state.json',dict(iteration=8,signature=dict(config=old)))
            with self.assertRaisesRegex(ValueError,'committed (reward|one-SE)'):phase.startup(8)
            self.assertTrue(p.buffer.path(8,'critic.npy').exists())
            write_json(run/'state.json',dict(iteration=8,signature=dict(config=new)))
            cleanup=phase.startup(8);self.assertGreater(cleanup['deleted_bytes'],0)
            self.assertEqual(cleanup['next_phase'],16)
            stage_only(model,phase,9)
            phase.pending_schedule=P.advance(phase.schedule,9,False);phase.pending_epoch_control=phase.epoch_control
            phase.state=gae.initial(new['advantage'],9);phase.critic_state['generation']=9
            next_raw=dict(model_config=model.config.to_dict(),training={KEY:phase.payload()})
            write_json(run/'state.json',dict(iteration=9,signature=dict(config=new)));phase.buffer.committed(9)
            restored=TrainingPhases(model,new,raw=next_raw,generation=9,root=SOURCE_ROOT,run=run)
            self.assertIsNone(restored.startup(9));self.assertTrue(restored.buffer.path(9,'critic.npy').exists())
            self.assertEqual(restored.cache_origin,8);self.assertEqual(restored.schedule['origin'],8)
            self.assertEqual(restored.buffer.receipts[0]['reward_id'],digest(reward_spec(new)))

    def test_interrupted_composed_g540_adoption_preserves_original_checkpoint(self):
        old=read(EXPORT/'recorded-config.json');old['model_size']='small'
        new=vp_cap_config(read(BASE));new['model_size']='small'
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(old)))
        opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate']);value_adam=torch.optim.Adam(model.parameters(),lr=3e-6)
        for optimizer in (opt,value_adam):
            p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
        options={k:copy.deepcopy(old[k]) for k in ('advantage','value_phase','critic_data','seat_descriptors')}
        saved=dict(contract='actor-value-phases/9.0.0',options=options,gae=gae.initial(old['advantage'],1),
            critic_data=dict(critic_data.initial(old['critic_data']),generation=1),critic_evidence=ppo_evidence(1),
            beta_clone=8.,kl_trips=[],optimizer=value_adam.state_dict(),schedule=P.initial(1),
            epoch_control=dict(ppg_epochs.initial(1,old['value_phase']),cap=1),rolling_buffer=[])
        signature=dict(read(EXPORT/'state.json')['signature'],config=old)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';path=run/'generation-0001.pt'
            identity=save_checkpoint(path,model,opt,step=1,training={KEY:saved,'signature':digest(signature),
                'fresh_run':run.relative_to(root).as_posix()})
            checksum=sha(path);entry=dict(path=path.relative_to(root).as_posix(),sha256=checksum,model_id=identity,generation=1)
            state=dict(format='step5-recovery-league/2.2.0',iteration=1,games_done=320,signature=signature,
                history=[entry],population=[],evaluation_anchor=entry)
            write_json(run/'state.json',state);target=dict(signature,config=new,implementation=implementation_id())
            def restore():
                net,raw,_=load_resume(root,run,state,target,reviewed=True)
                adam=torch.optim.AdamW(net.parameters());resume_optimizer(raw,adam);migrate(net,adam,reward_spec(new))
                phase=TrainingPhases(net,new,raw=raw,generation=1,root=root,run=run)
                return net,adam,phase,torch.get_rng_state().clone()
            a,adam,p,rng=restore();state=commit_transition(run,state,target);p.startup(1)
            b,other,q,rng2=restore();q.startup(1)
            self.assertTrue(_same_payload(a.state_dict(),model.state_dict()))
            self.assertTrue(_same_payload(a.state_dict(),b.state_dict()));self.assertTrue(_same_payload(adam.state_dict(),other.state_dict()))
            self.assertTrue(_same_payload(p.optimizer.state_dict(),q.optimizer.state_dict()));self.assertTrue(torch.equal(rng,rng2))
            self.assertEqual(p.schedule,P.initial(1));self.assertEqual(p.epoch_control['cap'],2)
            self.assertEqual(p.state,q.state);self.assertEqual(sha(path),checksum)
            self.assertEqual((state['iteration'],state['games_done']),(1,320))


if __name__=='__main__':unittest.main()
