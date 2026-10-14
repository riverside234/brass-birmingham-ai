"""Iteration-60 review: resume settings and diagnostic correctness."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from fixtures import game
from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import resume_optimizer, save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery.config import UPDATE_V21, validate_config
from brass_ai.recovery.diagnostics import summarize, merge_summaries
from brass_ai.recovery.history import compact
from brass_ai.recovery.resume import apply_optimizer_config, check_signature, load_resume, commit_transition
from brass_ai.recovery.review_metrics import CategoricalProbe, reward_distribution
from brass_ai.recovery.updates import Dataset, prepared
from brass_ai.step5.league import implementation_id, write_json


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reviewed_lr_applies_after_restore_without_resetting_moments_or_rng(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        opt=torch.optim.AdamW([p],lr=1e-5,weight_decay=1e-4)
        p.square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
        raw=dict(optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state().clone(),cuda_rng=[])
        restored=torch.optim.AdamW([p],lr=5e-6)
        resume_optimizer(raw,restored)
        self.assertEqual(restored.param_groups[0]['lr'],1e-5)  # The actual regression.
        receipt=apply_optimizer_config(restored,dict(learning_rate=5e-6,weight_decay=1e-4))
        self.assertEqual(receipt['after'],[dict(lr=5e-6,weight_decay=1e-4)])
        self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
        for name,value in raw['optimizer']['state'][0].items():
            torch.testing.assert_close(value,restored.state_dict()['state'][0][name],rtol=0,atol=0)
        p.square().sum().backward();restored.step()
        self.assertEqual(restored.state_dict()['state'][0]['step'].item(),2)

    def test_real_retention_keeps_correct_ppo_field_names_and_work_evidence(self):
        source=read(SOURCE_ROOT/'results/selfplay-joint/iter59.json')
        before=copy.deepcopy(source);out=compact(source)
        self.assertEqual(source,before)
        self.assertEqual(out['summary_version'],4)
        self.assertEqual(out['actor']['learner_decisions'],28645)
        self.assertEqual(out['actor']['learning_rates'],[1e-5])
        self.assertEqual(out['actor']['last_epoch']['actor'],source['actor']['epochs'][-1]['actor'])
        self.assertEqual(out['search']['phases'],source['search']['teaching']['phases'])
        self.assertEqual(out['search']['fit_before'],source['search']['fit_before'])
        self.assertEqual(out['utilization']['inference_positions'],27616)
        self.assertLess(len(json.dumps(out)),40000)
        self.assertEqual(compact(out),out)
        longer=copy.deepcopy(source)
        longer['actor']['epochs']=[dict(source['actor']['epochs'][-1],epoch=i) for i in range(12)]
        bounded=compact(longer)['actor']
        self.assertEqual(len(bounded['epochs']),8)
        self.assertEqual(bounded['last_epoch']['epoch'],11)
        old=read(SOURCE_ROOT/'results/selfplay-joint/retention-history.json')['iteration-0000/report.json']['summary']
        self.assertEqual(compact(old),old)
        self.assertNotIn('learning_rates',compact(old)['actor'])

    def test_categorical_kl_is_full_distribution_on_fixed_probe_and_rng_neutral(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.logits=torch.nn.Parameter(torch.log(torch.tensor([.2,.3,.5])))
            def forward(self,states,actions,legal,mapping):
                return (self.logits.expand(len(states),-1).masked_fill(~legal,-torch.inf),)
        dataset=Dataset(prepared([game()]),'cpu')
        model=Model();rng=torch.get_rng_state().clone()
        probe=CategoricalProbe(model,dataset,[0,2],SimpleNamespace(exploration=0.,temperature=1.),UPDATE_V21)
        self.assertAlmostEqual(probe.measure(model)['categorical_kl'],0.,places=6)
        with torch.no_grad():model.logits.copy_(torch.log(torch.tensor([.4,.2,.4])))
        report=probe.measure(model)
        expected=sum(p*np.log(p/q) for p,q in zip([.2,.3,.5],[.4,.2,.4]))
        self.assertAlmostEqual(report['categorical_kl'],expected,places=6)
        self.assertEqual(report['positions'],2)
        self.assertTrue(0<report['entropy_over_log_legal']<=1)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(CategoricalProbe(model,dataset,[],SimpleNamespace(exploration=0.,temperature=1.),UPDATE_V21).measure(model)['positions'],0)

    def test_reward_variability_is_per_trajectory_and_preserves_covariance(self):
        spec=dict(win=1.,placement=.1,vp=.001,insolvency=dict(event_cost=.005,max_cost=.025))
        trajectories=[dict(win=1.,placement=1.,vp=100,cost=0.,utility=1.2),
                      dict(win=0.,placement=0.,vp=50,cost=.025,utility=.025)]
        result=reward_distribution(trajectories,spec)
        self.assertEqual(result['capped_trajectories'],1)
        self.assertAlmostEqual(result['weighted']['win']['std'],.5)
        self.assertAlmostEqual(result['weighted']['cost']['mean'],-.0125)
        self.assertAlmostEqual(sum(sum(row.values()) for row in result['covariance'].values()),result['utility']['std']**2)

    def test_seat_and_gameplay_diagnostics_merge_without_opponent_leakage(self):
        first=game();first['records'][0]['chosen']=1
        second=copy.deepcopy(first);second['game_id']='second'
        for g in (first,second):g['opponents']=['learner','other','other','other']
        full=summarize([first,second],{'learner'})
        merged=merge_summaries([summarize([first],{'learner'}),summarize([second],{'learner'})])
        self.assertEqual(full,merged)
        self.assertEqual(full['economy']['passes_with_nonpass_alternative'],2)
        self.assertEqual(full['learner_by_seat']['0']['seat_games'],2)
        self.assertEqual(full['learner_by_seat']['0']['win_share_sum'],1.)
        self.assertEqual(full['learner_by_seat']['1'],{})

    def test_joint7_pinned_continuation_and_lr_arm_require_review(self):
        state=read(SOURCE_ROOT/'results/selfplay-joint/state.json')
        for name in ('joint7-control','joint7-review','joint7-lr-half'):
            config=read(SOURCE_ROOT/f'configs/step5/{name}.json');validate_config(config)
            signature=dict(state['signature'],implementation=implementation_id(),config=config)
            self.assertTrue(check_signature(state,signature,True))
            with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
            bad=copy.deepcopy(signature);bad['config']['reward']['vp']=.002
            with self.assertRaisesRegex(ValueError,'unchanged'):check_signature(state,bad,True)
            expected=copy.deepcopy(state['signature']['config']['search']['reward'])
            self.assertEqual(expected['insolvency'],dict(event_cost=.005,max_cost=.025))
            expected['insolvency']['max_cost']=.075
            self.assertEqual(config['search']['reward'],expected)

    def test_generation60_fixture_restores_adam_and_counters_then_repeats_resume(self):
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet, utility_config
        state=read(SOURCE_ROOT/'results/selfplay-joint/state.json')
        state['signature']['config']['model_size']='small'
        spec=state['signature']['config']['search']['reward']
        model=UtilityNet(utility_config(ModelConfig.preset('small'),spec))
        model.utility_calibration=dict(ready=True,reward_id=digest(spec))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
        sum(p.square().mean() for p in model.parameters()).backward();opt.step()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp).resolve();path=run/'generation-0060.pt'
            identity=save_checkpoint(path,model,opt,step=60,training=dict(fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),
                mode='step5-population-ppo',signature=digest(state['signature'])))
            entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=60)
            state.update(history=[entry],population=[entry],resume=[entry],evaluation_anchor=entry,champion=entry)
            write_json(run/'state.json',state);original=path.read_bytes()
            config=read(SOURCE_ROOT/'configs/step5/joint7-lr-half.json');config['model_size']='small'
            signature=dict(state['signature'],config=config,implementation=implementation_id())
            restored,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
            self.assertTrue(changed);self.assertFalse(restored.utility_calibration['ready'])
            for key,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            optimizer=torch.optim.AdamW(restored.parameters(),lr=5e-6);resume_optimizer(raw,optimizer)
            apply_optimizer_config(optimizer,config)
            for key,values in opt.state_dict()['state'].items():
                for name,value in values.items():torch.testing.assert_close(value,optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
            updated=commit_transition(run,state,signature)
            for key in ('iteration','games_done','next_seed','context_counter','next_search_games','population','champion','origin','imitation'):
                self.assertEqual(updated[key],state[key])
            self.assertEqual(path.read_bytes(),original)
            self.assertFalse(load_resume(SOURCE_ROOT,run,updated,signature)[2])


if __name__=='__main__':unittest.main()
