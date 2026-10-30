"""Explicit 0.10 group entropy with unchanged DOMiNO sampling and value training."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import exploration as E,updates
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.domino_auxiliary import Diversity
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec
from test_domino import fixture,collect
from test_exploration_revision import measured

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z'


class Entropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_migration_fresh_defaults_and_strict_versions(self):
        old=read(EXPORT/'recorded-config.json');before=copy.deepcopy(old)
        new=validate_config(E.fixed_010_config(old));self.assertEqual(old,before)
        self.assertEqual(E.fixed_010_config(new),new)
        self.assertEqual(new,dict(old,update=dict(old['update'],fixed_entropy=E.FIXED_ENTROPY_010)))
        self.assertEqual(validate_config(old),old)
        for source in (old,new):
            fresh=latest_model_config(source,source['imitation_checkpoint'])
            self.assertEqual(fresh['update']['fixed_entropy'],E.FIXED_ENTROPY_010)
        for key,value in [('canal_action',.099),('canal_other',float('nan')),('rail_action',True),('rail_other',.005)]:
            bad=copy.deepcopy(new);bad['update']['fixed_entropy']['weights'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(new);bad['entropy_control']['mode']='apply'
        with self.assertRaises(ValueError):validate_config(bad)

    def test_raw_entropy_gradients_reports_and_saved_baseline_preserved(self):
        state=read(EXPORT/'state.json');c=E.fixed_010_config(state['signature']['config'])
        restored=E.restore(c,state,state['exploration']);before=copy.deepcopy(restored)
        settings=E.effective(c,restored,1326)
        self.assertEqual(settings['alphas'],dict.fromkeys(E.GROUPS,.10))
        advanced,report=E.advance(restored,c,measured(.01),generation=1327,warmup_done=True,healthy=True)
        self.assertEqual(restored,before);self.assertEqual(advanced['baseline'],before['baseline'])
        for g in E.GROUPS:
            self.assertEqual(report['groups'][g]['effective_alpha'],.10)
            self.assertEqual(advanced['groups'][g]['alpha'],before['groups'][g]['alpha'])
        contexts=[dict(era=era,phase=phase) for era in ('CanalEra','RailEra') for phase in ('action','debt')]
        ds=SimpleNamespace(rows=[dict(view=dict(context=x)) for x in contexts])
        behavior=SimpleNamespace(raw_entropy=True,entropy_alphas=settings['alphas'])
        logits=torch.tensor([[2.,0.,float('-inf')]]*4,requires_grad=True)
        raw,loss=updates.entropy_term(logits,torch.ones_like(logits)/3,ds,list(range(4)),behavior,c['update'])
        torch.testing.assert_close(loss,.10*raw);loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertEqual(float(logits.grad[:,2].abs().sum()),0.)
        for i in range(1,4):torch.testing.assert_close(logits.grad[0],logits.grad[i])
        behavior.entropy_alphas=E.FIXED_ENTROPY['weights']
        with self.assertRaisesRegex(ValueError,'matching raw-policy'):updates.entropy_term(logits,logits,ds,list(range(4)),behavior,c['update'])

    def test_isolated_ppo_ppg_smoke_with_new_entropy(self):
        model,opt,c=fixture();c=E.fixed_010_config(c)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(model,c,root=SOURCE_ROOT,run=Path(tmp))
            p.diversity=Diversity(model,c['diversity'],regime='entropy-smoke')
            b,rows=collect(model,p,1,games=42);reward=reward_spec(c)
            report=updates.ppo_update(model,opt,[dict(split='train')],b,c['update'],learner_data=rows,
                insolvency=reward['insolvency'],reward=reward,phases=p)
            self.assertEqual(report['optimizer_steps'],1)
            before=copy.deepcopy(opt.state_dict());value=p.fit(model)
            self.assertTrue(_same_payload(before,opt.state_dict()));self.assertLessEqual(value['probe_kl'],.002+1e-7)
            self.assertEqual(b.entropy_alphas,dict.fromkeys(E.GROUPS,.10))


if __name__=='__main__':unittest.main()
