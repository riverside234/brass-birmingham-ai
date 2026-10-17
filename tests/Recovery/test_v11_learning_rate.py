"""Warmup boundaries, real Adam restart continuity, and recipe isolation."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch

from brass_ai.learning.contracts import SOURCE_ROOT,read_json
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.recovery.learning_rate import configure,rate,step,report,validate_warmup,DEFAULT_WARMUP,KEY
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.resume import apply_optimizer_config


class V11LearningRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_linear_boundaries_disabled_single_and_invalid(self):
        state=dict(base_lr=6e-6,steps=100,start_factor=.1)
        self.assertAlmostEqual(rate(state,1),6e-7)
        self.assertAlmostEqual(rate(state,100),6e-6)
        self.assertAlmostEqual(rate(state,101),6e-6)
        self.assertAlmostEqual(rate(dict(state,steps=0),1),6e-6)
        self.assertAlmostEqual(rate(dict(state,steps=1),1),6e-6)
        for key,value in [('imitation_steps',True),('selfplay_steps',-1),('start_factor',0),('start_factor',float('nan'))]:
            with self.assertRaises(ValueError):validate_warmup(dict(DEFAULT_WARMUP,**{key:value}))

    def test_checkpoint_restores_schedule_adam_rng_and_exact_next_update(self):
        torch.manual_seed(112);model=FourSeatNet(ModelConfig.preset('small'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=6e-6,weight_decay=1e-4)
        configure(optimizer,6e-6,100,.1,'selfplay')
        from brass_ai.recovery.lr_controller import configure as configure_controller,DEFAULT_CONTROLLER
        configure_controller(optimizer,DEFAULT_CONTROLLER)
        def update(net,opt):
            opt.zero_grad(set_to_none=True)
            sum(p.square().mean() for p in net.parameters()).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            step(opt)
        for _ in range(3):update(model,optimizer)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
            path=Path(folder)/'warmup.pt';save_checkpoint(path,model,optimizer,step=2,training=dict(ordinary_games=640))
            loaded,raw=load_checkpoint(path);restored=torch.optim.AdamW(loaded.parameters(),lr=1e-3)
            resume_optimizer(raw,restored)
            config=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json')
            apply_optimizer_config(restored,config)
            self.assertEqual(report(restored),report(optimizer))
            self.assertEqual(report(restored)['completed_steps'],3)
            self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
            update(model,optimizer);update(loaded,restored)
            for a,b in zip(model.parameters(),loaded.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(report(restored)['completed_steps'],4)
            with self.assertRaisesRegex(ValueError,'checkpoint/config'):configure(restored,6e-6,101,.1,'selfplay')
            restored.param_groups[0].pop(KEY)
            with self.assertRaisesRegex(ValueError,'Missing warmup state'):configure(restored,6e-6,100,.1,'selfplay')

    def test_new_config_scales_once_and_preserves_other_recipe_fields(self):
        current=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json');validate_config(current)
        self.assertEqual(current['learning_rate'],6e-6)
        self.assertEqual(current['imitation']['learning_rate'],6e-5)
        self.assertEqual(current['update']['grad_clip'],1.)
        self.assertEqual(latest_model_config(current,current['imitation_checkpoint']),current)
        old=copy.deepcopy(current);old['architecture']='brass-four-seat-mlp-resnet/10.0.0';old['parameter_limit']=3000000
        old['learning_rate']=1e-5;old['imitation']['learning_rate']=1e-4;old['update']['grad_clip']=5.
        old.pop('learning_rate_warmup')
        generated=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(generated,current)
        self.assertEqual(old['learning_rate'],1e-5)


if __name__=='__main__':unittest.main()
