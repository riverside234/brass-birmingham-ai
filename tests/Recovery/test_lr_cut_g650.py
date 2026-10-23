"""Smaller PPO feedback cuts, with unchanged triggers and exact Adam adoption."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.recovery import lr_controller as lr
from brass_ai.recovery.config import latest_model_config
from brass_ai.recovery.learning_rate import configure as warmup,KEY as WARMUP_KEY
from brass_ai.recovery.resume import apply_optimizer_config
from brass_ai.recovery.retention import _same_payload
from test_ppo_lr_controller import ready,measurements
from test_gae_value_phase import fixture

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0650-20261023T202940235880Z'


class SmallerCut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_narrow_change_fresh_defaults_and_explicit_legacy_divisor(self):
        old=read(EXPORT/'state.json')['signature']['config'];new=lr.cut_config(old)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'ppo_lr_controller'})
        self.assertEqual(old['ppo_lr_controller']['cut_divisor'],1.5)
        self.assertEqual(new['ppo_lr_controller'],dict(old['ppo_lr_controller'],cut_divisor=1.25))
        self.assertEqual(new,lr.cut_config(new))
        self.assertEqual(latest_model_config(old,old['imitation_checkpoint'])['ppo_lr_controller']['cut_divisor'],1.25)
        with self.assertRaises(ValueError):
            lr.cut_config(dict(old,ppo_lr_controller=dict(old['ppo_lr_controller'],cut_divisor=2.)))
        for divisor,expected in ((1.5,2/3),(1.25,.8)):
            for sequence in ([measurements(early_stop=True,steps_done=4)],
                [measurements(kl=.015001)],
                [measurements(early_stop=True),measurements(),measurements(early_stop=True)]):
                s=ready();s['config']['cut_divisor']=divisor
                for m in sequence:s,r=lr.decide(s,m)
                self.assertEqual(r['decision'],'down');self.assertAlmostEqual(s['multiplier'],expected)
                self.assertEqual((s['cooldown'],s['window']),(3,[]));self.assertEqual(r['cut_divisor'],divisor)
        # The change does not turn a single late stop into a new cut trigger.
        s,r=lr.decide(ready(),measurements(early_stop=True,steps_done=5,kl=.015))
        self.assertEqual(r['decision'],'hold');self.assertEqual(s['multiplier'],1.)
        s=ready();s['multiplier']=.6
        s,r=lr.decide(s,measurements(kl=.02));self.assertEqual(s['multiplier'],.5)

    def test_real_checkpoint_adopts_divisor_without_resetting_any_evidence(self):
        model,_,cfg,_=fixture(games=2)
        cfg=dict(cfg,ppo_lr_controller=dict(lr.DEFAULT_CONTROLLER,cut_divisor=1.5))
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        warmup(opt,cfg['learning_rate'],100,.1,'selfplay');lr.configure(opt,cfg['ppo_lr_controller'])
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step();opt.zero_grad(set_to_none=True)
        control=ready();control.update(config=copy.deepcopy(cfg['ppo_lr_controller']),multiplier=1.25,cooldown=2,
            data_signature='a'*64,window=[dict(kl=.00475,early_stop=False)]*2)
        opt.param_groups[0][lr.KEY]=control;opt.param_groups[0][WARMUP_KEY]['completed_steps']=1000
        training=dict(untouched=dict(generation=650,games_done=208000,ppg_next=658,beta=1.,entropy_baseline=[1.,2.,3.,4.]))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
            path=Path(folder)/'generation-0650.pt';save_checkpoint(path,model,opt,step=650,training=training)
            checkpoint_hash=sha(path);net,raw=load_checkpoint(path)
            restored=torch.optim.AdamW(net.parameters(),lr=1.);resume_optimizer(raw,restored)
            before=copy.deepcopy(restored.state_dict());weights=copy.deepcopy(net.state_dict())
            saved_training=copy.deepcopy(raw['training'])
            rng=torch.get_rng_state().clone();py=random.getstate();np_rng=np.random.get_state()
            apply_optimizer_config(restored,lr.cut_config(cfg))
            expected=copy.deepcopy(before);expected['param_groups'][0][lr.KEY]['config']['cut_divisor']=1.25
            expected['param_groups'][0]['lr']=cfg['learning_rate']
            self.assertTrue(_same_payload(expected,restored.state_dict()))
            self.assertTrue(_same_payload(weights,net.state_dict()));self.assertEqual(raw['training'],saved_training)
            self.assertEqual(sha(path),checkpoint_hash);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(np_rng[1],np.random.get_state()[1]))
            s,r=lr.decide(restored.param_groups[0][lr.KEY],measurements(kl=.02))
            self.assertEqual(r['decision'],'down');self.assertEqual(s['multiplier'],1.)


if __name__=='__main__':unittest.main()
