"""Strict half-target decision boundary and continuation with existing Adam evidence."""
import copy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.recovery import lr_controller as lr
from brass_ai.recovery.learning_rate import configure as warmup,KEY as WARMUP_KEY
from brass_ai.recovery.resume import apply_optimizer_config,check_signature
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.local_scoring_auxiliary import Companion,KEY as AUX,DEFAULT
from brass_ai.recovery.local_scoring_adoption import configure as adoption
from brass_ai.step5.league import write_json,implementation_id
from test_ppo_lr_controller import ready,measurements
from test_gae_value_phase import fixture

PREVIOUS='cb165c0d852ca79dfd3df2c40483dc7e06bf6dd988d94755ecd1cb597cc3a6af'
EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0560-20261022T202410715648Z'


class Threshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_strict_half_target_and_legacy_config(self):
        for ratio,kl,expected in ((.45,.00475,'hold'),(.50,.00475,'up'),(.50,.004999,'up'),(.50,.005,'hold'),(.50,.005001,'hold')):
            state=ready();state['config']['low_ratio']=ratio
            for _ in range(3):state,report=lr.decide(state,measurements(kl=kl))
            self.assertEqual(report['decision'],expected)
            self.assertEqual(report['raise_below_kl'],ratio*.01)
        from brass_ai.recovery.config import latest_model_config
        cfg=read(EXPORT/'recorded-config.json');fresh=latest_model_config(cfg,cfg['imitation_checkpoint'])
        self.assertEqual(fresh['ppo_lr_controller']['low_ratio'],.5)
        self.assertEqual(cfg['ppo_lr_controller']['low_ratio'],.45)
        sampling=read(EXPORT/'iteration-0559/report.json')['sampling']
        self.assertEqual(lr.collection_signature(cfg,sampling),lr.collection_signature(lr.threshold_config(cfg),sampling))

    def test_real_checkpoint_keeps_streak_moments_rng_and_companion(self):
        model,_,cfg,(_,data)=fixture(games=2)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        warmup(opt,cfg['learning_rate'],100,.1,'selfplay')
        old=dict(lr.DEFAULT_CONTROLLER,low_ratio=.45);lr.configure(opt,old)
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step();opt.zero_grad(set_to_none=True)
        control=ready();control.update(config=old,multiplier=1.25,cooldown=1,data_signature='a'*64,
            window=[dict(kl=.00475,early_stop=False)]*2)
        opt.param_groups[0][lr.KEY]=control
        opt.param_groups[0][WARMUP_KEY]['completed_steps']=1000
        heads=Companion(model,copy.deepcopy(DEFAULT));heads.generation=580;heads.age=580
        # Nonempty companion moments are part of the exact restore assertion.
        for phase in ('ppo','ppg'):
            for p in heads.heads.parameters():p.grad=torch.ones_like(p)
            heads.optimizers[phase].step();heads.optimizers[phase].zero_grad(set_to_none=True)
        training={AUX:heads.payload(580),'preservation_fixture':dict(ppg_origin=560,next_phase=584,gae=.95,games_done=185600)}
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            file=Path(tmp)/'generation-0580.pt';save_checkpoint(file,model,opt,step=580,training=training);before=sha(file)
            net,raw=load_checkpoint(file);saved_training=copy.deepcopy(raw['training'])
            restored=torch.optim.AdamW(net.parameters(),lr=1.)
            resume_optimizer(raw,restored);adam=copy.deepcopy(restored.state_dict());rng=torch.get_rng_state().clone()
            py=random.getstate();np_rng=np.random.get_state()
            new=lr.threshold_config(dict(cfg,ppo_lr_controller=old))
            apply_optimizer_config(restored,new)
            expected=copy.deepcopy(adam);expected['param_groups'][0][lr.KEY]['config']['low_ratio']=.5
            # Standard resume restores the configured base in `lr`; warmup and
            # the persisted multiplier determine each subsequent PPO step.
            expected['param_groups'][0]['lr']=cfg['learning_rate']
            self.assertTrue(_same_payload(expected,restored.state_dict()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(py,random.getstate())
            self.assertTrue(np.array_equal(np_rng[1],np.random.get_state()[1]))
            companion=Companion(net,copy.deepcopy(DEFAULT),raw=raw,generation=580)
            self.assertTrue(_same_payload(training[AUX],companion.payload(580)))
            self.assertTrue(_same_payload(saved_training,raw['training']));self.assertEqual(sha(file),before)
            s=restored.param_groups[0][lr.KEY]
            s,r=lr.decide(s,measurements(kl=.00475));self.assertEqual(r['decision'],'cooldown')
            s,r=lr.decide(s,measurements(kl=.00475));self.assertEqual(r['decision'],'up')
            self.assertAlmostEqual(s['multiplier'],1.25**2)

    def test_reviewed_g560_helper_narrow_changes_and_idempotence(self):
        state=read(EXPORT/'state.json');cfg=adoption(state);cfg['ppo_lr_controller']['low_ratio']=.45
        state.update(iteration=560,signature=dict(state['signature'],config=cfg,implementation=PREVIOUS))
        signature=dict(state['signature'],config=lr.threshold_config(cfg),implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaisesRegex(ValueError,'560'):check_signature(dict(state,iteration=559),signature,reviewed=True)
        with self.assertRaises(ValueError):check_signature(state,signature)
        bad=copy.deepcopy(signature);bad['config']['exploration']=.02
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        spec=importlib.util.spec_from_file_location('lr_threshold_helper',SOURCE_ROOT/'scripts/configure-ppo-lr-threshold.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);checkpoint=run/'generation-0560.pt';checkpoint.write_bytes(b'authored checksum fixture; not a trained model')
            state['history'][-1]=dict(generation=560,path=checkpoint.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(checkpoint),model_id='a'*64)
            write_json(run/'state.json',state);before=sha(run/'state.json')
            output=SOURCE_ROOT/'configs/step5'/('lr-half-'+run.name+'.json');self.addCleanup(lambda:output.unlink(missing_ok=True))
            for _ in range(2):
                r=helper.generate(SOURCE_ROOT,run,output,560)
                self.assertFalse(r['run_modified']);self.assertEqual(r['first_collection_generation'],561)
                self.assertEqual(read(output),signature['config']);self.assertEqual(sha(run/'state.json'),before)
            with self.assertRaisesRegex(ValueError,'Stale'):helper.generate(SOURCE_ROOT,run,output,561)
            after=dict(state,signature=signature)
            self.assertFalse(check_signature(after,signature,reviewed=True))
            # Existing local-scoring mode switches remain independently reviewed.
            off=copy.deepcopy(signature);off['config'][AUX]['mode']='off'
            self.assertTrue(check_signature(after,off,reviewed=True))


if __name__=='__main__':unittest.main()
