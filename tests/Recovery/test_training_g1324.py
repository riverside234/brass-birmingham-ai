"""Versioned 50/50 branch sampling, replay compatibility and continuation."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import branching as B,exploration as E,gae,snapshot_r2
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.training_revision_g1324 import configure
from brass_ai.recovery.training_revision_g1322 import configure as prior_recipe
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import implementation_id,write_json
from brass_ai.step5.policy import NeuralPolicy,exploration_distribution
import test_training_g1322 as prior
import test_training_g1320 as reward_tests
import test_training_g1120 as opening_tests
from test_insolvency_recovery import row
from test_ppg_eight import stage_only

PREVIOUS_SOURCE='8746640733360fd984d0a35a38aa8de6748bf058a9cc53952e86fc54dc1c58b1'


def old_recipe():return prior_recipe(prior.old_recipe())


def smoke_config():
    m,opt,c=prior.smoke_config();c=prior_recipe(c)
    m.config=replace(m.config,utility=reward_spec(c))
    return m,opt,c


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_recipe_fresh_defaults_and_old_configs(self):
        old=old_recipe();original=copy.deepcopy(old);new=validate_config(configure(old))
        self.assertEqual(old,original);self.assertEqual(new,configure(new))
        self.assertEqual({k for k in new if new[k]!=old[k]},{'branching'})
        self.assertEqual(new['branching'],dict(old['branching'],contract='selfplay-branching/6.0.0',
            move_distribution=dict(temperature_2=.5,exploration_q=.5)))
        for c in (old,new):
            self.assertEqual(validate_config(c),c)
            self.assertEqual(latest_model_config(c,c['imitation_checkpoint'])['branching'],E.BALANCED_BRANCH)
        for changes in (dict(move_distribution=E.LEGACY_BRANCH['move_distribution']),dict(rate_per_decision=.015),
            dict(skip_opening_decisions=2)):
            bad=copy.deepcopy(new);bad['branching'].update(changes)
            with self.assertRaises(ValueError):validate_config(bad)

    def test_equal_components_conditional_marginal_and_determinism(self):
        raw=np.array([.6,.3,.1]);q=np.array([.1,.2,.7]);contract=E.BALANCED_BRANCH['contract']
        t2=np.sqrt(raw);t2[0]=0;t2/=t2.sum();excluded=q.copy();excluded[0]=0;excluded/=excluded.sum()
        rng=Mock();rng.choices.side_effect=[[0],[1]]
        B.alternative(raw,q,0,rng,contract=contract)
        self.assertEqual(rng.choices.call_args_list[0].kwargs['weights'],[.5,.5])
        seen=set()
        for seed in range(200):
            a=B.alternative(raw,q,0,random.Random(seed),contract=contract)
            self.assertEqual(a,B.alternative(raw,q,0,random.Random(seed),contract=contract))
            seen.add(a['component']);self.assertNotEqual(a['chosen'],0)
            self.assertEqual(a['component_probability'],.5)
            p=t2 if a['component']=='temperature_2' else excluded
            self.assertAlmostEqual(a['probability'],p[a['chosen']])
            self.assertAlmostEqual(a['marginal_probability'],(.5*t2+.5*excluded)[a['chosen']])
        self.assertEqual(seen,{'temperature_2','exploration_q'})

    def test_empty_support_and_legacy_sampling_unchanged(self):
        contract=E.BALANCED_BRANCH['contract']
        a=B.alternative(np.array([.9,.1]),np.array([1.,0.]),0,random.Random(5),contract=contract)
        self.assertEqual((a['component'],a['component_probability'],a['chosen'],a['marginal_probability']),('temperature_2',1.,1,1.))
        self.assertIsNone(B.alternative(np.ones(1),np.ones(1),0,random.Random(5),contract=contract))
        raw=np.array([.7,.2,.1]);q=np.array([.6,.4,0])
        for seed in range(30):
            # Reproduce the original exact draws and probabilities.
            rng=random.Random(seed);parts=[]
            for name,w,values in (('temperature_1',.7,raw),('temperature_2',.25,np.sqrt(raw)),('exploration_q',.05,q)):
                p=values.copy();p[0]=0;p/=p.sum();parts.append((name,w,p))
            n=rng.choices(range(3),weights=[x[1] for x in parts],k=1)[0]
            name,w,p=parts[n];i=rng.choices(range(3),weights=p,k=1)[0]
            expected=dict(chosen=i,component=name,probability=float(p[i]),component_probability=w,
                marginal_probability=float(sum(w*v[i] for _,w,v in parts)))
            for c in (E.LEGACY_BRANCH['contract'],E.THREE_T2_BRANCH['contract']):
                self.assertEqual(B.alternative(raw,q,0,random.Random(seed),contract=c),expected)
        with self.assertRaises(ValueError):B.alternative(raw,q,0,random.Random(0),contract='unknown')

    def test_real_capture_replay_and_tamper_rejection(self):
        raw=np.array([.7,.2,.1]);choices=[dict(kind=k) for k in ('loan','develop','pass')]
        # A non-tempered parent decision after the existing opening window.
        r=row();r.update(step=3,choices=choices,style=[0.]*4,intent=[0.]*8)
        r['view'].update(deck_count=1,own_hand=[1]);q=exploration_distribution(choices,r['intent'],3,'action')
        r['behavior']=dict(kind='model',policy_id='current',probability=float((.92*raw+.08*q)[0]),
            temperature=1.,exploration=.08,exploration_version=3,component='policy',action_decision=3)
        parent=dict(game_id='authored',setup_group='setup',split='train',opponents=['current']*4,records=[copy.deepcopy(r) for _ in range(4)])
        rngs=[random.Random(i) for i in range(4)]
        policies=[NeuralPolicy(None,evaluator=SimpleNamespace(model_id='current'),exploration=.08,
            exploration_version=3,opening=E.THREE_T2_OPENING) for _ in range(4)]
        for p,rng in zip(policies,rngs):
            p.last_raw[rng]=raw;p.contexts[rng]=None;p.action_counts[rng]=4
        for cfg in (E.THREE_T2_BRANCH,E.BALANCED_BRANCH):
            packet=B.capture(Mock(),parent,r,policies,rngs,branching=cfg)
            meta=packet['metadata'];fork=copy.deepcopy(r);fork.update(step=0,chosen=meta['chosen'])
            fork['behavior'].update(component='branch_move',probability=meta['marginal_probability'])
            child=dict(game_id=parent['game_id']+'/branch/'+meta['seed'],setup_group='setup',split='train',
                opponents=parent['opponents'],records=[fork],branch=meta)
            B.validate_child(parent,child)
            bad=copy.deepcopy(child);bad['branch']['marginal_probability']*=.5
            with self.assertRaises(ValueError):B.validate_child(parent,bad)
            if cfg==E.BALANCED_BRANCH:
                bad=copy.deepcopy(child);bad['branch']['component']='temperature_1'
                with self.assertRaisesRegex(ValueError,'sampling evidence'):B.validate_child(parent,bad)
                bad=copy.deepcopy(child);bad['branch']['contract']=E.THREE_T2_BRANCH['contract']
                with self.assertRaises(ValueError):B.validate_child(parent,bad)
        bad=copy.deepcopy(E.BALANCED_BRANCH);bad['move_distribution']=E.LEGACY_BRANCH['move_distribution']
        with self.assertRaisesRegex(ValueError,'versioned contract'):B.capture(Mock(),parent,r,policies,rngs,branching=bad)

    def test_new_forks_preserve_three_decision_reset(self):
        with patch.object(E,'THREE_OPENING',E.THREE_T2_OPENING),patch.object(E,'THREE_BRANCH',E.BALANCED_BRANCH):
            opening_tests.Revision().test_fork_skips_three_and_resets_all_four_seats()

    def test_resume_preserves_valid_cache_all_state_and_no_reset(self):
        m,opt,old=smoke_config();new=configure(old)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            for optimizer in (opt,p.optimizer):
                next(m.parameters()).grad=torch.ones_like(next(m.parameters()));optimizer.step();optimizer.zero_grad(set_to_none=True)
            p.beta=2.;p.trips=[True,False];p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            raw=dict(model_config=m.config.to_dict(),training={KEY:copy.deepcopy(p.payload())})
            before=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            resumed=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertIsNone(resumed.startup(1));self.assertEqual(resumed.buffer.receipts,p.buffer.receipts)
            self.assertEqual((resumed.beta,resumed.trips,resumed.cache_origin),(p.beta,p.trips,p.cache_origin))
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(before,(m.state_dict(),opt.state_dict(),resumed.optimizer.state_dict(),resumed.state,resumed.critic_state,torch.get_rng_state())))

    def test_helper_simulated1324_is_readonly_idempotent_and_generation_bound(self):
        spec=importlib.util.spec_from_file_location('g1324_helper',SOURCE_ROOT/'scripts/configure-training-g1324.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        # User reports g1324; actual checkpoint is verified by the lab helper.
        state=reward_tests.exported_state();state.update(iteration=1324,games_done=423680)
        state['signature'].update(config=old_recipe(),implementation=PREVIOUS_SOURCE)
        state['history'][-1].update(generation=1324,path='artifacts/step5/joint7-v11-2-grouped-lr-critic/generation-1324.pt')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated checkpoint fixture')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            output='configs/step5/g1324.json';before=(sha(run/'state.json'),sha(checkpoint))
            report=helper.generate(root,run,output,1324)
            self.assertEqual(report,helper.generate(root,run,output,1324));self.assertFalse(report['cache_refresh'])
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            signature=dict(state['signature'],implementation=implementation_id(),config=configure(state['signature']['config']))
            self.assertTrue(check_signature(state,signature,reviewed=True))
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=False)
            bad=copy.deepcopy(signature);bad['config']['exploration']=.05
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
            with self.assertRaisesRegex(ValueError,'Stale expected generation'):helper.generate(root,run,output,1322)
            state['signature']=signature;write_json(run/'state.json',state)
            self.assertEqual(report,helper.generate(root,run,output,1324))
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint missing/changed'):helper.generate(root,run,output,1324)


if __name__=='__main__':unittest.main()
