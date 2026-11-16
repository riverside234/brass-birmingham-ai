"""Cotton quotas, PPO probabilities, resumed PPG, heldout EV and shared packing."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT as ROOT
from brass_ai.recovery import ordinary_cotton as C, training_revision_g2400 as G
from brass_ai.step5.policy import COTTON_CANDIDATES, NeuralPolicy


def recipe():
    # Portable authored recipe; do not require the results repo or lab weights.
    from test_utility_width import recipe as initial
    from brass_ai.recovery import search_head_refresh, teacher_search_retune, training_revision_g2090, training_revision_g2240
    c=initial();c['teacher_phase']['mode']='on'
    from brass_ai.recovery import training_revision_g1491 as R
    c=search_head_refresh.configure(c,adaptive_beta=True)
    c=teacher_search_retune.configure(c)
    c=teacher_search_retune.configure(R.configure_teacher_games(c,0),split_c_scale=True)
    c=search_head_refresh.configure(c,reset=True)
    c=training_revision_g2090.configure(c)
    c=training_revision_g2240.configure(c)
    return teacher_search_retune.configure(c,update_scales=True)


def ordinary():
    i=next(i for i in range(20) if C.planned_assignment(C.NETWORK,i,4400))
    proof=dict(kind='selfplay',purpose='training',ordinary_game_index=i,sampling_seed=4400,
               **{C.KEY:C.planned_assignment(C.NETWORK,i,4400)})
    actor=C.TableExploration(proof);seat=actor.seats[0]
    view=dict(seat=seat,context=dict(phase='action',era='CanalEra'),
              players=[dict(mat=[dict(type='CottonMill',tiles=[dict(level=1)])]) for _ in range(4)],
              slots=[dict(id='A:0',types=['CottonMill','Manufacturer']),dict(id='B:0',types=['Pottery'])],
              links=[dict(id='AB',regions=['A','B']),dict(id='BC',regions=['B','C'])])
    choices=[dict(id='netA',kind='network',target='AB'),dict(id='netB',kind='network',target='BC'),
             dict(id='build1',kind='build',industry='CottonMill'),dict(id='dev',kind='develop',industry='CottonMill'),
             dict(id='loan',kind='loan')]
    return proof,actor,view,choices


class CottonNetwork(unittest.TestCase):
    def test_quota_is_independent_and_resets_each_era(self):
        proof,actor,view,choices=ordinary();self.assertEqual(actor.targets(view,choices),[0,3])
        for _ in range(3):actor.commit(view,dict(component='cotton'),choices[3])
        self.assertEqual(actor.targets(view,choices),[0])
        actor.commit(view,dict(component='cotton'),choices[0])
        self.assertEqual(actor.targets(view,choices),[])
        view['context'].update(era='RailEra',phase='continuation')
        self.assertEqual(actor.targets(view,choices),[0,3])
        actor.commit(view,dict(component='cotton'),choices[0])
        self.assertEqual(actor.targets(view,choices),[3])
        view['slots'][0]['tile']=dict(seat=view['seat'],type='CottonMill',flipped=False)
        self.assertEqual(actor.targets(view,choices),[])
        view['slots'][0]['tile']['flipped']=True
        self.assertEqual(actor.targets(view,choices),[3])
        view['seat']=(view['seat']+1)%4
        self.assertEqual(actor.targets(view,choices),[])

    def test_mixed_probability_matches_ppo_and_replay(self):
        from brass_ai.recovery.exploration import check_behavior
        from brass_ai.recovery.updates import ExplorationMixture,row_probabilities
        proof,actor,view,choices=ordinary();replay=C.TableExploration(proof)
        policy=NeuralPolicy(None,evaluator=SimpleNamespace(model_id='learner'),exploration=.08,exploration_version=3)
        logits=torch.tensor([-1.,1.,.2,.3,-.4]);raw=logits.double().softmax(-1).numpy()
        rng=random.Random(11)
        chosen,p,_=policy.sample_training(view,choices,rng,raw,None,cotton_probability=.2,cotton_targets=actor.targets(view,choices))
        b=dict(policy.last_behavior[rng],kind='model',policy_id=policy.policy_id,probability=p)
        check_behavior(b,'action');self.assertEqual(b[COTTON_CANDIDATES],[0,3])
        row=dict(view=view,seat=view['seat'],choices=choices,chosen=chosen,behavior=b,intent=None)
        replay.check(row)
        dataset=SimpleNamespace(temperatures=torch.tensor([[b['temperature']]]),cotton_rates=torch.tensor([[.2]]))
        result=row_probabilities(dataset,[0],logits[None],torch.tensor(ExplorationMixture([row])[0])[None],policy)
        self.assertAlmostEqual(float(result[0,chosen]),p,places=6)
        b[COTTON_CANDIDATES]=[0,1,3]
        with self.assertRaisesRegex(ValueError,'eligibility'):C.TableExploration(proof).check(row)

    def test_teacher_and_heldout_mix_is_exact_over_a_cycle(self):
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.recovery import training_revision_g1491 as R
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.search_value_schedule import validation_plan
        c=G.configure(recipe());validate_config(c)
        c=R.configure_teacher_games(c,0)
        counts=dict(train=[0,0],heldout=[0,0])
        for generation in range(2400,2425):
            state=dict(iteration=generation,signature=dict(config=c),search_value=dict(stage='C',w=1.))
            jobs,_=plan_jobs(c,state,model_id='m',export_hash='e',epoch=generation,ready=True)
            self.assertEqual(sum(j['split']=='train' for j in jobs),32)
            for j in jobs:
                R.validate_job(j)
                self.assertEqual(j['search_value']['w'],0.)
                cotton=j['teacher_revision']['cotton_seats']
                counts[j['split']][0]+=bool(cotton);counts[j['split']][1]+=1
                self.assertLessEqual(len(cotton),1)
        self.assertEqual(counts,dict(train=[120,800],heldout=[5,20]))
        self.assertEqual((c['value_phase']['max_epochs'],c['teacher_phase']['search_value']['max_epochs']),(2,1))
        self.assertEqual(c['value_phase']['beta_clone_min'],.5)

    def test_config_transition_preserves_labels_weights_and_schedule(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        before=recipe();after=G.configure(before)
        self.assertIs(G.configure(after),after)
        validate_config(after)
        self.assertEqual(buffer_identity(before),buffer_identity(after))
        self.assertIs(before['teacher_phase']['search_value'],after['teacher_phase']['search_value'])
        self.assertTrue(signature_transition(dict(config=before,implementation=QUALIFIED),dict(config=after,implementation=QUALIFIED)))
        self.assertFalse(signature_transition(dict(config=after,implementation=QUALIFIED),dict(config=before,implementation=QUALIFIED)))

    def test_ppg_resume_changes_only_the_epoch_option(self):
        from test_utility_width import learned
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.retention import _same_payload
        model=learned();before=recipe();after=G.configure(before)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as folder:
            run=Path(folder)
            first=TrainingPhases(model,before,root=ROOT,run=run)
            # Populate Adam state so the resume checks moments as well as options.
            for group in first.optimizer.param_groups:
                for p in group['params']:p.grad=torch.ones_like(p)*.01
            first.optimizer.step();first.optimizer.zero_grad(set_to_none=True)
            first.buffer.pending=[]
            saved=first.payload()
            resumed=TrainingPhases(model,after,raw={'training':{KEY:saved}},root=ROOT,run=run)
            resumed.buffer.pending=[]
            out=resumed.payload()
            self.assertEqual(out['options']['value_phase']['max_epochs'],2)
            self.assertTrue(_same_payload({k:v for k,v in saved.items() if k!='options'},
                                          {k:v for k,v in out.items() if k!='options'}))


class Diagnostics(unittest.TestCase):
    # The g2400 PPO preparation and teacher-term changes were rolled back at g2654
    # (test_training_g2600.Staging covers the restored path).

    def test_ev_uses_the_same_heldout_targets_and_all_seats_as_mse(self):
        from brass_ai.recovery.search_value_schedule import diagnostics
        target=np.arange(1024,dtype=float).reshape(256,4)/1024
        rows=[dict(split='heldout',game_id=str(i//64),u=t) for i,t in enumerate(target)]
        main=target+.2;delta=np.sin(target)*.1
        result=diagnostics(rows,main,delta)
        self.assertAlmostEqual(result['main_ev'],1.)
        self.assertAlmostEqual(result['corrected_ev'],1-np.var(target-main-delta)/np.var(target))
        self.assertAlmostEqual(result['main_mse'],.04)
        flat=[dict(r,u=np.ones(4)) for r in rows]
        self.assertIsNone(diagnostics(flat,main,delta)['main_ev'])
        self.assertEqual(diagnostics([],[],[]),dict(recorded=False,games=0,rows=0))

