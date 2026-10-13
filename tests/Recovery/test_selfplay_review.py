"""Regression checks for the supplied selfplay-v1 losses and sparse teachers."""
import copy
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.bank import read
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.replay import split_for
from brass_ai.recovery.config import DISTILLATION,PPO_LEARNING_RATE,UPDATE_V21,validate_config
from brass_ai.recovery.teacher_schedule import PHASES,ScheduledMasked,training_teacher
from brass_ai.recovery.updates import distill_update,phase_probe,prepared,supervised_epoch
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import MaskedSearchPolicy,visit_teacher
from brass_ai.step5.league import seeds_for


def teacher_games(count=4):
    games=[]
    for i,seed in enumerate(seeds_for(17120000,count)[0]):
        episode=game();base=episode['records'];episode['records']=[]
        episode['game_id']=f'teacher-{i}'
        episode['setup_group']=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
        episode['split']=split_for(episode['setup_group'])
        for phase,era,first,last in PHASES:
            for source in base:
                row=copy.deepcopy(source);row['step']=len(episode['records'])
                row['view']['context'].update(era=era,round=first,revision=row['step'])
                choices=row['choices'];stats=dict(expansion=[True,True,False],root={choices[0]['id']:dict(visits=3),choices[1]['id']:dict(visits=2)})
                row['teacher']=visit_teacher(choices,stats,[.6,.3,.1],model='a'*64,observer=row['seat'],config=SearchConfig(seconds=12,workers=4))
                episode['records'].append(row)
        games.append(episode)
    return games


class SelfplayReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_probe_represents_logged_phase_mixture_and_is_seeded(self):
        counts=dict(action=21741,bonus=151,continuation=3545,debt=291,payment=1705)
        rows=[dict(view=dict(context=dict(phase=phase,era='CanalEra' if i%2 else 'RailEra')))
              for phase,n in counts.items() for i in range(n)]
        ds=SimpleNamespace(rows=rows);indices=list(range(len(rows)))
        torch.manual_seed(816);probe=phase_probe(ds,indices,2048)
        torch.manual_seed(816);self.assertEqual(probe,phase_probe(ds,indices,2048))
        self.assertEqual(len(probe),2048);self.assertEqual(len(set(probe)),2048)
        sample=Counter(rows[i]['view']['context']['phase'] for i in probe)
        for phase,n in counts.items():self.assertLessEqual(abs(sample[phase]-2048*n/len(rows)),2)
        # The old equal-phase probe would see about .0088, concealing .0319 KL.
        kl=lambda i:.04 if rows[i]['view']['context']['phase']=='action' else .001
        self.assertAlmostEqual(sum(map(kl,probe))/len(probe),sum(map(kl,indices))/len(indices),delta=1e-4)
        self.assertEqual(set(phase_probe(ds,indices,len(rows)+1)),set(indices))
        self.assertEqual(phase_probe(ds,[],2048),[])

    def test_one_teacher_as_in_lab_report_cannot_update_weights_or_optimizer(self):
        episodes=teacher_games();model=FourSeatNet(ModelConfig.preset('small'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=PPO_LEARNING_RATE)
        # Seed optimizer moments to ensure the sparse gate does not move them.
        supervised_epoch(model,optimizer,prepared(episodes[:1]),dict(UPDATE_V21,effective_batch_size=16,microbatch_size=8))
        before=model_id(model);steps=[v['step'].clone() for v in optimizer.state.values()]
        for i,episode in enumerate(episodes):
            for j,row in enumerate(episode['records']):
                if i or j:row['teacher']=None
        report=distill_update(model,optimizer,episodes,UPDATE_V21)
        self.assertFalse(report['updated']);self.assertEqual(report['teacher_rows'],1)
        self.assertEqual(report['optimizer_steps'],0);self.assertEqual(model_id(model),before)
        self.assertTrue(all(torch.equal(a,v['step']) for a,v in zip(steps,optimizer.state.values())))

    def test_missing_late_rail_or_single_setup_cannot_pass_diversity_gate(self):
        model=FourSeatNet(ModelConfig.preset('small'));optimizer=torch.optim.AdamW(model.parameters())
        episodes=teacher_games()
        for episode in episodes:
            for row in episode['records']:
                if row['view']['context']['era']=='RailEra' and row['view']['context']['round']==7:row['teacher']=None
        report=distill_update(model,optimizer,episodes,UPDATE_V21)
        self.assertEqual(report['teacher_rows'],48);self.assertFalse(report['updated'])
        self.assertEqual(report['samples_by_phase']['rail_late'],0)
        report=distill_update(model,optimizer,teacher_games(1),UPDATE_V21)
        self.assertEqual(report['teacher_rows'],16);self.assertEqual(report['teacher_setups'],1)
        self.assertFalse(report['updated'])

    def test_policy_only_distillation_preserves_value_head_and_bounds_reuse(self):
        episodes=teacher_games(2);model=FourSeatNet(ModelConfig.preset('small'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=PPO_LEARNING_RATE)
        cfg=dict(UPDATE_V21,effective_batch_size=2048,microbatch_size=8)
        supervised_epoch(model,optimizer,prepared(episodes),cfg) # nonzero critic Adam moments
        head={k:v.clone() for k,v in model.outcomes.state_dict().items()}
        moments={p:{k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()} for p in model.outcomes.parameters()}
        policy_before=[v.clone() for v in model.policy_scorer.parameters()]
        report=distill_update(model,optimizer,episodes,cfg,steps=8)
        self.assertTrue(report['updated']);self.assertEqual(report['optimizer_steps'],4)
        self.assertEqual(report['sample_presentations'],64);self.assertEqual(report['max_observed_reuse'],2)
        self.assertFalse(report['value_loss_training'])
        for update in report['updates']:
            self.assertAlmostEqual(update['loss'],.25*update['policy_loss'])
            self.assertFalse(update['value_loss_training'])
        for k,v in model.outcomes.state_dict().items():self.assertTrue(torch.equal(v,head[k]))
        for p in model.outcomes.parameters():
            self.assertIsNone(p.grad)
            for k,v in optimizer.state[p].items():self.assertTrue(torch.equal(v,moments[p][k]))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(policy_before,model.policy_scorer.parameters())))

    def test_only_teacher_replicas_skip_duplicate_incumbent(self):
        parent=object.__new__(MaskedSearchPolicy);parent.policy_id='root'
        worker=object.__new__(MaskedSearchPolicy);worker.policy_id='worker';parent.replicas=[worker]
        training_teacher(parent)
        self.assertFalse(parent.teacher_replica);self.assertTrue(worker.teacher_replica)
        self.assertEqual(worker.incumbent(None,None,None,None,None),{})
        parent.teacher_work_deadline=0
        with patch('brass_ai.recovery.teacher_schedule.time.monotonic',return_value=1):
            self.assertEqual(parent.incumbent(None,None,None,None,None),{})
        self.assertIsInstance(parent,ScheduledMasked)

    def test_reviewed_presets_keep_il_and_existing_training_limits(self):
        for path in (SOURCE_ROOT/'configs/step5').glob('lab*.json'):
            cfg=validate_config(read(path))
            self.assertEqual(cfg['learning_rate'],1e-5);self.assertEqual(cfg['imitation']['learning_rate'],1e-4)
            self.assertEqual(cfg['distillation'],DISTILLATION)
            self.assertEqual(cfg['update']['kl_probe_rows'],2048);self.assertEqual(cfg['update']['target_kl'],.01)
            self.assertEqual((cfg['search']['seconds'],cfg['search']['pending_paths'],cfg['search']['batch_size']),(12.,2,2))
            self.assertEqual((cfg['search_every_games'],cfg['search_ready_games'],cfg['search_games'],cfg['distill_steps']),(160,640,5,8))
            self.assertEqual(cfg['search_wall_fraction'],None);self.assertEqual(cfg['legacy_budget']['seconds'],1.)


if __name__=='__main__':unittest.main()
