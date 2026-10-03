"""Current fresh-run quotas, effective batches and Masked teacher memory bounds."""
import copy
from collections import Counter
from pathlib import Path
import random
import tempfile
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import split_for
from brass_ai.step5.league import seeds_for, train, validate_config
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.step5.updates import bounded_batches
from brass_ai.recovery.config import CURRICULUM, UPDATE_V21, table_kind
from brass_ai.recovery.updates import ppo_update, distill_update
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import visit_teacher
from fixtures import game


def onpolicy_game(policy):
    result=game();seed=seeds_for(551000,1)[0][0]
    group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
    result.update(setup_group=group,split=split_for(group),opponents=[policy.policy_id]*4)
    for row in result['records']:
        rng=random.Random(row['seat'])
        selected,probability,teacher=policy.select_batch([(row['view'],row['choices'],rng)])[0]
        style,intent=policy.context_features(row['view'],rng)
        row.update(chosen=selected,teacher=teacher,style=style,intent=intent,
                   behavior=dict(kind='model',policy_id=policy.policy_id,probability=probability,temperature=1.))
    return result


class ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_presets_and_strict_new_fields(self):
        large=validate_config(read(SOURCE_ROOT/'configs/step5/lab.json'))
        small=validate_config(read(SOURCE_ROOT/'configs/step5/lab-small.json'))
        self.assertEqual(large['games_per_iteration'],320)
        self.assertEqual((large['execution']['heavy_processes'],large['execution']['search_workers_per_turn']),(24,4))
        self.assertEqual((large['update']['effective_batch_size'],large['update']['microbatch_size']),(2048,512))
        self.assertEqual(small['format'],'step5-training/2.2.0')
        self.assertEqual(small['games_per_iteration'],160)
        for config in (dict(large,num_workers=64),read(SOURCE_ROOT/'configs/step5/previous-lab.json')):
            with self.assertRaises(ValueError):validate_config(config)
        for field in ('effective_batch_size','microbatch_size','max_candidate_cells'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_config(dict(large,update=dict(large['update'],**{field:0})))

    def test_quota_counts_survive_collection_and_resume_boundaries(self):
        for start in (0,2400,6000,11000):
            scheduled=[table_kind(start+i,CURRICULUM) for i in range(320)]
            resumed=[table_kind(start+i,CURRICULUM) for i in range(137)]+[
                table_kind(start+i,CURRICULUM) for i in range(137,320)]
            self.assertEqual(Counter(scheduled),Counter(resumed))
            self.assertEqual(scheduled,resumed)

    def test_padded_cell_limit_preserves_all_rows(self):
        from types import SimpleNamespace
        data=[(None,None,None,SimpleNamespace(choice_ids=list(range(n)))) for n in (2,5,3,8)]
        batches=list(bounded_batches(data,4,10))
        self.assertEqual([(start,len(batch)) for start,batch in batches],[(0,2),(2,1),(3,1)])
        self.assertEqual([entry for _,batch in batches for entry in batch],data)
        with self.assertRaises(ValueError):list(bounded_batches(data,4,7))

    def test_microbatch_size_preserves_effective_update(self):
        torch.manual_seed(181);first=FourSeatNet(ModelConfig.preset('small'));second=copy.deepcopy(first)
        policy=NeuralPolicy(first);games=[onpolicy_game(policy)]
        config=dict(UPDATE_V21,effective_batch_size=4,critic_batch_size=4,microbatch_size=4,epochs=2)
        torch.manual_seed(721)
        a=ppo_update(first,torch.optim.SGD(first.parameters(),lr=1e-3),games,policy,config)
        torch.manual_seed(721)
        b=ppo_update(second,torch.optim.SGD(second.parameters(),lr=1e-3),games,policy,dict(config,microbatch_size=1,max_candidate_cells=3))
        self.assertEqual(a['optimizer_steps'],b['optimizer_steps'])
        for left,right in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(left,right,rtol=1e-3,atol=2e-6)

    def test_new_runs_reject_prior_checkpoints_and_run_imports(self):
        config=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            for keyword in ('initial','from_run'):
                with self.subTest(keyword=keyword),self.assertRaisesRegex(ValueError,'forbids'):
                    train(SOURCE_ROOT,Path(temporary)/keyword,config,**{keyword:'old-run'})

    def test_distillation_memory_splits_preserve_teacher_weights(self):
        torch.manual_seed(271);first=FourSeatNet(ModelConfig.preset('small'));second=copy.deepcopy(first)
        games=[onpolicy_game(NeuralPolicy(first))]
        for row in games[0]['records']:
            choices=row['choices'];stats=dict(completed=4,expansion=[True,True,False],
                root={choices[0]['id']:dict(visits=3),choices[1]['id']:dict(visits=1)})
            row['teacher']=visit_teacher(choices,stats,[.4,.4,.2],model=model_id(first),observer=row['seat'],config=SearchConfig())
        config=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=4)
        torch.manual_seed(721)
        a=distill_update(first,torch.optim.AdamW(first.parameters(),lr=1e-4),games,config,steps=2)
        torch.manual_seed(721)
        b=distill_update(second,torch.optim.AdamW(second.parameters(),lr=1e-4),games,dict(config,microbatch_size=1,max_candidate_cells=3),steps=2)
        self.assertTrue(a['updated'] and b['updated']);self.assertEqual(a['teacher_rows'],b['teacher_rows'])
        for left,right in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(left,right,rtol=1e-3,atol=2e-6)


if __name__=='__main__':unittest.main()
