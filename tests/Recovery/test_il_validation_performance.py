"""One IL validation forward pass preserves both reports and best-checkpoint selection."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import demonstrations as demos
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.updates import Dataset,prepared,supervised_epoch


def validation_rows():
    episodes=[]
    for index in range(3):
        episode=game();episode['game_id']=f'validation-{index}'
        for seat,row in enumerate(episode['records']):
            row['choices']=row['choices'][:1+(index+seat)%3]
            row['behavior']['policy_id']='rare' if index==0 and seat==1 else 'common'
        episodes.append(episode)
    return prepared(episodes,neutral=True)


def metrics(report):
    return {k:v for k,v in report.items() if k not in ('timings','memory')}


class ILValidationPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def assert_reports_match(self,device,precision='fp32'):
        torch.manual_seed(72411)
        model=FourSeatNet(ModelConfig.preset('small')).to(device)
        rows=validation_rows();balanced=demos.balance(rows)
        weights=[row['_bc_weight'] for row,_,_,_ in balanced]
        self.assertGreater(max(weights)-min(weights),.5)
        ordinary=Dataset(rows,device,policy=False);weighted=Dataset(balanced,device,policy=False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone()
        for nonforced,bucket,policy_weight in ((True,True,1.7),(False,False,1.7),(True,False,0.)):
            config=dict(UPDATE_V21,effective_batch_size=5,microbatch_size=2,
                        actor_nonforced=nonforced,bucket=bucket,precision=precision)
            left=supervised_epoch(model,None,ordinary,config,training=False,weight=policy_weight)
            right=supervised_epoch(model,None,weighted,config,training=False,weight=policy_weight)
            with patch.object(model,'forward',wraps=model.forward) as forward:
                actual=supervised_epoch(model,None,ordinary,config,training=False,weight=policy_weight,
                                        validation_weights=weights)
            batches=sum(len(list(ordinary.microbatches(range(start,min(start+5,len(rows))),2,
                        config['max_candidate_cells'],bucket))) for start in range(0,len(rows),5))
            self.assertEqual(forward.call_count,batches)
            self.assertEqual(metrics(actual[0]),metrics(left))
            self.assertEqual(metrics(actual[1]),metrics(right))
            self.assertIsNot(actual[0]['by_teacher_era_phase_action'],actual[1]['by_teacher_era_phase_action'])
            first=next(iter(actual[0]['by_teacher_era_phase_action']))
            self.assertIsNot(actual[0]['by_teacher_era_phase_action'][first],actual[1]['by_teacher_era_phase_action'][first])
        uniform=supervised_epoch(model,None,ordinary,config,training=False,validation_weights=[1.]*len(rows))
        self.assertEqual(metrics(uniform[0]),metrics(uniform[1]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))

    def test_both_validation_reports_are_exact_with_one_forward_pass_cpu(self):
        self.assert_reports_match('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
    def test_both_validation_reports_are_exact_with_one_forward_pass_cuda(self):
        self.assert_reports_match('cuda')
        if torch.cuda.is_bf16_supported():self.assert_reports_match('cuda','bf16')

    def test_reweighting_is_validation_only_and_checks_weights(self):
        model=FourSeatNet(ModelConfig.preset('small'));dataset=Dataset(validation_rows(),'cpu',policy=False)
        for weights in ([1.],[-1.]*12,[float('nan')]*12):
            with self.assertRaisesRegex(ValueError,'Invalid validation weights'):
                supervised_epoch(model,None,dataset,UPDATE_V21,training=False,validation_weights=weights)
        with self.assertRaisesRegex(ValueError,'validation-only'):
            supervised_epoch(model,None,dataset,UPDATE_V21,validation_weights=[1.]*12)

    def test_training_weights_rng_optimizer_and_selection_match_two_validation_passes(self):
        rows=validation_rows()
        training=copy.deepcopy(rows)
        for row in training:row[0]['behavior']['policy_id']='training'
        training=[(row,target,('training-'+identity[0],identity[1]),encoded)
                  for row,target,identity,encoded in training]
        config=dict(UPDATE_V21,effective_batch_size=5,microbatch_size=2)

        def two_pass(model,optimizer,data,config,*,validation_weights=None,**kwargs):
            ordinary=supervised_epoch(model,optimizer,data,config,**kwargs)
            if validation_weights is None:return ordinary
            weighted=[(dict(row,_bc_weight=weight),target,identity,encoded)
                      for (row,target,identity,encoded),weight in zip(data.data,validation_weights)]
            balanced=supervised_epoch(model,optimizer,Dataset(weighted,'cpu',policy=False),config,**kwargs)
            return ordinary,balanced

        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as name:
            directory=Path(name);manifest=directory/'manifest.json';manifest.write_text('{}',encoding='utf-8')
            with patch.object(demos,'load_dataset',side_effect=lambda root,path,split,**kwargs:
                              (training if split=='train' else rows,{})),contextlib.redirect_stdout(io.StringIO()):
                with patch.object(demos,'supervised_epoch',side_effect=two_pass):
                    demos.train_bc(SOURCE_ROOT,directory/'old',manifest,None,config,
                                   epochs=2,min_epochs=1,patience=2)
                demos.train_bc(SOURCE_ROOT,directory/'new',manifest,None,config,
                               epochs=2,min_epochs=1,patience=2)
            old,new=(read(directory/name/'state.json') for name in ('old','new'))
            for key in ('epoch','best_loss','bad_epochs','stopped'):
                self.assertEqual(old[key],new[key])
            self.assertEqual(old['best']['path'],new['best']['path'])
            for left,right in zip(old['history'],new['history']):
                for key in ('train','validation','balanced_validation'):
                    self.assertEqual(metrics(left[key]),metrics(right[key]))
            left,right=(torch.load(directory/name/state['last']['path'],weights_only=True)
                        for name,state in (('old',old),('new',new)))
            self.assertEqual(left['model_id'],right['model_id'])
            self.assertTrue(torch.equal(left['rng'],right['rng']))
            self.assertEqual(left['optimizer']['param_groups'],right['optimizer']['param_groups'])
            for parameter,state in left['optimizer']['state'].items():
                for key,value in state.items():
                    self.assertTrue(torch.equal(value,right['optimizer']['state'][parameter][key]))


if __name__=='__main__':unittest.main()
