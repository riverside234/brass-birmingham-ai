"""Exact preceding v9 continuation only; no blanket source/config admission."""
import copy
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.recovery.config import TEACHER_PHASE_TIME_BUDGET_V4
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.step5.league import implementation_id
import test_training_performance_resume as resume_fixtures


PREVIOUS='63c9da11e4aa582af8ad41ea32c85781728ff6418c953a2df1170f7718d5d9b4'


class EncoderReuseResumeTests(unittest.TestCase):
    state=resume_fixtures.TrainingPerformanceResumeTests.state
    fixture=resume_fixtures.TrainingPerformanceResumeTests.fixture

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def signatures(self):
        config=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');config['model_size']='small'
        current=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),
                     legacy_pool={},teachers={})
        previous=copy.deepcopy(current);previous['implementation']=PREVIOUS
        previous['config']['teacher_work_budget']=copy.deepcopy(TEACHER_PHASE_TIME_BUDGET_V4)
        return previous,current

    def test_only_exact_source_and_budget_change_are_admitted(self):
        previous,current=self.signatures();state=self.state(previous);snapshot=copy.deepcopy(state)
        self.assertTrue(check_signature(state,current,reviewed=True))
        unchanged=copy.deepcopy(current);unchanged['config']=copy.deepcopy(previous['config'])
        self.assertTrue(check_signature(state,unchanged,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,current)
        for key in ('learning_rate','search_every_games','seed','exploration'):
            bad=copy.deepcopy(current);bad['config'][key]+=1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'pinned review changes'):
                check_signature(state,bad,reviewed=True)
        bad=copy.deepcopy(current);bad['config']['teacher_work_budget']['seconds_per_phase']['rail_late']=20.
        with self.assertRaisesRegex(ValueError,'pinned review changes'):check_signature(state,bad,reviewed=True)
        bad=copy.deepcopy(state);bad['signature']['implementation']='0'*64
        with self.assertRaisesRegex(ValueError,'not been reviewed'):check_signature(bad,current,reviewed=True)
        bad=copy.deepcopy(current);bad['compatibility']['encoder_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'learning contracts'):check_signature(state,bad,reviewed=True)
        self.assertEqual(state,snapshot)

    def test_checkpoint_adam_rng_progress_population_and_lineage_survive_restart(self):
        run,state,current,original,optimizer=self.fixture()
        checkpoint=run/'generation-0070.pt';saved=checkpoint.read_bytes();state_bytes=(run/'state.json').read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
        self.assertTrue(changed);self.assertEqual((run/'state.json').read_bytes(),state_bytes)
        restored=torch.optim.AdamW(model.parameters(),lr=current['config']['learning_rate'])
        resume_optimizer(raw,restored);self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
        for name,value in original.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        wanted,actual=optimizer.state_dict(),restored.state_dict()
        self.assertEqual(wanted['param_groups'],actual['param_groups'])
        for key,values in wanted['state'].items():
            for name,value in values.items():torch.testing.assert_close(value,actual['state'][key][name],rtol=0,atol=0)
        # Existing source-change freshness safeguard requires new calibration.
        self.assertFalse(model.utility_calibration['ready'])
        updated=commit_transition(run,state,current)
        for key,value in state.items():
            if key!='signature':self.assertEqual(value,updated[key],key)
        self.assertEqual(set(updated['source_transitions'][-1]['config_changes']),{'teacher_work_budget'})
        self.assertEqual(checkpoint.read_bytes(),saved)
        again,raw2,changed=load_resume(SOURCE_ROOT,run,updated,current)
        self.assertFalse(changed);self.assertFalse(again.utility_calibration['ready'])
        self.assertEqual(raw2['model_id'],raw['model_id']);self.assertEqual(checkpoint.read_bytes(),saved)

    def test_pending_collection_still_blocks_the_source_transition(self):
        run,state,current,_,_=self.fixture()
        pending=run/'iteration-0070';pending.mkdir()
        (pending/'collection.json').write_text('{}\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):
            load_resume(SOURCE_ROOT,run,state,current,reviewed=True)


if __name__=='__main__':unittest.main()
