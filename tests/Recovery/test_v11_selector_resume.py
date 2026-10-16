"""V11 selector receipts cannot bridge the different v11.1 architecture."""
import copy
import unittest
from pathlib import Path
import tempfile
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json

PREVIOUS='c4b6b899d4c28aa83fc91706dbf2812233c0ca2ad3f59b4bee07e9ace3c12168'
V11_FINGERPRINT='c7df5bf97fbe02f605aabeea89c9cc24d74981c5a4355a15f050f6b97f0cae39'


class V11SelectorResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_review_flag_cannot_bridge_previous_v11_semantics(self):
        config=read(SOURCE_ROOT/'configs/step5/joint7-latest.json')
        current=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
        previous=copy.deepcopy(current)
        previous['config']['architecture']='brass-four-seat-mlp-resnet/11.0.0'
        previous['compatibility']['architecture_sha256']=V11_FINGERPRINT
        previous['implementation']=PREVIOUS
        from brass_ai.recovery.curriculum import activate_boost
        state=dict(format='step5-recovery-league/2.2.0',signature=previous,iteration=1,games_done=0)
        activate_boost(state,config)
        for reviewed in (False,True):
            with self.assertRaises(ValueError):check_signature(state,current,reviewed=reviewed)
        self.assertFalse(check_signature(dict(state,signature=current),current))

    def test_completed_previous_source_not_admitted_as_v11_1(self):
        from latest_model_fixture import imitation_fixture
        from brass_ai.recovery.demonstrations import completed_imitation
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            run,_,_=imitation_fixture(Path(directory))
            for name in ('state.json','report.json'):
                data=read(run/name);data['signature']['implementation']=PREVIOUS
                write_json(run/name,data)
            before={p.name:sha(p) for p in run.iterdir() if p.is_file()}
            with self.assertRaisesRegex(ValueError,'reviewed compatible source'):
                completed_imitation(SOURCE_ROOT,run)
            self.assertEqual(before,{p.name:sha(p) for p in run.iterdir() if p.is_file()})

    def test_completed_current_il_retains_weights_origin_and_source_files(self):
        from latest_model_fixture import imitation_fixture
        from brass_ai.recovery.demonstrations import completed_imitation
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            run,original,config=imitation_fixture(Path(directory))
            before={p.name:sha(p) for p in run.iterdir() if p.is_file()}
            model,origin=completed_imitation(SOURCE_ROOT,run)
            self.assertEqual(origin,config['imitation_checkpoint'])
            for name,value in original.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(before,{p.name:sha(p) for p in run.iterdir() if p.is_file()})


if __name__=='__main__':unittest.main()
