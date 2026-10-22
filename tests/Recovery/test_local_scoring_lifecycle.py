"""Exercise the production diagnostic/release boundary after actual PPO work."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_config,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY as PHASES
from brass_ai.recovery.league import finish_training_diagnostics
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT,KEY
from brass_ai.recovery.retention import _same_payload
from test_gae_value_phase import ppo,finish
from test_local_scoring import annotate
from test_ppg_eight import setup,collect


class Lifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_post_update_diagnostics_release_then_checkpoint_preserves_training_state(self):
        for mode in ('apply','shadow'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                model,opt,cfg=setup();cfg=ppg_config.coverage_options(cfg)
                cfg[KEY]=dict(copy.deepcopy(DEFAULT),mode=mode)
                run=Path(tmp);phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
                companion=Companion(model,cfg[KEY]);phase.local_scoring=companion
                behavior,data=collect(model,phase,1,branches=True)
                annotate(data);annotate(phase.snapshot_data)
                update=ppo(model,opt,cfg,behavior,data,phase);phase.fit(model);finish(phase,update,1)
                self.assertGreater(update['optimizer_steps'],0)
                self.assertFalse(phase.value_report['scheduled'])
                ds=phase.critic_dataset
                eligible=[i for i in phase.heldout_ids if ds.roles[i]=='learner' and not ds.rows[i].get('_branch')]
                self.assertTrue(eligible)
                before=copy.deepcopy(dict(model=model.state_dict(),adam=opt.state_dict(),
                    phase=phase.payload(),aux=companion.payload(1)))
                rng=torch.get_rng_state().clone()
                diagnostics=companion.diagnostics
                def checked(net,dataset,ids):
                    self.assertIs(dataset,phase.critic_dataset)
                    self.assertIsNotNone(dataset)
                    self.assertEqual(ids,phase.heldout_ids)
                    return diagnostics(net,dataset,ids)
                with patch.object(companion,'diagnostics',side_effect=checked) as measured:
                    report=finish_training_diagnostics(model,phase,companion)
                measured.assert_called_once()
                self.assertEqual(report['heldout']['rows'],len(eligible))
                self.assertEqual(report['heldout']['games'],len({ds.data[i][2][0] for i in eligible}))
                self.assertTrue(report['heldout']['groups'])
                json.dumps(report,allow_nan=False)
                self.assertIsNone(phase.dataset);self.assertIsNone(phase.critic_dataset)
                self.assertIsNone(phase.ppo_probe);self.assertEqual(phase.snapshot_data,[])
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(_same_payload(before,dict(model=model.state_dict(),adam=opt.state_dict(),
                    phase=phase.payload(),aux=companion.payload(1))))
                checkpoint=run/'generation-0001.pt'
                save_checkpoint(checkpoint,model,opt,step=1,training={PHASES:phase.payload(),KEY:companion.payload(1)})
                _,raw=load_checkpoint(checkpoint)
                self.assertEqual(raw['step'],1)
                self.assertTrue(_same_payload(before['phase'],raw['training'][PHASES]))
                self.assertTrue(_same_payload(before['aux'],raw['training'][KEY]))

    def test_disabled_and_absent_companion_still_release_without_diagnostics(self):
        for mode in ('off',None):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                model,_,cfg=setup();phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=Path(tmp))
                _,data=collect(model,phase,1);phase.prepare(updates.Dataset(data,'cpu'))
                companion=Companion(model,dict(copy.deepcopy(DEFAULT),mode='off')) if mode else None
                if companion:
                    with patch.object(companion,'diagnostics',side_effect=AssertionError('disabled diagnostics')):
                        report=finish_training_diagnostics(model,phase,companion)
                    self.assertEqual(report['mode'],'off');self.assertNotIn('heldout',report)
                else:self.assertIsNone(finish_training_diagnostics(model,phase,None))
                self.assertIsNone(phase.critic_dataset)
        self.assertIsNone(finish_training_diagnostics(None,None,None))

    def test_missing_dataset_fails_explicitly_instead_of_suppressing_diagnostics(self):
        model,_,_=setup();companion=Companion(model,copy.deepcopy(DEFAULT))
        with self.assertRaisesRegex(ValueError,'live critic dataset before release'):
            finish_training_diagnostics(model,None,companion)


if __name__=='__main__':unittest.main()
