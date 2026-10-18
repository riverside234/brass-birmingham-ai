"""The reported generation-one run can adopt the performance fix without reset."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition,apply_optimizer_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json
from test_critic_descriptors import mixed_fixture
from test_gae_value_phase import ppo,finish


PREVIOUS='de2aac3edd606da9c23fad732c6bc4d8dc6118ad2d801af8b9ae6ffb9326f314'
REPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0001-20261018T215854542444Z'


class Resume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_report_requires_explicit_review_but_no_config_or_probe_change(self):
        state=read(REPORT/'state.json');before=copy.deepcopy(state)
        signature=copy.deepcopy(state['signature']);signature['implementation']=implementation_id()
        self.assertEqual(state['signature']['implementation'],PREVIOUS)
        self.assertEqual(signature['compatibility'],compatibility())
        self.assertEqual(signature['config']['audit_workers'],16)
        self.assertEqual(signature['config'],read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic.json'))
        probe=read(REPORT/'il-kl/manifest.json')
        self.assertEqual(probe['source']['compatibility'],compatibility())
        self.assertEqual(probe['config'],signature['config']['il_kl_limit'])
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        self.assertTrue(check_signature(state,signature,True))
        changed=copy.deepcopy(signature);changed['config']['rehearsal_sampling']['rows_per_iteration']+=1
        with self.assertRaisesRegex(ValueError,'unchanged config'):check_signature(state,changed,True)
        changed=copy.deepcopy(signature);changed['implementation']='0'*64
        with self.assertRaisesRegex(ValueError,'not been reviewed'):check_signature(state,changed,True)
        self.assertEqual(state,before)

    def test_checkpoint_moments_rng_gates_and_progress_survive_reviewed_transition(self):
        model,optimizer,cfg,behavior,data,snap,_=mixed_fixture()
        phase=TrainingPhases(model,cfg);phase.snapshot_data=snap
        update=ppo(model,optimizer,cfg,behavior,data,phase);phase.fit(model);finish(phase,update,1)
        previous=dict(config=cfg,compatibility=compatibility(),implementation=PREVIOUS,teachers={},legacy_pool={})
        current=dict(previous,implementation=implementation_id())
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);path=run/'generation-0001.pt'
            identity=save_checkpoint(path,model,optimizer,step=1,training=dict(mode='step5-population-ppo',
                signature=digest(previous),fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),**{KEY:phase.payload()}))
            entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=1)
            state=dict(format='step5-recovery-league/2.2.0',signature=previous,iteration=1,games_done=320,
                previous_games=0,next_seed=18570640,context_counter=960,history=[entry],population=[entry],
                evaluation_anchor=entry,ordinary_seconds=1070.,search_seconds=0.,next_search_games=320)
            write_json(run/'state.json',state);checkpoint_hash=sha(path)
            restored,raw,changed=load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
            self.assertTrue(changed);adam=torch.optim.AdamW(restored.parameters())
            resume_optimizer(raw,adam);apply_optimizer_config(adam,cfg)
            other=TrainingPhases(restored,cfg,raw=raw,generation=1)
            self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()))
            self.assertTrue(_same_payload(optimizer.state_dict()['state'],adam.state_dict()['state']))
            self.assertTrue(_same_payload(phase.payload(),other.payload()))
            self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
            updated=commit_transition(run,state,current)
            for key in state:
                if key!='signature':self.assertEqual(updated[key],state[key])
            self.assertEqual(sha(path),checkpoint_hash)
            # Interruption immediately after the source commit still binds the
            # original generation, so it can resume again without resetting.
            _,_,changed=load_resume(SOURCE_ROOT,run,updated,current)
            self.assertFalse(changed)


if __name__=='__main__':unittest.main()
