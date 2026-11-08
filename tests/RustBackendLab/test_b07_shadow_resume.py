"""Real checkpoint-loader coverage for the exact 2->4 diagnostic transition."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,compatibility,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery import resume,teacher_gradient_qualification as Q
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.teacher_term import TeacherTerm
from brass_ai.recovery.utility import reward_spec
from brass_ai.rust_backend import lab_shadow_config as C,lab_replay_compat as R,lab_backend as B
from brass_ai.step5.league import write_json


class PendingResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        base=read(ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json')
        config=C.configure(base)
        self.after=dict(config=config,implementation=R.QUALIFIED,compatibility=compatibility(),
            imitation_source={'bc_run':'unchanged'},teachers={},legacy_pool={})
        before=copy.deepcopy(self.after);before['config']['teacher_phase']['term']['shadow_iterations']=2
        self.before=before
        self.preset=ModelConfig.preset('small')
        self.model=UtilityNet(utility_config(self.preset,reward_spec(config)))
        self.model.utility_calibration=dict(ready=True,reward_id=digest(reward_spec(config)),version='debug-fit')
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config['learning_rate'])
        sum(p.square().mean() for p in self.model.parameters()).backward()
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        observation=dict(generation=1257,steps=1,valid_steps=1,raw_shares=[1.75],cosines=[.2],
            actor_norms=[1.],teacher_norms=[1.75],context=dict(tables={'train':28,'heldout':4},
                labels_by_phase={p:128 for p in ('canal_early','canal_late','rail_early','rail_late')}))
        self.phase=dict(contract='teacher-phase/1.1.0',generation=1258,teacher_seed_cursor=40,
            weight=dict(mode='shadow',beta=1.,calibrated=False,share_target=.25,shadow_shares=[[1.75]],outside_streak=0,
                gradient_qualification=Q.seal(dict(contract=Q.CONTRACT,passed=False,target=.25,
                    observations=[observation],reason='Need two observations',beta=None,raw_median=None))))
        rollback=self.run/'rollback.json';rollback.write_bytes(b'{}')
        backend=dict(contract=B.CONTRACT,mode='rust',implementation=R.QUALIFIED,config_sha256=digest(before['config']),
            shadow=dict(contract=C.CONTRACT,tables={'train':28,'held_out':4}),build={'unchanged':'runtime'},
            qualification={'unchanged':'qualification'},rollback=dict(path=str(rollback),sha256=sha(rollback)))
        self.checkpoint=self.run/'generation-1258.pt'
        self.previous_execution=next(e for e in read(R.REVIEW)['predecessor_executions']
            if e['implementation']=='3e1de866b56e821c9d8492152c27afff924ffb52fe2c7e75089c45a7a69f0c98')
        identity=save_checkpoint(self.checkpoint,self.model,self.optimizer,step=1258,training=dict(
            mode='step5-population-ppo',fresh_run=self.run.relative_to(ROOT).as_posix(),signature=digest(before),
            teacher_phase=self.phase,native_backend=backend,native_backend_execution=self.previous_execution))
        entry=dict(path=self.checkpoint.relative_to(ROOT).as_posix(),sha256=sha(self.checkpoint),model_id=identity,generation=1258)
        self.state=dict(format='step5-recovery-league/2.2.0',signature=before,iteration=1258,games_done=402560,
            history=[entry],population=[entry],evaluation_anchor=entry,native_backend=backend,
            teacher_phase=copy.deepcopy(self.phase),teacher_seed_cursor=40)
        write_json(self.run/'state.json',self.state)
        self.pending=self.run/'iteration-1258';self.pending.mkdir()
        for name,data in [('native-ordinary/replay/frozen.gz',b'original completed family'),
                          ('teacher-shadow/table.json.gz',b'original completed teacher table')]:
            path=self.pending/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        self.files=self.pending_bytes();self.state_bytes=(self.run/'state.json').read_bytes()

    def pending_bytes(self):
        return {p.relative_to(self.pending).as_posix():p.read_bytes() for p in self.pending.rglob('*') if p.is_file()}

    def load(self,signature=None,*,state=None,reviewed=True):
        # Bound the fixture's allocation only. Signature checks, the real strict
        # checkpoint loader, pending guards and source-transition code all run.
        with patch.object(resume.ModelConfig,'preset',return_value=self.preset):
            return resume.load_resume(ROOT,self.run,self.state if state is None else state,
                self.after if signature is None else signature,reviewed=reviewed)

    def unchanged(self):
        self.assertEqual(self.pending_bytes(),self.files)
        self.assertEqual((self.run/'state.json').read_bytes(),self.state_bytes)

    def test_real_loader_restores_pending_upgrade_with_exact_weights_adam_rng_and_first_measurement(self):
        model,raw,changed=self.load()
        self.assertTrue(changed);self.unchanged()
        self.assertTrue(_same_payload(model.state_dict(),self.model.state_dict()))
        self.assertEqual(model.utility_calibration,self.model.utility_calibration)
        optimizer=torch.optim.AdamW(model.parameters());resume_optimizer(raw,optimizer)
        self.assertTrue(_same_payload(optimizer.state_dict(),self.optimizer.state_dict()))
        self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
        self.assertEqual(raw['training']['teacher_phase'],self.phase)
        term=TeacherTerm(model,self.after['config']['teacher_phase'],[],seed=7,generation=1258,state=self.phase['weight'])
        self.assertFalse(term.qualification['passed']);self.assertEqual(term.qualification['required_iterations'],4)
        self.assertEqual(term.qualification['observations'],self.phase['weight']['gradient_qualification']['observations'])

    def test_transition_commit_and_crash_retry_keep_old_checkpoint_and_pending_bindings(self):
        model,raw,changed=self.load();self.assertTrue(changed)
        optimizer=torch.optim.AdamW(model.parameters());resume_optimizer(raw,optimizer)
        checkpoint_bytes=self.checkpoint.read_bytes()
        with patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value={'receipt':{'existing':'prepared teacher'}}):
            updated=resume.commit_transition(self.run,self.state,self.after)
        self.assertEqual(updated['teacher_phase'],self.phase)
        self.assertEqual(updated['native_backend']['config_sha256'],digest(self.after['config']))
        # Exercise real checkpoint/backend transition binding; the private lab
        # qualification receipt is the sole external dependency in this fixture.
        with patch.object(B,'checked_pin',return_value={'passed':True,'implementation':R.QUALIFIED}):
            B.validate_state(updated,raw,check_runtime=False)
        for _ in range(2):
            reloaded,retry,revised=self.load(state=updated)
            self.assertFalse(revised);self.assertEqual(retry['training']['teacher_phase'],self.phase)
            self.assertTrue(_same_payload(reloaded.state_dict(),model.state_dict()))
            self.assertEqual(reloaded.utility_calibration,self.model.utility_calibration)
            self.assertEqual(self.pending_bytes(),self.files)
            self.assertEqual(self.checkpoint.read_bytes(),checkpoint_bytes)

    def test_next_checkpoint_is_still_rejected_without_touching_pending_work(self):
        path=self.run/'generation-1259.pt.partial';path.write_bytes(b'uncommitted checkpoint')
        with self.assertRaisesRegex(ValueError,'Uncommitted checkpoint'):self.load()
        self.unchanged();self.assertEqual(path.read_bytes(),b'uncommitted checkpoint')

    def test_pending_file_is_never_accepted_as_a_directory(self):
        self.pending.rename(self.run/'retained-pending');self.pending.write_bytes(b'file, not a directory')
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):self.load()
        self.assertEqual(self.pending.read_bytes(),b'file, not a directory')

    def test_target_change_still_requires_finishing_pending_work(self):
        target=copy.deepcopy(self.after);target['config']['teacher_phase']['term']['share_target']=.4
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):self.load(target)
        self.unchanged()

    def test_unreviewed_config_source_missing_permission_and_backend_are_rejected(self):
        bad=copy.deepcopy(self.after);bad['config']['learning_rate']*=2
        source=copy.deepcopy(self.after);source['implementation']='f'*64
        for signature in (bad,source):
            with self.subTest(signature=signature['implementation']),self.assertRaises(ValueError):self.load(signature)
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):self.load(reviewed=False)
        for mode in ('reference',None):
            state=copy.deepcopy(self.state)
            if mode:state['native_backend']['mode']=mode
            else:state.pop('native_backend')
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.load(state=state)
        self.unchanged()

    def test_checkpoint_hash_and_signature_still_reject_tampering(self):
        broken=copy.deepcopy(self.state);broken['history'][-1]['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'Snapshot changed'):self.load(state=broken)
        raw=torch.load(self.checkpoint,weights_only=True);raw['training']['signature']='f'*64
        torch.save(raw,self.checkpoint);broken=copy.deepcopy(self.state)
        broken['history'][-1]['sha256']=sha(self.checkpoint)
        with self.assertRaisesRegex(ValueError,'not bound to this run signature'):self.load(state=broken)
        self.unchanged()

    def test_unreviewed_execution_cannot_claim_the_diagnostic_pending_exception(self):
        with patch.object(R,'implementation_id',return_value='f'*64),self.assertRaises(ValueError):self.load()
        self.unchanged()


if __name__=='__main__':unittest.main()
