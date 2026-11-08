"""Destination-only reward adoption, exact zero charges and old-state reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.learning import post_transfer_reward as P
from brass_ai.learning.reward import CostLedger,terminal_utility,validate
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.learning.search_value import ordinary_parameters
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.search_value_training import Companion
from brass_ai.recovery.utility import migrate,reviewed_reward_upgrade,reward_spec
from brass_ai.recovery import transfer_resume as R
from brass_ai.rust_backend import lab_shadow_config as C,lab_teacher_activation as A
from brass_ai.step5.league import write_json

EXPORT=ROOT/'results/joint7-v11-2-g1290-trained-restart-g1284-20261108T193427364056Z'


class Rewards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.state=read(EXPORT/'state.json');cls.old=cls.state['signature']['config']

    def result(self,tie=False):
        return dict(objective='win-share/1.0.0',seats=[dict(seat=s,vp=0,income=0,cash=4-(0 if tie and s==1 else s),
            rank=1 if tie and s<2 else s+1,utility=.5 if tie and s<2 else 1. if not tie and s==0 else 0.) for s in range(4)])

    def test_exact_half_placement_config_and_all_search_rewards_match_without_mutating_source(self):
        original=copy.deepcopy(self.old);new=P.revise(self.old);validate_config(new)
        self.assertEqual(self.old,original);self.assertEqual(new['reward'],dict(win=1.,placement=.05,vp=.001,vp_cap=180,vp_timing=P.VP_TIMING))
        self.assertEqual(new['insolvency'],dict(event_cost=0.,max_cost=.025))
        self.assertEqual({k for k in new if new[k]!=self.old[k]},{'reward','insolvency','search','teacher_phase'})
        self.assertEqual(new['search']['reward'],reward_spec(new))
        self.assertEqual(new['teacher_phase']['search']['reward'],reward_spec(new))
        self.assertEqual(new['teacher_phase']['fast_search']['reward'],reward_spec(new))
        self.assertEqual(P.previous(new),self.old);self.assertEqual(P.revise(new),new)
        self.assertTrue(A.is_on(new));self.assertTrue(P.transition(self.old,new))

    def test_unique_and_tied_finish_utilities_and_zero_event_charge_exact(self):
        new=P.specification(True);validate(new)
        np.testing.assert_allclose(terminal_utility(self.result(),new),[1.05,.05*2/3,.05/3,0.],atol=1e-15,rtol=0)
        np.testing.assert_allclose(terminal_utility(self.result(True),new),[.5+.05*2.5/3,.5+.05*2.5/3,.05/3,0.],atol=1e-15,rtol=0)
        ledger=CostLedger(new)
        for _ in range(100):
            for seat in range(4):ledger.charge(seat)
        self.assertEqual(ledger.spent.tolist(),[0.]*4);self.assertEqual(ledger.remaining,[.025]*4)
        old=CostLedger(P.specification(False));old.charge(0);self.assertEqual(old.spent[0],.0025)
        bad=copy.deepcopy(new);bad['insolvency']['max_cost']=0
        with self.assertRaises(ValueError):validate(bad)

    def test_only_exact_requested_reward_upgrade_is_reviewed(self):
        old,new=P.specification(False),P.specification(True)
        self.assertTrue(reviewed_reward_upgrade(old,new));self.assertFalse(reviewed_reward_upgrade(new,old))
        for key,value in [('win',1.01),('placement',.09),('vp',.002)]:
            changed=dict(new,**{key:value});self.assertFalse(reviewed_reward_upgrade(old,changed))

    def test_metadata_migration_preserves_weights_all_adams_and_rng_and_invalidates_readiness(self):
        model=UtilityNet(utility_config(ModelConfig.preset('small'),P.specification(False)));head=Companion(model)
        primary=torch.optim.AdamW(ordinary_parameters(model),lr=1e-5);value=torch.optim.Adam(ordinary_parameters(model),lr=1e-5)
        for optimizer in (primary,value):
            first=next(model.parameters());first.grad=torch.ones_like(first);optimizer.step();optimizer.zero_grad(set_to_none=True)
        for p in model.search_value_head.parameters():p.grad=torch.ones_like(p)
        head.optimizer.step();head.optimizer.zero_grad(set_to_none=True);head.steps=1
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility));model._reviewed_cap_upgrade=True
        before=copy.deepcopy((model.state_dict(),primary.state_dict(),value.state_dict(),head.snapshot(),torch.get_rng_state()))
        report=migrate(model,primary,P.specification(True))
        self.assertEqual(report['version'],P.REVISION);self.assertIsNone(model.utility_calibration)
        self.assertEqual(model.config.utility,P.specification(True))
        self.assertTrue(_same_payload(before,(model.state_dict(),primary.state_dict(),value.state_dict(),head.snapshot(),torch.get_rng_state())))

    def ticket(self,*,gap=False):
        tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(tmp.cleanup)
        root=Path(tmp.name);run=root/'artifacts/step5/joint7-v11-2-g1290-trained-restart';run.mkdir(parents=True)
        state=copy.deepcopy(self.state);state['history'][-1]['path']=(run/'generation-1284.pt').relative_to(root).as_posix()
        if gap:
            before=copy.deepcopy(state['signature']);state['signature']['config']=P.revise(before['config'])
            state['source_transitions'].append(dict(iteration=1284,checkpoint=state['history'][-1],from_signature=before,to_signature=digest(state['signature'])))
        write_json(run/'state.json',state);cfg=root/'resume.json';write_json(cfg,P.revise(state['signature']['config']))
        manifest=dict(id='a'*64,run=run.relative_to(root).as_posix(),config='resume.json',state_sha256=sha(run/'state.json'),
            buffer_rebuild=dict(contract='fresh-buffers/1.0.0',origin_generation=1284,auxiliary_buffer_iterations=8,il_probe_positions=2048,reward_revision=P.REVISION))
        return root,run,state,manifest

    def test_reward_change_requires_restored_checkpoint_instruction_and_survives_transition_gap(self):
        from brass_ai.learning import contracts
        for gap in (False,True):
            with self.subTest(gap=gap):
                root,run,state,manifest=self.ticket(gap=gap)
                old=copy.deepcopy(state['signature']);old['config']=copy.deepcopy(self.old)
                target=dict(old,config=P.revise(old['config']))
                context=dict(state,signature=old)
                with patch.object(contracts,'SOURCE_ROOT',root):
                    self.assertFalse(P.authorized(context,target))
                    R.install_instruction(root,manifest)
                    self.assertTrue(P.authorized(context,target));self.assertTrue(C.approved(context,target))
                record,reset=R.prepare(root,run,state,dict(training={}))
                self.assertTrue(reset);self.assertEqual(record['reward_before'],P.specification(False))
                self.assertEqual(record['reward_after'],P.specification(True))
                # Retry after startup state publication repeats the exact reset.
                self.assertEqual(R.prepare(root,run,state,dict(training={})),(record,True))

    def test_old_label_reward_id_is_retired_with_both_optimizer_and_stage_progress_preserved(self):
        model=UtilityNet(utility_config(ModelConfig.preset('small'),P.specification(False)));head=Companion(model)
        saved=dict(stage=dict(stage='A',stage_a_iterations=4),companion=head.snapshot(),label_ages=[dict(old=True)],
            reward_id=digest(P.specification(False)),buffer=dict(reward_id=digest(P.specification(False)),receipts=[dict(old=True)],retired=[dict(old=True)]))
        original=copy.deepcopy(saved)
        record=dict(origin_generation=1284,instruction_id='x',reward_revision=P.REVISION,reward_before=P.specification(False),reward_after=P.specification(True))
        new=R.search_payload(saved,record)
        self.assertEqual(new['stage']['stage_a_iterations'],4);self.assertEqual(new['buffer']['receipts'],[])
        self.assertEqual(new['reward_id'],digest(P.specification(True)));self.assertEqual(new['label_ages'],[])
        self.assertIs(new['companion'],saved['companion']);self.assertTrue(_same_payload(saved,original))

    def test_completed_entropy_clock_is_preserved_and_incomplete_or_reverse_cost_change_is_rejected(self):
        from brass_ai.recovery import entropy_schedule as S
        new=P.revise(self.old);before=S.restore_run(self.old,self.state)
        after=S.restore(new,before,1284)
        expected=copy.deepcopy(before);expected['insolvency']=P.ZERO_COST
        self.assertEqual(after,expected);self.assertEqual(after['progress'],100)
        self.assertIsNone(S.automatic_receipt(new,self.state))
        updated,report=S.advance(after,new,generation=1285,healthy=True)
        self.assertEqual(updated['progress'],100);self.assertEqual(report['reason'],'complete')
        with self.assertRaisesRegex(ValueError,'post-transfer charge removal'):
            S.restore(self.old,after,1284)
        incomplete=copy.deepcopy(before);incomplete['progress']=99;S.seal(incomplete)
        with self.assertRaisesRegex(ValueError,'post-transfer charge removal'):
            S.restore(new,incomplete,1284)
        edited=copy.deepcopy(new);edited['reward']['placement']=.04
        with self.assertRaisesRegex(ValueError,'exact post-transfer reward'):
            S.validate(edited)

    def test_strict_checkpoint_resume_and_precommit_retry_preserve_training_state(self):
        from brass_ai.learning import contracts
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.learning.batching import model_id
        from brass_ai.recovery.resume import load_resume
        root,run,state,manifest=self.ticket()
        small=ModelConfig.preset('small');model=UtilityNet(utility_config(small,P.specification(False)))
        Companion(model);optimizer=torch.optim.AdamW(ordinary_parameters(model),lr=1e-5)
        first=next(model.parameters());first.grad=torch.ones_like(first);optimizer.step();optimizer.zero_grad(set_to_none=True)
        signature=copy.deepcopy(state['signature']);signature['config']=P.revise(signature['config'])
        path=root/state['history'][-1]['path']
        save_checkpoint(path,model,optimizer,step=1284,training=dict(fresh_run=run.relative_to(root).as_posix(),
            signature=digest(state['signature'])))
        state['history'][-1].update(sha256=sha(path),model_id=model_id(model));state['evaluation_anchor']=state['history'][-1]
        state['population']=[];state['source_transitions']=[];write_json(run/'state.json',state)
        manifest['state_sha256']=sha(run/'state.json');R.install_instruction(root,manifest)
        original_bytes=path.read_bytes();original_weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(optimizer.state_dict())
        authorize=P.authorized
        def authorize_fixture(context,target):
            # Relocate only ticket access; checkpoint compatibility still reads
            # the real project contracts and remains independently strict.
            with patch.object(contracts,'SOURCE_ROOT',root):return authorize(context,target)
        for retry in (False,True):
            working=copy.deepcopy(state)
            if retry:
                working['signature']=signature
                working['source_transitions']=[dict(iteration=1284,checkpoint=working['history'][-1],
                    from_signature=state['signature'],to_signature=digest(signature))]
            with patch.object(P,'authorized',side_effect=authorize_fixture),patch.object(ModelConfig,'preset',return_value=small):
                restored,raw,changed=load_resume(root,run,working,signature,reviewed=True)
            self.assertEqual(changed,not retry)
            resumed=torch.optim.AdamW(ordinary_parameters(restored),lr=1e-5)
            resume_optimizer(raw,resumed);rng=torch.get_rng_state().clone()
            migrate(restored,resumed,P.specification(True))
            self.assertTrue(_same_payload(restored.state_dict(),original_weights))
            self.assertTrue(_same_payload(resumed.state_dict(),adam));self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertIsNone(restored.utility_calibration);self.assertEqual(path.read_bytes(),original_bytes)

    def test_export_queues_new_config_but_restores_original_checkpoint_and_state_bytes(self):
        from test_native_transfer import NativeTransfer,transfer
        fixture=NativeTransfer();fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=fixture.state['signature']['config']
        config.update(reward=copy.deepcopy(P.BEFORE),insolvency=copy.deepcopy(P.OLD_COST),
            search=dict(reward=P.specification(False)))
        fixture.put('artifacts/step5/selfplay/state.json',fixture.state)
        with self.assertRaisesRegex(ValueError,'before exporting'):
            transfer.inventory(fixture.root,fixture.run)
        fixture.put('distribution/rust-lab-g1250/inputs.zip',b'sealed source input fixture')
        original=(fixture.run/'state.json').read_bytes()
        fixture.export();manifest=transfer.inspect_bundle(fixture.bundle)
        self.assertEqual(manifest['buffer_rebuild']['reward_revision'],P.REVISION)
        self.assertEqual((fixture.run/'state.json').read_bytes(),original)
        fixture.move_source();result=transfer.restore(fixture.root,fixture.bundle)
        self.assertEqual((fixture.run/'state.json').read_bytes(),original)
        self.assertEqual(read(fixture.root/result['config'])['reward'],P.AFTER)
        self.assertIn('prepare-rust-post-transfer-reward.py',result['prepare_reward'])
        self.assertEqual(R.instruction(fixture.root,fixture.run)['reward_revision'],P.REVISION)


class NativeSupport(unittest.TestCase):
    def test_exact_core_patch_and_literal_numeric_tests_are_bound_to_recipe(self):
        from brass_ai.rust_backend import lab_reward_runtime as Z
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as t:
            root=Path(t);path=root/'brass-ai-core/src/search/reward.rs';path.parent.mkdir(parents=True)
            path.write_text('header\n'+Z.OLD+Z.OLD_VERSION+'\nfooter\n')
            Z.patch(root);text=path.read_text();self.assertNotIn(Z.OLD,text);self.assertIn(Z.NEW,text)
            self.assertTrue(text.endswith(Z.TESTS));self.assertEqual(text.count('#[test]'),4)
            with self.assertRaisesRegex(ValueError,'patch drift'):Z.patch(root)
        self.assertNotEqual(Z.BASE_CORE,Z.ZERO_CORE);self.assertEqual(len(Z.fingerprint()),64)

    def test_zero_teacher_reuses_exact_parent_even_after_relocation_and_rejects_changed_binary(self):
        from brass_ai.rust_backend import lab_reward_runtime as Z,lab_paths
        from brass_ai.learning.paths import make_relocation
        from brass_ai.rust_backend.lab_contract import pin
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as t:
            parent=Path(t);old=parent/'old';root=parent/'new';root.mkdir();old.mkdir()
            def item(name):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixed '+name.encode())
                return dict(path=str(old/name),bytes=p.stat().st_size,sha256=sha(p))
            verification=item('verification.json');binary=item('teacher');tests=item('tests.log');logs=item('build.log');numeric=item('reward-tests.log')
            original=dict(core_source=Z.BASE_CORE,producer='fixed',bootstrap={},receipt=verification)
            new=dict(original,contract='brass-lab-b07-teacher-build/1.0.0',core_source=Z.ZERO_CORE,implementation=QUALIFIED,binary=binary,tests=tests,build_log=logs,reward_tests=numeric,
                reward_support=dict(contract=Z.CONTRACT,source=Z.fingerprint(),base_core_source=Z.BASE_CORE,core_source=Z.ZERO_CORE),
                original_verification_receipt=verification,wire_regressions_passed=True)
            write_json(root/Z.SELECTOR,new);write_json(root/'.step5-relocation.json',make_relocation(root,[str(old)],'b'*64))
            with patch.object(lab_paths,'SOURCE_ROOT',root):
                moved=dict(original,receipt=pin(root/'verification.json'))
                self.assertEqual(Z.resolve(root,moved)['core_source'],Z.ZERO_CORE)
                (root/'teacher').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'evidence changed'):Z.resolve(root,moved)


if __name__=='__main__':unittest.main()
