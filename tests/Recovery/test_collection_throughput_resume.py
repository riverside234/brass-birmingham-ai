"""Narrow execution migration and exact checkpoint/optimizer continuation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.throughput_config import configure
from brass_ai.recovery.ppo_kl_control import configure_sparse
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition,apply_optimizer_config
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json
from test_compact_il_resume import CompactILResumeTests,BUNDLE


class ThroughputResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_only_reviewed_execution_edits_are_admitted(self):
        state=read(BUNDLE/'state.json');before=copy.deepcopy(state)
        for c in (configure(state['signature']['config']),configure(configure_sparse(state['signature']['config']))):
            requested=dict(state['signature'],implementation=implementation_id(),config=c)
            self.assertTrue(check_signature(state,requested,reviewed=True))
            with self.assertRaises(ValueError):check_signature(state,requested)
            for section,key in [('update','target_kl'),('insolvency','event_cost')]:
                bad=copy.deepcopy(requested);bad['config'][section][key]*=2
                with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        self.assertEqual(state,before)

    def test_full_checkpoint_roundtrip_preserves_weights_optimizer_rng_and_calibration(self):
        fixture=CompactILResumeTests();fixture.original=read(BUNDLE/'state.json');fixture.current=implementation_id()
        self.addCleanup(fixture.doCleanups);run,state,signature=fixture.fixture()
        signature['config']=configure(configure_sparse(signature['config']))
        original_state=copy.deepcopy(state);path=SOURCE_ROOT/state['history'][-1]['path'];checksum=sha(path)
        before=torch.load(path,map_location='cpu',weights_only=True)
        for retry in range(2):
            model,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=retry==0)
            optimizer=torch.optim.AdamW(model.parameters(),lr=signature['config']['learning_rate'])
            resume_optimizer(raw,optimizer);apply_optimizer_config(optimizer,signature['config'])
            self.assertTrue(_same_payload(before['state_dict'],dict(model.state_dict())))
            self.assertTrue(_same_payload(before['optimizer'],optimizer.state_dict()))
            self.assertTrue(_same_payload(before['training'],raw['training']))
            self.assertTrue(torch.equal(before['rng'],torch.get_rng_state()))
            self.assertEqual(model.utility_calibration,before['training']['utility_calibration'])
            self.assertEqual(changed,retry==0);self.assertEqual(sha(path),checksum)
            if retry==0:state=commit_transition(run,state,signature)
        for key in original_state.keys()-{'signature','source_transitions'}:self.assertEqual(state[key],original_state[key])

    def test_configuration_helper_is_retry_safe_and_rejects_pending_work(self):
        spec=importlib.util.spec_from_file_location('throughput_helper',SOURCE_ROOT/'scripts/configure-selfplay-throughput.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0921.pt';checkpoint.write_bytes(b'fixture')
            state=read(BUNDLE/'state.json');state['history'][-1].update(path='artifacts/run/generation-0921.pt',sha256=sha(checkpoint))
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes()
            args=(root,'artifacts/run','configs/step5/throughput.json',921)
            result=helper.generate(*args);self.assertEqual(result,helper.generate(*args))
            self.assertEqual(result['rating_compute']['workers'],16)
            self.assertEqual((run/'state.json').read_bytes(),before)
            (run/'iteration-0921').mkdir();(run/'iteration-0921/partial').write_text('pending')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(*args)


if __name__=='__main__':unittest.main()
