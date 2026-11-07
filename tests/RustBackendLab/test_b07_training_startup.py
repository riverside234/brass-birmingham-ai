"""Public training-entry regressions; stop before models, collection or updates."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.recovery import entropy_schedule as S,league
from brass_ai.rust_backend import lab_replay_compat as R
from brass_ai.step5.league import implementation_id,write_json


class ReachedTrainer(Exception):
    pass


class Startup(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)
        self.config=read(ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json')
        self.assertTrue(S.automatic(self.config))
        c=copy.deepcopy(self.config['update'][S.KEY])
        schedule=S.seal(dict(contract=S.STATE,config=c,started_at=c['start_generation'],progress=100,
            held=0,last_generation=1250,insolvency=copy.deepcopy(self.config['insolvency']),
            insolvency_steps_applied=S.step_count(c,self.config['insolvency']),reset_count=0))
        S.validate_state(schedule,1250)
        self.state=dict(format='step5-recovery-league/2.2.0',iteration=1250,games_done=400000,
            signature=dict(config=copy.deepcopy(self.config),implementation=R.qualified_implementation(),
                imitation_source=copy.deepcopy(self.config['imitation_checkpoint'])),
            native_backend=dict(mode='rust',implementation=R.qualified_implementation()),
            exploration={S.KEY:schedule})
        boost=self.config.get('legacy_boost')
        if boost:
            self.state['legacy_boost']=dict(start_games=0,
                until_games=sum(s['games'] for s in boost['stages']),config=digest(boost))
        write_json(self.run/'state.json',self.state)
        # An interrupted iteration must survive the public entry/lock/precheck.
        self.pending=self.run/'iteration-1250/native-ordinary/replay/frozen.jsonl.gz'
        self.pending.parent.mkdir(parents=True);self.pending.write_bytes(b'unchanged pending replay fixture')
        self.calls=[]

    def stop(self,root,run,config,**kwargs):
        self.calls.append((root,run,copy.deepcopy(config),kwargs))
        # Exercise the inner trainer's shared signature handling too, while
        # deliberately not running its model loading or any training work.
        incoming=dict(self.state['signature'],config=config,implementation=implementation_id())
        if 'native_backend' in self.state:
            self.assertEqual(R.training_signature(self.state,incoming),self.state['signature'])
        raise ReachedTrainer('stopped before collection')

    def invoke(self,config=None,**overrides):
        args=dict(iterations=1,device='cpu',expected_generation=1250,
            imitation_run=self.config['imitation_checkpoint']['bc_run'])
        args.update(overrides)
        with patch.object(league,'_train',side_effect=self.stop):
            return league.train(ROOT,str(self.run),self.config if config is None else config,**args)

    def unchanged(self,before):
        self.assertEqual((self.run/'state.json').read_bytes(),before)
        self.assertEqual(self.pending.read_bytes(),b'unchanged pending replay fixture')
        self.assertFalse((self.run/'.active-run.json').exists())

    def test_public_entry_with_actual_g1250_config_reaches_trainer_and_retries_unchanged(self):
        before=(self.run/'state.json').read_bytes()
        for _ in range(2):
            with self.assertRaises(ReachedTrainer):self.invoke()
            self.unchanged(before)
        self.assertEqual(len(self.calls),2)
        for _,_,config,kwargs in self.calls:
            self.assertEqual(config,self.config)
            self.assertEqual(kwargs['iterations'],1)
            self.assertFalse(kwargs['resume_reviewed_source'])
            self.assertEqual(kwargs['imitation_run'],self.config['imitation_checkpoint']['bc_run'])

    def test_continuation_command_through_real_cli_and_scheduler_reaches_trainer(self):
        from brass_ai.step5 import cli
        import torch
        spec=importlib.util.spec_from_file_location('b07_startup_launcher',ROOT/'scripts/continue-rust-lab-b07.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        config_path=self.run/'committed-config.json';write_json(config_path,self.config)
        argv=launcher.training_command(ROOT,self.run,config_path,1250,'mono',state=self.state)
        argv[argv.index('--device')+1]='cpu'
        before=(self.run/'state.json').read_bytes();threads=torch.get_num_threads()
        try:
            # The private engine asset gate is unrelated to this fixture; all
            # CLI/config, lock, generation and scheduler checks execute normally.
            with patch.object(cli,'require_preflight'),patch.object(league,'_train',side_effect=self.stop):
                with self.assertRaises(ReachedTrainer):cli.main(argv[2:])
        finally:torch.set_num_threads(threads)
        self.assertEqual(len(self.calls),1);self.unchanged(before)

    def test_unreviewed_execution_stops_before_trainer_even_with_override(self):
        before=(self.run/'state.json').read_bytes()
        for reviewed in (False,True):
            with patch.object(R,'implementation_id',return_value='f'*64):
                with self.assertRaisesRegex(ValueError,'exact replay-location review'):
                    self.invoke(resume_reviewed_source=reviewed)
            self.unchanged(before)
        self.assertEqual(self.calls,[])

    def test_changed_recipe_stops_before_trainer_and_preserves_pending_records(self):
        before=(self.run/'state.json').read_bytes();config=copy.deepcopy(self.config)
        config['learning_rate']*=2
        with self.assertRaisesRegex(ValueError,'exact replay-location review'):self.invoke(config)
        self.assertEqual(self.calls,[]);self.unchanged(before)

    def test_missing_backend_cannot_claim_reviewed_native_resume(self):
        self.state.pop('native_backend');write_json(self.run/'state.json',self.state)
        before=(self.run/'state.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'configuration/source changed'):self.invoke()
        self.assertEqual(self.calls,[]);self.unchanged(before)

    def test_stale_generation_stops_before_scheduler_or_trainer(self):
        before=(self.run/'state.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'Expected committed generation 1251'):
            self.invoke(expected_generation=1251)
        self.assertEqual(self.calls,[]);self.unchanged(before)

    def test_current_reference_source_still_reaches_trainer(self):
        self.state.pop('native_backend');self.state['signature']['implementation']=implementation_id()
        write_json(self.run/'state.json',self.state);before=(self.run/'state.json').read_bytes()
        with self.assertRaises(ReachedTrainer):self.invoke()
        self.assertEqual(len(self.calls),1);self.unchanged(before)


if __name__=='__main__':unittest.main()
