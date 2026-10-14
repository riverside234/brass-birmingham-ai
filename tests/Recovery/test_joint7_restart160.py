"""Exact rollback bytes, fail-closed restore and audited counter reconstruction."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.retention import run_lock,slim_checkpoint,_same_payload
from brass_ai.recovery.resume import load_resume
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json,implementation_id

sys.path.insert(0,str(SOURCE_ROOT/'scripts'))
from joint7_restart160 import reconstruct,resolve_models,publish,EXPORT,TARGET_SHA


class RestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_export_reconstructs160_and_cross_checks170(self):
        state,history,receipt=reconstruct(SOURCE_ROOT/EXPORT)
        self.assertEqual(state['iteration'],160);self.assertEqual(state['games_done'],51200)
        self.assertEqual(state['next_seed'],18575575);self.assertEqual(state['next_search_games'],51360)
        self.assertEqual(state['history'][-1]['sha256'],TARGET_SHA)
        self.assertEqual(len(state['population']),64)
        self.assertTrue(all(e['generation']<=160 for e in state['population']))
        self.assertLess(receipt['counter_max_error'],1e-6)
        self.assertEqual(state['signature']['config']['insolvency']['max_cost'],.025)
        self.assertNotIn('champion_mix',state)
        self.assertTrue(all(v['summary']['games_done']<=51200 for v in history.values()))

    def fixture(self):
        tmp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(tmp.cleanup)
        parent=Path(tmp.name);run=parent/'league';run.mkdir();backup=parent/'backup';backup.mkdir()
        cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json');cfg['model_size']='small'
        signature=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(cfg)))
        opt=torch.optim.AdamW(model.parameters());entries=[]
        for gen in (0,160):
            sum(p.square().mean() for p in model.parameters()).backward();opt.step();opt.zero_grad()
            path=run/f'generation-{gen:04d}.pt'
            identity=save_checkpoint(path,model,opt,step=gen,training=dict(signature=digest(signature),fresh_run=run.relative_to(SOURCE_ROOT).as_posix()))
            entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=gen))
        slim=slim_checkpoint(SOURCE_ROOT,run,entries[-1])
        state=dict(format='step5-recovery-league/2.2.0',signature=signature,iteration=160,games_done=51200,
            previous_games=0,next_seed=18575575,next_search_games=51360,context_counter=51200,
            ordinary_seconds=105472.,search_seconds=70394.,history=entries,population=[entries[0],slim],
            evaluation_anchor=entries[0],champion=slim,champion_gate=dict(accepted_games=51200,recent=[],accepted_gate=None),evaluations=[])
        later=copy.deepcopy(state);later.update(iteration=179,games_done=57280)
        write_json(run/'state.json',later);future=run/'iteration-0160';future.mkdir();(future/'report.json').write_bytes(b'future evidence')
        receipt=dict(source_state_digest=digest(later),checkpoint=entries[-1],target_generation=160)
        return run,backup,state,receipt

    def test_full_optimizer_missing_is_not_replaced_by_best_weights(self):
        run,backup,state,receipt=self.fixture();full=SOURCE_ROOT/state['history'][-1]['path'];saved=full.read_bytes()
        full.unlink()
        files,missing=resolve_models(SOURCE_ROOT,run,state)
        self.assertIn(state['history'][-1]['path'],missing)
        (backup/full.name).write_bytes(b'wrong checkpoint')
        _,missing=resolve_models(SOURCE_ROOT,run,state,checkpoint_backup=backup/full.name)
        self.assertIn(state['history'][-1]['path'],missing)
        (backup/full.name).write_bytes(saved)
        files,missing=resolve_models(SOURCE_ROOT,run,state,checkpoint_backup=backup/full.name)
        self.assertFalse(missing)

    def test_swap_preserves_original_bytes_adam_rng_lineage_and_future_evidence(self):
        run,backup,state,receipt=self.fixture();full=SOURCE_ROOT/state['history'][-1]['path'];saved=full.read_bytes()
        before=read(run/'state.json');old_state_bytes=(run/'state.json').read_bytes()
        files,missing=resolve_models(SOURCE_ROOT,run,state);self.assertFalse(missing)
        with run_lock(run):result=publish(SOURCE_ROOT,run,state,{},files,receipt)
        archive=SOURCE_ROOT/result['archive']
        self.assertEqual((archive/'state.json').read_bytes(),old_state_bytes)
        self.assertEqual((archive/'iteration-0160/report.json').read_bytes(),b'future evidence')
        self.assertFalse((run/'iteration-0160').exists());self.assertFalse((archive/'.active-run.json').exists())
        self.assertEqual(full.read_bytes(),saved);self.assertEqual((archive/full.name).read_bytes(),saved)
        restored=read(run/'state.json');self.assertEqual(restored['iteration'],160)
        model,raw,changed=load_resume(SOURCE_ROOT,run,restored,state['signature'])
        self.assertFalse(changed)
        _,old=load_checkpoint(archive/full.name);self.assertTrue(_same_payload(raw,old))
        opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
        self.assertTrue(_same_payload(raw['optimizer'],opt.state_dict()));self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
        self.assertEqual(restored['history'],state['history']);self.assertEqual(restored['population'],state['population'])
        self.assertEqual(read(run/'best.json')['checkpoint'],state['champion'])

    def test_live_progress_change_cannot_be_overwritten(self):
        run,backup,state,receipt=self.fixture();files,missing=resolve_models(SOURCE_ROOT,run,state)
        later=read(run/'state.json');later['iteration']=180;write_json(run/'state.json',later)
        with run_lock(run):
            with self.assertRaisesRegex(ValueError,'advanced'):publish(SOURCE_ROOT,run,state,{},files,receipt)
        self.assertEqual(read(run/'state.json')['iteration'],180)


if __name__=='__main__':unittest.main()
