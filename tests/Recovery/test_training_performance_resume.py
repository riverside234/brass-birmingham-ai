"""Exact v7 continuation through the IL/self-play performance-only revision."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.curriculum import activate_boost
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.recovery.source_reviews import expand_source_reviews
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import implementation_id,write_json


PREVIOUS='34a62056d352342da5d33662b34e4c35eeb11146c5d1a5be4e6ee3cc6a7141f7'
PRE_V7='c3e45029ffbbfab1d3a0410d4cd5eb597ff7c3275ede907882132393cdcadcc4'


class TrainingPerformanceResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def signatures(self):
        config=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');config['model_size']='small'
        current=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),
                     legacy_pool={},teachers={})
        previous=copy.deepcopy(current);previous['implementation']=PREVIOUS
        return previous,current

    def state(self,signature,entry=None):
        entry=entry or dict(path='unused.pt',sha256='0'*64,model_id='0'*64,generation=70)
        state=dict(format='step5-recovery-league/2.2.0',signature=signature,iteration=70,
            games_done=0,next_seed=18570000,context_counter=876,search_seed_cursor=18970000,
            next_search_games=22560,ordinary_seconds=12345.,search_seconds=54321.,
            history=[entry],population=[entry],evaluation_anchor=entry,resume=[entry])
        activate_boost(state,signature['config']);state['games_done']=22400
        return state

    def fixture(self):
        temporary=tempfile.TemporaryDirectory(prefix='v7-performance-resume-',dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(temporary.cleanup);run=Path(temporary.name).resolve()
        previous,current=self.signatures()
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(previous['config'])))
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility),scope='fixture')
        optimizer=torch.optim.AdamW(model.parameters(),lr=previous['config']['learning_rate'])
        sum(parameter.square().mean() for parameter in model.parameters()).backward();optimizer.step()
        path=run/'generation-0070.pt'
        identity=save_checkpoint(path,model,optimizer,step=70,training=dict(mode='step5-population-ppo',
            fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(previous)))
        entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=70)
        state=self.state(previous,entry);write_json(run/'state.json',state)
        return run,state,current,model,optimizer

    def test_existing_approval_prefixes_and_definitions_are_unchanged(self):
        # Digests pin the normalized JSON bytes captured immediately before this
        # review, including compact receipt references and every prior definition.
        raw=read(SOURCE_ROOT/'configs/learning/selfplay-resume-sources.json')
        expanded=expand_source_reviews(raw)['transitions']
        self.assertEqual(digest(expanded[:2138]),
            '8dc350d87f62ff57daf1f2d516e570c8b34072cdaf667c11f00e20df6d8798b6')
        counts=dict(reason=93,config_changes=78,config_migrations=13,config_migration=3,learning_compatibility=1)
        preserved=dict(format=raw['format'],transitions=raw['transitions'][:2138],
            definitions={name:{str(index):raw['definitions'][name][str(index)] for index in range(count)}
                         for name,count in counts.items()})
        self.assertEqual(digest(preserved),'54ce35de5abfd1a267bb72fcb003c246aba02b1177a0194f8e7ef2ed7cd34b97')
        origins=read(SOURCE_ROOT/'configs/learning/imitation-handoff-sources.json')['compatible_sources']
        self.assertEqual(digest(origins[:36]),
            '1ae769e2cf911da7d2c46dc3d5c75e6f56edf5eb0b19ea3812068aa8ea954bd2')
        added=[entry for entry in expanded if entry['to_implementation']==implementation_id()]
        self.assertEqual(len(added),1)
        self.assertEqual(added[0]['from_implementation'],PREVIOUS)
        self.assertEqual(added[0]['learning_compatibility'],digest(compatibility()))
        for key in ('config_changes','config_migration','config_migrations'):
            self.assertFalse(added[0].get(key))
        self.assertTrue(any(entry['implementation']==PREVIOUS and
                            entry['learning_compatibility']==digest(compatibility()) for entry in origins))

    def test_reviewed_commit_preserves_weights_adam_rng_population_and_counters(self):
        run,state,current,original,optimizer=self.fixture()
        checkpoint=run/'generation-0070.pt';before=checkpoint.read_bytes()
        old_state_bytes=(run/'state.json').read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
        self.assertTrue(changed)
        self.assertEqual((run/'state.json').read_bytes(),old_state_bytes)
        restored=torch.optim.AdamW(model.parameters(),lr=current['config']['learning_rate'])
        resume_optimizer(raw,restored)
        self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        expected,actual=optimizer.state_dict(),restored.state_dict()
        self.assertEqual(expected['param_groups'],actual['param_groups'])
        for key,values in expected['state'].items():
            for name,value in values.items():torch.testing.assert_close(value,actual['state'][key][name],rtol=0,atol=0)
        self.assertTrue(raw['training']['utility_calibration']['ready'])
        self.assertFalse(model.utility_calibration['ready'])
        updated=commit_transition(run,state,current)
        for key,value in state.items():
            if key!='signature':self.assertEqual(value,updated[key],key)
        self.assertEqual(updated['source_transitions'][-1]['config_changes'],{})
        self.assertEqual(checkpoint.read_bytes(),before)
        restarted,again,changed=load_resume(SOURCE_ROOT,run,updated,current)
        self.assertFalse(changed);self.assertFalse(restarted.utility_calibration['ready'])
        self.assertEqual(again['model_id'],raw['model_id'])
        self.assertEqual(checkpoint.read_bytes(),before)

    def test_review_flag_settings_unknown_sources_and_old_contracts_are_rejected(self):
        previous,current=self.signatures();state=self.state(previous)
        self.assertTrue(check_signature(state,current,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):
            check_signature(state,current)
        for key in ('learning_rate','search_every_games','seed','exploration'):
            changed=copy.deepcopy(current);changed['config'][key]+=1
            with self.subTest(setting=key),self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)
        for source in ('0'*64,PRE_V7):
            old=copy.deepcopy(state);old['signature']['implementation']=source
            with self.subTest(source=source),self.assertRaisesRegex(ValueError,'not been reviewed'):
                check_signature(old,current,reviewed=True)
        changed=copy.deepcopy(current);changed['implementation']='0'*64
        with self.assertRaisesRegex(ValueError,'not been reviewed'):check_signature(state,changed,reviewed=True)
        changed=copy.deepcopy(current);changed['config']['selfplay_recipe']='different-recipe'
        with self.assertRaisesRegex(ValueError,'fresh IL run'):check_signature(state,changed,reviewed=True)
        changed=copy.deepcopy(current);changed['config']['architecture']='brass-four-seat-mlp-resnet/4.0.0'
        with self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)
        changed=copy.deepcopy(current);changed['compatibility']['architecture_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'learning contracts'):check_signature(state,changed,reviewed=True)
        old=copy.deepcopy(state);old['signature']['compatibility']=copy.deepcopy(changed['compatibility'])
        with self.assertRaisesRegex(ValueError,'not been reviewed'):check_signature(old,changed,reviewed=True)

    def test_uncommitted_collection_and_checkpoint_are_not_reused_or_modified(self):
        run,state,current,_,_=self.fixture()
        pending=run/'iteration-0070';pending.mkdir()
        partial=pending/'search-000000.jsonl.gz';partial.write_bytes(b'prior source diagnostic labels')
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):
            load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
        self.assertEqual(partial.read_bytes(),b'prior source diagnostic labels')
        partial.unlink();pending.rmdir()
        uncommitted=run/'generation-0071.pt';uncommitted.write_bytes(b'uncommitted checkpoint')
        with self.assertRaisesRegex(ValueError,'Uncommitted checkpoint'):
            load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
        self.assertEqual(uncommitted.read_bytes(),b'uncommitted checkpoint')

    def test_source_review_does_not_admit_an_old_checkpoint_architecture(self):
        run,state,current,_,_=self.fixture();path=run/'generation-0070.pt'
        raw=torch.load(path,weights_only=True)
        raw['model_config']['architecture']='brass-four-seat-mlp-resnet/4.0.0'
        torch.save(raw,path);state['history'][-1]['sha256']=sha(path);before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'fresh version-9'):
            load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
        self.assertEqual(path.read_bytes(),before)

    def test_completed_v7_il_source_handoff_is_accepted_without_rewriting_weights(self):
        with tempfile.TemporaryDirectory(prefix='v7-il-source-',dir=SOURCE_ROOT/'artifacts') as name:
            run=Path(name).resolve();manifest=run/'manifest.json';write_json(manifest,{})
            model=FourSeatNet(ModelConfig.preset('small'))
            signature=dict(format='bc-training/2.2.0',initialization='random',implementation=PREVIOUS,
                architecture=model.config.to_dict(),epochs=1,
                manifest_path=manifest.relative_to(SOURCE_ROOT).as_posix(),manifest_sha256=sha(manifest))
            checkpoint=run/'epoch-0001.pt'
            identity=save_checkpoint(checkpoint,model,step=1,training=dict(mode='step5-v3-bc',
                initialization='random',parent=None,fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),
                signature=digest(signature),manifest_sha256=sha(manifest)))
            entry=dict(path=checkpoint.name,sha256=sha(checkpoint))
            state=dict(signature=signature,epoch=1,stopped=False,best=entry,last=entry)
            report=dict(format='bc-training-report/1.0.0',signature=signature,epochs=1,candidate_sha256=entry['sha256'])
            write_json(run/'state.json',state);write_json(run/'report.json',report)
            before={path.name:path.read_bytes() for path in (checkpoint,run/'state.json',run/'report.json')}
            rng=torch.get_rng_state().clone();restored,origin=completed_imitation(SOURCE_ROOT,run)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(origin['model_id'],identity)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(before,{name:(run/name).read_bytes() for name in before})
            signature['implementation']='0'*64
            write_json(run/'state.json',state);write_json(run/'report.json',report)
            with self.assertRaisesRegex(ValueError,'reviewed compatible source'):completed_imitation(SOURCE_ROOT,run)


if __name__=='__main__':unittest.main()
