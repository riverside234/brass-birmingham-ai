import copy
import gzip
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,assert_compatible,read_json,digest
from brass_ai.learning.features import Encoder,Encoded
from brass_ai.learning.batching import collate,predict_encoded,model_id
from brass_ai.learning.model import ModelConfig,FourSeatNet,ARCHITECTURE
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.replay import check_game,write_shard,read_shards,cache_receipt,load_cache,cache_version
from brass_ai.recovery.updates import Dataset,prepared
from brass_ai.recovery.shared_transport import SharedSlot
from brass_ai.recovery.utility import reward_spec
from brass_ai.feature_engineering.contracts import STATE_FIELDS,ACTION_FIELDS,specification
from brass_ai.feature_engineering.data import raw_compatible,convert
from brass_ai.bank import sha
from fixtures import game,rotate
from fe_fixtures import benchmark_positions


def legacy_contract():
    raw=compatibility();raw.pop('feature_version');raw.pop('derived_source_sha256')
    raw.update(learning_contract='four-seat-learning/1.0.0',
        features_sha256=digest(read_json(SOURCE_ROOT/'configs/learning/features-1.0.0.json')),
        encoder_sha256='a'*64,architecture_sha256='b'*64)
    return raw


class FeatureSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.positions=benchmark_positions()

    def test_reference_optimized_parity_dynamic_candidates_context_and_all_seats(self):
        reference=Encoder();fast=PreparedEncoder()
        self.assertEqual((reference.state_dim,reference.action_dim),(3128,327))
        for row in self.positions:
            for shift in range(4):
                view=rotate(row['view'],shift);choices=row['choices'];style=[.2,0,0,0];intent=[0,1,0,0,0,0,0,0]
                a=reference.encode(view,choices,style,intent);b=fast.encode(view,choices,style,intent)
                for name in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
                self.assertEqual(a.choice_ids,tuple(c['id'] for c in choices))
                np.testing.assert_array_equal(b.state[-12:],np.asarray(style+intent,dtype=np.float32))
                self.assertTrue(np.isfinite(b.state).all() and np.isfinite(b.actions).all())
                self.assertIs(fast.encode(view,choices,style,intent),b)
                self.assertFalse(b.state.flags.writeable)

    def test_state_groups_reach_all_state_heads_and_action_block_is_not_categorical_cache(self):
        row=self.positions[0];fast=PreparedEncoder();a=fast.encode(row['view'],row['choices'])
        view=copy.deepcopy(row['view']);view['players'][view['seat']]['cash']+=19
        b=fast.encode(view,row['choices'])
        np.testing.assert_array_equal(a.actions[:,:184],b.actions[:,:184])
        self.assertFalse(np.array_equal(a.actions[:,184:],b.actions[:,184:]))
        model=UtilityNet(utility_config(ModelConfig.preset(),reward_spec(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json'))))
        # The same augmented state embedding feeds policy, official and utility heads.
        self.assertEqual(model.state_encoder.input_dim,3128)
        self.assertEqual(model.action_encoder[0].in_features,327)
        with torch.no_grad():outputs=model(*collate([a,b]))
        for values in outputs[1:]:self.assertEqual(tuple(values.shape),(2,4))
        self.assertFalse(torch.equal(outputs[1][0],outputs[1][1]))

    def test_frozen_layout_width_and_explicit_parameter_budgets(self):
        schema=read_json(SOURCE_ROOT/'configs/learning/features-4.0.0.json')
        self.assertEqual(schema['derived'],json.loads(json.dumps(specification())))
        self.assertEqual((len(STATE_FIELDS),len(ACTION_FIELDS)),(326,52))
        model=FourSeatNet();self.assertEqual(model.config.hidden,512)
        self.assertEqual(model.parameter_count,5904644);self.assertLess(model.parameter_count,model.config.parameter_limit)
        self.assertEqual(model.config.parameter_limit,6500000)
        bad=model.config.to_dict();bad['parameter_limit']=2000000
        with self.assertRaisesRegex(ValueError,'budget'):ModelConfig.from_dict(bad)
        bad=model.config.to_dict();bad['architecture']='brass-four-seat-mlp-resnet/1.0.0'
        with self.assertRaisesRegex(ValueError,'old architectures'):ModelConfig.from_dict(bad)

    def test_terminal_missing_source_information_private_fields_and_deadline_fail_explicitly(self):
        row=self.positions[0];fast=PreparedEncoder();view=copy.deepcopy(row['view'])
        view['context']['terminal']=True
        with self.assertRaises(ValueError):fast.encode(view,[])
        view=copy.deepcopy(row['view']);del view['players'][0]['mat']
        with self.assertRaises(Exception):fast.encode(view,row['choices'])
        view=copy.deepcopy(row['view']);view['opponent_hands']={}
        with self.assertRaises(Exception):fast.encode(view,row['choices'])
        with self.assertRaisesRegex(ValueError,'deadline'):fast.encode(row['view'],row['choices'],deadline=time.monotonic()-1)

    def test_mixed_old_arrays_and_versions_rejected_before_batch_or_update(self):
        row=self.positions[0];current=PreparedEncoder().encode(row['view'],row['choices'])
        for old in (Encoded(current.state[:2539],current.actions[:,:184],current.stable_to_relative,current.choice_ids),
                    Encoded(current.state,current.actions,current.stable_to_relative,current.choice_ids,'permitted-features/1.0.0')):
            with self.assertRaisesRegex(ValueError,'mixed-version'):collate([current,old])
            r=game()['records'][0]
            with self.assertRaisesRegex(ValueError,'mixed-version'):Dataset([(r,dict(win_share=[0]*4,placement=[1]*4,vp=[0]*4),('g',0),old)],'cpu')

    def test_checkpoint_roundtrip_adam_rng_and_old_contract_rejected(self):
        model=FourSeatNet();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        sum(p.square().mean() for p in model.parameters()).backward();optimizer.step()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'new.pt';save_checkpoint(path,model,optimizer,step=17,training=dict(progress=123))
            restored,raw=load_checkpoint(path);adam=torch.optim.AdamW(restored.parameters(),lr=1e-4)
            resume_optimizer(raw,adam)
            self.assertEqual(model_id(model),model_id(restored));self.assertEqual(raw['step'],17)
            self.assertEqual(len(adam.state),len(optimizer.state));self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
            legacy=legacy_contract()
            raw['compatibility']=legacy;torch.save(raw,path)
            with self.assertRaisesRegex(ValueError,'Incompatible'):load_checkpoint(path)

    def test_encoded_cache_version_receipts_and_neutral_supervision_preserve_labels(self):
        episode=check_game(game())
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'raw.jsonl.gz';write_shard(path,[episode],dict(purpose='training'))
            receipt=cache_receipt(path);self.assertEqual(receipt['validator'],cache_version())
            self.assertTrue(path.with_name(path.name+'.features-v4.npz').exists())
            self.assertFalse(path.with_name(path.name+'.features.npz').exists())
            old=dict(receipt,validator='0'*64);self.assertIsNone(load_cache(path,[episode],old))
            restored=read_shards([path],receipts={str(path):receipt})[0]
            self.assertEqual(episode['records'],restored['records'])
            styled=restored['records'][0];styled['style']=[1.,0,0,0]
            # Altering an attested context cannot be passed off as original PPO.
            with self.assertRaisesRegex(ValueError,'context changed'):restored.encoded(styled,PreparedEncoder())
            neutral=prepared([restored],neutral=True)
            self.assertEqual(neutral[0][0]['chosen'],episode['records'][0]['chosen'])
            np.testing.assert_array_equal(neutral[0][3].state[-12:],[0]*12)

    def test_explicit_raw_conversion_checks_rules_and_does_not_admit_old_weights(self):
        legacy=legacy_contract()
        raw_compatible(legacy)
        with self.assertRaises(ValueError):assert_compatible(legacy)
        for field in ('rules','features_sha256','wire_sha256','loss'):
            bad=dict(legacy);bad[field]='unknown'
            with self.assertRaises(ValueError):raw_compatible(bad)

    def test_conversion_rejects_neural_labels_and_missing_raw_facts_without_touching_sources(self):
        for mode in ('neural','teacher','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
                directory=Path(directory);path=directory/'source.jsonl.gz';episode=game()
                for row in episode['records']:row['behavior']['kind']='external'
                if mode=='neural':episode['records'][0]['behavior']['kind']='model'
                elif mode=='teacher':episode['records'][0]['teacher']={}
                else:del episode['records'][0]['view']['own_hand']
                with gzip.open(path,'wt',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(format='four-seat-replay/1.0.0',compatibility=legacy_contract(),provenance=dict(purpose='training',seed=5)))+'\n')
                    stream.write(json.dumps(episode)+'\n')
                manifest=directory/'manifest.json';token=sha(path)
                manifest.write_text(json.dumps(dict(format='bc-dataset/1.0.0',label_kind='qualified-v3-choice/1.0.0',
                    context='neutral-predecision/1.0.0',compatibility=legacy_contract(),teachers=['fixture'],calibration=None,
                    entries=[dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=token,game_id=episode['game_id'],
                        split=episode['split'],setup_group=episode['setup_group'],transitions_verified=True,semantic_sha256='0'*64)])))
                with self.assertRaisesRegex(ValueError,'conversion failed'):convert(SOURCE_ROOT,manifest,directory/'converted')
                self.assertEqual(sha(path),token);self.assertFalse((directory/'converted/manifest.json').exists())
                self.assertTrue((directory/'converted/failure.json').exists())

    def test_production_source_reviews_do_not_authorize_v1_to_v2_continuation(self):
        from brass_ai.recovery.source_reviews import expand_source_reviews
        from brass_ai.step5.league import implementation_id
        approvals=expand_source_reviews(read_json(SOURCE_ROOT/'configs/learning/selfplay-resume-sources.json'))
        source=implementation_id();token=digest(compatibility())
        self.assertFalse(any(e['to_implementation']==source and
            e['learning_compatibility']==token for e in approvals['transitions']))

    def test_dynamic_batched_oversized_inference_and_shared_memory_fallback(self):
        encoder=PreparedEncoder();rows=[encoder.encode(r['view'],r['choices']) for r in self.positions[:3]]
        model=FourSeatNet().eval()
        with torch.inference_mode():
            together=predict_encoded(model,rows,'cpu',100000)
            for i,row in enumerate(rows):
                single=predict_encoded(model,[row],'cpu',7)
                for k in range(len(single)):
                    expected=together[k][i,:len(row.choice_ids)] if k==0 else together[k][i]
                    np.testing.assert_allclose(expected,single[k][0],rtol=2e-4,atol=2e-4)
        slot=SharedSlot(size=2*1024*1024)
        try:
            frames=[dict(encoded=rows[0])];packed=slot.pack_frames(frames,'a'*32)
            unpacked=slot.unpack_frames(packed,'a'*32)[0]['encoded']
            np.testing.assert_array_equal(rows[0].actions,unpacked.actions)
            packed[0]['encoded']['feature_version']='old'
            with self.assertRaisesRegex(ValueError,'mixed-version'):slot.unpack_frames(packed,'a'*32)
            huge=Encoded(rows[0].state,np.zeros((2000,327),np.float32),rows[0].stable_to_relative,tuple(map(str,range(2000))))
            self.assertIsNone(slot.pack_frames([dict(encoded=huge)],'b'*32))
        finally:slot.close()

    def test_official_onnx_export_dynamic_parity_and_personality_budget(self):
        from brass_ai.learning.export import export_onnx,load_export
        from brass_ai.personality.model import PersonalityNet,PersonalityConfig
        from brass_ai.personality.checkpoint import save,load
        from brass_ai.personality.contracts import CONTEXT_DIM
        encoder=PreparedEncoder();rows=[encoder.encode(r['view'],r['choices']) for r in self.positions[:3]]
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            directory=Path(directory);report=export_onnx(FourSeatNet(),directory/'official.onnx',[[rows[0]],rows[1:]])
            self.assertLess(report['parity_max_absolute_error'],.005);load_export(directory/'official.onnx')
            utility=UtilityNet(utility_config(ModelConfig.preset(),reward_spec(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json'))))
            model=PersonalityNet(PersonalityConfig(utility.config.to_dict()))
            self.assertEqual(model.parameter_count,6023137)
            self.assertEqual(model.config.parameter_limit,6600000)
            save(directory/'goals.pt',model);restored,_=load(directory/'goals.pt')
            with torch.no_grad():
                output=restored(*collate(rows[:2]),personality=torch.zeros(2,CONTEXT_DIM),remaining=torch.full((2,4),.075))
            self.assertEqual(tuple(output['remaining_utility_4'].shape),(2,4))
            self.assertEqual(tuple(output['goals']['family'].shape),(2,6))


if __name__=='__main__':unittest.main()
