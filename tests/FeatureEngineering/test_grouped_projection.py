"""Versioned grouping semantics, joint fusion, strict admission and named export."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT,CONFIG_DIR,read_json,digest
from brass_ai.learning.features import Encoder
from brass_ai.learning.batching import collate
from brass_ai.learning.model import ModelConfig,FourSeatNet,ResidualBlock,dense_multiply_adds
from brass_ai.learning.state_layout import state_layout,build_state_layout,PROJECTION_WIDTHS,LAYOUT_FILE,LAYOUT_VERSION
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.export import export_onnx,load_export
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.personality.model import PersonalityNet,PersonalityConfig
from brass_ai.personality.checkpoint import save as save_personality,load as load_personality
from brass_ai.personality.laptop.export import export,Graph,validate_metadata
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.utility import reward_spec
from fe_fixtures import benchmark_positions


class GroupedProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.layout=state_layout()
        cls.positions=benchmark_positions()
        cls.reward=reward_spec(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json'))

    def models(self):
        base=FourSeatNet();utility=UtilityNet(utility_config(base.config,self.reward))
        actor=PersonalityNet(PersonalityConfig(utility.config.to_dict()))
        return base,utility,actor

    def test_complete_disjoint_deterministic_mapping_and_whole_measures(self):
        schema=read_json(CONFIG_DIR/'features-4.0.0.json');a=self.layout;b=build_state_layout(schema)
        self.assertEqual(a,b);self.assertEqual(a.identity,b.identity)
        self.assertEqual(sorted(a.permutation),list(range(3128)))
        self.assertEqual(len(set(a.permutation)),3128)
        self.assertEqual(a.raw_dimension,2539)
        self.assertEqual([len(g.indices) for g in a.groups],[31,218,109,1314,1247,197,12])
        self.assertEqual([g.width for g in a.groups[:-1]],list(PROJECTION_WIDTHS))
        self.assertEqual(a.groups[-1].indices,tuple(range(3116,3128)))
        self.assertEqual(PROJECTION_WIDTHS,(32,64,48,192,224,48))
        self.assertEqual(sum(PROJECTION_WIDTHS),608)
        self.assertEqual(sum(g.width for g in a.groups),620)
        owner={i:g.name for g in a.groups for i in g.indices}
        for segment in a.segments:
            self.assertEqual({owner[i] for i in range(segment.start,segment.stop)},{segment.group})
            if segment.source=='derived':
                field=next(f for f in schema['derived']['state'] if f['name']==segment.name)
                self.assertEqual(segment.stop-segment.start,len(field['components']))
                self.assertEqual(segment.components,tuple(field['components']))

    def test_mapping_matches_reference_prepared_values_masks_and_policy_tail(self):
        reference=Encoder();fast=PreparedEncoder();segments={s.name:s for s in self.layout.segments}
        fields={f['name']:f for f in reference.schema['derived']['state']}
        style=[.2,.1,-.1,0];intent=[0,.1,.2,.3,.4,.5,.6,.7]
        for row in self.positions:
            v=row['view'];c=row['choices'];a=reference.encode(v,c,style,intent);b=fast.encode(v,c,style,intent)
            np.testing.assert_array_equal(a.state,b.state);np.testing.assert_array_equal(a.actions,b.actions)
            self.assertEqual(a.choice_ids,b.choice_ids);np.testing.assert_array_equal(a.stable_to_relative,b.stable_to_relative)
            np.testing.assert_array_equal(a.state[-12:],np.asarray(style+intent,np.float32))
            context=reference.derived.build(v,c,style,intent)
            for name,field in fields.items():
                s=segments[name];value=context.state[field['fact']]
                expected=value.tensor(field['scale']) if len(field['components'])==7 else value.compact_tensor(field['scale'])
                np.testing.assert_array_equal(a.state[s.start:s.stop],np.asarray(expected,np.float32))
            s=segments['hand.counts'];counts=Counter(v['own_hand'])
            np.testing.assert_array_equal(a.state[s.start:s.stop],np.asarray([counts[k]/8 for k in reference.cards],np.float32))
            for offset in range(4):
                p=v['players'][(v['seat']+offset)%4];s=segments[f'player.{offset}.cash']
                self.assertAlmostEqual(float(a.state[s.start]),p['cash']/reference.schema['player_fields']['cash'],places=5)

    def test_every_projection_and_policy_context_reaches_joint_fusion(self):
        torch.manual_seed(57719);model=FourSeatNet();x=torch.randn(2,3128,requires_grad=True)
        fusion_inputs=[]
        hook=model.state_encoder.fusion[0].register_forward_pre_hook(lambda _,args:fusion_inputs.append(args[0]))
        h=model.state_encoder(x);hook.remove()
        self.assertEqual(tuple(h.shape),(2,304));self.assertEqual(tuple(fusion_inputs[0].shape),(2,620))
        self.assertEqual(model.state_encoder.fusion[0].in_features,620)
        self.assertEqual([p[0].out_features for p in model.state_encoder.projections],list(PROJECTION_WIDTHS))
        torch.testing.assert_close(fusion_inputs[0][:,-12:],x[:,-12:])
        (h*torch.randn_like(h)).sum().backward()
        for projection,group in zip(model.state_encoder.projections,self.layout.groups):
            self.assertGreater(float(projection[0].weight.grad.abs().sum()),0.)
            self.assertGreater(float(x.grad[:,list(group.indices)].abs().sum()),0.)
        self.assertGreater(float(x.grad[:,-12:].abs().sum()),0.)
        self.assertGreater(float(model.state_encoder.fusion[0].weight.grad.abs().sum()),0.)
        self.assertEqual(sum(isinstance(m,ResidualBlock) for m in model.modules()),6)

    def test_parameter_and_linear_mac_budgets_for_all_heads(self):
        base,utility,actor=self.models()
        for model,count,fixed in ((base,2005917,1899168),(utility,2015937,1909152),(actor,2081770,1974800)):
            self.assertEqual(model.parameter_count,count)
            mac=dense_multiply_adds(model)
            self.assertEqual(mac['fixed_per_position'],fixed);self.assertEqual(mac['per_padded_candidate'],97280)
        self.assertEqual(sum(p.numel() for p in base.state_encoder.parameters()),1867536)
        self.assertEqual([sum(p.numel() for p in g.parameters()) for g in base.state_encoder.projections],
                         [1024,14016,5280,252480,279552,9504])
        self.assertEqual(sum(p.numel() for p in base.state_encoder.fusion.parameters()),188784)
        for small,count,fixed in (
                (FourSeatNet(ModelConfig.preset('small')),934605,855136),
                (UtilityNet(utility_config(ModelConfig.preset('small'),self.reward)),938993,859488),
                (PersonalityNet(PersonalityConfig(utility_config(ModelConfig.preset('small'),self.reward).to_dict())),974378,894688)):
            self.assertEqual(small.parameter_count,count)
            self.assertEqual(dense_multiply_adds(small)['fixed_per_position'],fixed)
            self.assertEqual(dense_multiply_adds(small)['per_padded_candidate'],74752)
        config=ModelConfig.from_dict(json.loads(json.dumps(base.config.to_dict())))
        self.assertEqual(config,base.config)
        self.assertEqual(config.group_layout,self.layout.identity)

    def test_wrong_layout_widths_and_flat_architecture_are_rejected(self):
        config=ModelConfig.preset().to_dict()
        for key,value in (('group_layout','0'*64),('projection_widths',[64,48,64,112,160,64])):
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaisesRegex(ValueError,'layout|widths'):ModelConfig.from_dict(bad)
        flat=copy.deepcopy(config);flat['architecture']='brass-four-seat-mlp-resnet/4.0.0'
        flat.pop('group_layout');flat.pop('projection_widths')
        with self.assertRaisesRegex(ValueError,'old architectures'):ModelConfig.from_dict(flat)
        meta=dict(config=utility_config(ModelConfig.preset(),self.reward).to_dict())
        meta['config']['group_layout']='0'*64
        with self.assertRaisesRegex(ValueError,'layout'):validate_metadata(meta)

    def test_target_identity_and_same_total_wrong_widths_stale_manifest_rejected(self):
        record=self.layout.specification()
        self.assertEqual(record['format'],LAYOUT_VERSION)
        self.assertEqual(record,read_json(CONFIG_DIR/LAYOUT_FILE))
        self.assertEqual(record['layout_sha256'],digest({k:v for k,v in record.items() if k!='layout_sha256'}))
        self.assertEqual(record['feature_version'],'permitted-features/4.0.0')
        self.assertEqual([g['input_dim'] for g in record['groups']],[31,218,109,1314,1247,197,12])
        config=ModelConfig.preset().to_dict()
        for key,value in (('group_layout','f'*64),
                          ('projection_widths',[64,32,48,160,192,48])):
            bad=dict(config,**{key:value})
            with self.assertRaisesRegex(ValueError,'layout|widths'):ModelConfig.from_dict(bad)
        bad=copy.deepcopy(record);bad['groups'][0]['indices'].reverse()
        state_layout.cache_clear()
        original_read=read_json
        try:
            with patch('brass_ai.learning.state_layout.read_json',side_effect=lambda p:bad if p.name==LAYOUT_FILE else original_read(p)):
                with self.assertRaisesRegex(ValueError,'Frozen grouped state layout'):state_layout()
        finally:state_layout.cache_clear()

    def test_incompatible_checkpoint_metadata_rejected_before_source_review_or_tensor_loading(self):
        base=FourSeatNet(ModelConfig.preset('small'))
        utility=UtilityNet(utility_config(base.config,self.reward))
        actor=PersonalityNet(PersonalityConfig(utility.config.to_dict()))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            for name,model in (('base',base),('utility',utility),('actor',actor)):
                path=Path(directory)/(name+'.pt')
                (save_personality if name=='actor' else save_checkpoint)(path,model)
                raw=torch.load(path,weights_only=True)
                config=raw['config']['base'] if name=='actor' else raw['model_config']
                config.update(architecture='brass-four-seat-mlp-resnet/5.0.0',
                    group_layout='f'*64,projection_widths=[64,32,48,160,192,48])
                # Deliberately incompatible source evidence must not obscure
                # the actionable architecture error or approve migration.
                if name=='actor':raw['contract']={}
                else:raw['compatibility']={}
                raw['state_dict']={};torch.save(raw,path);before=path.read_bytes()
                with self.assertRaisesRegex(ValueError,'fresh version-9.*old architectures'):
                    (load_personality if name=='actor' else load_checkpoint)(path)
                self.assertEqual(path.read_bytes(),before)

    def test_widened_checkpoint_roundtrip_preserves_optimizer_rng_and_progress(self):
        model=FourSeatNet(ModelConfig.preset('small'));optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        sum(p.square().mean() for p in model.parameters()).backward();optimizer.step()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'current.pt'
            save_checkpoint(path,model,optimizer,step=23,training=dict(ordinary_games=960,lineage='isolated-fixture'))
            restored,raw=load_checkpoint(path);adam=torch.optim.AdamW(restored.parameters(),lr=1e-4)
            resume_optimizer(raw,adam)
            self.assertEqual(raw['step'],23);self.assertEqual(raw['training']['ordinary_games'],960)
            self.assertEqual(raw['training']['lineage'],'isolated-fixture')
            self.assertEqual(raw['model_config']['projection_widths'],list(PROJECTION_WIDTHS))
            self.assertEqual(raw['model_config']['group_layout'],self.layout.identity)
            self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
            for a,b in zip(model.state_dict().values(),restored.state_dict().values()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for parameter,loaded in zip(model.parameters(),restored.parameters()):
                state=optimizer.state[parameter];other=adam.state[loaded]
                for key,value in state.items():torch.testing.assert_close(value,other[key],rtol=0,atol=0)

    def test_residual_initialization_heads_masks_and_one_state_pass(self):
        for model in self.models():
            core=model.core if isinstance(model,PersonalityNet) else model
            blocks=list(core.state_encoder.backbone.children())
            self.assertEqual(len(blocks),7);self.assertIsInstance(blocks[-1],torch.nn.LayerNorm)
            for block in blocks[:-1]:
                self.assertEqual([type(m) for m in block.branch],
                    [torch.nn.LayerNorm,torch.nn.Linear,torch.nn.GELU,torch.nn.Linear])
                self.assertEqual(block.branch[1].in_features,304)
                self.assertEqual(block.branch[-1].out_features,304)
                self.assertEqual(torch.count_nonzero(block.branch[-1].weight).item(),0)
                self.assertEqual(torch.count_nonzero(block.branch[-1].bias).item(),0)
            calls=[];hook=core.state_encoder.register_forward_hook(lambda *args:calls.append(1))
            states=torch.randn(2,3128);actions=torch.randn(2,13,327)
            mask=torch.ones(2,13,dtype=torch.bool);mask[1,7:]=False
            mapping=torch.tensor([[0,1,2,3],[3,0,1,2]])
            extra={}
            if isinstance(model,PersonalityNet):
                from brass_ai.personality.contracts import CONTEXT_DIM
                extra=dict(personality=torch.zeros(2,CONTEXT_DIM),remaining=torch.full((2,4),.075))
            with torch.inference_mode():output=model(states,actions,mask,mapping,**extra)
            hook.remove();self.assertEqual(calls,[1])
            logits=output['policy_logits'] if isinstance(output,dict) else output[0]
            self.assertTrue(torch.isneginf(logits[~mask]).all());self.assertTrue(torch.isfinite(logits[mask]).all())
            self.assertEqual((core.action_encoder[0].in_features,core.action_encoder[0].out_features),(327,128))
            self.assertEqual((core.policy_scorer[0].in_features,core.policy_scorer[0].out_features),(432,128))

    def test_semantic_component_composition(self):
        segments=self.layout.segments
        count=lambda prefix:sum(s.stop-s.start for s in segments if s.name.startswith(prefix))
        self.assertEqual(count('hand.counts'),27)
        self.assertEqual(sum(s.stop-s.start for s in segments if s.name.startswith('player.') and s.group=='cards_history'),116)
        self.assertEqual(count('history.'),72)
        for prefix,total in (('mat.',864),('development.',450),('realization.',37),
                             ('slot.',882),('link.',276),('merchant.',63),('pending.',155),('opportunities.',42)):
            self.assertEqual(count(prefix),total,prefix)

    def test_same_shaped_different_group_index_buffer_rejected_on_load(self):
        model=FourSeatNet(ModelConfig.preset('small'))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'model.pt';save_checkpoint(path,model)
            raw=torch.load(path,weights_only=True);index=raw['state_dict']['state_encoder.permutation']
            j=len(self.layout.groups[0].indices);index[[0,j]]=index[[j,0]]
            torch.save(raw,path)
            with self.assertRaisesRegex(ValueError,'index buffer'):load_checkpoint(path)

    def test_named_utility_and_goal_exports_dynamic_shapes_and_layout_metadata(self):
        _,utility,actor=self.models()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            for name,model in (('utility',utility),('actor',actor)):
                path=Path(directory)/(name+'.onnx');meta=export(model,path)
                self.assertTrue(meta['parity']['passed']);self.assertEqual(meta['config']['group_layout'],self.layout.identity)
                self.assertEqual(meta['config']['residual_blocks'],6)
                self.assertEqual(meta['config']['architecture'],'brass-four-seat-mlp-resnet/9.0.0')
                self.assertEqual(meta['parity']['shapes'],[[1,1],[2,7],[4,31],[1,257]])
                self.assertEqual(Graph(path).meta['config']['projection_widths'],list(PROJECTION_WIDTHS))
                old=copy.deepcopy(meta);old['config']['architecture']='brass-four-seat-mlp-resnet/5.0.0'
                with self.assertRaisesRegex(ValueError,'fresh version-9'):validate_metadata(old)

    def test_base_export_dynamic_parity_and_old_grouped_metadata_rejected(self):
        model=FourSeatNet();encoded=[Encoder().encode(r['view'],r['choices']) for r in self.positions[:3]]
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            path=Path(directory)/'base.onnx';meta=export_onnx(model,path,[encoded[:1],encoded[1:]])
            self.assertLess(meta['parity_max_absolute_error'],2e-4)
            _,loaded=load_export(path);self.assertEqual(loaded['model_config']['group_layout'],self.layout.identity)
            meta['model_config']['architecture']='brass-four-seat-mlp-resnet/5.0.0';meta['compatibility']={}
            path.with_suffix('.json').write_text(json.dumps(meta),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'fresh version-9'):load_export(path)


if __name__=='__main__':unittest.main()
