"""Six-group feature rollback, current grouped model and retained lab binding fix."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.feature_engineering.context import DerivedContextBuilder
from brass_ai.feature_engineering.contracts import STATE_FIELDS,ACTION_FIELDS,specification
from brass_ai.feature_engineering.data import raw_compatible
from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,assert_compatible,read_json,digest
from brass_ai.learning.features import Encoder,Encoded
from brass_ai.learning.model import ModelConfig,FourSeatNet,ResidualBlock
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.personality.model import PersonalityNet,PersonalityConfig
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.utility import reward_spec
from fe_fixtures import benchmark_positions


class SixGroupRollbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.position=benchmark_positions()[0]

    def test_exact_six_group_layout_without_hand_coverage(self):
        schema=read_json(SOURCE_ROOT/'configs/learning/features-4.0.0.json')
        self.assertEqual(digest(specification()),digest(schema['derived']))
        self.assertEqual((len(STATE_FIELDS),len(ACTION_FIELDS)),(326,52))
        self.assertFalse(any(f.name.startswith('hand.') for f in STATE_FIELDS+ACTION_FIELDS))
        p=self.position;context=DerivedContextBuilder().build(p['view'],p['choices'])
        self.assertFalse(any(k.startswith('hand.') for k in context.state))
        self.assertTrue(all(not any(k.startswith('hand.') for k in a) for a in context.actions))
        a=Encoder().encode(p['view'],p['choices']);b=PreparedEncoder().encode(p['view'],p['choices'])
        self.assertEqual((a.state.shape,a.actions.shape),((3128,),(len(p['choices']),327)))
        np.testing.assert_array_equal(a.state,b.state);np.testing.assert_array_equal(a.actions,b.actions)
        self.assertEqual(a.feature_version,'permitted-features/4.0.0')

    def test_eight_real_residual_blocks_for_default_and_presets(self):
        for size,count in (('small',1499780),('medium',5904644)):
            model=FourSeatNet(ModelConfig.preset(size))
            self.assertEqual(model.config.residual_blocks,8)
            self.assertEqual(sum(isinstance(m,ResidualBlock) for m in model.modules()),8)
            self.assertEqual(model.parameter_count,count)
            self.assertEqual(model.config.architecture,'brass-four-seat-mlp-resnet/11.1.0')
        model=FourSeatNet();self.assertEqual(model.config.hidden,512)
        utility=UtilityNet(utility_config(model.config,reward_spec(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json'))))
        self.assertEqual(utility.parameter_count,5921320)
        self.assertEqual(PersonalityNet(PersonalityConfig(utility.config.to_dict())).parameter_count,6023137)

    def test_archived_hand_tensors_and_old_model_versions_rejected(self):
        p=self.position;e=Encoder().encode(p['view'],p['choices'])
        hand=Encoded(np.zeros(5685,np.float32),np.zeros((len(e.choice_ids),576),np.float32),
            e.stable_to_relative,e.choice_ids,'permitted-features/3.0.0')
        with self.assertRaisesRegex(ValueError,'mixed-version'):collate([hand])
        for old in (1,2,3,4,5):
            raw=ModelConfig.preset().to_dict();raw.update(architecture=f'brass-four-seat-mlp-resnet/{old}.0.0',residual_blocks=4)
            with self.assertRaisesRegex(ValueError,'old architectures'):ModelConfig.from_dict(raw)

    def test_archived_hand_raw_can_be_reaudited_but_not_relabelled_as_neural(self):
        old=compatibility();old.update(learning_contract='four-seat-learning/3.0.0',
            feature_version='permitted-features/3.0.0',
            features_sha256=digest(read_json(SOURCE_ROOT/'configs/learning/features-3.0.0.json')))
        raw_compatible(old)
        with self.assertRaises(ValueError):assert_compatible(old)
        bad=dict(old,feature_version='permitted-features/2.0.0')
        with self.assertRaisesRegex(ValueError,'Unknown raw feature version'):raw_compatible(bad)


class FeatureLabConfigTests(unittest.TestCase):
    def test_configure_binds_fresh_imitation_rehearsal_and_preserves_recipe(self):
        base=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json');before=copy.deepcopy(base)
        origin=dict(base['imitation_checkpoint'],bc_run='artifacts/six-group-five-block-il',
            path='artifacts/six-group-five-block-il/epoch-0001.pt',manifest='artifacts/six-group-data/manifest.json')
        main=runpy.run_path(str(SOURCE_ROOT/'scripts/feature-engineering.py'))['main']
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            output=Path(directory)/'recipe.json'
            with patch.object(sys,'argv',['feature-engineering.py','configure',
                    '--imitation-run',origin['bc_run'],'--output',str(output)]), \
                 patch('brass_ai.recovery.demonstrations.completed_imitation',
                    return_value=(SimpleNamespace(config=ModelConfig.preset('medium')),origin)), \
                 redirect_stdout(io.StringIO()):main()
            result=read_json(output)
            self.assertEqual(result['imitation_checkpoint'],origin)
            self.assertEqual(result['rehearsal_manifest'],origin['manifest'])
            for key in base.keys()-{'imitation_checkpoint','rehearsal_manifest'}:self.assertEqual(result[key],base[key],key)
        self.assertEqual(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json'),before)


if __name__=='__main__':unittest.main()
