"""The performance update admits exact continuations, never recipe edits."""
import copy
import unittest

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, digest
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id


class InputCacheResume(unittest.TestCase):
    def test_same_settings_and_completed_il_layout_keep_their_identity(self):
        export=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0320-20261021T052141479749Z'
        saved=read(export/'state.json')
        config=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-entropy95.json')
        old=dict(saved['signature'],config=config,
                 implementation='e0dc28ad559ddcd2d6b0eff95d1dac61c4ede949f70bf341a16d08e2ca71dca3')
        state=dict(format=saved['format'],iteration=saved['iteration'],signature=old)
        original=copy.deepcopy(state);new=dict(old,implementation=implementation_id())
        self.assertTrue(check_signature(state,new,reviewed=True));self.assertEqual(state,original)
        self.assertEqual(digest(compatibility()),'ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54')
        with self.assertRaises(ValueError):check_signature(state,new)
        for key,value in (('low_ratio',.6),('target_kl',.015)):
            changed=copy.deepcopy(new);changed['config']['ppo_lr_controller'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)
        changed=dict(new,implementation='0'*64)
        with self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)

    def test_existing_g320_hotfix_route_remains_available(self):
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0320-20261021T052141479749Z/state.json')
        new=dict(state['signature'],implementation=implementation_id(),
                 config=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-entropy95.json'))
        self.assertTrue(check_signature(state,new,reviewed=True))


if __name__=='__main__':unittest.main()
