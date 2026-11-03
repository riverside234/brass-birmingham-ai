"""Bounded transfer checks, including the extra-module failure seen on the lab."""
import copy
import json
from pathlib import Path
import unittest

from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.rust_backend.lab_transfer import check_producer,implementation_inventory


class Transfer(unittest.TestCase):
    def setUp(self):
        self.sources={'src/brass_ai/recovery/league.py':'# reviewed\n'}
        self.manifest=dict(implementation=digest(self.sources),producer='p'*64,
            implementation_files=implementation_inventory(self.sources))

    def test_exact_source_passes(self):
        self.assertEqual(check_producer(self.manifest,self.sources,'p'*64),self.manifest['implementation'])

    def test_added_module_is_named_even_when_every_listed_file_matches(self):
        changed=dict(self.sources,**{'src/brass_ai/recovery/entropy_schedule.py':'# new source\n'})
        with self.assertRaisesRegex(ValueError,r'added \(1\): src/brass_ai/recovery/entropy_schedule.py') as error:
            check_producer(self.manifest,changed,'p'*64)
        self.assertIn('implementation expected '+self.manifest['implementation'],str(error.exception))
        self.assertIn('actual '+digest(changed),str(error.exception))

    def test_changed_missing_and_native_producer_stay_rejected(self):
        for sources,producer,fragment in (({next(iter(self.sources)):'# changed\n'},'p'*64,'changed (1)'),
                ({},'p'*64,'missing (1)'),(self.sources,'q'*64,'native producer expected')):
            with self.subTest(fragment=fragment),self.assertRaises(ValueError) as error:
                check_producer(self.manifest,sources,producer)
            self.assertIn(fragment,str(error.exception))

    def test_inventory_cannot_omit_a_hashed_dependency(self):
        bad=copy.deepcopy(self.manifest);bad['implementation_files']={}
        with self.assertRaisesRegex(ValueError,'added'):
            check_producer(bad,self.sources,'p'*64)

    def test_g570_report_keeps_the_recorded_recipe(self):
        folder=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z'
        state=json.loads((folder/'state.json').read_text(encoding='utf-8'))
        config=json.loads((folder/'recorded-config.json').read_text(encoding='utf-8'))
        self.assertEqual(state['iteration'],570)
        self.assertEqual(state['history'][-1]['sha256'],'06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9')
        self.assertEqual(config,state['signature']['config'])
        from brass_ai.recovery.config import validate_config
        self.assertEqual(validate_config(copy.deepcopy(config)),config)
        self.assertNotIn('entropy_schedule',config['update'])
        self.assertEqual(config['branching']['contract'],'selfplay-branching/8.0.0')
        self.assertEqual(config['diversity']['mode'],'off')


if __name__=='__main__':unittest.main()
