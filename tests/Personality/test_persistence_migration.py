"""Strict reviewed P0/P1 source migration, without reusing P2 qualification."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from personality_fixtures import episode,parent
from brass_ai.learning.contracts import CONFIG_DIR,SOURCE_ROOT,read_json
from brass_ai.learning.reward import specification
from brass_ai.personality import checkpoint
from brass_ai.personality.contracts import accepted_contract,contract
from brass_ai.personality.model import import_parent
from brass_ai.personality.training import fit
from brass_ai.personality.traces import examples


class PersistenceMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def previous_contract(self):
        receipt=read_json(CONFIG_DIR.parent/'personality/source-migrations.json')
        self.assertEqual(receipt['format'],'personality-source-migrations/1.0.0')
        previous=receipt['migrations'][0]['from_contract']
        matches=[m for m in receipt['migrations'] if m['from_contract']==previous and m['to_contract']==contract()]
        self.assertEqual(len(matches),1)
        return previous

    def test_exact_reviewed_source_load_preserves_weights_optimizer_and_rng(self):
        old=self.previous_contract();model,_=import_parent(parent())
        data=examples([episode()],specification(),roots_only=True)
        optimizer,_=fit(model,data,'cpu',epochs=1)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'old.pt'
            with patch.object(checkpoint,'contract',return_value=old):
                original_id=checkpoint.save(path,model,optimizer,training=dict(phase='warmup'))
            original=torch.load(path,map_location='cpu',weights_only=True)
            restored,raw=checkpoint.load(path)
            self.assertEqual(raw['model_id'],original_id)
            self.assertEqual(raw['source_migration']['from_model_id'],original_id)
            self.assertEqual(raw['source_migration']['to_model_id'],checkpoint.model_id(restored))
            self.assertNotEqual(original_id,checkpoint.model_id(restored))
            self.assertTrue(torch.equal(raw['rng'],original['rng']))
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,restored.state_dict()[key]))
            resumed,_=fit(restored,data,'cpu',epochs=1,resume=raw)
            self.assertGreater(float(next(iter(resumed.state.values()))['step']),
                               float(next(iter(optimizer.state.values()))['step']))
            checkpoint.save(Path(tmp)/'current.pt',restored,resumed,training=dict(phase='warmup'))
            _,current=checkpoint.load(Path(tmp)/'current.pt')
            self.assertEqual(current['contract'],contract())
            self.assertEqual(current['training']['source_migration']['from_model_id'],original_id)

    def test_unreviewed_source_and_changed_semantics_are_rejected(self):
        old=self.previous_contract()
        self.assertIsNotNone(accepted_contract(old))
        for name,value in [('source','0'*64),('goal_schema','0'*64)]:
            changed=copy.deepcopy(old);changed[name]=value
            with self.assertRaisesRegex(ValueError,'contract/source differs'):accepted_contract(changed)

    def test_reviewed_source_does_not_excuse_changed_weights(self):
        old=self.previous_contract();model,_=import_parent(parent())
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'old.pt'
            with patch.object(checkpoint,'contract',return_value=old):checkpoint.save(path,model)
            raw=torch.load(path,map_location='cpu',weights_only=True)
            next(iter(raw['state_dict'].values())).add_(.1);torch.save(raw,path)
            with self.assertRaisesRegex(ValueError,'weights changed'):checkpoint.load(path)

    def test_receipt_must_pin_this_exact_target_source(self):
        old=self.previous_contract()
        receipt=read_json(CONFIG_DIR.parent/'personality/source-migrations.json')
        for entry in receipt['migrations']:
            if entry['from_contract']==old and entry['to_contract']==contract():
                entry['to_contract']['source']='0'*64
        with patch('brass_ai.personality.contracts.read_json',return_value=receipt):
            with self.assertRaisesRegex(ValueError,'contract/source differs'):accepted_contract(old)


if __name__=='__main__':unittest.main()
