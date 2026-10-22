"""Bounded label sidecars, shared maps, holdout binding and post-commit expiry."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.local_scoring_auxiliary import DEFAULT,KEY
from brass_ai.recovery.ppg_replay import Rows
from brass_ai.step5.league import write_json
import test_ppg_eight as base
from test_local_scoring import annotate


class Cache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_shared_sidecar_tamper_rejection_and_ninth_age_expiry_after_commit(self):
        model,_,cfg=base.setup();cfg[KEY]=copy.deepcopy(DEFAULT)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            collect=base.collect
            def labeled(*args,**kwargs):
                b,data=collect(*args,**kwargs);annotate(data);annotate(p.snapshot_data);return b,data
            with patch.object(base,'collect',labeled):
                for g in range(1,10):
                    base.stage_only(model,p,g)
                    if g==1:
                        r=p.buffer.pending[0];file=p.buffer.path(1,'local-scoring.json');original=file.read_bytes()
                        packed=read(file);first=next(iter(packed['games'].values()))
                        self.assertIsInstance(first['receipt']['eras'][0]['slots'][0],list)
                        with p.buffer.datasets('cpu') as (ds,_,_):
                            self.assertTrue(all('_local_scoring' in ds.rows[i] for i in range(len(ds.data))))
                            self.assertEqual(len({id(ds.rows[i]['_local_scoring']['envelope']) for i in range(len(ds.data))}),len(r['game_ids']))
                            from brass_ai.recovery.ppg_auxiliary import layout
                            train,held_ids,_,_,_=layout(ds,[r],cfg['value_phase'],1.)
                            held=set(r['holdout_selection']['heldout_games'])
                            self.assertTrue(all(ds.data[i][2][0] in held for i in held_ids))
                            self.assertTrue(all(ds.data[i][2][0] not in held for group in train.values() for i in group))
                        file.write_bytes(original+b' ')
                        with self.assertRaisesRegex(ValueError,'sidecar changed'):Rows(p.buffer.path(1,'critic.npy'),r)
                        file.write_bytes(original)
                    if g==9:
                        self.assertTrue(p.buffer.path(1,'local-scoring.json').exists())
                        with self.assertRaisesRegex(ValueError,'commit'):p.buffer.committed(9)
                    write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
                self.assertFalse(p.buffer.path(1,'local-scoring.json').exists())
                self.assertTrue(p.buffer.path(2,'local-scoring.json').exists())
                self.assertEqual([r['generation'] for r in p.buffer.receipts],list(range(2,10)))


if __name__=='__main__':unittest.main()
