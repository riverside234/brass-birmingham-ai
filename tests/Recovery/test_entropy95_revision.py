"""Frozen entropy floors, fresh branch openings and exact g320 continuation."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import exploration as E,branching
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json
from test_exploration_revision import measured

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0320-20261021T052141479749Z'


class BranchOpenings(unittest.TestCase):

    def test_per_seat_counter_admission_including_intervening_subdecisions(self):
        rows=[]
        for n in range(3):
            for seat in (2,0,3,1):
                rows.append(dict(seat=seat,view=dict(context=dict(phase='bonus')),
                                 behavior=dict(action_decision=None,temperature=1.)))
                rows.append(dict(seat=seat,view=dict(context=dict(phase='action')),
                                 behavior=dict(action_decision=n,temperature=2. if n<2 else 1.)))
        branching.validate_opening_rows(rows)
        for field,value in (('action_decision',5),('temperature',1.)):
            bad=copy.deepcopy(rows);bad[1]['behavior'][field]=value
            with self.assertRaisesRegex(ValueError,'Branch opening'):branching.validate_opening_rows(bad)

    def test_old_fork_seed_preserved_and_new_contract_is_distinct(self):
        self.assertEqual(branching.seed_for('p',3),digest(dict(contract=E.LEGACY_BRANCH['contract'],parent='p',decision=3)))
        self.assertNotEqual(branching.seed_for('p',3),branching.seed_for('p',3,E.BRANCH['contract']))


if __name__=='__main__':unittest.main()
