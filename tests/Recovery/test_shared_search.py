"""Game/IL/self-play use one search core and the official (not Network) v4."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.recovery.config import LEGACY_BUDGET,LEGACY_TIME_BUDGET,LEGACY_OPPONENT_WORK_BUDGET,v4_search_enabled
from brass_ai.recovery.legacy_search import legacy_spec
from brass_ai.recovery.parallel import merge_statistics
from brass_ai.search.deadline import configure_clock
from brass_ai.search.legacy_runtime import reference,source_paths,validate_reference,runtime_directory,GAME_POLICY

ROOT=SOURCE_ROOT
BANK='bank-3dc65253a610191d66a5'
PARENT=BANK+'/balanced-v3-masked-economy'


class SharedSearchTests(unittest.TestCase):
    def test_canonical_files_are_used_once_by_lab(self):
        paths=source_paths(ROOT)
        for name in GAME_POLICY:
            self.assertEqual([p for p in paths if p.name==name],[ROOT/'src/BrassLocalAI/Policy'/name])
        for name in ('MaskedSearchExecution.cs','HeuristicBatchQueue.cs','DeadlineAdmission.cs','RootForest.cs'):
            self.assertEqual(sum(p.name==name for p in paths),1)

    def test_official_v4_private_strategy_is_preserved(self):
        original=(ROOT/'training/opponents/bank'/BANK/'source/src/BrassMaskedSearch/MaskedSearch.cs').read_text()
        current=(ROOT/'src/BrassMaskedSearch/MaskedSearch.cs').read_text()
        first='    public sealed partial class Planner'
        self.assertEqual(original[original.index(first):original.index('        public Decision Decide(')],
                         current[current.index(first):current.index('        // Official IL/game v4 strategy')])
        self.assertNotIn('NetworkOptions',current)
        for p in ('src/BrassLegacyAI/Planner.cs','src/BrassLocalAI/Policy/Planner.cs'):
            self.assertNotIn('NetworkOptions',(ROOT/p).read_text())

    def test_game_lab_and_teacher_timing_but_not_fixed_quotas(self):
        for role in ('laptop','lab','reference'):
            for work,timed,expected in ((None,False,True),({'mode':'time'},True,True),({'mode':'simulations'},False,False)):
                policy=SimpleNamespace(cpu_role=role,teacher_work_budget=work,time_budget_primary=timed)
                bot=SimpleNamespace();configure_clock(bot,policy)
                self.assertEqual(bot.deadline_admission,expected)
                self.assertEqual(bot.time_budget_primary,timed)

    def test_timed_fixed_and_fast_specs_pin_shared_runtime(self):
        specs=[legacy_spec(ROOT,PARENT,LEGACY_BUDGET,work_budget=w,search_free=fast)
               for w in (LEGACY_TIME_BUDGET,LEGACY_OPPONENT_WORK_BUDGET) for fast in (False,True)]
        ref=reference(ROOT,BANK)
        self.assertTrue(all(s['search_runtime']==ref for s in specs))
        self.assertEqual(len({s['policy_id'] for s in specs}),4)
        for spec in specs:
            old=legacy_spec(ROOT,PARENT,LEGACY_BUDGET,work_budget=spec['work_budget'],search_free=spec['search_free'],search_runtime=False)
            self.assertNotIn('search_runtime',old);self.assertNotEqual(old['policy_id'],spec['policy_id'])

    def test_source_change_invalidates_collection_and_runtime(self):
        ref=reference(ROOT,BANK);changed=copy.deepcopy(ref)
        changed['recipe']['sources']['src/BrassMaskedSearch/DeadlineAdmission.cs']='f'*64
        with self.assertRaisesRegex(ValueError,'reference changed'):validate_reference(changed)
        from brass_ai.search import legacy_runtime
        with patch.object(legacy_runtime,'recorded',return_value=ref),patch.object(legacy_runtime,'reference',return_value=changed):
            with self.assertRaisesRegex(ValueError,'source changed'):runtime_directory(ROOT,ref)

    def test_actor_factory_preserves_pinned_or_archived_runtime_choice(self):
        from brass_ai.recovery.execution import make_policy
        for runtime in (None,False):
            spec=legacy_spec(ROOT,PARENT,LEGACY_BUDGET,work_budget=LEGACY_TIME_BUDGET,search_runtime=runtime)
            with patch('brass_ai.recovery.legacy_search.LabLegacyPolicy',return_value=SimpleNamespace(policy_id=spec['policy_id'])) as factory:
                make_policy(spec,ROOT,None,None,None)
                self.assertEqual(factory.call_args.kwargs['search_runtime'],spec.get('search_runtime',False))

    def test_forest_reports_only_actual_attempts_and_sums_worker_tail_time(self):
        item=dict(completed=4,interrupted=0,opponent_nodes=0,max_pending=1,opponent_choices=0,root={},plans={},expansion=[False],
                  batching=dict(attempted=4,deadline_stops=1,deadline_remaining_seconds=.3,simulation_estimate_seconds=.5))
        result=merge_statistics([item,item],[{'id':'x'}])
        self.assertEqual((result['completed'],result['interrupted']),(8,0))
        self.assertEqual(result['batching'],dict(attempted=8,deadline_stops=2,deadline_remaining_seconds=.6,simulation_estimate_seconds=.5))

    def test_completed_IL_admission_does_not_require_search_binary_or_current_source(self):
        from brass_ai.recovery.demonstrations import dataset_tasks
        for shared in (False,True):
            spec=legacy_spec(ROOT,PARENT,LEGACY_BUDGET,work_budget=LEGACY_TIME_BUDGET,search_runtime=None if shared else False)
            calibration=dict(format='direct-masked-budget/1.0.0',budget=LEGACY_BUDGET,parents=[PARENT],work_budgets=[LEGACY_TIME_BUDGET])
            if shared:calibration['search_runtimes']=[spec['search_runtime']]
            manifest=dict(format='bc-dataset/1.0.0',compatibility=compatibility(),label_kind='qualified-v3-choice/1.0.0',
                          context='neutral-predecision/1.0.0',teachers=[spec['policy_id']],entries=[],calibration=calibration)
            with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
                p=Path(temp)/'manifest.json';p.write_text(json.dumps(manifest),encoding='utf-8')
                with patch('brass_ai.search.legacy_runtime.reference',side_effect=AssertionError('old BC must not bind current code')):
                    # No entries deliberately: admission must pass before the
                    # expected empty-partition failure, without executing search.
                    with self.assertRaisesRegex(ValueError,'No train demonstration'):dataset_tasks(ROOT,p,'train')

    def test_new_collection_provenance_retains_shared_reference(self):
        from brass_ai.recovery.demonstrations import shared_teacher_references
        spec=legacy_spec(ROOT,PARENT,LEGACY_BUDGET,work_budget=LEGACY_TIME_BUDGET)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            p=Path(temp)/'spec.json';p.write_text(json.dumps(dict(format='v3-demonstration-collection/1.0.0',teachers={PARENT:spec})),encoding='utf-8')
            self.assertEqual(shared_teacher_references(ROOT,[Path(temp)/'game.jsonl.gz'],[PARENT],LEGACY_BUDGET),[spec['search_runtime']])

    def test_self_play_search_remains_disabled(self):
        config=read(ROOT/'configs/step5/joint7-latest.json')
        self.assertFalse(v4_search_enabled(config))
        self.assertFalse(config['teacher_enabled'])


if __name__=='__main__':unittest.main()
