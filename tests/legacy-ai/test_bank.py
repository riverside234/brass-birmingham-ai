"""Admission and statistical checks; no synthetic result is a strength claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from brass_ai.bank import admission,below,digest,freeze,load_bank,sha
from brass_ai.qualification import grouped_interval,publish_pool,write_new,summarize


class BankTests(unittest.TestCase):
    def test_cluster_interval_keeps_correlated_rotations_together(self):
        rows=[dict(seed=seed,win=seed) for seed in (0,1) for _ in range(4)]
        self.assertEqual([0,1],grouped_interval(rows,'win'))
        self.assertIsNone(grouped_interval(rows[:4],'win'))
        self.assertEqual(grouped_interval(rows,'win'),grouped_interval(rows,'win'))

    def test_freeze_rejects_path_escape_and_duplicate_names(self):
        for names in (['../x'],['a','a'],[]):
            with self.assertRaises(ValueError):freeze('.',names)
        with self.assertRaises(ValueError):below('.', '../outside')

    def test_content_address_and_source_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'training/opponents/bank'
            payload=dict(player_count=4,wire='0.2.0',rules='steam-brass-patch-1.1.0',source_files={},adapter_files=[])
            identity='bank-'+digest(payload)[:20];file=source/identity/'manifest.json'
            write_new(file,dict(id=identity,payload=payload))
            self.assertEqual(identity,load_bank(root,identity)['id'])
            payload['player_count']=3;file.write_text(json.dumps(dict(id=identity,payload=payload)))
            with self.assertRaisesRegex(ValueError,'changed'):load_bank(root,identity)

    def test_pool_admission_is_immutable_disjoint_and_evidence_gated(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);policy='bank-test/cotton-v2'
            report=dict(bank='bank-test',profiles={'cotton-v2':dict(policy_id=policy,training_eligible=True,
                evaluation_eligible=False,niche_qualified=True)})
            path=root/'training/opponents/evaluations/test.json';write_new(path,report)
            publish_pool(root,report,path);admission(root,policy)
            with self.assertRaisesRegex(ValueError,'not admitted'):admission(root,policy,'evaluation')
            changed=copy.deepcopy(report);changed['profiles']['cotton-v2']['training_eligible']=False
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'changed'):admission(root,policy)
            path.write_text(json.dumps(report,indent=2)+'\n')
            other=root/'training/opponents/evaluations/other.json';write_new(other,report)
            with self.assertRaisesRegex(ValueError,'Cannot change'):publish_pool(root,report,other)
            report['profiles']['cotton-v2'].update(training_eligible=False,evaluation_eligible=True)
            other.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'disjoint'):publish_pool(root,report,other)

    def test_failed_match_is_neither_a_win_nor_pool_eligibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            profiles={n:dict(tactic='balanced',milliseconds=10000) for n in ('balanced-v2','balanced-s2-v2','balanced-s1-10s-v2')}
            payload=dict(player_count=4,wire='0.2.0',rules='steam-brass-patch-1.1.0',source_files={},adapter_files=[],profiles=profiles)
            bank='bank-'+digest(payload)[:20];write_new(root/'training/opponents/bank'/bank/'manifest.json',dict(id=bank,payload=payload))
            job=dict(id='failure-fixture',stage='guarded',seed=1,table=['balanced-v2',None,None,None],target=0)
            body=dict(bank=bank,jobs=[job],bootstrap_seed=731,bootstrap_samples=100,admission=dict(min_guarded_games=16,
                min_mixed_games=4,min_identical_games=1,minimum_guarded_mean_rank=2.5,held_out_family='pottery-flex',general_hard_min_setup_groups=30),
                equal_budget_pair=['balanced-s2-v2','balanced-s1-10s-v2'])
            spec=dict(id='qualification-'+digest(body)[:20],payload=body)
            output=root/'matches';write_new(output/(job['id']+'.json'),dict(job=job,spec_id=spec['id'],match=dict(completed=False,failure='injected test failure')))
            report=summarize(root,spec,output,dict(passed=True,bank=bank));profile=report['profiles']['balanced-v2']
            self.assertEqual(1,report['failed_games']);self.assertIsNone(profile['guarded']['win_share'])
            self.assertFalse(profile['training_eligible']);self.assertFalse(report['reply_comparison']['benefit_demonstrated'])


if __name__=='__main__':unittest.main()
