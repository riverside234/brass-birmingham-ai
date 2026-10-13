"""Remaining portable-package semantic boundaries; authored fixtures only."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_p6_package_audit as prior_package_tests
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.personality.laptop import package,contracts
from brass_ai.personality.difficulty import contracts as difficulty


class PackageSemanticsTests(unittest.TestCase):
    def test_portable_calibration_is_bound_to_roster_readiness(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);original,files=prior_package_tests.PackageAuditTests().fixture(directory)
            for change in ('temperature','readiness'):
                documents=copy.deepcopy(original)
                if change=='temperature':
                    documents['manifest.json']['calibration']=dict(documents['manifest.json']['calibration'],temperature=2.)
                else:
                    documents['readiness.json']=contracts.seal(dict(
                        {k:v for k,v in documents['readiness.json'].items() if k!='id'},
                        goal_calibration=dict(documents['readiness.json']['goal_calibration'],temperature=2.)))
                documents['manifest.json']=contracts.seal({k:v for k,v in documents['manifest.json'].items() if k!='id'})
                with self.subTest(change=change), \
                     patch.object(package,'read',side_effect=lambda p:documents[Path(p).relative_to(directory).as_posix()]), \
                     patch.object(package,'inventory',return_value=files):
                    with self.assertRaisesRegex(ValueError,'Readiness|calibration'):
                        package.validate(directory)

    def test_tier_profiles_must_match_the_supplied_qualification(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);documents,files=prior_package_tests.PackageAuditTests().fixture(directory)
            report=contracts.seal(dict(format=difficulty.REPORT,contract=difficulty.identity(),
                roster_id=documents['roster.json']['id'],checks={k:False for k in difficulty.RELEASE_CHECKS},
                gates=asdict(difficulty.Gates()),p5_qualified=False,qualified_tiers=[],release_qualified=False,
                profiles=dict(documents['profiles.json'],hint_mode='intent-hints')))
            documents['qualification.json']=report
            with patch.object(package,'read',side_effect=lambda p:documents[Path(p).relative_to(directory).as_posix()]), \
                 patch.object(package,'inventory',return_value=files):
                with self.assertRaisesRegex(ValueError,'profile'):
                    package.validate(directory)

    def test_portable_roster_cannot_keep_stale_id_after_profile_edit(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);documents,files=prior_package_tests.PackageAuditTests().fixture(directory)
            alpha=documents['roster.json']['members'][0]['profile']['alpha']
            changed_alpha=0. if alpha else .1
            documents['roster.json']['members'][0]['profile']['alpha']=changed_alpha
            documents['manifest.json']['roster']['members'][0]['profile']['alpha']=changed_alpha
            documents['manifest.json']=contracts.seal({k:v for k,v in documents['manifest.json'].items() if k!='id'})
            with patch.object(package,'read',side_effect=lambda p:documents[Path(p).relative_to(directory).as_posix()]), \
                 patch.object(package,'inventory',return_value=files):
                with self.assertRaisesRegex(ValueError,'Roster|roster'):
                    package.validate(directory)

    def test_export_reward_cannot_differ_from_qualified_search(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);original,files=prior_package_tests.PackageAuditTests().fixture(directory)
            def validate(documents):
                with patch.object(package,'read',side_effect=lambda p:documents[Path(p).relative_to(directory).as_posix()]), \
                     patch.object(package,'inventory',return_value=files):
                    return package.validate(directory)
            self.assertEqual(validate(original)['reward'],original['roster.json']['search']['reward'])
            changed=copy.deepcopy(original)
            # A producer may consistently label all exports with one reward while
            # retaining a different qualified search recipe in the roster.
            changed['manifest.json']['reward']=copy.deepcopy(original['manifest.json']['reward'])
            changed['manifest.json']['reward']['insolvency']['event_cost']=.004
            for name,meta in changed.items():
                if name.startswith('models/'):
                    meta['config']['utility']=copy.deepcopy(changed['manifest.json']['reward'])
                    meta['reward_id']=digest(meta['config']['utility'])
            changed['manifest.json']=contracts.seal({k:v for k,v in changed['manifest.json'].items() if k!='id'})
            with self.assertRaisesRegex(ValueError,'reward'):
                validate(changed)


if __name__=='__main__':unittest.main()
