"""Run R2 contract/model checks; --integration requires the qualified R1 host."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'src'))
sys.path.insert(0, str(SOURCE/'tests/R2'))
from brass_ai.learning.contracts import LAB_ROOT, project_path, require

parser=argparse.ArgumentParser()
parser.add_argument('--project-root', default=str(LAB_ROOT))
parser.add_argument('--integration',action='store_true')
parser.add_argument('--mono')
args=parser.parse_args()
require(Path(args.project_root).is_absolute(), 'Provide an absolute --project-root for this platform')
root=Path(args.project_root).resolve()
suite=unittest.defaultTestLoader.discover(str(SOURCE/'tests/R2'),pattern='test_learning.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
report={'format':'r2-exit-checks/1.0.0','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'passed':False,'unit_tests':result.testsRun,'unit_passed':result.wasSuccessful(),
        'failures':len(result.failures),'errors':len(result.errors),'integration_requested':args.integration}
target=project_path(root,'artifacts/r2/exit-checks.json')
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if result.wasSuccessful() and args.integration:
    try:
        from integration import check_integration
        report['integration']=check_integration(root,mono=args.mono)
    except Exception as exc:
        report['integration']={'passed':False,'error':f'{type(exc).__name__}: {exc}'}
report['passed']=result.wasSuccessful() and (not args.integration or report.get('integration',{}).get('passed',False))
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(target),'unit_passed':result.wasSuccessful(),
                  'integration_passed':report.get('integration',{}).get('passed')},indent=2))
sys.exit(0 if report['passed'] else 1)
