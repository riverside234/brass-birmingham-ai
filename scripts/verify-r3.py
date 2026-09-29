"""R3 exit runner: focused tests, frozen source/runtime identity and integration evidence.

Run the separate .NET/Unity and replay diagnostics first. Tournament promotion
is deliberately separate, so a passing correctness report is not a strength claim.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT,load_bank,read,require,sha
from brass_ai.qualification import write_new

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--bank',required=True)
    p.add_argument('--output',required=True);p.add_argument('--unit-report',help='Reuse a completed check-frozen-bank.py report for this exact bank')
    a=p.parse_args();root=a.root.resolve()
    payload=load_bank(root,a.bank)['payload']
    # This test suite executes the current build; verify it is the bank build.
    for path,h in payload['source_files'].items():
        if path.startswith('src/Brass'):
            require(sha(root/path)==h,'Current test source differs from frozen bank: '+path)
    for f in payload['runtime_files']:
        require(sha(root/'artifacts/legacy-ai/runtime'/f['name'])==f['sha256'],'Current test runtime differs')
    if a.unit_report:
        unit=read(root/a.unit_report)
        require(unit['passed'] and unit['bank']==a.bank and unit['tests']>0 and unit['failures']==unit['errors']==0,'Focused R3 checks failed or mismatched')
        require(unit['manifest_sha256']==sha(root/'training/opponents/bank'/a.bank/'manifest.json'),'Unit report belongs to another manifest')
        test_count=unit['tests']
    else:
        suite=unittest.defaultTestLoader.discover(str(root/'tests/legacy-ai'),pattern='test*.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        require(result.wasSuccessful(),'Focused R3 tests failed');test_count=result.testsRun
    files={
        'dotnet':'artifacts/legacy-ai/bridge-result.json',
        'unity':'artifacts/legacy-ai/unity/legacy-unity-result.json',
        'unity_routing':'artifacts/legacy-ai/unity/overlay-ui-result.json',
        'replay':'artifacts/legacy-ai/r3-integration-result.json',
        'diversity':'artifacts/legacy-ai/style-diversity.json',
        'r2':'artifacts/r2/exit-checks.json'}
    if a.unit_report:files['units']=a.unit_report
    evidence={k:read(root/path) for k,path in files.items()}
    for k in ('dotnet','unity','unity_routing','replay','r2'):require(evidence[k]['passed'],k+' failed')
    require(evidence['replay']['bank']==a.bank and evidence['diversity']['bank']==a.bank,'Integration evidence belongs to another bank')
    require(evidence['diversity']['each_specialist_differed'],'Tactical distinction was not demonstrated')
    binary=next(f['sha256'] for f in payload['runtime_files'] if f['name']=='BrassLegacyAI.exe')
    require(sha(root/'artifacts/legacy-ai/rebuild-check/BrassLegacyAI.exe')==binary,'Snapshot rebuild differs from qualified binary')
    for f in payload['runtime_files']:
        if f['name']=='BrassLegacyAI.exe':continue
        require(sha(root/'BrassBirminghamUnity_Data/Managed'/f['name'])==f['sha256'],'Installed rules dependency changed')
    host=read(root/'configs/learning/host-1.0.0.json')
    require(sha(root/'artifacts/r1-host/guarded/BrassHost.exe')==host['host_sha256'],'R1 host changed')
    report=dict(format='r3-correctness/1.0.0',passed=True,checked_at_utc=datetime.now(timezone.utc).isoformat(),bank=a.bank,
        unit_tests=test_count,unit_failures=0,unit_errors=0,
        bank_manifest_sha256=sha(root/'training/opponents/bank'/a.bank/'manifest.json'),worker_sha256=binary,
        evidence={k:dict(path=path,sha256=sha(root/path)) for k,path in files.items()},
        source_hashes={path:sha(root/path) for path in ('src/brass_ai/qualification.py','scripts/verify-r3.py','tests/legacy-ai/test_advanced.py','tests/legacy-ai/test_bank.py','tests/legacy-ai/BridgeChecks.cs')},
        dotnet_assertions=evidence['dotnet']['assertions'],unity_assertions=evidence['unity']['result']['assertions'],
        complete_live_turns=evidence['unity']['result']['full_match']['turns'],
        replay_decisions=evidence['replay']['decisions'],r2_regressions=evidence['r2']['unit_tests'],
        tactical_differences=evidence['diversity']['differences_from_balanced'],
        identical_snapshot_rebuild=True,installed_rules_unchanged=True,r1_host_unchanged=True,
        human_play_acceptance=False,linux_qualified=False,strength_qualification='separate prespecified tournament')
    write_new(root/a.output,report);print(json.dumps(report))

if __name__=='__main__':main()
