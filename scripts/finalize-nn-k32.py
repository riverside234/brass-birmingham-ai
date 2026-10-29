"""Verify installed K32 files and both bounded live smokes, then record release."""
from pathlib import Path
import runpy
import shutil

B = runpy.run_path(str(Path(__file__).with_name('build-nn-k32.py')))
ROOT, OUT = B['ROOT'], B['OUT']
read, write, sha, pins = (B[k] for k in ('read', 'write', 'sha', 'pins'))


def main():
    destination = OUT.parent
    assert not (destination / 'installed-verification.json').exists()
    m, q = read(OUT / 'build-manifest.json'), read(OUT / 'qualification.json')
    installed = read(ROOT / 'artifacts/patch-installation.json')
    assert q['passed'] and installed['version'] == m['version'] == '1.4.43-rust-local'
    assert installed['searchMethod'] == 'search_v1' and not installed['v205Enabled']
    assert installed['nnSearchKChoices'] == [8,16,32] and installed['nnSearchKDefault'] == 16
    assert installed['nnTemperatureChoices'] == [1.0,1.5] and installed['savePolicy'] == 'completed-turn-checkpoint/1.0.0'
    assert installed['nnK32QualificationSha256'].lower() == sha(OUT / 'qualification.json')
    assert q['build_manifest_sha256'] == sha(OUT / 'build-manifest.json')
    for pin in m['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / pin['name']) == pin['sha256']
    assert pins(ROOT / 'BrassNative') == m['nativeFiles']
    for base_name, name in [('policy.json','policy-k32.json'), ('policy-t15.json','policy-k32-t15.json')]:
        base = read(ROOT / 'BrassNative/alpha-g890' / base_name)
        policy = read(ROOT / 'BrassNative/alpha-g890' / name)
        assert policy['method'] == 'search_v1' and set(policy['k_by_allowance_ms'].values()) == {32}
        policy['k_by_allowance_ms'] = base['k_by_allowance_ms']
        assert policy == base
    recovery = Path(installed['nnK32Recovery'])
    assert recovery.resolve().is_relative_to((ROOT / 'artifacts/patch-recovery').resolve())
    assert sha(recovery / 'patch-installation.json') == m['baseline_receipt_sha256']
    upgrade = read(destination / 'upgrade-test.json')
    assert upgrade['passed'] and upgrade['saves_unchanged'] and upgrade['interrupted_install_recovered'] and upgrade['install_and_rollback_passed']
    smokes = []
    for label, temperature in [('installed-t10',1.0), ('installed-t15',1.5)]:
        directory = destination / label
        verification, result = read(directory / 'verification.json'), read(directory / 'result.json')
        assert verification['passed'] and verification['version'] == installed['version']
        assert verification['checks'] >= 740 and verification['searchK'] == 32 and verification['temperature'] == temperature
        assert verification['qualificationSha256'].lower() == sha(OUT / 'qualification.json')
        assert verification['resultSha256'].lower() == sha(directory / 'result.json')
        assert result['passed'] and result['temperature'] == temperature and result['search_k'] == 32
        assert result['search_method'] == 'search_v1' and not result['focused']
        assert len(result['commits']) == 2 and result['commits'][-1]['remaining'] == 0 and result['delivery_ms'] < 4000
        assert result['readiness']['k32_host_delivery_reserve_ns'] == 1_200_000_000
        smokes.append(dict(temperature=temperature, searchK=32, checks=verification['checks'],
            deliveryMs=result['delivery_ms'], verification=str(directory / 'verification.json'), sha256=sha(directory / 'verification.json')))
    shutil.copyfile(ROOT / 'artifacts/patch-installation.json', destination / 'deployment.json')
    verification = dict(passed=True, version=installed['version'], searchMethod='search_v1', v205Enabled=False,
        searchKChoices=[8,16,32], defaultSearchK=16, allAllowancesMs=[4000,8000,10000,12000],
        temperatures=[1.0,1.5], nativeFiles=len(m['nativeFiles']), managedAssemblies=len(m['files']),
        nativeSourceSha256=m['native_source_sha256'], sharedGameSearchCoreUnchanged=True, modelExportAndOrtUnchanged=True,
        k32HostReserveNs=1_200_000_000, hostReserveTests=4, uiChecks=q['checks']['ui-write-final'],
        uiRestartChecks=q['checks']['ui-read-final'], installedSmokes=smokes, recovery=str(recovery),
        upgradeRecoveryAndRollbackPassed=True, qualificationSha256=sha(OUT / 'qualification.json'), deploymentSha256=sha(destination / 'deployment.json'))
    write(destination / 'installed-verification.json', verification)
    text = f'''# NN search K32 — local release 1.4.43

AI selection → NN → Alpha Test now offers **K8 / K16 / K32** in the existing cream/gold control style. The choice applies to the whole NN table at every **4/8/10/12-second** difficulty and both **1.0/1.5** temperatures. Seats can retain different time budgets. K16 remains the default; existing saved choices are preserved.

Apply saves the setting; Cancel discards the draft. K32 is included in the existing assignment integrity hash and restored from both menu preferences and completed-turn saves. Older saves without a K field still use K16 without being rewritten. K32 saves require this release to resume.

The installed backend remains **search_v1**, with V205 disabled, two independent CPU sessions and one compute thread each. The existing g890 test model, ONNX export, ONNX Runtime provider, and shared Rust rules/search core are preserved. The local worker adds an admission-time **1.2-second delivery reserve for K32** within the original turn deadline; K8/K16 keep their previous reserve. The maximum 12-second budget already reserved 1.2 seconds. No retry can extend the original deadline.

This reserve addresses the retained pre-fix smoke: one run requested action two after its clock expired, and another returned its native plan at 3.63 seconds but finished both Unity commits at 4.40 seconds. The measured extra interval included snapshot work, validation, frame scheduling and commit; it was not all validation CPU. Wider roots can leave less support for a full two-action plan, but that was not established as the cause of the first retry. Earlier work cutoff gives the existing legal completion and guarded commit path time to finish.

Short qualification passed {q['checks']['ui-write-final']} UI/preference checks and {q['checks']['ui-read-final']} fresh-process restart checks, all K/temperature/difficulty combinations, invalid/tampered settings, legacy saves, Apply/Cancel, and large-text layout at 1920×1080 and 1920×1200. Four Rust reserve tests cover unchanged deadlines, monotonic narrowing, preserved long-budget reserve and invalid reserves. Interrupted installation recovery and exact rollback passed with an unchanged save sentinel.

The actual installed game completed both actions and completed-turn save/load at K32 in **{smokes[0]['deliveryMs']:.2f} ms at T1.0** and **{smokes[1]['deliveryMs']:.2f} ms at T1.5**, with {smokes[0]['checks']} checks per smoke. These are bounded functional checks, not a playing-strength or latency-tail guarantee. Installation was verified against all {len(m['nativeFiles'])} native files and three managed assemblies.

Evidence: [installed verification](../artifacts/nn-k32/installed-verification.json), [qualification](../artifacts/nn-k32/candidate-v2/qualification.json), [reserve tests/build](../artifacts/nn-k32/native-build.json), [recovery test](../artifacts/nn-k32/upgrade-test.json), and [panel preview](../artifacts/nn-k32/candidate-v2/evidence/images/setup-nn-1080-largest.png). Initial failed/late smokes remain under `artifacts/nn-k32/live-t10*`.

With the game closed, exact rollback is `& ./scripts/install-nn-k32.ps1 -Action Rollback`; use `-Action Recover` after an interrupted installation. User saves are preserved. The separately frozen lab/training bundles are not changed.
'''
    (ROOT / 'ai-research/NN_K32_GAME_UPDATE.md').write_text(text, encoding='utf-8')
    print(verification)


if __name__ == '__main__':
    main()
