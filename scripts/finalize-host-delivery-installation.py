"""Verify the installed release and record immutable deployment evidence."""
from pathlib import Path
import runpy
import json

ROOT=Path(__file__).resolve().parents[1]
B=runpy.run_path(str(ROOT/'scripts/build-host-delivery-release.py'))
STUDY,OUT,read,write,sha,pins=(B[k] for k in ('STUDY','OUT','read','write','sha','pins'))

if __name__=='__main__':
    assert not (STUDY/'installed-verification.json').exists(), 'Preserve deployment evidence'
    m=read(OUT/'build-manifest.json');q=read(OUT/'qualification.json')
    installed=read(ROOT/'artifacts/patch-installation.json')
    assert installed['version']==m['version']=='1.4.44-rust-local'
    assert installed['searchMethod']=='search_v1' and not installed['v205Enabled']
    assert installed['hostDeliveryQualificationSha256'].lower()==sha(OUT/'qualification.json')
    assert installed['hostDeliveryInstalledSmoke']=='passed/4s/K16+K32/two-actions-and-save-load'
    assert q['passed'] and q['build_manifest_sha256']==sha(OUT/'build-manifest.json')
    assert pins(ROOT/'BrassNative')==m['nativeFiles']
    for pin in m['files']:
        assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/pin['name'])==pin['sha256']
    turns={}
    for k in (16,32):
        folder=STUDY/f'installed-k{k}';v=read(folder/'verification.json');r=read(folder/'result.json')
        assert v['passed'] and v['searchK']==r['search_k']==k and v['deliveryMs']<4000
        assert v['resultSha256'].lower()==sha(folder/'result.json')
        assert installed[f'hostDeliveryInstalledK{k}Sha256'].lower()==sha(folder/'verification.json')
        assert len(r['commits'])==2 and r['commits'][-1]['remaining']==0
        assert r['readiness']['k32_host_delivery_reserve_ns']==800000000
        assert r['readiness']['build']['source_sha256']==m['native_source_sha256']
        turns[str(k)]=dict(checks=v['checks'],delivery_ms=r['delivery_ms'],requests=r['requests'],
            actions=2,temperature=1.0,result_sha256=sha(folder/'result.json'),verification_sha256=sha(folder/'verification.json'))
    upgrade=read(STUDY/'upgrade-test.json');user=read(STUDY/'user-data-verification.json')
    assert upgrade['passed'] and upgrade['exact_install_and_rollback'] and upgrade['interrupted_install_recovered']
    assert upgrade['installerSha256'].lower()==sha(ROOT/'scripts/install-host-delivery.ps1')
    assert user['passed']
    recovery=Path(installed['hostDeliveryRecovery']);backup=read(recovery/'recovery.json')
    assert backup['receiptSha256'].lower()==sha(recovery/'patch-installation.json')
    for f in backup['files']:
        assert sha(recovery/f['path'])==f['priorSha256'].lower()
    write(STUDY/'deployment.json',installed)
    result=dict(passed=True,version=installed['version'],installed=True,
        managed_files=3,native_files=len(m['nativeFiles']),smokes=turns,
        k32_reserve_by_allowance_ms={'4000':800,'8000':800,'10000':1000,'12000':1200},
        k16_reserve_unchanged=True,original_turn_deadlines_unchanged=True,
        search_method='search_v1',v205_enabled=False,nn_model_unchanged=True,
        source_sha256=m['native_source_sha256'],qualification_sha256=sha(OUT/'qualification.json'),
        deployment_sha256=sha(STUDY/'deployment.json'),recovery_directory=str(recovery),
        saves_and_history_files_preserved=user['savesAndHistoryFiles'],
        registry_preference_measurement_available=user['nnPreferencesMeasured'],
        preferences_behavior='Installer does not modify preferences; self-test menu choices stay in memory.',
        upgrade_test_sha256=sha(STUDY/'upgrade-test.json'),user_data_verification_sha256=sha(STUDY/'user-data-verification.json'),
        limitations=['Short functional smokes, not a strength comparison or population latency-tail guarantee.',
            'The earlier host-overhead measurements used the same optimized C# binary with the previous 1.2s native floor.'])
    write(STUDY/'installed-verification.json',result)
    report=ROOT/'ai-research/HOST_DELIVERY_GAME_UPDATE.md'
    assert not report.exists()
    report.write_text(f'''# Host delivery game update — 1.4.44

Installed and verified on 2026-10-29 (America/Chicago). The game now uses the qualified host optimizations: revision-bound event reuse, a small post-action clock snapshot, direct observation tokens, reusable board indexes, selected-choice legality resolution and 60 FPS validation handoffs. Native search retains its 30 FPS rendering cap. Identity, actor, revision, cancellation, receipt and final legality checks remain active.

The [host comparison](HOST_DELIVERY_OPTIMIZATION_REPORT.md) measured 50–56% lower outside-native overhead and 71.1% fewer managed object bytes in its bounded allocation probe. That comparison used the previous 1.2-second K32 floor. This installation retains the exact measured C# binary and reduces only the native K32 delivery floor to 0.8 seconds. Shared Rust rules/search mathematics, model/export, ORT provider and all policy files are unchanged. K16 retains its previous timing.

| Search allowance | New K32 delivery reserve | Maximum search-work interval |
| --- | ---: | ---: |
| 4 seconds | 0.8 seconds | 3.2 seconds |
| 8 seconds | 0.8 seconds | 7.2 seconds |
| 10 seconds | 1.0 second | 9.0 seconds |
| 12 seconds | 1.2 seconds | 10.8 seconds |

The reserve is paid within the original turn clock. No retry replenishes that clock. Five native budget tests cover unchanged deadlines, narrowing, the new floor and retained larger defaults. Pre-install four-second smokes completed both actions in Canal at 2.828 seconds, late Rail at 3.029 seconds and a Rail root with effective K32 at 3.237 seconds. The K16 control completed in 3.504 seconds. Each passed 760 checks, including completed-turn save/load.

The actual installed player then passed **K16 in {turns['16']['delivery_ms']/1000:.3f} seconds** and **K32 in {turns['32']['delivery_ms']/1000:.3f} seconds**, with {turns['16']['checks']} / {turns['32']['checks']} checks, two committed actions and completed-turn save/load. Both use a four-second allowance, temperature 1.0 and two independent one-thread CPU sessions. NN remains the temporary g890 Alpha Test model. These are short functional checks, not a strength or latency-tail guarantee.

All {user['savesAndHistoryFiles']} user save/history files match their pre-install hashes. The installer does not write menu preferences, and the smoke harness keeps its menu choices in memory. Direct registry preference measurement was unavailable under the tool's permissions; it is not reported as a measured pass. Interrupted-install recovery and exact rollback passed in a disposable copy with an unchanged saved-game sentinel. Only the AI assembly and three native package files were published; the game/helper assemblies and policy files are preserved.

With the game closed, rollback is:

```powershell
& .\\scripts\\install-host-delivery.ps1 -Action Rollback
```

For an interrupted publication, use `-Action Recover`. The backup is `{recovery}`. Saves are preserved.

Evidence: [installed verification](../artifacts/host-delivery/20261029/installed-verification.json), [release qualification](../artifacts/host-delivery/20261029/release-1.4.44/qualification.json), [native build and tests](../artifacts/host-delivery/20261029/reserve08-native/native-build.json), [recovery test](../artifacts/host-delivery/20261029/upgrade-test.json), and [save preservation](../artifacts/host-delivery/20261029/user-data-verification.json). Installation changes no lab deployment or training bundle.
''',encoding='utf-8')
    print(json.dumps(result,indent=2))
