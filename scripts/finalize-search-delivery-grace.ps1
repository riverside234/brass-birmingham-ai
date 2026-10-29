$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=Split-Path -Parent $PSScriptRoot
$study=Join-Path $root 'artifacts/search-delivery-grace/20261029'
$build=Join-Path $study 'release-1.4.45'
$record=Join-Path $root 'artifacts/patch-installation.json'
$final=Join-Path $study 'installed-verification.json'
if(Test-Path -LiteralPath $final){throw 'Preserve final installation evidence.'}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Test processes must retire before finalization.'}
if(Test-Path -LiteralPath (Join-Path $root 'artifacts/search-delivery-grace.pending.json')){throw 'Recover the pending installation first.'}
$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
$a=Read-BrassJson (Join-Path $study 'assessment.json')
$u=Read-BrassJson (Join-Path $study 'upgrade-test.json')
$installed=Read-BrassJson $record
$qualificationHash=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash
if($installed.version -ne '1.4.45-rust-local' -or -not $q.passed -or -not $a.passed -or -not $u.passed -or
 $installed.searchDeliveryGraceQualificationSha256 -ne $qualificationHash -or $u.qualificationSha256 -ne $qualificationHash -or
 $u.installerSha256 -ne (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'install-search-delivery-grace.ps1')).Hash -or
 $q.assessment_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $study 'assessment.json')).Hash -or
 $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $installed.deliveryGraceMs -ne 1000 -or $installed.searchWorkReserveWithinAllowanceMs -ne 0 -or $installed.v205Enabled){throw 'Release qualification or installed policy differs.'}
$pins=@($m.files|ForEach-Object{@{path=('BrassBirminghamUnity_Data/Managed/'+$_.name);sha256=$_.sha256}})
$pins+=@($m.nativeFiles|ForEach-Object{@{path=('BrassNative/'+$_.path);sha256=$_.sha256}})
foreach($pin in $pins){if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw ('Installed runtime differs: '+$pin.path)}}
if(@(Get-ChildItem -LiteralPath (Join-Path $root 'BrassNative') -Recurse -File).Count -ne @($m.nativeFiles).Count){throw 'Native inventory differs.'}
$smokes=@()
foreach($entry in @(@{ms=4000;k=8;late=$false},@{ms=10000;k=16;late=$true},@{ms=12000;k=32;late=$true})){
 $path=Join-Path $study ('installed-'+$entry.ms+'-k'+$entry.k+'/verification.json')
 $r=Read-BrassJson $path
 $key='searchDeliveryGraceInstalled'+$entry.ms+'K'+$entry.k
 if(-not $r.passed -or $r.qualificationSha256 -ne $qualificationHash -or $r.searchK -ne $entry.k -or $r.allowanceMs -ne $entry.ms -or
  $r.lateProbe -ne $entry.late -or $r.deliveryGraceMs -ne 1000 -or $r.actions -ne 2 -or $r.deliveryMs -ge $entry.ms+1000 -or
  $installed.$key -ne $path -or $installed.($key+'Sha256') -ne (Get-FileHash -LiteralPath $path).Hash){throw 'Installed smoke receipt differs.'}
 $smokes+=@{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash;searchMs=$entry.ms;searchK=$entry.k;lateProbe=$r.lateProbe;
  deliveryMs=$r.deliveryMs;tailMs=$r.tailMs;clockMs=$r.turnClockMs;headroomMs=$r.headroomMs;checks=$r.checks}
}
& (Join-Path $PSScriptRoot 'check-search-delivery-grace-user-data.ps1') -Action Verify
$data=Read-BrassJson (Join-Path $study 'user-data-verification.json')
if(-not $data.passed){throw 'User-data preservation failed.'}
Write-BrassJsonAtomic $final @{passed=$true;version=$installed.version;searchAllowancesMs=@(4000,8000,10000,12000);searchKChoices=@(8,16,32);
 deliveryGraceMs=1000;searchWorkReserveWithinAllowanceMs=0;deadlinePolicy=$installed.deadlinePolicy;
 qualificationSha256=$qualificationHash;assessmentSha256=$q.assessment_sha256;installedSmokes=$smokes;installedFiles=$pins;
 userDataVerificationSha256=(Get-FileHash -LiteralPath (Join-Path $study 'user-data-verification.json')).Hash;
 rollbackBackup=$installed.searchDeliveryGraceRecovery;upgradeTestSha256=(Get-FileHash -LiteralPath (Join-Path $study 'upgrade-test.json')).Hash}
$installed|Add-Member -NotePropertyName searchDeliveryGraceInstalledSmoke -NotePropertyValue 'passed' -Force
$installed|Add-Member -NotePropertyName searchDeliveryGraceInstalledVerification -NotePropertyValue $final -Force
$installed|Add-Member -NotePropertyName searchDeliveryGraceInstalledVerificationSha256 -NotePropertyValue ((Get-FileHash -LiteralPath $final).Hash) -Force
Write-BrassJsonAtomic $record $installed
$rows=@('| Search | K | Delivery after cutoff | Clock headroom |','| --- | --- | --- | --- |')
foreach($r in $a.rows){$rows+=('| '+($r.search_ms/1000)+' s | '+$r.search_k+' | '+([Math]::Round($r.tail_ms,1))+' ms | '+([Math]::Round($r.headroom_ms,1))+' ms |')}
$installedRows=@('| Search | K | Probe | Total delivery | Clock headroom |','| --- | --- | --- | --- | --- |')
foreach($r in $smokes){$installedRows+=('| '+($r.searchMs/1000)+' s | '+$r.searchK+' | '+$(if($r.lateProbe){'Delayed delivery'}else{'Normal'})+' | '+([Math]::Round($r.deliveryMs,1))+' ms | '+([Math]::Round($r.headroomMs,1))+' ms |')}
$report=@"
# NN full search allowance and fixed delivery grace

Installed release: **1.4.45-rust-local**, 2026-10-29.

The selected 4/8/10/12-second allowance is now available to NN search. Every K choice (8, 16, 32) receives one additional delivery second, giving total turn-clock deadlines of 5/9/11/13 seconds. This replaces the previous delivery reserve deducted from search time. The AI selection panel explains the additional second.

The search clock retains its existing boundary: model startup and initial live-board snapshot preparation happen before it starts. These totals therefore describe the running AI turn clock, rather than time from clicking a menu button. A selected allowance is a maximum; search can finish earlier.

## Deadline and correctness contract

- Rust freezes the search cutoff and the total deadline before starting either owner. NN simulation cannot consume the delivery second. Native retirement, completion, IPC and host delivery share the remaining time.
- Unity accepts delivery only until the earlier of the original total deadline or one second after receiving the result. It requires 100 ms of measured deadline headroom before starting a main-thread commit.
- Retries and replans keep the same running clock and frozen total deadline. Loading a legacy partial-turn clock uses its saved remaining time without adding another second. A new completed-turn save starts a new turn normally.
- Identity, generation, revision, actor, cancellation, detached legality checks and second-action revalidation remain in place. An expired result stops the NN turn before mutation.
- This update uses search_v1, V205 disabled, the existing g890 test NN and ONNX Runtime CPU inference with two independent sessions and one compute thread per session. K and temperature choices remain available.

## Short qualification tests

All 12 combinations of K8/K16/K32 and 4/8/10/12 seconds passed. Cases cover Canal, branching Rail and late Rail with a long event history. The probe pauses only the test bridge until the search cutoff; for an already-completed reply it sets test-only delivery timing at resume. Original native timing and the unmodified turn clock are recorded separately. These are delivery stress probes, not strength matches or a population latency guarantee.

$($rows -join "`n")

The slowest measured delivery tail was $([Math]::Round($a.maximum_delivery_tail_ms,1)) ms; minimum remaining clock headroom was $([Math]::Round($a.minimum_headroom_ms,1)) ms. An additional expired-reply probe crossed the total deadline and committed no actions. Ordinary 4-second K8 and K32 turns also passed.

The exact shipped Mono runtime passed 64 budget/clock checks. Native reserve tests passed all 9 cases. Unity lifecycle tests passed $($a.lifecycle_checks) checks, including cancellation, stale replies, actor/revision changes, second-action validation, save/load and completed-turn checkpoints. The panel screenshot was inspected. Disposable installation tests verified interrupted-install recovery and exact rollback to 1.4.44.

## Installed game verification

$($installedRows -join "`n")

All installed smokes executed both actions and verified save/load against the installed DLL and native binary. Installed runtime hashes match the qualified candidate. All $($data.savesAndHistoryFiles) user save/history files remain byte-for-byte unchanged. Registry-backed preference measurement was unavailable in the sandbox; the installer does not write preferences and test menu settings use the existing in-memory self-test store.

Evidence: artifacts/search-delivery-grace/20261029/assessment.json, release-1.4.45/qualification.json, upgrade-test.json, installed-verification.json and user-data-verification.json. Test source snapshots, test DLL hashes and raw results are retained alongside each case.

## Recovery

Close the game before recovery. Run scripts/install-search-delivery-grace.ps1 with -Action Rollback to restore the exact previous 1.4.44 DLL, native package and installation receipt. Use -Action Recover if an installation was interrupted. The installed receipt identifies the retained backup at $($installed.searchDeliveryGraceRecovery).
"@
$report|Set-Content -LiteralPath (Join-Path $root 'ai-research/NN_SEARCH_DELIVERY_GRACE_GAME_UPDATE.md') -Encoding utf8
Write-Output 'Finalized installed 1.4.45: full NN search allowance plus one delivery second; all short smokes passed.'
