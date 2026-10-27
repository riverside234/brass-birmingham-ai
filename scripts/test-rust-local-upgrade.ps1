param([string]$StudyDirectory='artifacts/rust-backend/20261026-local-release')
$ErrorActionPreference='Stop'
$upgradeRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$upgradeStudy=Join-Path $upgradeRoot $StudyDirectory
$upgradeOut=Join-Path $upgradeStudy 'upgrade-test'
if(Test-Path -LiteralPath $upgradeOut){throw 'Preserve existing upgrade evidence.'}
$upgradeGame=Join-Path $upgradeOut 'game';$upgradeManaged=Join-Path $upgradeGame 'BrassBirminghamUnity_Data/Managed'
[void](New-Item -ItemType Directory -Path $upgradeManaged)
[void](New-Item -ItemType Directory -Path (Join-Path $upgradeGame 'artifacts'))
$upgradeRecord=Join-Path $upgradeGame 'artifacts/patch-installation.json'
$upgradePrior=Read-BrassJson (Join-Path $upgradeRoot 'artifacts/patch-installation.json')
if($upgradePrior.version -ne '1.4.35-alpha-test'){throw 'Test must begin from current L13.'}
$upgradePrior.gameRoot=$upgradeGame;$upgradePrior.saveDataPaths=@()
Write-BrassJsonAtomic $upgradeRecord $upgradePrior
$upgradeHash=(Get-FileHash -LiteralPath $upgradeRecord).Hash
foreach($file in $upgradePrior.files){Copy-Item -LiteralPath (Join-Path $upgradeRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)) -Destination (Join-Path $upgradeManaged $file.name)}
Copy-Item -LiteralPath (Join-Path $upgradeRoot 'BrassNative') -Destination $upgradeGame -Recurse
$upgradeSentinel=Join-Path $upgradeGame 'nn-save-sentinel.json';[IO.File]::WriteAllText($upgradeSentinel,'{"family":"nn","checkpoint":"human-end"}')
$upgradeSaveHash=(Get-FileHash -LiteralPath $upgradeSentinel).Hash
$upgradeInstaller=Join-Path $PSScriptRoot 'install-rust-local-release.ps1'
function Assert-Prior {
 if((Get-FileHash -LiteralPath $upgradeRecord).Hash -ne $upgradeHash){throw 'Prior receipt differs.'}
 foreach($file in $upgradePrior.files){if((Get-FileHash -LiteralPath (Join-Path $upgradeManaged $file.name)).Hash -ne $file.sha256){throw 'Prior managed file differs.'}}
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $upgradeRoot 'BrassNative') -Recurse -File){
  $relative=$file.FullName.Substring((Join-Path $upgradeRoot 'BrassNative').Length+1)
  if((Get-FileHash -LiteralPath $file.FullName).Hash -ne (Get-FileHash -LiteralPath (Join-Path $upgradeGame ('BrassNative/'+$relative))).Hash){throw 'Prior native file differs.'}
 }
 if((Get-FileHash -LiteralPath $upgradeSentinel).Hash -ne $upgradeSaveHash){throw 'Save modified.'}
}
$rejected=$false
try{& $upgradeInstaller -BuildDirectory 'artifacts/rust-backend/20261026-l15/candidate' -GameRoot $upgradeGame}catch{$rejected=$true}
if(-not $rejected){throw 'Unqualified candidate admitted.'};Assert-Prior
# A modified qualified binary must be rejected without publication.
$tampered=Join-Path $upgradeOut 'tampered';Copy-Item -LiteralPath (Join-Path $upgradeStudy 'candidate') -Destination $tampered -Recurse
[IO.File]::AppendAllText((Join-Path $tampered 'Managed/BrassLegacyAI.dll'),'tamper')
$rejected=$false
try{& $upgradeInstaller -BuildDirectory ($StudyDirectory+'/upgrade-test/tampered') -GameRoot $upgradeGame}catch{$rejected=$true}
if(-not $rejected){throw 'Tampered candidate admitted.'};Assert-Prior
foreach($point in 'NativePublished','ManagedPublished'){
 $interrupted=$false
 try{& $upgradeInstaller -BuildDirectory ($StudyDirectory+'/candidate') -GameRoot $upgradeGame -TestInterruptAt $point}catch{
  if($_.Exception.Message -notlike 'TEST INTERRUPT:*'){throw};$interrupted=$true
 }
 if(-not $interrupted -or -not(Test-Path -LiteralPath (Join-Path $upgradeGame 'artifacts/rust-local-release.pending.json'))){throw 'Fault injection failed.'}
 & $upgradeInstaller -Action Recover -GameRoot $upgradeGame;Assert-Prior
}
& $upgradeInstaller -BuildDirectory ($StudyDirectory+'/candidate') -GameRoot $upgradeGame
$installed=Read-BrassJson $upgradeRecord
if(-not $installed.localReleaseQualified -or $installed.version -ne '1.4.36-rust-local'){throw 'Full release not installed.'}
& $upgradeInstaller -Action Rollback -GameRoot $upgradeGame;Assert-Prior
Write-BrassJsonAtomic (Join-Path $upgradeOut 'receipt.json') ([ordered]@{passed=$true;unqualifiedRejected=$true;tamperedRejected=$true;
 interruptedNativeRecovery=$true;interruptedManagedRecovery=$true;successfulUpgradeAndExactRollback=$true;saveUnchanged=$true;
 candidateManifestSha256=(Get-FileHash -LiteralPath (Join-Path $upgradeStudy 'candidate/build-manifest.json')).Hash;
 qualificationSha256=(Get-FileHash -LiteralPath (Join-Path $upgradeStudy 'candidate/qualification.json')).Hash;
 installerSha256=(Get-FileHash -LiteralPath $upgradeInstaller).Hash;priorReceiptSha256=$upgradeHash})
Write-Output 'L13 upgrade, exact rollback, both interrupted stages and rejection controls passed.'
