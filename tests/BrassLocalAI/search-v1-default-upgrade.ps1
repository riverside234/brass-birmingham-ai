$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$evidence=Join-Path $root 'artifacts/search-v1-default'
$target=Join-Path $evidence ('upgrade-test/'+[Guid]::NewGuid().ToString('N'))
$managed=Join-Path $target 'BrassBirminghamUnity_Data/Managed'
[void](New-Item -ItemType Directory -Path $managed,(Join-Path $target 'artifacts'),(Join-Path $target 'saves'))
$record=Get-Content -LiteralPath (Join-Path $evidence 'before-installation.json') -Raw|ConvertFrom-Json
if($record.version -ne '1.4.41-rust-local'){throw 'Expected exact 1.4.41 base.'}
foreach($entry in $record.files){Copy-Item -LiteralPath (Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$entry.name)) -Destination $managed}
Copy-Item -LiteralPath (Join-Path $root 'BrassNative') -Destination (Join-Path $target 'BrassNative') -Recurse
$record.gameRoot=$target;$record.saveDataPaths=@((Join-Path $target 'saves'))
$record|ConvertTo-Json -Depth 15|Set-Content -LiteralPath (Join-Path $target 'artifacts/patch-installation.json') -Encoding utf8
'Preserve this saved-game sentinel.'|Set-Content -LiteralPath (Join-Path $target 'saves/untouched.storage')
$saved=(Get-FileHash -LiteralPath (Join-Path $target 'saves/untouched.storage')).Hash
$prior=(Get-FileHash -LiteralPath (Join-Path $target 'artifacts/patch-installation.json')).Hash
$base=Get-Content -LiteralPath (Join-Path $evidence 'candidate/build-manifest.json') -Raw|ConvertFrom-Json
$base.nativeFiles=$base.baseNative
$installer=Join-Path $root 'scripts/install-search-v1-default.ps1'
function VerifyPrior {
 if((Get-FileHash -LiteralPath (Join-Path $target 'artifacts/patch-installation.json')).Hash -ne $prior){throw 'Prior receipt was not restored exactly.'}
 foreach($entry in $record.files){if((Get-FileHash -LiteralPath (Join-Path $managed $entry.name)).Hash -ne $entry.sha256){throw 'Prior managed runtime differs.'}}
 foreach($pin in $base.nativeFiles){if((Get-FileHash -LiteralPath (Join-Path $target ('BrassNative/'+$pin.path))).Hash -ne $pin.sha256){throw 'Prior native package differs.'}}
 if((Get-FileHash -LiteralPath (Join-Path $target 'saves/untouched.storage')).Hash -ne $saved){throw 'Save sentinel changed.'}
}
$interrupted=$false
try{& $installer -GameRoot $target -TestInterrupt}catch{if($_.Exception.Message -notlike '*TEST INTERRUPT*'){throw};$interrupted=$true}
if(-not $interrupted){throw 'Expected interruption.'}
& $installer -GameRoot $target -Action Recover
VerifyPrior
& $installer -GameRoot $target
$installed=Get-Content -LiteralPath (Join-Path $target 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
if($installed.version -ne '1.4.42-rust-local' -or $installed.searchMethod -ne 'search_v1' -or $installed.v205Enabled -or ($installed.nnTemperatureChoices -join ',') -ne '1,1.5' -or $installed.savePolicy -ne 'completed-turn-checkpoint/1.0.0'){throw 'Upgrade receipt mismatch.'}
& $installer -GameRoot $target -Action Rollback
VerifyPrior
@{passed=$true;interrupted_install_recovered=$true;install_and_rollback_passed=$true;saves_unchanged=$true;target=$target;
 installerSha256=(Get-FileHash -LiteralPath $installer).Hash;qualificationSha256=(Get-FileHash -LiteralPath (Join-Path $evidence 'candidate/qualification.json')).Hash}|
 ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $evidence 'upgrade-test.json') -Encoding utf8
