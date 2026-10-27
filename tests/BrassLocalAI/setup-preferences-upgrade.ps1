$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$evidence=Join-Path $root 'artifacts/setup-preferences'
$target=Join-Path $evidence ('upgrade-test/'+[Guid]::NewGuid().ToString('N'))
$managed=Join-Path $target 'BrassBirminghamUnity_Data/Managed'
[void](New-Item -ItemType Directory -Path $managed,(Join-Path $target 'artifacts'),(Join-Path $target 'saves'))
$record=Get-Content -LiteralPath (Join-Path $evidence 'before-installation.json') -Raw|ConvertFrom-Json
if($record.version -ne '1.4.37-rust-local'){throw 'Expected 1.4.37 base.'}
foreach($entry in $record.files){Copy-Item -LiteralPath (Join-Path $evidence ('baseline/Managed/'+$entry.name)) -Destination $managed}
Copy-Item -LiteralPath (Join-Path $root 'BrassNative') -Destination (Join-Path $target 'BrassNative') -Recurse
$record.gameRoot=$target;$record.saveDataPaths=@((Join-Path $target 'saves'))
$record|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $target 'artifacts/patch-installation.json') -Encoding utf8
'Preserve this saved-game sentinel.'|Set-Content -LiteralPath (Join-Path $target 'saves/untouched.storage')
$saved=(Get-FileHash -LiteralPath (Join-Path $target 'saves/untouched.storage')).Hash
$prior=(Get-FileHash -LiteralPath (Join-Path $target 'artifacts/patch-installation.json')).Hash
$installer=Join-Path $root 'scripts/install-setup-preferences-patch.ps1'
$interrupted=$false
try{& $installer -GameRoot $target -TestInterrupt}catch{if($_.Exception.Message -notlike '*TEST INTERRUPT*'){throw};$interrupted=$true}
if(-not $interrupted){throw 'Expected interrupted publication.'}
& $installer -GameRoot $target -Action Recover
if((Get-FileHash -LiteralPath (Join-Path $target 'artifacts/patch-installation.json')).Hash -ne $prior){throw 'Recovery did not restore receipt.'}
foreach($entry in $record.files){if((Get-FileHash -LiteralPath (Join-Path $managed $entry.name)).Hash -ne $entry.sha256){throw 'Recovery did not restore runtime.'}}
& $installer -GameRoot $target
$installed=Get-Content -LiteralPath (Join-Path $target 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
if($installed.version -ne '1.4.38-rust-local' -or $installed.savePolicy -ne 'completed-turn-checkpoint/1.0.0'){throw 'Upgrade receipt mismatch.'}
& $installer -GameRoot $target -Action Rollback
if((Get-FileHash -LiteralPath (Join-Path $target 'artifacts/patch-installation.json')).Hash -ne $prior){throw 'Rollback did not restore receipt.'}
foreach($entry in $record.files){if((Get-FileHash -LiteralPath (Join-Path $managed $entry.name)).Hash -ne $entry.sha256){throw 'Rollback did not restore runtime.'}}
if((Get-FileHash -LiteralPath (Join-Path $target 'saves/untouched.storage')).Hash -ne $saved){throw 'Save sentinel changed.'}
@{passed=$true;interrupted_install_recovered=$true;install_and_rollback_passed=$true;saves_unchanged=$true;target=$target;
 installerSha256=(Get-FileHash -LiteralPath $installer).Hash;qualificationSha256=(Get-FileHash -LiteralPath (Join-Path $evidence 'candidate/qualification.json')).Hash}|
 ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $evidence 'upgrade-test.json') -Encoding utf8
