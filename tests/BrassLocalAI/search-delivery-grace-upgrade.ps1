$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$study=Join-Path $root 'artifacts/search-delivery-grace/20261029'
$target=Join-Path $study ('install-tests/'+[Guid]::NewGuid().ToString('N'))
$managed=Join-Path $target 'BrassBirminghamUnity_Data/Managed'
[void](New-Item -ItemType Directory -Path $managed,(Join-Path $target 'artifacts'),(Join-Path $target 'saves'))
$record=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
if($record.version -ne '1.4.44-rust-local'){throw 'Exact 1.4.44 base required.'}
foreach($entry in $record.files){Copy-Item -LiteralPath (Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$entry.name)) -Destination $managed}
Copy-Item -LiteralPath (Join-Path $root 'BrassNative') -Destination (Join-Path $target 'BrassNative') -Recurse
$record.gameRoot=$target;$record.saveDataPaths=@((Join-Path $target 'saves'))
$receipt=Join-Path $target 'artifacts/patch-installation.json'
$record|ConvertTo-Json -Depth 15|Set-Content -LiteralPath $receipt -Encoding utf8
'Saved-game sentinel: preserve byte for byte.'|Set-Content -LiteralPath (Join-Path $target 'saves/unchanged.storage')
$saved=(Get-FileHash -LiteralPath (Join-Path $target 'saves/unchanged.storage')).Hash
$prior=(Get-FileHash -LiteralPath $receipt).Hash
$installer=Join-Path $root 'scripts/install-search-delivery-grace.ps1'
function VerifyPrior {
 if((Get-FileHash -LiteralPath $receipt).Hash -ne $prior){throw 'Prior receipt not restored exactly.'}
 foreach($pin in $record.files){if((Get-FileHash -LiteralPath (Join-Path $managed $pin.name)).Hash -ne $pin.sha256){throw 'Prior managed runtime differs.'}}
 if((Get-FileHash -LiteralPath (Join-Path $target 'saves/unchanged.storage')).Hash -ne $saved){throw 'Save sentinel changed.'}
}
$interrupted=$false
try{& $installer -GameRoot $target -TestInterrupt}catch{if($_.Exception.Message -notlike '*TEST INTERRUPT*'){throw};$interrupted=$true}
if(-not $interrupted){throw 'Interruption injection did not run.'}
& $installer -GameRoot $target -Action Recover
VerifyPrior
& $installer -GameRoot $target
$installed=Get-Content -LiteralPath $receipt -Raw|ConvertFrom-Json
if($installed.version -ne '1.4.45-rust-local' -or $installed.deliveryGraceMs -ne 1000 -or ($installed.nnSearchKChoices -join ',') -ne '8,16,32'){throw 'Updated installation receipt differs.'}
& $installer -GameRoot $target -Action Rollback
VerifyPrior
@{passed=$true;interrupted_install_recovered=$true;exact_install_and_rollback=$true;saves_unchanged=$true;target=$target;
 installerSha256=(Get-FileHash -LiteralPath $installer).Hash;qualificationSha256=(Get-FileHash -LiteralPath (Join-Path $study 'release-1.4.45/qualification.json')).Hash}|
 ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $study 'upgrade-test.json') -Encoding utf8
