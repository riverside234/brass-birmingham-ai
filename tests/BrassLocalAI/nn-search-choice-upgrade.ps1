$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$evidence=Join-Path $root 'artifacts/nn-search-choice'
$target=Join-Path $evidence ('upgrade-test/'+[Guid]::NewGuid().ToString('N'))
$managed=Join-Path $target 'BrassBirminghamUnity_Data/Managed'
[void](New-Item -ItemType Directory -Path $managed,(Join-Path $target 'artifacts'),(Join-Path $target 'saves'))
$record=Get-Content -LiteralPath (Join-Path $evidence 'before-installation.json') -Raw|ConvertFrom-Json
if($record.version -ne '1.4.39-rust-local'){throw 'Expected exact 1.4.39 base.'}
foreach($entry in $record.files){Copy-Item -LiteralPath (Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$entry.name)) -Destination $managed}
Copy-Item -LiteralPath (Join-Path $root 'BrassNative') -Destination (Join-Path $target 'BrassNative') -Recurse
$record.gameRoot=$target;$record.saveDataPaths=@((Join-Path $target 'saves'))
$record|ConvertTo-Json -Depth 15|Set-Content -LiteralPath (Join-Path $target 'artifacts/patch-installation.json') -Encoding utf8
'Preserve this saved-game sentinel.'|Set-Content -LiteralPath (Join-Path $target 'saves/untouched.storage')
$saved=(Get-FileHash -LiteralPath (Join-Path $target 'saves/untouched.storage')).Hash
$prior=(Get-FileHash -LiteralPath (Join-Path $target 'artifacts/patch-installation.json')).Hash
$installer=Join-Path $root 'scripts/install-nn-search-choice.ps1'
function VerifyPrior {
 if((Get-FileHash -LiteralPath (Join-Path $target 'artifacts/patch-installation.json')).Hash -ne $prior){throw 'Prior receipt was not restored exactly.'}
 foreach($entry in $record.files){if((Get-FileHash -LiteralPath (Join-Path $managed $entry.name)).Hash -ne $entry.sha256){throw 'Prior managed runtime differs.'}}
 if(Test-Path -LiteralPath (Join-Path $target 'BrassNative/alpha-g890/policy-k8.json')){throw 'K8 addition was not removed by recovery.'}
 if((Get-FileHash -LiteralPath (Join-Path $target 'saves/untouched.storage')).Hash -ne $saved){throw 'Save sentinel changed.'}
}
$interrupted=$false
try{& $installer -GameRoot $target -TestInterrupt}catch{if($_.Exception.Message -notlike '*TEST INTERRUPT*'){throw};$interrupted=$true}
if(-not $interrupted){throw 'Expected interruption.'}
& $installer -GameRoot $target -Action Recover
VerifyPrior
& $installer -GameRoot $target
$installed=Get-Content -LiteralPath (Join-Path $target 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
if($installed.version -ne '1.4.40-rust-local' -or ($installed.nnSearchKChoices -join ',') -ne '8,16' -or $installed.savePolicy -ne 'completed-turn-checkpoint/1.0.0'){throw 'Upgrade receipt mismatch.'}
& $installer -GameRoot $target -Action Rollback
VerifyPrior
@{passed=$true;interrupted_install_recovered=$true;install_and_rollback_passed=$true;saves_unchanged=$true;target=$target;
 installerSha256=(Get-FileHash -LiteralPath $installer).Hash;qualificationSha256=(Get-FileHash -LiteralPath (Join-Path $evidence 'candidate/qualification.json')).Hash}|
 ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $evidence 'upgrade-test.json') -Encoding utf8
