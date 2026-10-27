param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$BuildDirectory='artifacts/setup-preferences/candidate',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [switch]$TestInterrupt
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$managed=Join-Path $game 'BrassBirminghamUnity_Data/Managed'
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/setup-preferences.pending.json'
function Inside([string]$Path,[string]$Parent){
 $absolute=[IO.Path]::GetFullPath($Path)
 if(-not $absolute.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Hotfix path escapes its boundary.'}
 return $absolute
}
function Verify($Entries,[string]$Directory){
 foreach($entry in $Entries){$path=Inside (Join-Path $Directory $entry.path) $Directory
  if(-not(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path).Hash -ne $entry.sha256){throw ('Hotfix file changed: '+$entry.path)}}
}
function ManagedPins($Files){@($Files|ForEach-Object{@{path=$_.name;sha256=$_.sha256}})}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $game 'artifacts/patch-recovery')
 $recovery=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if(-not $recovery -or $recovery.gameRoot -ne $game){throw 'Recovery belongs to another installation.'}
 Verify $recovery.priorFiles (Join-Path $Backup 'Managed')
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $recovery.receiptSha256){throw 'Recovery receipt changed.'}
 foreach($entry in $recovery.priorFiles){
  $hash=(Get-FileHash -LiteralPath (Inside (Join-Path $managed $entry.path) $managed)).Hash
  $candidate=@($recovery.candidateFiles|Where-Object path -eq $entry.path)
  if($hash -ne $entry.sha256 -and $hash -notin @($candidate.sha256)){throw 'Unrecognized runtime change; recovery stopped.'}
 }
 foreach($entry in $recovery.priorFiles){Copy-BrassAtomic (Join-Path $Backup ('Managed/'+$entry.path)) (Join-Path $managed $entry.path)}
 Verify $recovery.priorFiles $managed
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $record
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and wait for the AI worker to retire.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves @($prior.saveDataPaths)
if($TestInterrupt){
 [void](Inside $game (Join-Path $repo 'artifacts/setup-preferences/upgrade-test'))
 if($Action -ne 'Install'){throw 'Interruption injection is for disposable installation tests only.'}
}
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching hotfix intent.'}
 Restore $pending.backup;Remove-Item -LiteralPath $intent
 Write-Output 'Recovered exact prior managed runtime and receipt.';return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the interrupted hotfix first.'}
if($Action -eq 'Rollback'){
 if($prior.version -ne '1.4.38-rust-local' -or -not $prior.setupPreferencesRecovery){throw 'No setup-preferences recovery receipt.'}
 Restore $prior.setupPreferencesRecovery
 Write-Output 'Restored 1.4.37; native engine and saved games unchanged.';return
}
$build=[IO.Path]::GetFullPath((Join-Path $repo $BuildDirectory))
$qualified=Read-BrassJson (Join-Path $build 'qualification.json')
$manifest=Read-BrassJson (Join-Path $build 'build-manifest.json')
if(-not $qualified.passed -or $qualified.format -ne 'brass-setup-preferences/1.0.0' -or
 -not $qualified.local_release_qualified -or $qualified.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $manifest.version -ne '1.4.38-rust-local' -or $manifest.baseVersion -ne '1.4.37-rust-local' -or -not $manifest.local_release_qualified){throw 'Qualified 1.4.38 hotfix required.'}
Verify $qualified.files $build
if($prior.version -ne $manifest.baseVersion -or $prior.stockSha256 -ne $manifest.stockSha256 -or -not $prior.localReleaseQualified){throw 'Qualified installed 1.4.37 required.'}
if((@($manifest.files.name|Sort-Object)-join '|') -ne 'Assembly-CSharp.dll|BrassLegacyAI.dll|BrassPatch.dll'){throw 'Unexpected runtime file list.'}
Verify (ManagedPins $manifest.baseFiles) $managed
Verify (ManagedPins $prior.files) $managed
Verify (ManagedPins $manifest.files) (Join-Path $build 'Managed')
Verify $qualified.native_files (Join-Path $game 'BrassNative')
if(@(Get-ChildItem -LiteralPath (Join-Path $game 'BrassNative') -Recurse -File).Count -ne @($qualified.native_files).Count){throw 'Unexpected native package files.'}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/setup-preferences-'+[Guid]::NewGuid().ToString('N'))) $game
[void](New-Item -ItemType Directory -Path (Join-Path $backup 'Managed'))
foreach($entry in $prior.files){Copy-Item -LiteralPath (Join-Path $managed $entry.name) -Destination (Join-Path $backup ('Managed/'+$entry.name))}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') ([ordered]@{gameRoot=$game;priorFiles=(ManagedPins $prior.files);candidateFiles=(ManagedPins $manifest.files);receiptSha256=(Get-FileHash -LiteralPath $record).Hash})
Write-BrassJsonAtomic $intent ([ordered]@{gameRoot=$game;backup=$backup;operation='setup-preferences-hotfix'})
try{
 foreach($entry in $manifest.files){Copy-BrassAtomic (Join-Path $build ('Managed/'+$entry.name)) (Join-Path $managed $entry.name)}
 if($TestInterrupt){throw 'TEST INTERRUPT: managed published'}
 Verify (ManagedPins $manifest.files) $managed
 $prior|Add-Member -NotePropertyName version -NotePropertyValue $manifest.version -Force
 $prior|Add-Member -NotePropertyName files -NotePropertyValue $manifest.files -Force
 $prior|Add-Member -NotePropertyName installedUtc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
 $prior|Add-Member -NotePropertyName setupPreferencesRecovery -NotePropertyValue $backup -Force
 $prior|Add-Member -NotePropertyName qualificationSha256 -NotePropertyValue ((Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash) -Force
 $prior|Add-Member -NotePropertyName savePolicy -NotePropertyValue $manifest.save_policy -Force
 Write-BrassJsonAtomic $record $prior
 Remove-Item -LiteralPath $intent
}catch{
 $failure=$_
 if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}}
 throw $failure
}
Write-Output 'Installed 1.4.38: remembered AI setup and nested NN selection.'
