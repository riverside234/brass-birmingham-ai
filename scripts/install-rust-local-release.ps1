param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$BuildDirectory='artifacts/rust-backend/20261026-local-release/candidate',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [ValidateSet('None','NativePublished','ManagedPublished')][string]$TestInterruptAt='None'
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$releaseRepo=Split-Path -Parent $PSScriptRoot
$releaseGame=(Resolve-Path -LiteralPath $GameRoot).Path
$releaseManaged=Join-Path $releaseGame 'BrassBirminghamUnity_Data/Managed'
$releaseNative=Join-Path $releaseGame 'BrassNative'
$releaseRecord=Join-Path $releaseGame 'artifacts/patch-installation.json'
$releaseIntent=Join-Path $releaseGame 'artifacts/rust-local-release.pending.json'
function Inside([string]$Path,[string]$Parent){
 $absolute=[IO.Path]::GetFullPath($Path)
 if(-not $absolute.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Release path escapes its boundary.'}
 return $absolute
}
function Pins([string]$Directory){
 @(Get-ChildItem -LiteralPath $Directory -Recurse -File|ForEach-Object {
  if($_.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse points are not supported.'}
  @{path=$_.FullName.Substring($Directory.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
 })
}
function Verify($Entries,[string]$Directory,[switch]$Exact){
 foreach($entry in $Entries){$p=Inside (Join-Path $Directory $entry.path) $Directory
  if(-not(Test-Path -LiteralPath $p -PathType Leaf) -or (Get-FileHash -LiteralPath $p).Hash -ne $entry.sha256){throw ('Release file changed: '+$entry.path)}}
 if($Exact -and @(Get-ChildItem -LiteralPath $Directory -Recurse -File).Count -ne @($Entries).Count){throw 'Unexpected native package file.'}
}
function ManagedPins($Files){@($Files|ForEach-Object{@{path=$_.name;sha256=$_.sha256}})}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $releaseGame 'artifacts/patch-recovery')
 $recovery=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if($recovery.gameRoot -ne $releaseGame){throw 'Recovery belongs to another installation.'}
 Verify $recovery.managed (Join-Path $Backup 'Managed')
 Verify $recovery.native (Join-Path $Backup 'BrassNative') -Exact
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $recovery.recordSha256){throw 'Recovery receipt changed.'}
 # Reject unrelated edits before touching any current runtime file.
 foreach($entry in $recovery.managed){
  $p=Inside (Join-Path $releaseManaged $entry.path) $releaseManaged
  $hash=(Get-FileHash -LiteralPath $p).Hash
  $candidate=@($recovery.candidateManaged|Where-Object path -eq $entry.path)
  if($hash -ne $entry.sha256 -and $hash -notin @($candidate.sha256)){throw 'Unrecognized managed recovery target.'}
 }
 if(Test-Path -LiteralPath $releaseNative){
  $known=$false
  try{Verify $recovery.native $releaseNative -Exact;$known=$true}catch{}
  if(-not $known){Verify $recovery.candidateNative $releaseNative -Exact}
  $retained=Inside (Join-Path $Backup ('retired-'+[Guid]::NewGuid().ToString('N'))) $releaseGame
  $from=Inside $releaseNative $releaseGame;Move-Item -LiteralPath $from -Destination $retained
 }
 Copy-Item -LiteralPath (Join-Path $Backup 'BrassNative') -Destination $releaseNative -Recurse
 foreach($entry in $recovery.managed){Copy-BrassAtomic (Join-Path $Backup ('Managed/'+$entry.path)) (Join-Path $releaseManaged $entry.path)}
 Verify $recovery.managed $releaseManaged;Verify $recovery.native $releaseNative -Exact
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $releaseRecord
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and wait for native workers to retire.'}
$prior=Read-BrassJson $releaseRecord
if(-not $prior -or $prior.gameRoot -ne $releaseGame){throw 'Matching installed receipt required.'}
Assert-BrassNoPendingSaves @($prior.saveDataPaths)
if($TestInterruptAt -ne 'None'){
 [void](Inside $releaseGame (Join-Path $releaseRepo 'artifacts/rust-backend'))
 if($Action -ne 'Install'){throw 'Fault injection is for disposable install tests only.'}
}
if($Action -eq 'Recover'){
 $intent=Read-BrassJson $releaseIntent
 if(-not $intent -or $intent.gameRoot -ne $releaseGame){throw 'No matching release intent.'}
 Restore $intent.backup;Remove-Item -LiteralPath $releaseIntent
 Write-Output 'Recovered exact previous runtime and receipt; saves untouched.';return
}
if(Test-Path -LiteralPath $releaseIntent){throw 'Recover the interrupted release before another operation.'}
if($Action -eq 'Rollback'){
 if($prior.version -ne '1.4.36-rust-local' -or -not $prior.localReleaseRecovery){throw 'No local release recovery receipt.'}
 Restore $prior.localReleaseRecovery
 Write-Output 'Restored exact prior L13 runtime and receipt; saves untouched.';return
}
$build=[IO.Path]::GetFullPath((Join-Path $releaseRepo $BuildDirectory))
$q=Read-BrassJson (Join-Path $build 'qualification.json');$audit=Read-BrassJson (Join-Path $build 'final-verification.json')
$manifest=Read-BrassJson (Join-Path $build 'build-manifest.json')
if(-not $q.passed -or -not $q.local_release_qualified -or $q.format -ne 'brass-local-release/1.0.0' -or
 $q.scope -ne 'local-game-g890' -or -not $audit.passed -or
 $audit.qualification_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash -or
 $manifest.version -ne '1.4.36-rust-local' -or -not $manifest.local_release_qualified){throw 'Sealed local release qualification required.'}
Verify $q.files $build
if($q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash){throw 'Release manifest changed.'}
if($prior.version -ne $manifest.baseVersion -or $prior.stockSha256 -ne $manifest.stockSha256){throw 'Qualified installed L13 base required.'}
Verify (ManagedPins $prior.files) $releaseManaged
Verify (ManagedPins $manifest.files) (Join-Path $build 'Managed')
$oldNative=Pins $releaseNative;$newNative=Pins (Join-Path $build 'BrassNative')
# The installed native source package must match its existing qualified package.
$base=Join-Path $releaseRepo 'artifacts/rust-backend/20261026-l13-qualified/BrassNative'
Verify (Pins $base) $releaseNative -Exact
$backup=Inside (Join-Path $releaseGame ('artifacts/patch-recovery/local-'+[Guid]::NewGuid().ToString('N'))) $releaseGame
[void](New-Item -ItemType Directory -Path (Join-Path $backup 'Managed'))
foreach($entry in $prior.files){Copy-Item -LiteralPath (Join-Path $releaseManaged $entry.name) -Destination (Join-Path $backup ('Managed/'+$entry.name))}
Copy-Item -LiteralPath $releaseNative -Destination (Join-Path $backup 'BrassNative') -Recurse
Copy-Item -LiteralPath $releaseRecord -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') ([ordered]@{gameRoot=$releaseGame;managed=(ManagedPins $prior.files);native=$oldNative;
 candidateManaged=(ManagedPins $manifest.files);candidateNative=$newNative;recordSha256=(Get-FileHash -LiteralPath $releaseRecord).Hash})
$stage=Inside (Join-Path $backup 'staged-native') $releaseGame
Copy-Item -LiteralPath (Join-Path $build 'BrassNative') -Destination $stage -Recurse;Verify $newNative $stage -Exact
Write-BrassJsonAtomic $releaseIntent ([ordered]@{gameRoot=$releaseGame;backup=$backup;operation='local-release-upgrade'})
try{
 $from=Inside $releaseNative $releaseGame;$retained=Inside (Join-Path $backup 'prior-native') $releaseGame
 Move-Item -LiteralPath $from -Destination $retained;Move-Item -LiteralPath $stage -Destination $releaseNative
 if($TestInterruptAt -eq 'NativePublished'){throw 'TEST INTERRUPT: native published'}
 foreach($entry in $manifest.files){Copy-BrassAtomic (Join-Path $build ('Managed/'+$entry.name)) (Join-Path $releaseManaged $entry.name)}
 if($TestInterruptAt -eq 'ManagedPublished'){throw 'TEST INTERRUPT: managed published'}
 Verify (ManagedPins $manifest.files) $releaseManaged;Verify $newNative $releaseNative -Exact
 $record=[ordered]@{version=$manifest.version;status='installed';installedUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$releaseGame;
  stockBackup=$prior.stockBackup;stockSha256=$prior.stockSha256;files=$manifest.files;onlineEnabled=$false;saveDataPaths=$prior.saveDataPaths;
  localReleaseRecovery=$backup;nativeBuildSha256=$manifest.native_build_sha256;
  qualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;qualificationScope=$q.scope;
  localReleaseQualified=$true;finalModel=$false;part2Deployed=$false}
 Write-BrassJsonAtomic $releaseRecord $record;Remove-Item -LiteralPath $releaseIntent
}catch{
 $failure=$_
 if($TestInterruptAt -eq 'None'){try{Restore $backup;Remove-Item -LiteralPath $releaseIntent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}}
 throw $failure
}
Write-Output 'Installed 1.4.36 local Rust release. ONNX Runtime CPU; temporary g890 model; lab unchanged.'
