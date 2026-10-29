param([ValidateSet('Install','Rollback','Recover')][string]$Action='Install',[string]$GameRoot=(Split-Path -Parent $PSScriptRoot),[switch]$TestInterrupt)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$study=Join-Path $repo 'artifacts/fixed-reserve/20261029'
$build=Join-Path $study 'release-1.4.45-v2'
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/fixed-reserve.pending.json'
$changes=@('BrassNative/alpha-g890/build.json','BrassNative/alpha-g890/brass-ai-local.exe','BrassNative/alpha-g890/native-source.zip')
function Inside([string]$Path,[string]$Parent){
 $p=[IO.Path]::GetFullPath($Path)
 if(-not $p.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Fixed-reserve path escapes its boundary.'}
 return $p
}
if($game -ne $repo){[void](Inside $game (Join-Path $study 'install-tests'))}
if($TestInterrupt -and ($game -eq $repo -or $Action -ne 'Install')){throw 'Interruption injection requires a disposable copy.'}
function Verify($Entries,[string]$Directory,[switch]$Exact){
 foreach($entry in $Entries){
  $path=Inside (Join-Path $Directory $entry.path) $Directory
  if(-not(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path).Hash -ne $entry.sha256){throw ('File changed: '+$entry.path)}
 }
 if($Exact -and @(Get-ChildItem -LiteralPath $Directory -File -Recurse).Count -ne @($Entries).Count){throw 'Native inventory changed.'}
}
function Managed($Files){@($Files|ForEach-Object{@{path=$_.name;sha256=$_.sha256}})}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $game 'artifacts/patch-recovery')
 $r=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if(-not $r -or $r.gameRoot -ne $game -or (@($r.files.path|Sort-Object)-join '|') -ne (@($changes|Sort-Object)-join '|')){throw 'Recovery identity differs.'}
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $r.receiptSha256){throw 'Recovery receipt changed.'}
 foreach($f in $r.files){
  if((Get-FileHash -LiteralPath (Inside (Join-Path $Backup $f.path) $Backup)).Hash -ne $f.priorSha256){throw 'Recovery file changed.'}
  if((Get-FileHash -LiteralPath (Inside (Join-Path $game $f.path) $game)).Hash -notin @($f.priorSha256,$f.candidateSha256)){throw 'Unrelated runtime change; recovery stopped.'}
 }
 foreach($f in $r.files){Copy-BrassAtomic (Join-Path $Backup $f.path) (Join-Path $game $f.path)}
 Verify $r.priorNative (Join-Path $game 'BrassNative') -Exact
 $old=Read-BrassJson (Join-Path $Backup 'patch-installation.json')
 Verify (Managed $old.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $record
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and let its native worker retire first.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves @($prior.saveDataPaths)
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted installation.'}
 Restore $pending.backup;Remove-Item -LiteralPath $intent;return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the pending fixed-reserve installation first.'}
if($Action -eq 'Rollback'){
 if($prior.version -ne '1.4.45-rust-local' -or -not $prior.fixedReserveRecovery){throw 'No matching fixed-reserve rollback.'}
 Restore $prior.fixedReserveRecovery;Write-Output 'Restored exact 1.4.44 runtime and receipt.';return
}
$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-fixed-reserve-release/1.0.0' -or $m.version -ne '1.4.45-rust-local' -or $m.baseVersion -ne '1.4.44-rust-local' -or $prior.version -ne $m.baseVersion -or
 $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $q.assessment_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $study 'assessment.json')).Hash -or
 -not $prior.localReleaseQualified -or $m.stockSha256 -ne $prior.stockSha256 -or $m.search_method -ne 'search_v1' -or $m.v205_enabled){throw 'Qualified 1.4.45 candidate and exact 1.4.44 base required.'}
if($game -eq $repo -and (Get-FileHash -LiteralPath $record).Hash -ne $m.installed_baseline_receipt_sha256){throw 'Baseline receipt changed.'}
Verify $q.files $build
Verify (Managed $prior.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify (Managed $m.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify $m.baseNative (Join-Path $game 'BrassNative') -Exact
Verify $m.nativeFiles (Join-Path $build 'BrassNative') -Exact
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/fixed-reserve-'+[Guid]::NewGuid().ToString('N'))) $game
[void](New-Item -ItemType Directory -Path $backup)
$files=@()
foreach($path in $changes){
 $target=Inside (Join-Path $game $path) $game;$saved=Inside (Join-Path $backup $path) $backup
 [void](New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force)
 Copy-Item -LiteralPath $target -Destination $saved
 $files+=@{path=$path;priorSha256=(Get-FileHash -LiteralPath $target).Hash;candidateSha256=(Get-FileHash -LiteralPath (Join-Path $build $path)).Hash}
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') @{gameRoot=$game;files=$files;priorNative=$m.baseNative;receiptSha256=(Get-FileHash -LiteralPath $record).Hash}
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;operation='fixed-reserve-1.4.45'}
try{
 foreach($f in $files){Copy-BrassAtomic (Join-Path $build $f.path) (Join-Path $game $f.path)}
 if($TestInterrupt){throw 'TEST INTERRUPT: fixed-reserve files published'}
 Verify $m.nativeFiles (Join-Path $game 'BrassNative') -Exact
 foreach($entry in @{version=$m.version;installedUtc=[DateTime]::UtcNow.ToString('o');nativeBuildSha256=$m.native_build_sha256;
  fixedReserveRecovery=$backup;fixedReserveQualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;
  fixedReserveInstalledSmoke='pending';fixedDeliveryReserveMsByK=@{'8'=600;'16'=600;'32'=800};
  k32DeliveryReserveByAllowanceMs=@{'4000'=800;'8000'=800;'10000'=800;'12000'=800};
  deliveryReservePolicy='fixed-per-k/admission-only';hostDeliveryInstalledSmoke='historical/superseded-by-fixed-reserve'}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force
 }
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_
 if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}}
 throw $failure
}
Write-Output ('Installed 1.4.45 fixed reserves. Rollback backup: '+$backup)
