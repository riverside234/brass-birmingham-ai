param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [switch]$TestInterrupt
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$study=Join-Path $repo 'artifacts/host-delivery/20261029'
$build=Join-Path $study 'release-1.4.44'
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/host-delivery.pending.json'
$changes=@('BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll','BrassNative/alpha-g890/build.json','BrassNative/alpha-g890/brass-ai-local.exe','BrassNative/alpha-g890/native-source.zip')
function Inside([string]$Path,[string]$Parent){
 $p=[IO.Path]::GetFullPath($Path)
 if(-not $p.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Host-delivery path escapes its boundary.'}
 return $p
}
if($game -ne $repo){[void](Inside $game (Join-Path $study 'install-tests'))}
if($TestInterrupt -and ($game -eq $repo -or $Action -ne 'Install')){throw 'Interruption injection requires a disposable installation test.'}
function Verify($Pins,[string]$Directory,[switch]$Exact){
 foreach($pin in $Pins){
  $path=Inside (Join-Path $Directory $pin.path) $Directory
  if(-not(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw ('File changed: '+$pin.path)}
 }
 if($Exact -and @(Get-ChildItem -LiteralPath $Directory -Recurse -File).Count -ne @($Pins).Count){throw 'Native inventory differs.'}
}
function Managed($Files){@($Files|ForEach-Object{@{path=$_.name;sha256=$_.sha256}})}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $game 'artifacts/patch-recovery')
 $r=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if(-not $r -or $r.gameRoot -ne $game -or (@($r.files.path|Sort-Object)-join '|') -ne (@($changes|Sort-Object)-join '|')){throw 'Recovery identity differs.'}
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $r.receiptSha256){throw 'Recovery receipt changed.'}
 foreach($f in $r.files){
  $target=Inside (Join-Path $game $f.path) $game
  $saved=Inside (Join-Path $Backup $f.path) $Backup
  if((Get-FileHash -LiteralPath $saved).Hash -ne $f.priorSha256){throw 'Recovery file changed.'}
  if(-not(Test-Path -LiteralPath $target) -or (Get-FileHash -LiteralPath $target).Hash -notin @($f.priorSha256,$f.candidateSha256)){throw 'Unrelated live runtime change; recovery stopped.'}
 }
 foreach($f in $r.files){Copy-BrassAtomic (Join-Path $Backup $f.path) (Join-Path $game $f.path)}
 $old=Read-BrassJson (Join-Path $Backup 'patch-installation.json')
 Verify (Managed $old.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
 Verify $r.priorNative (Join-Path $game 'BrassNative') -Exact
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $record
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and let its native worker retire before installation.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves @($prior.saveDataPaths)
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted host-delivery installation.'}
 Restore $pending.backup;Remove-Item -LiteralPath $intent;return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the pending host-delivery installation first.'}
if($Action -eq 'Rollback'){
 if($prior.version -ne '1.4.44-rust-local' -or -not $prior.hostDeliveryRecovery){throw 'No matching host-delivery rollback receipt.'}
 Restore $prior.hostDeliveryRecovery;Write-Output 'Restored exact previous 1.4.43 runtime and receipt.';return
}
$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
$baseline=Read-BrassJson (Join-Path $study 'baseline/build-manifest.json')
if(-not $q.passed -or $q.format -ne 'brass-host-delivery-release/1.0.0' -or $m.version -ne '1.4.44-rust-local' -or $m.baseVersion -ne '1.4.43-rust-local' -or
 $prior.version -ne $m.baseVersion -or $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $q.host_qualification_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $study 'candidate/qualification.json')).Hash -or
 $q.native_build_receipt_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $study 'reserve08-native/native-build.json')).Hash -or
 $m.stockSha256 -ne $prior.stockSha256 -or -not $prior.localReleaseQualified -or $m.search_method -ne 'search_v1' -or $m.v205_enabled -or
 $m.k32_host_reserve_ns -ne 800000000){throw 'Qualified 1.4.44 release and exact 1.4.43 base required.'}
if($game -eq $repo -and (Get-FileHash -LiteralPath $record).Hash -ne $m.installed_baseline_receipt_sha256){throw 'Installed baseline receipt changed.'}
Verify $q.files $build
Verify (Managed $prior.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify (Managed $baseline.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify $baseline.nativeFiles (Join-Path $game 'BrassNative') -Exact
Verify (Managed $m.files) (Join-Path $build 'Managed')
Verify $m.nativeFiles (Join-Path $build 'BrassNative') -Exact
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){
 if(($m.files|Where-Object name -eq $name).sha256 -ne ($prior.files|Where-Object name -eq $name).sha256){throw 'Unrelated managed binary changed.'}
}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/host-delivery-'+[Guid]::NewGuid().ToString('N'))) $game
[void](New-Item -ItemType Directory -Path $backup)
$files=@()
foreach($path in $changes){
 $target=Inside (Join-Path $game $path) $game
 $source=if($path.StartsWith('BrassBirminghamUnity_Data/Managed/')){Join-Path $build 'Managed/BrassLegacyAI.dll'}else{Join-Path $build $path}
 $saved=Inside (Join-Path $backup $path) $backup
 [void](New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force)
 Copy-Item -LiteralPath $target -Destination $saved
 $files+=@{path=$path;priorSha256=(Get-FileHash -LiteralPath $target).Hash;candidateSha256=(Get-FileHash -LiteralPath $source).Hash;source=$source}
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') @{gameRoot=$game;files=$files;priorNative=$baseline.nativeFiles;receiptSha256=(Get-FileHash -LiteralPath $record).Hash}
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;operation='host-delivery-1.4.44'}
try{
 foreach($f in $files){Copy-BrassAtomic $f.source (Join-Path $game $f.path)}
 if($TestInterrupt){throw 'TEST INTERRUPT: host-delivery files published'}
 Verify (Managed $m.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
 Verify $m.nativeFiles (Join-Path $game 'BrassNative') -Exact
 foreach($entry in @{version=$m.version;files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');nativeBuildSha256=$m.native_build_sha256;
  hostDeliveryRecovery=$backup;hostDeliveryQualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;
  hostDeliveryCandidateSha256=(Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash;hostDeliveryInstalledSmoke='pending';
  hostDeliveryOptimization=$true;k32MinimumDeliveryReserveMs=800;searchMethod='search_v1';v205Enabled=$false;
  nnK32InstalledSmoke='historical/superseded-by-host-delivery';searchV1DefaultInstalledSmoke='historical/superseded-by-host-delivery'}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force
 }
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_
 if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}}
 throw $failure
}
Write-Output ('Installed 1.4.44 host optimization and 0.8s K32 minimum reserve. Rollback backup: '+$backup)
