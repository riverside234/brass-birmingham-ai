param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$BuildDirectory='artifacts/search-v2/20261028/v205-cache-candidate',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$managed=Join-Path $game 'BrassBirminghamUnity_Data/Managed'
$native=Join-Path $game 'BrassNative'
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/search-v2.pending.json'
function Inside([string]$Path,[string]$Parent){
 $p=[IO.Path]::GetFullPath($Path)
 if(-not $p.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Search-v2 path escapes boundary.'}
 return $p
}
function Verify($Entries,[string]$Directory,[switch]$Exact){
 foreach($entry in $Entries){$p=Inside (Join-Path $Directory $entry.path) $Directory
  if(-not(Test-Path -LiteralPath $p -PathType Leaf) -or (Get-FileHash -LiteralPath $p).Hash -ne $entry.sha256){throw ('Changed file: '+$entry.path)}}
 if($Exact -and @(Get-ChildItem -LiteralPath $Directory -Recurse -File).Count -ne @($Entries).Count){throw 'Unexpected package file.'}
}
function Pins([string]$Directory){
 @(Get-ChildItem -LiteralPath $Directory -Recurse -File|ForEach-Object{
  if($_.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse point in package.'}
  @{path=$_.FullName.Substring($Directory.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
 })
}
function ManagedPins($Files){@($Files|ForEach-Object{@{path=$_.name;sha256=$_.sha256}})}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $game 'artifacts/patch-recovery')
 $r=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if(-not $r -or $r.gameRoot -ne $game){throw 'Recovery identity mismatch.'}
 Verify $r.managed (Join-Path $Backup 'Managed');Verify $r.native (Join-Path $Backup 'BrassNative') -Exact
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $r.receiptSha256){throw 'Recovery receipt changed.'}
 foreach($p in $r.managed){
  $hash=(Get-FileHash -LiteralPath (Inside (Join-Path $managed $p.path) $managed)).Hash
  if($hash -ne $p.sha256 -and $hash -notin @($r.candidateManaged|Where-Object path -eq $p.path|ForEach-Object sha256)){throw 'Unrelated managed change; recovery stopped.'}
 }
 if(Test-Path -LiteralPath $native){
  $known=$false;try{Verify $r.native $native -Exact;$known=$true}catch{}
  if(-not $known){Verify $r.candidateNative $native -Exact}
  $from=Inside $native $game;$retained=Inside (Join-Path $Backup ('retired-'+[Guid]::NewGuid().ToString('N'))) $game
  Move-Item -LiteralPath $from -Destination $retained
 }
 Copy-Item -LiteralPath (Join-Path $Backup 'BrassNative') -Destination $native -Recurse
 foreach($p in $r.managed){Copy-BrassAtomic (Join-Path $Backup ('Managed/'+$p.path)) (Join-Path $managed $p.path)}
 Verify $r.managed $managed;Verify $r.native $native -Exact
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $record
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Game or native worker is still running.'}
$prior=Read-BrassJson $record
if(-not $prior -or [IO.Path]::GetFullPath($prior.gameRoot) -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves @($prior.saveDataPaths)
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted installation.'}
 Restore $pending.backup;Remove-Item -LiteralPath $intent;return
}
if(Test-Path -LiteralPath $intent){throw 'Recover pending installation first.'}
if($Action -eq 'Rollback'){
 if(-not $prior.searchV2Recovery){throw 'No search-v2 rollback receipt.'}
 Restore $prior.searchV2Recovery;Write-Output 'Restored exact previous runtime and receipt.';return
}
$build=[IO.Path]::GetFullPath((Join-Path $repo $BuildDirectory))
$manifestPath=Join-Path $build 'build-manifest.json'
$m=Read-BrassJson $manifestPath
$evidence=Read-BrassJson (Join-Path $repo 'artifacts/search-v2/20261028/v205/decision.json')
if(-not $evidence.correctness_passed -or $evidence.selected_method -ne 'search_v2_focus_only' -or
 $evidence.selected_candidate.sha256 -ne (Get-FileHash -LiteralPath $manifestPath).Hash -or
 $m.search_method -ne 'search_v2_focus_only'){throw 'Tested cache-only candidate required; V205 is excluded.'}
foreach($pin in @($evidence.native_tests,$evidence.parity_cancellation,$evidence.validation)+@($evidence.unity)){
 if((Get-FileHash -LiteralPath (Inside (Join-Path $repo $pin.path) $repo)).Hash -ne $pin.sha256){throw 'Qualification evidence changed.'}
}
if($prior.version -ne '1.4.38-rust-local' -or -not $prior.localReleaseQualified -or $prior.stockSha256 -ne $m.stockSha256){throw 'Installed 1.4.38 base required.'}
Verify (ManagedPins $prior.files) $managed;Verify (ManagedPins $m.files) (Join-Path $build 'Managed')
if((@($m.files.name|Sort-Object)-join '|') -ne 'Assembly-CSharp.dll|BrassLegacyAI.dll|BrassPatch.dll'){throw 'Unexpected managed file list.'}
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){
 if(($m.files|Where-Object name -eq $name).sha256 -ne ($prior.files|Where-Object name -eq $name).sha256){throw 'Unrelated UI/game binary change.'}
}
$package=Join-Path $build 'BrassNative/alpha-g890'
$policy=Read-BrassJson (Join-Path $package 'policy.json')
$oldPolicy=Read-BrassJson (Join-Path $native 'alpha-g890/policy.json')
if($policy.method -ne 'search_v2_focus_only'){throw 'V205 must remain disabled.'}
$policyCopy=$policy|ConvertTo-Json -Depth 15|ConvertFrom-Json
$policyCopy.PSObject.Properties.Remove('method')
if(($policyCopy|ConvertTo-Json -Depth 15 -Compress) -ne ($oldPolicy|ConvertTo-Json -Depth 15 -Compress)){throw 'Unexpected model/difficulty/policy change.'}
$nativeBuild=Read-BrassJson (Join-Path $package 'build.json')
if((Get-FileHash -LiteralPath (Join-Path $package 'build.json')).Hash -ne $m.native_build_sha256){throw 'Native build pin changed.'}
Verify $nativeBuild.files $package
$oldNative=Pins $native;$newNative=Pins (Join-Path $build 'BrassNative')
if((@($oldNative.path|Sort-Object)-join '|') -ne (@($newNative.path|Sort-Object)-join '|')){throw 'Package membership changed.'}
foreach($p in $newNative){
 $old=@($oldNative|Where-Object path -eq $p.path)[0]
 if($p.sha256 -ne $old.sha256 -and $p.path -notin @('alpha-g890/brass-ai-local.exe','alpha-g890/policy.json','alpha-g890/build.json','alpha-g890/native-source.zip','alpha-g890/README.txt')){throw ('Unexpected native change: '+$p.path)}
}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/search-v2-'+[Guid]::NewGuid().ToString('N'))) $game
[void](New-Item -ItemType Directory -Path (Join-Path $backup 'Managed'))
foreach($f in $prior.files){Copy-Item -LiteralPath (Join-Path $managed $f.name) -Destination (Join-Path $backup ('Managed/'+$f.name))}
Copy-Item -LiteralPath $native -Destination (Join-Path $backup 'BrassNative') -Recurse
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') ([ordered]@{gameRoot=$game;managed=(ManagedPins $prior.files);native=$oldNative;
 candidateManaged=(ManagedPins $m.files);candidateNative=$newNative;receiptSha256=(Get-FileHash -LiteralPath $record).Hash})
$stage=Inside (Join-Path $backup 'staged-native') $game
Copy-Item -LiteralPath (Join-Path $build 'BrassNative') -Destination $stage -Recurse;Verify $newNative $stage -Exact
Write-BrassJsonAtomic $intent ([ordered]@{gameRoot=$game;backup=$backup;operation='search-v2-cache-only'})
try{
 $from=Inside $native $game;$retained=Inside (Join-Path $backup 'prior-native') $game
 Move-Item -LiteralPath $from -Destination $retained;Move-Item -LiteralPath $stage -Destination $native
 foreach($f in $m.files){if((Get-FileHash -LiteralPath (Join-Path $managed $f.name)).Hash -ne $f.sha256){Copy-BrassAtomic (Join-Path $build ('Managed/'+$f.name)) (Join-Path $managed $f.name)}}
 Verify (ManagedPins $m.files) $managed;Verify $newNative $native -Exact
 foreach($item in @{version='1.4.39-rust-local';files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');searchMethod='search_v2_focus_only';v205Enabled=$false;
  searchV2Recovery=$backup;searchV2CandidateSha256=(Get-FileHash -LiteralPath $manifestPath).Hash;nativeBuildSha256=$m.native_build_sha256;
  searchV2EvidenceSha256=(Get-FileHash -LiteralPath (Join-Path $repo 'artifacts/search-v2/20261028/v205/decision.json')).Hash;searchV2InstalledSmoke='pending'}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $item.Key -NotePropertyValue $item.Value -Force
 }
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_;try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)};throw $failure
}
Write-Output ('Installed cache-only search_v2, V205 disabled. Rollback: '+$backup)
