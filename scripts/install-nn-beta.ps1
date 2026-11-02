param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [switch]$TestInterrupt
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$study=Join-Path $repo 'artifacts/nn-beta/20261101'
$build=Join-Path $study 'release-1.4.48'
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/nn-beta.pending.json'
$betaCheckedNodes=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
function Inside([string]$Path,[string]$Parent){
 $full=[IO.Path]::GetFullPath($Path);$boundary=[IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\'
 if(-not $full.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Beta path escapes boundary.'}
 for($node=$full;$node.Length -ge $boundary.Length;$node=[IO.Path]::GetDirectoryName($node)){
  if($betaCheckedNodes.Contains($node)){break}
  if([IO.File]::Exists($node) -or [IO.Directory]::Exists($node)){
   if([IO.File]::GetAttributes($node) -band [IO.FileAttributes]::ReparsePoint){throw 'Linked Beta installation path refused.'}
   [void]$betaCheckedNodes.Add($node)
  }
 }
 return $full
}
function Verify($Pins,[string]$Directory,[switch]$Exact){
 foreach($pin in $Pins){$path=Inside (Join-Path $Directory $pin.path) $Directory
  if(-not(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw ('File changed: '+$pin.path)}}
 if($Exact -and @(Get-ChildItem -LiteralPath $Directory -Recurse -File).Count -ne @($Pins).Count){throw 'Unexpected native package member.'}
}
function Managed($Files){@($Files|ForEach-Object{@{path=$_.name;sha256=$_.sha256}})}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $game 'artifacts/patch-recovery')
 $r=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if(-not $r -or $r.gameRoot -ne $game -or (Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $r.receiptSha256){throw 'Recovery identity changed.'}
 foreach($f in $r.files){
  if($f.path -ne 'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll' -and -not $f.path.StartsWith('BrassNative/beta-g1290/')){throw 'Unexpected recovery member.'}
  $target=Inside (Join-Path $game $f.path) $game
  if($f.existed -and (Get-FileHash -LiteralPath (Inside (Join-Path $Backup $f.path) $Backup)).Hash -ne $f.priorSha256){throw 'Recovery file changed.'}
  if(Test-Path -LiteralPath $target){$hash=(Get-FileHash -LiteralPath $target).Hash
   if($hash -ne $f.candidateSha256 -and (-not $f.existed -or $hash -ne $f.priorSha256)){throw 'Unrelated runtime change; recovery stopped.'}
  }elseif($f.existed){throw 'Existing runtime file is missing.'}
 }
 foreach($f in $r.files){$target=Inside (Join-Path $game $f.path) $game
  if($f.existed){Copy-BrassAtomic (Join-Path $Backup $f.path) $target}elseif(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target}}
 $old=Read-BrassJson (Join-Path $Backup 'patch-installation.json')
 Verify (Managed $old.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
 Verify $r.priorNative (Join-Path $game 'BrassNative') -Exact
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $record
}
if($game -ne $repo){[void](Inside $game (Join-Path $study 'install-tests'))}
if($TestInterrupt -and ($game -eq $repo -or $Action -ne 'Install')){throw 'Interruption injection requires a disposable test.'}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and let the native worker retire before installation.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves @($prior.saveDataPaths)
if($Action -eq 'Recover'){$pending=Read-BrassJson $intent;if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted Beta install.'};Restore $pending.backup;Remove-Item -LiteralPath $intent;return}
if(Test-Path -LiteralPath $intent){throw 'Recover the pending Beta installation first.'}
if($Action -eq 'Rollback'){if($prior.version -ne '1.4.48-rust-local' -or -not $prior.nnBetaRecovery){throw 'No matching Beta rollback receipt.'};Restore $prior.nnBetaRecovery;Write-Output 'Restored exact 1.4.47 runtime and receipt.';return}
$m=Read-BrassJson (Join-Path $build 'build-manifest.json');$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-nn-beta-release/1.0.0' -or $m.version -ne '1.4.48-rust-local' -or $m.baseVersion -ne $prior.version -or
 $prior.version -ne '1.4.47-rust-local' -or -not $prior.localReleaseQualified -or $m.stockSha256 -ne $prior.stockSha256 -or
 $m.beta_model_id -ne '73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff' -or
 $m.beta_checkpoint_sha256 -ne 'ed8edb039a1257d591cc0d2e2b87eb7a2d7aa411ce0f7710a2d7f8ad487dc3b0' -or
 $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $q.beta_native_build_receipt_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $study 'native-build/native-build.json')).Hash -or
 $m.search_method -ne 'search_v1' -or $m.v205_enabled -or ($m.nn_ui_choices -join '|') -ne 'alpha-test|beta'){throw 'Qualified exact g1290 Beta package and installed 1.4.47 required.'}
if($game -eq $repo -and (Get-FileHash -LiteralPath $record).Hash -ne $m.installed_baseline_receipt_sha256){throw 'Installed baseline receipt changed.'}
Verify $q.files $build
Verify (Managed $prior.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify (Managed $m.baseFiles) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify $m.baseNative (Join-Path $game 'BrassNative') -Exact
Verify (Managed $m.files) (Join-Path $build 'Managed')
Verify $m.nativeFiles (Join-Path $build 'BrassNative') -Exact
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){if(($m.files|Where-Object name -eq $name).sha256 -ne ($prior.files|Where-Object name -eq $name).sha256){throw 'Unrelated managed change.'}}
foreach($pin in $m.baseNative){if(@($m.nativeFiles|Where-Object {$_.path -eq $pin.path -and $_.sha256 -eq $pin.sha256}).Count -ne 1){throw 'Existing native package changed.'}}
$added=@($m.nativeFiles|Where-Object {$_.path.StartsWith('beta-g1290/')})
if(@($m.nativeFiles).Count -ne @($m.baseNative).Count+$added.Count -or -not $added.Count){throw 'Only the additive Beta package is permitted.'}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/nn-beta-'+[Guid]::NewGuid().ToString('N'))) $game
[void](New-Item -ItemType Directory -Path $backup)
$files=@();$changes=@('BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll')+@($added|ForEach-Object{'BrassNative/'+$_.path})
foreach($path in $changes){
 $target=Inside (Join-Path $game $path) $game
 $source=if($path.StartsWith('BrassBirminghamUnity_Data/Managed/')){Join-Path $build 'Managed/BrassLegacyAI.dll'}else{Join-Path $build $path}
 $existed=Test-Path -LiteralPath $target;$oldHash=$null
 if($existed){$oldHash=(Get-FileHash -LiteralPath $target).Hash;$saved=Inside (Join-Path $backup $path) $backup;[void](New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force);Copy-Item -LiteralPath $target -Destination $saved}
 $files+=@{path=$path;existed=$existed;priorSha256=$oldHash;candidateSha256=(Get-FileHash -LiteralPath $source).Hash;source=$source}
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') @{gameRoot=$game;files=$files;priorNative=$m.baseNative;receiptSha256=(Get-FileHash -LiteralPath $record).Hash}
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;operation='nn-beta-1.4.48'}
try{
 foreach($f in $files){$target=Inside (Join-Path $game $f.path) $game;[void](New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force);Copy-BrassAtomic $f.source $target}
 if($TestInterrupt){throw 'TEST INTERRUPT: Beta files published'}
 Verify (Managed $m.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed');Verify $m.nativeFiles (Join-Path $game 'BrassNative') -Exact
 foreach($entry in @{version=$m.version;files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');nnBetaRecovery=$backup;
  nnBetaQualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;nnBetaCandidateSha256=(Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash;
  nnBetaModel=$m.beta_model_id;nnBetaCheckpoint=$m.beta_checkpoint_sha256;nnBetaNativeBuildSha256=$m.beta_native_build_sha256;
  nnBetaInstalledSmoke='pending';nnUiModels=@('alpha-test','beta');nnIndustryFocusChoices=@('neutral');nnNeutralUiInstalledSmoke='historical/Beta-added';searchMethod='search_v1';v205Enabled=$false}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force}
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_;if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}};throw $failure
}
Write-Output ('Installed 1.4.48: Beta g1290. Recovery backup: '+$backup)
