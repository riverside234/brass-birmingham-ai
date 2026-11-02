param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [switch]$TestInterrupt
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$study=Join-Path $repo 'artifacts/nn-il/20261102'
$build=Join-Path $study 'release-1.4.50'
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/nn-il.pending.json'
$ilCheckedNodes=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
function Inside([string]$Path,[string]$Parent){
 $full=[IO.Path]::GetFullPath($Path);$boundary=[IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\'
 if(-not $full.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'IL path escapes boundary.'}
 for($node=$full;$node.Length -ge $boundary.Length;$node=[IO.Path]::GetDirectoryName($node)){
  if($ilCheckedNodes.Contains($node)){break}
  if([IO.File]::Exists($node) -or [IO.Directory]::Exists($node)){
   if([IO.File]::GetAttributes($node) -band [IO.FileAttributes]::ReparsePoint){throw 'Linked IL installation path refused.'}
   [void]$ilCheckedNodes.Add($node)
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
  if($f.path -ne 'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll' -and -not $f.path.StartsWith('BrassNative/il-v11-2/')){throw 'Unexpected recovery member.'}
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
if($Action -eq 'Recover'){$pending=Read-BrassJson $intent;if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted IL install.'};Restore $pending.backup;Remove-Item -LiteralPath $intent;return}
if(Test-Path -LiteralPath $intent){throw 'Recover the pending IL installation first.'}
if($Action -eq 'Rollback'){if($prior.version -ne '1.4.50-rust-local' -or -not $prior.nnIlRecovery){throw 'No matching IL rollback receipt.'};Restore $prior.nnIlRecovery;Write-Output 'Restored exact 1.4.49 runtime and receipt.';return}
$m=Read-BrassJson (Join-Path $build 'build-manifest.json');$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-nn-il-release/1.0.0' -or $m.version -ne '1.4.50-rust-local' -or $m.baseVersion -ne $prior.version -or
 $prior.version -ne '1.4.49-rust-local' -or -not $prior.localReleaseQualified -or $m.stockSha256 -ne $prior.stockSha256 -or
 $m.il_model_id -ne 'c4cdf78a1b90ae9ade7156878009e34ee355ec71ee67226f3fd8a95d09335a94' -or
 $m.il_checkpoint_sha256 -ne '657fe1901ab35d144e5d86e16b42e795d281946829c87bcb8e0245927b70a263' -or
 $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $q.il_native_build_receipt_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $study 'native-build/native-build.json')).Hash -or
 $m.search_method -ne 'search_v1' -or $m.v205_enabled -or ($m.nn_ui_choices -join '|') -ne 'alpha-test|beta|il'){throw 'Qualified exact v11.2 IL package and installed 1.4.49 required.'}
if($game -eq $repo -and (Get-FileHash -LiteralPath $record).Hash -ne $m.installed_baseline_receipt_sha256){throw 'Installed baseline receipt changed.'}
Verify $q.files $build
Verify (Managed $prior.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify (Managed $m.baseFiles) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify $m.baseNative (Join-Path $game 'BrassNative') -Exact
Verify (Managed $m.files) (Join-Path $build 'Managed')
Verify $m.nativeFiles (Join-Path $build 'BrassNative') -Exact
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){if(($m.files|Where-Object name -eq $name).sha256 -ne ($prior.files|Where-Object name -eq $name).sha256){throw 'Unrelated managed change.'}}
foreach($pin in $m.baseNative){if(@($m.nativeFiles|Where-Object {$_.path -eq $pin.path -and $_.sha256 -eq $pin.sha256}).Count -ne 1){throw 'Existing native package changed.'}}
$added=@($m.nativeFiles|Where-Object {$_.path.StartsWith('il-v11-2/')})
if(@($m.nativeFiles).Count -ne @($m.baseNative).Count+$added.Count -or -not $added.Count){throw 'Only the additive IL package is permitted.'}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/nn-il-'+[Guid]::NewGuid().ToString('N'))) $game
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
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;operation='nn-il-1.4.50'}
try{
 foreach($f in $files){$target=Inside (Join-Path $game $f.path) $game;[void](New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force);Copy-BrassAtomic $f.source $target}
 if($TestInterrupt){throw 'TEST INTERRUPT: IL files published'}
 Verify (Managed $m.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed');Verify $m.nativeFiles (Join-Path $game 'BrassNative') -Exact
 foreach($entry in @{version=$m.version;files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');nnIlRecovery=$backup;
  nnIlQualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;nnIlCandidateSha256=(Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash;
  nnIlModel=$m.il_model_id;nnIlCheckpoint=$m.il_checkpoint_sha256;nnIlNativeBuildSha256=$m.il_native_build_sha256;
  nnIlInstalledSmoke='pending';nnUiModels=@('alpha-test','beta','il');nnIndustryFocusChoices=@('neutral');nnNeutralUiInstalledSmoke='historical/IL-added';searchMethod='search_v1';v205Enabled=$false}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force}
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_;if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}};throw $failure
}
Write-Output ('Installed 1.4.50: IL v11.2. Recovery backup: '+$backup)
