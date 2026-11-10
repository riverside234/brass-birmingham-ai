param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [switch]$TestInterrupt
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$study=Join-Path $repo 'artifacts/nn-beta2/20261110'
$build=Join-Path $study 'release-1.4.55'
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/nn-beta2.pending.json'
$beta2CheckedNodes=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
function Inside([string]$Path,[string]$Parent){
 $full=[IO.Path]::GetFullPath($Path);$boundary=[IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\'
 if(-not $full.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'beta-2 path escapes boundary.'}
 for($node=$full;$node.Length -ge $boundary.Length;$node=[IO.Path]::GetDirectoryName($node)){
  if($beta2CheckedNodes.Contains($node)){break}
  if([IO.File]::Exists($node) -or [IO.Directory]::Exists($node)){
   if([IO.File]::GetAttributes($node) -band [IO.FileAttributes]::ReparsePoint){throw 'Linked beta-2 installation path refused.'}
   [void]$beta2CheckedNodes.Add($node)
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
  if($f.path -ne 'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll' -and -not $f.path.StartsWith('BrassNative/beta2-g1190/')){throw 'Unexpected recovery member.'}
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
if($Action -eq 'Recover'){$pending=Read-BrassJson $intent;if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted beta-2 install.'};Restore $pending.backup;Remove-Item -LiteralPath $intent;return}
if(Test-Path -LiteralPath $intent){throw 'Recover the pending beta-2 installation first.'}
if($Action -eq 'Rollback'){if($prior.version -ne '1.4.55-rust-local' -or -not $prior.nnBeta2Recovery){throw 'No matching beta-2 rollback receipt.'};Restore $prior.nnBeta2Recovery;Write-Output 'Restored exact 1.4.54 runtime and receipt.';return}
$m=Read-BrassJson (Join-Path $build 'build-manifest.json');$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-nn-beta2-release/1.0.0' -or $m.version -ne '1.4.55-rust-local' -or $m.baseVersion -ne $prior.version -or
 $prior.version -ne '1.4.54-rust-local' -or -not $prior.localReleaseQualified -or $m.stockSha256 -ne $prior.stockSha256 -or
 $m.beta2_model_id -ne 'd872ea925d6643ee3a3eeec99a52d16c6a78c2de1f757617f1aaaa47d175f864' -or
 $m.beta2_checkpoint_sha256 -ne 'f228f4dd7f1a913ea11e908cb2ccb53b76cd14ca302da9e2e5479c505d1294f1' -or
 $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $q.beta2_native_build_receipt_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $study 'native-build/native-build.json')).Hash -or
 $m.search_method -ne 'search_v1' -or $m.v205_enabled -or ($m.nn_ui_choices -join '|') -ne 'alpha-test|beta|il|beta-2'){throw 'Qualified exact v11.2 beta-2 package and installed 1.4.54 required.'}
if($game -eq $repo -and (Get-FileHash -LiteralPath $record).Hash -ne $m.installed_baseline_receipt_sha256){throw 'Installed baseline receipt changed.'}
Verify $q.files $build
Verify (Managed $prior.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify (Managed $m.baseFiles) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify $m.baseNative (Join-Path $game 'BrassNative') -Exact
Verify (Managed $m.files) (Join-Path $build 'Managed')
Verify $m.nativeFiles (Join-Path $build 'BrassNative') -Exact
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){if(($m.files|Where-Object name -eq $name).sha256 -ne ($prior.files|Where-Object name -eq $name).sha256){throw 'Unrelated managed change.'}}
foreach($pin in $m.baseNative){if(@($m.nativeFiles|Where-Object {$_.path -eq $pin.path -and $_.sha256 -eq $pin.sha256}).Count -ne 1){throw 'Existing native package changed.'}}
$added=@($m.nativeFiles|Where-Object {$_.path.StartsWith('beta2-g1190/')})
if(@($m.nativeFiles).Count -ne @($m.baseNative).Count+$added.Count -or -not $added.Count){throw 'Only the additive beta-2 package is permitted.'}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/nn-beta2-'+[Guid]::NewGuid().ToString('N'))) $game
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
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;operation='nn-beta2-1.4.55'}
try{
 foreach($f in $files){$target=Inside (Join-Path $game $f.path) $game;[void](New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force);Copy-BrassAtomic $f.source $target}
 if($TestInterrupt){throw 'TEST INTERRUPT: beta-2 files published'}
 Verify (Managed $m.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed');Verify $m.nativeFiles (Join-Path $game 'BrassNative') -Exact
 foreach($entry in @{version=$m.version;files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');nnBeta2Recovery=$backup;
  nnBeta2QualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;nnBeta2CandidateSha256=(Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash;
  nnBeta2Model=$m.beta2_model_id;nnBeta2Checkpoint=$m.beta2_checkpoint_sha256;nnBeta2NativeBuildSha256=$m.beta2_native_build_sha256;
  nnBeta2InstalledSmoke='pending';nnUiModels=@('alpha-test','beta','il','beta-2');nnIndustryFocusChoices=@('neutral');nnNeutralUiInstalledSmoke='historical/beta-2-added';searchMethod='search_v1';v205Enabled=$false}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force}
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_;if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}};throw $failure
}
Write-Output ('Installed 1.4.55: beta-2 g1190. Recovery backup: '+$backup)
